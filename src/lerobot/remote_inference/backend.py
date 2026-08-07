# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
import hashlib
import json
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_STATE

from .schema import (
    EmbodimentManifest,
    ImageFrame,
    ModelManifest,
    PolicyActionChunk,
    PolicyObservation,
    ProtocolValidationError,
)

logger = logging.getLogger(__name__)


class PolicyBackend(abc.ABC):
    """Fixed policy loaded by a remote inference server."""

    @property
    @abc.abstractmethod
    def manifest(self) -> ModelManifest:
        pass

    def warmup(self) -> None:
        """Optional model warmup performed before accepting a robot session."""

        return None

    def prepare(self, embodiment: EmbodimentManifest, task: str) -> None:
        """Optional embodiment-specific warmup performed before a session is opened."""

        return None

    @abc.abstractmethod
    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    def close(self) -> None:
        """Optional backend cleanup."""

        return None


@dataclass(frozen=True)
class DeterministicPolicyBackendConfig:
    model_id: str = "lerobot/deterministic-test-policy"
    action_horizon: int = 30
    state_features: tuple[str, ...] = ()
    action_features: tuple[str, ...] = ()
    camera_keys: tuple[str, ...] = ()


class DeterministicPolicyBackend(PolicyBackend):
    """Small backend used for transport, timing, and fault tests."""

    def __init__(self, config: DeterministicPolicyBackendConfig):
        if not config.action_features or not config.state_features or not config.camera_keys:
            raise ValueError("deterministic backend requires explicit feature and camera names")
        fingerprint = _model_fingerprint(
            model_id=config.model_id,
            revision="test",
            policy_type="deterministic",
            norm_tag="",
            horizon=config.action_horizon,
            state_features=config.state_features,
            action_features=config.action_features,
            camera_keys=config.camera_keys,
        )
        self._manifest = ModelManifest(
            model_id=config.model_id,
            revision="test",
            policy_type="deterministic",
            norm_tag="",
            action_horizon=config.action_horizon,
            action_dim=len(config.action_features),
            state_features=config.state_features,
            action_features=config.action_features,
            camera_keys=config.camera_keys,
            fingerprint=fingerprint,
        )
        self._manifest.validate()

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        action_dim = self._manifest.action_dim
        base = np.zeros(action_dim, dtype=np.float32)
        copy_dim = min(action_dim, observation.state.size)
        base[:copy_dim] = observation.state[:copy_dim]
        actions = np.repeat(base[None, :], self._manifest.action_horizon, axis=0)
        return PolicyActionChunk(
            observation_sequence=observation.sequence,
            first_action_tick=observation.capture_tick,
            actions=actions,
            model_fingerprint=self._manifest.fingerprint,
        )

    def reset(self) -> None:
        return None


@dataclass(frozen=True)
class LeRobotPolicyBackendConfig:
    pretrained_name_or_path: str
    policy_type: str | None = None
    revision: str | None = None
    device: str = "cuda"
    model_dtype: str | None = None
    norm_tag: str | None = None
    inference_action_mode: str | None = None


class LeRobotPolicyBackend(PolicyBackend):
    """Runs a fixed LeRobot policy and all model-side processors."""

    def __init__(self, config: LeRobotPolicyBackendConfig):
        self._config = config
        self._device = torch.device(config.device)
        self._lock = threading.Lock()
        self._prepared_input: tuple[tuple[tuple[str, int, int], ...], str] | None = None
        self._policy_config = self._load_policy_config(config)
        self._dataset_stats = None
        if config.policy_type == "molmoact2":
            self._dataset_stats = self._configure_original_molmoact2(self._policy_config)
        policy_class = get_policy_class(self._policy_config.type)
        if config.policy_type == "molmoact2":
            # Original MolmoAct2 checkpoints are loaded by the wrapper during construction;
            # they are not serialized as a complete LeRobot policy directory.
            self._policy = policy_class(self._policy_config)
        else:
            self._policy = policy_class.from_pretrained(
                config.pretrained_name_or_path,
                config=self._policy_config,
                revision=self._policy_config.pretrained_revision,
            )
        self._policy.to(self._device).eval()
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy_config,
            pretrained_path=None if config.policy_type == "molmoact2" else config.pretrained_name_or_path,
            pretrained_revision=self._policy_config.pretrained_revision,
            dataset_stats=self._dataset_stats,
        )
        self._manifest = self._build_manifest()

    @staticmethod
    def _load_policy_config(config: LeRobotPolicyBackendConfig) -> PreTrainedConfig:
        if config.policy_type is None:
            policy_config = PreTrainedConfig.from_pretrained(
                config.pretrained_name_or_path,
                revision=config.revision,
            )
        elif config.policy_type == "molmoact2":
            from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

            if not str(config.norm_tag or "").strip():
                raise ValueError("original MolmoAct2 checkpoints require policy.norm_tag")
            policy_config = MolmoAct2Config(
                checkpoint_path=config.pretrained_name_or_path,
                checkpoint_revision=config.revision,
                norm_tag=config.norm_tag,
                inference_action_mode=config.inference_action_mode or "continuous",
                model_dtype=config.model_dtype or "bfloat16",
            )
            policy_config.pretrained_path = config.pretrained_name_or_path
            policy_config.pretrained_revision = config.revision
        else:
            raise ValueError(
                "policy_type is only needed for original checkpoints; currently only molmoact2 is supported"
            )
        if config.model_dtype is not None and hasattr(policy_config, "model_dtype"):
            policy_config.model_dtype = config.model_dtype
        if config.norm_tag is not None and hasattr(policy_config, "norm_tag"):
            policy_config.norm_tag = config.norm_tag
        if config.inference_action_mode is not None and hasattr(policy_config, "inference_action_mode"):
            policy_config.inference_action_mode = config.inference_action_mode
        policy_config.device = config.device
        return policy_config

    @staticmethod
    def _configure_original_molmoact2(policy_config: PreTrainedConfig) -> dict:
        """Populate the LeRobot feature schema from the released checkpoint metadata."""
        from lerobot.policies.molmoact2.processor_molmoact2 import _load_hf_norm_stats_for_tag

        dataset_stats, metadata = _load_hf_norm_stats_for_tag(
            policy_config.checkpoint_path,
            revision=policy_config.checkpoint_revision,
            force_download=bool(policy_config.checkpoint_force_download),
            norm_tag=policy_config.norm_tag,
        )
        state_stats = metadata.get("state_stats")
        action_stats = metadata.get("action_stats")
        camera_keys = metadata.get("camera_keys")
        if not isinstance(state_stats, dict) or not isinstance(action_stats, dict):
            raise ValueError("MolmoAct2 normalization metadata is missing state or action statistics")
        state_names = tuple(str(name) for name in state_stats.get("names", ()))
        action_names = tuple(str(name) for name in action_stats.get("names", ()))
        if not state_names or not action_names:
            raise ValueError("MolmoAct2 normalization metadata is missing state or action feature names")
        if not isinstance(camera_keys, list) or not camera_keys:
            raise ValueError("MolmoAct2 normalization metadata is missing camera keys")
        image_keys = tuple(str(key) for key in camera_keys)

        policy_config.dataset_feature_names = {
            OBS_STATE: list(state_names),
            ACTION: list(action_names),
        }
        policy_config.input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(len(state_names),)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for key in image_keys},
        }
        policy_config.output_features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(action_names),))
        }
        policy_config.image_keys = list(image_keys)
        if metadata.get("normalize_gripper") is not None:
            policy_config.normalize_gripper = bool(metadata["normalize_gripper"])
        return dataset_stats

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def warmup(self) -> None:
        """Run one synthetic chunk before the server starts accepting sessions."""
        camera_shapes = tuple(
            (camera_key, *self._configured_image_size(camera_key))
            for camera_key in self._manifest.camera_keys
        )
        self._run_warmup(camera_shapes, task="warm up the policy")

    def prepare(self, embodiment: EmbodimentManifest, task: str) -> None:
        """Warm input-shape-specific paths before the robot is allowed to arm."""
        camera_shapes = tuple((camera.key, camera.height, camera.width) for camera in embodiment.cameras)
        self._run_warmup(camera_shapes, task=task)

    def _run_warmup(self, camera_shapes: tuple[tuple[str, int, int], ...], *, task: str) -> None:
        prepared_input = (camera_shapes, task)
        if prepared_input == self._prepared_input:
            return
        state = np.zeros(len(self._manifest.state_features), dtype=np.float32)
        if self._dataset_stats is not None:
            state_stats = self._dataset_stats.get(OBS_STATE, {})
            center = state_stats.get("q50", state_stats.get("mean"))
            if center is not None and np.asarray(center).shape == state.shape:
                state = np.asarray(center, dtype=np.float32)
        images = tuple(
            ImageFrame(
                key=camera_key,
                array=np.zeros((height, width, 3), dtype=np.uint8),
                capture_monotonic_ns=0,
            )
            for camera_key, height, width in camera_shapes
        )
        observation = PolicyObservation(
            episode_id="server-warmup",
            sequence=0,
            capture_tick=0,
            capture_monotonic_ns=0,
            state=state,
            images=images,
            task=task,
            last_executed_tick=0,
            action_queue_depth=0,
        )
        started = time.perf_counter()
        self.reset()
        self.infer(observation)
        self.reset()
        self._prepared_input = prepared_input
        logger.info("Policy warmup completed in %.2fs", time.perf_counter() - started)

    def _configured_image_size(self, camera_key: str) -> tuple[int, int]:
        feature_key = f"observation.images.{camera_key}"
        for step in self._preprocessor.steps:
            rename_map = getattr(step, "rename_map", None)
            if rename_map and feature_key in rename_map:
                feature_key = rename_map[feature_key]
                break
        feature = self._policy_config.input_features.get(feature_key)
        shape = tuple(feature.shape) if feature is not None else ()
        if len(shape) == 3 and shape[0] in (1, 3):
            height, width = int(shape[1]), int(shape[2])
        elif len(shape) == 3 and shape[2] in (1, 3):
            height, width = int(shape[0]), int(shape[1])
        else:
            height = width = 224
        return height, width

    def _feature_names(self, key: str, fallback_dim: int) -> tuple[str, ...]:
        metadata = getattr(self._policy_config, "dataset_feature_names", {}) or {}
        names = metadata.get(key)
        if names:
            return tuple(names)
        configured = getattr(self._policy_config, "action_feature_names", None)
        uses_padded_state = (
            key == OBS_STATE and int(self._policy_config.input_features[OBS_STATE].shape[-1]) != fallback_dim
        )
        if configured and (key == ACTION or (uses_padded_state and len(configured) == fallback_dim)):
            return tuple(configured)
        return tuple(f"{key}.{index}" for index in range(fallback_dim))

    def _processor_feature_dim(self, key: str, fallback_dim: int) -> int:
        """Return the raw feature width represented by the saved processor state."""
        for step in self._preprocessor.steps:
            step_state = step.state_dict()
            for stat_name in ("q01", "mean", "min", "std"):
                value = step_state.get(f"{key}.{stat_name}")
                if value is not None and value.ndim == 1:
                    return int(value.shape[0])
        return fallback_dim

    def _external_camera_keys(self) -> tuple[str, ...]:
        configured_images = tuple(
            key
            for key, feature in self._policy_config.input_features.items()
            if feature.type is FeatureType.VISUAL
        )
        for step in self._preprocessor.steps:
            rename_map = getattr(step, "rename_map", None)
            if not rename_map:
                continue
            source_keys = tuple(
                source for source, target in rename_map.items() if target in configured_images
            )
            if source_keys:
                return tuple(key.removeprefix("observation.images.") for key in source_keys)
        return tuple(key.removeprefix("observation.images.") for key in configured_images)

    def _build_manifest(self) -> ModelManifest:
        state_feature = self._policy_config.input_features.get(OBS_STATE)
        action_feature = self._policy_config.output_features.get(ACTION)
        if state_feature is None or action_feature is None:
            raise ValueError("policy configuration does not declare state and action features")
        state_dim = self._processor_feature_dim(OBS_STATE, int(state_feature.shape[-1]))
        action_dim = int(action_feature.shape[-1])
        state_features = self._feature_names(OBS_STATE, state_dim)
        action_features = self._feature_names(ACTION, action_dim)
        if len(state_features) != state_dim or len(action_features) != action_dim:
            raise ValueError("policy feature metadata does not match its declared dimensions")
        camera_keys = self._external_camera_keys()
        horizon = int(
            getattr(
                self._policy_config,
                "n_action_steps",
                getattr(self._policy_config, "chunk_size", 1),
            )
        )
        revision = self._config.revision or "main"
        norm_tag = str(getattr(self._policy_config, "norm_tag", "") or "")
        fingerprint = _model_fingerprint(
            model_id=self._config.pretrained_name_or_path,
            revision=revision,
            policy_type=self._policy_config.type,
            norm_tag=norm_tag,
            horizon=horizon,
            state_features=state_features,
            action_features=action_features,
            camera_keys=camera_keys,
        )
        manifest = ModelManifest(
            model_id=self._config.pretrained_name_or_path,
            revision=revision,
            policy_type=self._policy_config.type,
            norm_tag=norm_tag,
            action_horizon=horizon,
            action_dim=action_dim,
            state_features=state_features,
            action_features=action_features,
            camera_keys=camera_keys,
            fingerprint=fingerprint,
        )
        manifest.validate()
        return manifest

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        raw_observation: dict[str, np.ndarray] = {OBS_STATE: observation.state.copy()}
        for frame in observation.images:
            raw_observation[f"observation.images.{frame.key}"] = frame.array.copy()
        batch = prepare_observation_for_inference(
            raw_observation,
            self._device,
            observation.task,
            "remote_robot",
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device.type == "cuda" and getattr(self._policy_config, "model_dtype", "") == "bfloat16"
            else nullcontext()
        )
        with self._lock, torch.inference_mode(), autocast:
            batch = self._preprocessor(batch)
            chunk = self._policy.predict_action_chunk(batch)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            processed = [self._postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])]
            actions = torch.stack(processed, dim=1).squeeze(0).detach().cpu().float().numpy()
        actions = actions[: self._manifest.action_horizon, : self._manifest.action_dim]
        if actions.shape != (self._manifest.action_horizon, self._manifest.action_dim):
            raise ProtocolValidationError(f"policy returned unexpected action shape {actions.shape}")
        return PolicyActionChunk(
            observation_sequence=observation.sequence,
            first_action_tick=observation.capture_tick,
            actions=np.asarray(actions, dtype=np.float32),
            model_fingerprint=self._manifest.fingerprint,
        )

    def reset(self) -> None:
        with self._lock:
            self._policy.reset()
            self._preprocessor.reset()
            self._postprocessor.reset()


def _model_fingerprint(
    *,
    model_id: str,
    revision: str,
    policy_type: str,
    norm_tag: str,
    horizon: int,
    state_features: tuple[str, ...],
    action_features: tuple[str, ...],
    camera_keys: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "model_id": model_id,
            "revision": revision,
            "policy_type": policy_type,
            "norm_tag": norm_tag,
            "horizon": horizon,
            "state_features": state_features,
            "action_features": action_features,
            "camera_keys": camera_keys,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
