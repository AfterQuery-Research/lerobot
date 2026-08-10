# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Inference engine configs and factory.

Selection is explicit via ``--inference.type=sync|rtc|remote``.  Adding a new
backend requires registering its config subclass and dispatching it in
:func:`create_inference_engine`.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from threading import Event

import draccus

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine
from .rtc import RTCInferenceEngine
from .sync import SyncInferenceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class InferenceEngineConfig(draccus.ChoiceRegistry, abc.ABC):
    """Abstract base for inference backend configuration.

    Use ``--inference.type=<name>`` on the CLI to select a backend.
    """

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@InferenceEngineConfig.register_subclass("sync")
@dataclass
class SyncInferenceConfig(InferenceEngineConfig):
    """Inline synchronous inference (one policy call per control tick)."""


@InferenceEngineConfig.register_subclass("rtc")
@dataclass
class RTCInferenceConfig(InferenceEngineConfig):
    """Real-Time Chunking: async policy inference in a background thread."""

    # Eagerly constructed so draccus exposes nested fields directly on the CLI
    # (e.g. ``--inference.rtc.execution_horizon=...``).
    rtc: RTCConfig = field(default_factory=RTCConfig)
    queue_threshold: int = 30


@InferenceEngineConfig.register_subclass("remote")
@dataclass
class RemoteInferenceConfig(InferenceEngineConfig):
    """Inference served by a fixed policy process on another machine."""

    server_address: str = "127.0.0.1:8081"
    schema_id: str = "lerobot-remote-v1"
    requested_model_id: str = ""
    client_instance_id: str = ""
    connect_timeout_s: float = 10.0
    inference_timeout_s: float = 10.0
    max_message_bytes: int = 16 * 1024 * 1024
    jpeg_quality: int = 95
    image_encoding: str = "jpeg"
    execution_horizon: int = 15
    tls_root_cert_path: str | None = None
    tls_client_cert_path: str | None = None
    tls_client_key_path: str | None = None
    tls_server_name_override: str | None = None
    camera_calibration_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.server_address.strip():
            raise ValueError("remote server_address must not be empty")
        if not self.schema_id.strip():
            raise ValueError("remote schema_id must not be empty")
        if self.connect_timeout_s <= 0 or self.inference_timeout_s <= 0:
            raise ValueError("remote timeouts must be positive")
        if self.max_message_bytes <= 0:
            raise ValueError("remote max_message_bytes must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("remote jpeg_quality must be between 1 and 100")
        if self.execution_horizon <= 0:
            raise ValueError("remote execution_horizon must be positive")
        if self.image_encoding.lower() not in {"raw_rgb", "png", "jpeg"}:
            raise ValueError("remote image_encoding must be raw_rgb, png, or jpeg")
        if bool(self.tls_client_cert_path) != bool(self.tls_client_key_path):
            raise ValueError("remote TLS client certificate and key must be configured together")
        if self.tls_client_cert_path and self.tls_root_cert_path is None:
            raise ValueError("remote TLS client certificates require a TLS root certificate")
        if self.tls_server_name_override and self.tls_root_cert_path is None:
            raise ValueError("remote TLS server name override requires a TLS root certificate")


@InferenceEngineConfig.register_subclass("remote_yam_currentrel_r6d")
@dataclass
class YamCurrentRelativeR6DRemoteInferenceConfig(RemoteInferenceConfig):
    """Remote inference for the 20D dual-UMI current-relative Rotation6D contract."""

    urdf_path: str = ""
    target_frame_name: str = "gripper"
    left_camera_key: str = "left"
    right_camera_key: str = "right"
    image_width: int = 800
    image_height: int = 600
    ik_iterations: int = 10
    max_position_residual_m: float = 2e-3
    max_orientation_residual_rad: float = 0.017453292519943295
    max_joint_delta_rad: float = 0.02
    max_gripper_delta: float = 0.05
    max_dispatches_per_waypoint: int = 4
    max_progress_hold_steps: int = 90
    progress_target_tolerance_rad: float = 2e-3
    min_progress_rad: float = 1e-4
    flange_to_tcp_z_m: float = -0.056683
    gripper_a_left: float = 2.2559
    gripper_b_left: float = -1.2290
    gripper_a_right: float = 2.7300
    gripper_b_right: float = -1.5781

    def __post_init__(self) -> None:
        super().__post_init__()
        from .yam_current_relative_r6d import YamCurrentRelativeR6DSettings

        YamCurrentRelativeR6DSettings(
            urdf_path=self.urdf_path,
            target_frame_name=self.target_frame_name,
            left_camera_key=self.left_camera_key,
            right_camera_key=self.right_camera_key,
            image_width=self.image_width,
            image_height=self.image_height,
            ik_iterations=self.ik_iterations,
            max_position_residual_m=self.max_position_residual_m,
            max_orientation_residual_rad=self.max_orientation_residual_rad,
            max_joint_delta_rad=self.max_joint_delta_rad,
            max_gripper_delta=self.max_gripper_delta,
            max_dispatches_per_waypoint=self.max_dispatches_per_waypoint,
            max_progress_hold_steps=self.max_progress_hold_steps,
            progress_target_tolerance_rad=self.progress_target_tolerance_rad,
            min_progress_rad=self.min_progress_rad,
            flange_to_tcp_z_m=self.flange_to_tcp_z_m,
            gripper_a_left=self.gripper_a_left,
            gripper_b_left=self.gripper_b_left,
            gripper_a_right=self.gripper_a_right,
            gripper_b_right=self.gripper_b_right,
        ).validate()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_inference_engine(
    config: InferenceEngineConfig,
    *,
    policy: PreTrainedPolicy | None,
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    robot_wrapper: ThreadSafeRobot,
    hw_features: dict,
    dataset_features: dict,
    ordered_action_keys: list[str],
    task: str,
    fps: float,
    device: str | None,
    use_torch_compile: bool = False,
    compile_warmup_inferences: int = 2,
    shutdown_event: Event | None = None,
) -> InferenceEngine:
    """Instantiate the appropriate inference engine from a config object."""
    logger.info("Creating inference engine: %s", config.type)
    if isinstance(config, YamCurrentRelativeR6DRemoteInferenceConfig):
        from .remote import RemoteEngineSettings, default_client_instance_id, parse_image_encoding
        from .yam_current_relative_r6d import (
            YamCurrentRelativeR6DRemoteInferenceEngine,
            YamCurrentRelativeR6DSettings,
        )

        client_instance_id = config.client_instance_id or default_client_instance_id(robot_wrapper.robot_type)
        return YamCurrentRelativeR6DRemoteInferenceEngine(
            settings=RemoteEngineSettings(
                server_address=config.server_address,
                schema_id=config.schema_id,
                requested_model_id=config.requested_model_id,
                client_instance_id=client_instance_id,
                connect_timeout_s=config.connect_timeout_s,
                inference_timeout_s=config.inference_timeout_s,
                max_message_bytes=config.max_message_bytes,
                jpeg_quality=config.jpeg_quality,
                tls_root_cert_path=config.tls_root_cert_path,
                tls_client_cert_path=config.tls_client_cert_path,
                tls_client_key_path=config.tls_client_key_path,
                tls_server_name_override=config.tls_server_name_override,
                execution_horizon=config.execution_horizon,
                image_encoding=parse_image_encoding(config.image_encoding),
                camera_calibration_sha256=dict(config.camera_calibration_sha256),
            ),
            yam_settings=YamCurrentRelativeR6DSettings(
                urdf_path=config.urdf_path,
                target_frame_name=config.target_frame_name,
                left_camera_key=config.left_camera_key,
                right_camera_key=config.right_camera_key,
                image_width=config.image_width,
                image_height=config.image_height,
                ik_iterations=config.ik_iterations,
                max_position_residual_m=config.max_position_residual_m,
                max_orientation_residual_rad=config.max_orientation_residual_rad,
                max_joint_delta_rad=config.max_joint_delta_rad,
                max_gripper_delta=config.max_gripper_delta,
                max_dispatches_per_waypoint=config.max_dispatches_per_waypoint,
                max_progress_hold_steps=config.max_progress_hold_steps,
                progress_target_tolerance_rad=config.progress_target_tolerance_rad,
                min_progress_rad=config.min_progress_rad,
                flange_to_tcp_z_m=config.flange_to_tcp_z_m,
                gripper_a_left=config.gripper_a_left,
                gripper_b_left=config.gripper_b_left,
                gripper_a_right=config.gripper_a_right,
                gripper_b_right=config.gripper_b_right,
            ),
            robot_wrapper=robot_wrapper,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            fps=fps,
            shutdown_event=shutdown_event,
        )
    if isinstance(config, RemoteInferenceConfig):
        from .remote import (
            RemoteEngineSettings,
            RemoteInferenceEngine,
            default_client_instance_id,
            parse_image_encoding,
        )

        client_instance_id = config.client_instance_id or default_client_instance_id(robot_wrapper.robot_type)
        return RemoteInferenceEngine(
            settings=RemoteEngineSettings(
                server_address=config.server_address,
                schema_id=config.schema_id,
                requested_model_id=config.requested_model_id,
                client_instance_id=client_instance_id,
                connect_timeout_s=config.connect_timeout_s,
                inference_timeout_s=config.inference_timeout_s,
                max_message_bytes=config.max_message_bytes,
                jpeg_quality=config.jpeg_quality,
                tls_root_cert_path=config.tls_root_cert_path,
                tls_client_cert_path=config.tls_client_cert_path,
                tls_client_key_path=config.tls_client_key_path,
                tls_server_name_override=config.tls_server_name_override,
                execution_horizon=config.execution_horizon,
                image_encoding=parse_image_encoding(config.image_encoding),
                camera_calibration_sha256=dict(config.camera_calibration_sha256),
            ),
            robot_wrapper=robot_wrapper,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            fps=fps,
            shutdown_event=shutdown_event,
        )

    if policy is None or preprocessor is None or postprocessor is None:
        raise ValueError(f"{config.type} inference requires a local policy and processors")
    if isinstance(config, SyncInferenceConfig):
        return SyncInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            device=device,
            robot_type=robot_wrapper.robot_type,
        )
    if isinstance(config, RTCInferenceConfig):
        return RTCInferenceEngine(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robot_wrapper=robot_wrapper,
            rtc_config=config.rtc,
            hw_features=hw_features,
            task=task,
            fps=fps,
            device=device,
            use_torch_compile=use_torch_compile,
            compile_warmup_inferences=compile_warmup_inferences,
            rtc_queue_threshold=config.queue_threshold,
            shutdown_event=shutdown_event,
        )
    raise ValueError(f"Unknown inference engine type: {type(config).__name__}")
