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

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.remote_inference.schema import (
    CameraSpec,
    EmbodimentManifest,
    ImageEncoding,
    ImageFrame,
    PolicyActionChunk,
    PolicyObservation,
    ProtocolValidationError,
)
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lerobot.remote_inference.client import RemotePolicyClient


@dataclass(frozen=True)
class RemoteEngineSettings:
    server_address: str
    schema_id: str
    requested_model_id: str
    client_instance_id: str
    connect_timeout_s: float
    inference_timeout_s: float
    max_message_bytes: int
    jpeg_quality: int
    tls_root_cert_path: str | None
    tls_client_cert_path: str | None
    tls_client_key_path: str | None
    tls_server_name_override: str | None
    execution_horizon: int
    image_encoding: ImageEncoding
    camera_calibration_sha256: dict[str, str]


@dataclass(frozen=True)
class _ObservationSnapshot:
    sequence: int
    capture_tick: int
    capture_monotonic_ns: int
    values: dict[str, Any]
    previous_values: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PendingChunk:
    chunk: PolicyActionChunk
    round_trip_ns: int


class RemoteInferenceEngine(InferenceEngine):
    """Action-chunk inference on a fixed remote policy server."""

    def __init__(
        self,
        *,
        settings: RemoteEngineSettings,
        robot_wrapper: ThreadSafeRobot,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        if settings.execution_horizon <= 0:
            raise ValueError("remote execution horizon must be positive")
        self._settings = settings
        self._robot = robot_wrapper
        self._dataset_features = dataset_features
        self._ordered_action_keys = tuple(ordered_action_keys)
        self._task = task
        self._fps = fps
        self._global_shutdown_event = shutdown_event

        self._manifest = self._build_embodiment_manifest()
        self._client: RemotePolicyClient | None = None
        self._action_queue: deque[torch.Tensor] = deque()
        self._pending_chunk: _PendingChunk | None = None
        self._latest_observation: _ObservationSnapshot | None = None
        self._sequence = 0
        self._last_submitted_sequence = -1
        self._current_tick = 0
        self._last_executed_tick = 0
        self._model_action_horizon: int | None = None
        self._active_chunk_sequence: int | None = None
        self._active_chunk_actions_sent = 0
        self._switch_tick: int | None = None
        self._request_tick: int | None = None
        self._request_in_flight = False
        self._round_trip_samples_ns: deque[int] = deque(maxlen=32)
        self._last_chunk_log_ns = 0
        self._lock = threading.Lock()
        self._observation_ready = threading.Event()
        self._policy_active = threading.Event()
        self._stop_event = threading.Event()
        self._failed = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def _build_embodiment_manifest(self) -> EmbodimentManifest:
        state_names = tuple(self._dataset_features[OBS_STATE]["names"])
        action_names = tuple(self._dataset_features[ACTION]["names"])
        if action_names != self._ordered_action_keys:
            raise ValueError("remote action key ordering must match dataset action features")
        cameras = []
        for key, shape in self._robot.observation_features.items():
            if not isinstance(shape, tuple):
                continue
            if len(shape) != 3 or shape[2] != 3:
                raise ValueError(f"remote inference only supports RGB cameras, got {key}: {shape}")
            cameras.append(
                CameraSpec(
                    key=key,
                    width=shape[1],
                    height=shape[0],
                    channels=shape[2],
                    encoding=self._settings.image_encoding,
                    calibration_sha256=self._settings.camera_calibration_sha256.get(key, ""),
                )
            )
        manifest = EmbodimentManifest(
            schema_id=self._settings.schema_id,
            robot_id=getattr(self._robot.inner, "id", None) or self._settings.client_instance_id,
            robot_type=self._robot.robot_type,
            control_hz=self._fps,
            state_features=state_names,
            action_features=action_names,
            cameras=tuple(cameras),
        ).signed()
        manifest.validate()
        return manifest

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def action_queue_depth(self) -> int:
        with self._lock:
            return len(self._action_queue)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        from lerobot.remote_inference.client import RemotePolicyClient, RemotePolicyClientConfig

        self._client = RemotePolicyClient(
            RemotePolicyClientConfig(
                server_address=self._settings.server_address,
                connect_timeout_s=self._settings.connect_timeout_s,
                inference_timeout_s=self._settings.inference_timeout_s,
                max_message_bytes=self._settings.max_message_bytes,
                jpeg_quality=self._settings.jpeg_quality,
                tls_root_cert_path=self._settings.tls_root_cert_path,
                tls_client_cert_path=self._settings.tls_client_cert_path,
                tls_client_key_path=self._settings.tls_client_key_path,
                tls_server_name_override=self._settings.tls_server_name_override,
            )
        )
        try:
            session = self._client.connect(
                self._manifest,
                task=self._task,
                requested_model_id=self._settings.requested_model_id,
                client_instance_id=self._settings.client_instance_id,
            )
            if self._settings.execution_horizon > session.model.action_horizon:
                raise ValueError(
                    "remote execution_horizon "
                    f"({self._settings.execution_horizon}) exceeds model action horizon "
                    f"({session.model.action_horizon})"
                )
            self._model_action_horizon = session.model.action_horizon
        except Exception:
            self._client.close()
            self._client = None
            raise
        self._stop_event.clear()
        self._failed.clear()
        self._ready.set()
        self._thread = threading.Thread(target=self._inference_loop, daemon=True, name="RemoteInference")
        self._thread.start()
        logger.info(
            "Remote inference connected to %s (prediction_horizon=%d, execution_horizon=%d)",
            self._settings.server_address,
            self._model_action_horizon,
            self._settings.execution_horizon,
        )

    def stop(self) -> None:
        self._policy_active.clear()
        self._stop_event.set()
        self._observation_ready.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(2.0, self._settings.inference_timeout_s + 0.5))
        self._thread = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self._ready.clear()
        with self._lock:
            self._clear_scheduler_locked()

    def reset(self) -> None:
        with self._lock:
            self._clear_scheduler_locked()
            self._latest_observation = None
            self._sequence = 0
            self._last_submitted_sequence = -1
            self._current_tick = 0
            self._last_executed_tick = 0
            self._request_in_flight = False
        if self._client is not None and self._client.connected:
            self._client.reset()

    def pause(self) -> None:
        self._policy_active.clear()
        with self._lock:
            self._clear_scheduler_locked()

    def resume(self) -> None:
        if not self.failed:
            self._policy_active.set()
            self._observation_ready.set()

    def notify_observation(self, obs: dict) -> None:
        if self.failed or not self._policy_active.is_set():
            return
        now_ns = time.monotonic_ns()
        with self._lock:
            snapshot = _ObservationSnapshot(
                sequence=self._sequence,
                capture_tick=self._current_tick,
                capture_monotonic_ns=now_ns,
                values={key: _copy_observation_value(value) for key, value in obs.items()},
                previous_values=self._previous_values_for_snapshot(),
            )
            self._sequence += 1
            self._latest_observation = snapshot
            should_request = self._should_request_locked()
        if should_request:
            self._observation_ready.set()

    def _previous_values_for_snapshot(self) -> dict[str, Any] | None:
        """Return optional history needed by a specialized observation encoder."""

        return None

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        del obs_frame
        with self._lock:
            self._activate_pending_if_due_locked()
            action = self._action_queue.popleft() if self._action_queue else None
            should_request = self._should_request_locked()
        if should_request:
            self._observation_ready.set()
        return action

    def notify_action_sent(self) -> None:
        with self._lock:
            if self._active_chunk_sequence is not None:
                self._active_chunk_actions_sent += 1
            self._last_executed_tick = self._current_tick
            self._current_tick += 1
            self._activate_pending_if_due_locked()
            should_request = self._should_request_locked()
        if should_request:
            self._observation_ready.set()

    def _clear_scheduler_locked(self) -> None:
        self._action_queue.clear()
        self._pending_chunk = None
        self._active_chunk_sequence = None
        self._active_chunk_actions_sent = 0
        self._switch_tick = None
        self._request_tick = None

    def _latency_budget_steps_locked(self) -> int:
        if not self._round_trip_samples_ns:
            return 1
        round_trip_ns = float(np.percentile(np.asarray(self._round_trip_samples_ns), 95))
        return max(1, math.ceil(round_trip_ns / 1e9 * self._fps) + 2)

    def _should_request_locked(self) -> bool:
        snapshot = self._latest_observation
        if (
            snapshot is None
            or snapshot.sequence <= self._last_submitted_sequence
            or self._request_in_flight
            or self._pending_chunk is not None
        ):
            return False
        if self._active_chunk_sequence is None or not self._action_queue:
            return True
        return self._request_tick is not None and self._current_tick >= self._request_tick

    def _activate_pending_if_due_locked(self) -> bool:
        if self._pending_chunk is None:
            return False
        if (
            self._active_chunk_sequence is not None
            and self._action_queue
            and self._switch_tick is not None
            and self._current_tick < self._switch_tick
        ):
            return False
        return self._activate_pending_chunk_locked()

    def _activate_pending_chunk_locked(self) -> bool:
        pending = self._pending_chunk
        if pending is None:
            return False
        self._pending_chunk = None
        chunk = pending.chunk
        elapsed_steps = max(0, self._current_tick - chunk.first_action_tick)
        if elapsed_steps >= chunk.actions.shape[0]:
            logger.warning(
                "Discarding fully stale action chunk for observation %d "
                "(round_trip=%.1fms, server=%.1fms, elapsed_steps=%d)",
                chunk.observation_sequence,
                pending.round_trip_ns / 1e6,
                chunk.server_compute_ns / 1e6,
                elapsed_steps,
            )
            return False

        previous_actions_sent = self._active_chunk_actions_sent
        previous_switch_tick = self._switch_tick
        future = self._future_actions_for_execution(chunk, elapsed_steps)
        self._action_queue.clear()
        self._action_queue.extend(torch.from_numpy(action.copy()) for action in future)
        self._active_chunk_sequence = chunk.observation_sequence
        self._active_chunk_actions_sent = 0

        committed_actions = self._commit_length_for_actions(len(self._action_queue))
        self._switch_tick = self._current_tick + committed_actions
        latency_budget_steps = self._latency_budget_steps_locked()
        self._request_tick = max(self._current_tick, self._switch_tick - latency_budget_steps)
        transition_late_steps = (
            max(0, self._current_tick - previous_switch_tick) if previous_switch_tick is not None else 0
        )

        now_ns = time.monotonic_ns()
        should_log_info = now_ns - self._last_chunk_log_ns >= 5_000_000_000
        log = logger.info if should_log_info else logger.debug
        log(
            "Remote chunk %d activated (round_trip=%.1fms, server=%.1fms, discarded=%d, "
            "queued=%d, commit=%d, request_tick=%d, previous_executed=%d, late=%d)",
            chunk.observation_sequence,
            pending.round_trip_ns / 1e6,
            chunk.server_compute_ns / 1e6,
            elapsed_steps,
            len(self._action_queue),
            committed_actions,
            self._request_tick,
            previous_actions_sent,
            transition_late_steps,
        )
        if should_log_info:
            self._last_chunk_log_ns = now_ns
        if committed_actions < self._settings.execution_horizon:
            logger.warning(
                "Remote chunk %d has only %d latency-aligned actions available for execution_horizon=%d",
                chunk.observation_sequence,
                committed_actions,
                self._settings.execution_horizon,
            )
        return True

    def _commit_length_for_actions(self, action_count: int) -> int:
        """Return the number of queued control ticks committed at activation."""

        return min(self._settings.execution_horizon, action_count)

    def _future_actions_for_execution(
        self,
        chunk: PolicyActionChunk,
        elapsed_steps: int,
    ) -> np.ndarray:
        """Select executable rows after latency alignment.

        Specialized embodiment adapters may impose a hard row horizon without
        changing the generic remote scheduler.
        """

        return chunk.actions[elapsed_steps:]

    def _prepare_chunk_for_execution(
        self,
        chunk: PolicyActionChunk,
        snapshot: _ObservationSnapshot,
    ) -> PolicyActionChunk:
        """Translate a validated model chunk into hardware action coordinates."""

        del snapshot
        return chunk

    def _make_policy_observation(
        self,
        snapshot: _ObservationSnapshot,
        queue_depth: int,
    ) -> PolicyObservation:
        frame = build_dataset_frame(self._dataset_features, snapshot.values, prefix=OBS_STR)
        images = tuple(
            ImageFrame(
                key=camera.key,
                array=np.asarray(frame[f"observation.images.{camera.key}"], dtype=np.uint8),
                capture_monotonic_ns=snapshot.capture_monotonic_ns,
            )
            for camera in self._manifest.cameras
        )
        return PolicyObservation(
            episode_id="rollout",
            sequence=snapshot.sequence,
            capture_tick=snapshot.capture_tick,
            capture_monotonic_ns=snapshot.capture_monotonic_ns,
            state=np.asarray(frame[OBS_STATE], dtype=np.float32),
            images=images,
            task=self._task,
            last_executed_tick=self._last_executed_tick,
            action_queue_depth=queue_depth,
        )

    def _inference_loop(self) -> None:
        from lerobot.remote_inference.client import RemotePolicyError

        assert self._client is not None
        while not self._stop_event.is_set():
            self._observation_ready.wait(timeout=0.1)
            self._observation_ready.clear()
            if self._stop_event.is_set() or not self._policy_active.is_set():
                continue
            with self._lock:
                if not self._should_request_locked():
                    continue
                snapshot = self._latest_observation
                assert snapshot is not None
                queue_depth = len(self._action_queue)
                self._last_submitted_sequence = snapshot.sequence
                self._request_in_flight = True
            try:
                observation = self._make_policy_observation(snapshot, queue_depth)
                request_started_ns = time.perf_counter_ns()
                chunk = self._client.infer(observation)
                chunk = self._prepare_chunk_for_execution(chunk, snapshot)
                round_trip_ns = time.perf_counter_ns() - request_started_ns
                if self._stop_event.is_set():
                    return
                with self._lock:
                    self._request_in_flight = False
                    if not self._policy_active.is_set():
                        continue
                    self._round_trip_samples_ns.append(round_trip_ns)
                    elapsed_steps = max(0, self._current_tick - chunk.first_action_tick)
                    if elapsed_steps >= chunk.actions.shape[0]:
                        logger.warning(
                            "Discarding fully stale action chunk for observation %d "
                            "(round_trip=%.1fms, server=%.1fms, elapsed_steps=%d)",
                            chunk.observation_sequence,
                            round_trip_ns / 1e6,
                            chunk.server_compute_ns / 1e6,
                            elapsed_steps,
                        )
                        if self._should_request_locked():
                            self._observation_ready.set()
                        continue
                    self._pending_chunk = _PendingChunk(chunk=chunk, round_trip_ns=round_trip_ns)
                    if self._active_chunk_sequence is None:
                        self._activate_pending_chunk_locked()
                    else:
                        logger.debug(
                            "Remote chunk %d ready for execution boundary at tick %s "
                            "(round_trip=%.1fms, currently_executed=%d)",
                            chunk.observation_sequence,
                            self._switch_tick,
                            round_trip_ns / 1e6,
                            self._active_chunk_actions_sent,
                        )
                    should_request = self._should_request_locked()
                if should_request:
                    self._observation_ready.set()
            except (RemotePolicyError, ProtocolValidationError, KeyError, ValueError) as exc:
                if self._stop_event.is_set():
                    return
                logger.error("Remote inference stopped: %s", exc)
                with self._lock:
                    self._request_in_flight = False
                    self._clear_scheduler_locked()
                self._failed.set()
                self._policy_active.clear()
                if self._global_shutdown_event is not None:
                    self._global_shutdown_event.set()
                return
            except Exception:
                if self._stop_event.is_set():
                    return
                logger.exception("Remote inference stopped after an unexpected error")
                with self._lock:
                    self._request_in_flight = False
                    self._clear_scheduler_locked()
                self._failed.set()
                self._policy_active.clear()
                if self._global_shutdown_event is not None:
                    self._global_shutdown_event.set()
                return


def parse_image_encoding(value: str) -> ImageEncoding:
    try:
        return ImageEncoding[value.strip().upper()]
    except KeyError as exc:
        choices = ", ".join(encoding.name.lower() for encoding in ImageEncoding)
        raise ValueError(f"unknown remote image encoding {value!r}; choose one of {choices}") from exc


def default_client_instance_id(robot_type: str) -> str:
    return f"{robot_type}-{uuid.uuid4().hex}"


def _copy_observation_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value
