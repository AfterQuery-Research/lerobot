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
from dataclasses import dataclass, replace

import numpy as np

from lerobot.remote_inference.schema import (
    CameraSpec,
    EmbodimentManifest,
    ImageFrame,
    PolicyActionChunk,
    PolicyObservation,
)
from lerobot.remote_inference.yam_current_relative_r6d import (
    YAM_CURRENTREL_ACTION_NAMES,
    YAM_CURRENTREL_CAMERA_KEYS,
    YAM_CURRENTREL_SCHEMA_ID,
    YAM_CURRENTREL_STATE_NAMES,
    YamCurrentRelativeQuery,
    YamCurrentRelativeR6DAdapter,
    YamJointProgressWatchdog,
    prepare_policy_image,
    rate_limit_action_chunk,
)
from lerobot.remote_inference.yam_umi_ee_bridge import (
    YAM_FLANGE_TO_FINGERTIP,
    YAM_SCALAR_KEYS,
    GripperMap,
)

from .remote import (
    RemoteEngineSettings,
    RemoteInferenceEngine,
    _ObservationSnapshot,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YamCurrentRelativeR6DSettings:
    urdf_path: str = ""
    target_frame_name: str = "gripper"
    left_camera_key: str = "left"
    right_camera_key: str = "right"
    image_width: int = 800
    image_height: int = 600
    ik_iterations: int = 10
    max_position_residual_m: float = 2e-3
    max_orientation_residual_rad: float = float(np.deg2rad(1.0))
    max_joint_delta_rad: float = 0.02
    max_gripper_delta: float = 0.05
    max_dispatches_per_waypoint: int = 16
    max_progress_hold_steps: int = 90
    progress_target_tolerance_rad: float = 2e-3
    min_progress_rad: float = 1e-4
    flange_to_tcp_z_m: float = float(YAM_FLANGE_TO_FINGERTIP[2, 3])
    gripper_a_left: float = 2.2559
    gripper_b_left: float = -1.2290
    gripper_a_right: float = 2.7300
    gripper_b_right: float = -1.5781

    def validate(self) -> None:
        if not self.target_frame_name.strip():
            raise ValueError("target_frame_name must not be empty")
        if not self.left_camera_key.strip() or not self.right_camera_key.strip():
            raise ValueError("left and right camera keys must not be empty")
        if self.left_camera_key == self.right_camera_key:
            raise ValueError("left and right camera keys must be different")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("policy image dimensions must be positive")
        if self.ik_iterations <= 0:
            raise ValueError("ik_iterations must be positive")
        if self.max_position_residual_m <= 0 or self.max_orientation_residual_rad <= 0:
            raise ValueError("IK residual limits must be positive")
        if self.max_joint_delta_rad <= 0 or self.max_gripper_delta <= 0:
            raise ValueError("current-relative action rate limits must be positive")
        if self.max_dispatches_per_waypoint <= 0:
            raise ValueError("max_dispatches_per_waypoint must be positive")
        if self.max_progress_hold_steps <= 0:
            raise ValueError("max_progress_hold_steps must be positive")
        if self.progress_target_tolerance_rad <= 0 or self.min_progress_rad < 0:
            raise ValueError("current-relative progress tolerances are invalid")
        numeric = (
            self.flange_to_tcp_z_m,
            self.gripper_a_left,
            self.gripper_b_left,
            self.gripper_a_right,
            self.gripper_b_right,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("YAM current-relative settings must be finite")
        if self.gripper_a_left == 0 or self.gripper_a_right == 0:
            raise ValueError("gripper scale coefficients must be non-zero")


class YamCurrentRelativeR6DRemoteInferenceEngine(RemoteInferenceEngine):
    """Remote engine that presents a 20D UMI policy as 14D YAM joint control."""

    def __init__(
        self,
        *,
        settings: RemoteEngineSettings,
        yam_settings: YamCurrentRelativeR6DSettings,
        robot_wrapper,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        fps: float,
        shutdown_event=None,
        adapter: YamCurrentRelativeR6DAdapter | None = None,
    ) -> None:
        yam_settings.validate()
        if tuple(ordered_action_keys) != YAM_SCALAR_KEYS:
            raise ValueError("current-relative YAM inference requires the canonical 14D YAM action order")
        if not np.isclose(fps, 30.0, atol=1e-6, rtol=0.0):
            raise ValueError("the current-relative onset-v3 policy must run at --fps=30")
        if settings.execution_horizon > 15:
            raise ValueError("current-relative onset-v3 execution_horizon cannot exceed 15")
        self._yam_settings = yam_settings
        self._queries: dict[int, YamCurrentRelativeQuery] = {}
        self._progress_watchdog = YamJointProgressWatchdog(
            max_hold_steps=yam_settings.max_progress_hold_steps,
            target_tolerance_rad=yam_settings.progress_target_tolerance_rad,
            min_progress_rad=yam_settings.min_progress_rad,
        )
        self._candidate_action: np.ndarray | None = None
        self._last_dispatched_action: np.ndarray | None = None
        self._progress_target: np.ndarray | None = None
        self._progress_measured_before: np.ndarray | None = None
        if adapter is None:
            flange_to_tcp = np.eye(4, dtype=np.float64)
            flange_to_tcp[2, 3] = yam_settings.flange_to_tcp_z_m
            adapter = YamCurrentRelativeR6DAdapter.from_urdf(
                yam_settings.urdf_path or None,
                target_frame_name=yam_settings.target_frame_name,
                ik_iterations=yam_settings.ik_iterations,
                max_position_residual_m=yam_settings.max_position_residual_m,
                max_orientation_residual_rad=yam_settings.max_orientation_residual_rad,
                flange_to_tcp=flange_to_tcp,
                gripper=GripperMap(
                    a_left=yam_settings.gripper_a_left,
                    b_left=yam_settings.gripper_b_left,
                    a_right=yam_settings.gripper_a_right,
                    b_right=yam_settings.gripper_b_right,
                ),
            )
        self._adapter = adapter
        super().__init__(
            settings=settings,
            robot_wrapper=robot_wrapper,
            dataset_features=dataset_features,
            ordered_action_keys=ordered_action_keys,
            task=task,
            fps=fps,
            shutdown_event=shutdown_event,
        )

    def _build_embodiment_manifest(self) -> EmbodimentManifest:
        cameras = tuple(
            CameraSpec(
                key=key,
                width=self._yam_settings.image_width,
                height=self._yam_settings.image_height,
                encoding=self._settings.image_encoding,
                calibration_sha256=self._settings.camera_calibration_sha256.get(key, ""),
            )
            for key in YAM_CURRENTREL_CAMERA_KEYS
        )
        manifest = EmbodimentManifest(
            schema_id=YAM_CURRENTREL_SCHEMA_ID,
            robot_id=getattr(self._robot.inner, "id", None) or self._settings.client_instance_id,
            robot_type="bi_yam_currentrel_r6d",
            control_hz=self._fps,
            state_features=YAM_CURRENTREL_STATE_NAMES,
            action_features=YAM_CURRENTREL_ACTION_NAMES,
            cameras=cameras,
        ).signed()
        manifest.validate()
        return manifest

    def _previous_values_for_snapshot(self) -> dict[str, float] | None:
        if self._latest_observation is None:
            return None
        return {key: float(self._latest_observation.values[key]) for key in YAM_SCALAR_KEYS}

    def _make_policy_observation(
        self,
        snapshot: _ObservationSnapshot,
        queue_depth: int,
    ) -> PolicyObservation:
        query = self._adapter.build_query(snapshot.values, snapshot.previous_values)
        self._queries[snapshot.sequence] = query
        for old_sequence in tuple(self._queries):
            if old_sequence < snapshot.sequence - 4:
                self._queries.pop(old_sequence, None)

        physical_camera_keys = (
            self._yam_settings.left_camera_key,
            self._yam_settings.right_camera_key,
        )
        missing = [key for key in physical_camera_keys if key not in snapshot.values]
        if missing:
            raise KeyError(f"observation is missing configured policy cameras: {missing}")
        images = tuple(
            ImageFrame(
                key=logical_key,
                array=prepare_policy_image(
                    snapshot.values[physical_key],
                    width=self._yam_settings.image_width,
                    height=self._yam_settings.image_height,
                ),
                capture_monotonic_ns=snapshot.capture_monotonic_ns,
            )
            for logical_key, physical_key in zip(
                YAM_CURRENTREL_CAMERA_KEYS,
                physical_camera_keys,
                strict=True,
            )
        )
        return PolicyObservation(
            episode_id="rollout",
            sequence=snapshot.sequence,
            capture_tick=snapshot.capture_tick,
            capture_monotonic_ns=snapshot.capture_monotonic_ns,
            state=query.state,
            images=images,
            task=self._task,
            last_executed_tick=self._last_executed_tick,
            action_queue_depth=queue_depth,
        )

    def _prepare_chunk_for_execution(
        self,
        chunk: PolicyActionChunk,
        snapshot: _ObservationSnapshot,
    ) -> PolicyActionChunk:
        query = self._queries.pop(chunk.observation_sequence, None)
        if query is None or chunk.observation_sequence != snapshot.sequence:
            raise ValueError(f"missing query anchor for policy observation {chunk.observation_sequence}")
        model_actions = chunk.actions
        decoded = self._adapter.decode_action_chunk(model_actions, query)
        initial_joints = np.concatenate((query.left_joints, query.right_joints))
        decoded_joints = np.concatenate((decoded.actions[:, :6], decoded.actions[:, 7:13]), axis=1)
        model_joint_steps = np.diff(np.vstack((initial_joints, decoded_joints)), axis=0)
        valid_model_actions = model_actions[: decoded.actions.shape[0]]
        model_translations = np.concatenate(
            (valid_model_actions[:, :3], valid_model_actions[:, 10:13]), axis=1
        )
        logger.info(
            "Decoded current-relative chunk %d: model_rows=%d/%d max_translation=%.4fm "
            "max_model_joint_step=%.4frad "
            "max_ik_position_residual=%.3fmm "
            "max_ik_orientation_residual=%.3fdeg",
            chunk.observation_sequence,
            decoded.actions.shape[0],
            chunk.actions.shape[0],
            float(np.abs(model_translations).max()),
            float(np.abs(model_joint_steps).max()),
            float(decoded.position_residual_m.max() * 1e3),
            float(np.rad2deg(decoded.orientation_residual_rad.max())),
        )
        return replace(chunk, actions=np.ascontiguousarray(decoded.actions, dtype=np.float32))

    def _future_actions_for_execution(
        self,
        chunk: PolicyActionChunk,
        elapsed_steps: int,
    ) -> np.ndarray:
        """Latency-align model rows before expanding them into bounded dispatches."""

        model_waypoints = chunk.actions[elapsed_steps : elapsed_steps + self._settings.execution_horizon]
        if not len(model_waypoints):
            return model_waypoints

        if self._last_dispatched_action is not None:
            initial_action = self._last_dispatched_action
        elif self._latest_observation is not None:
            initial_action = np.asarray(
                [self._latest_observation.values[key] for key in YAM_SCALAR_KEYS],
                dtype=np.float64,
            )
        else:
            raise ValueError("current-relative transition has no measured or dispatched YAM action")

        rate_limited = rate_limit_action_chunk(
            model_waypoints,
            initial_action,
            max_joint_delta=self._yam_settings.max_joint_delta_rad,
            max_gripper_delta=self._yam_settings.max_gripper_delta,
            max_dispatches_per_waypoint=self._yam_settings.max_dispatches_per_waypoint,
        )
        executable = rate_limited.actions
        model_joints = np.concatenate((model_waypoints[:, :6], model_waypoints[:, 7:13]), axis=1)
        initial_joints = np.concatenate((initial_action[:6], initial_action[7:13]))
        model_joint_steps = np.diff(np.vstack((initial_joints, model_joints)), axis=0)
        executable_joints = np.concatenate((executable[:, :6], executable[:, 7:13]), axis=1)
        dispatch_joint_steps = np.diff(np.vstack((initial_joints, executable_joints)), axis=0)
        logger.info(
            "Prepared current-relative chunk %d: stale_model_rows=%d model_rows=%d "
            "dispatches=%d max_dispatches_per_waypoint=%d "
            "max_model_joint_step=%.4frad max_dispatch_joint_step=%.4frad",
            chunk.observation_sequence,
            elapsed_steps,
            len(model_waypoints),
            len(executable),
            int(rate_limited.dispatches_per_waypoint.max()),
            float(np.abs(model_joint_steps).max()),
            float(np.abs(dispatch_joint_steps).max()),
        )
        return executable

    def get_action(self, obs_frame: dict | None):
        self._check_dispatch_progress()
        action = super().get_action(obs_frame)
        with self._lock:
            self._candidate_action = None if action is None else action.detach().cpu().numpy().copy()
        return action

    def notify_action_sent(self) -> None:
        with self._lock:
            if self._candidate_action is not None:
                self._last_dispatched_action = self._candidate_action.copy()
        super().notify_action_sent()
        with self._lock:
            snapshot = self._latest_observation
            if self._candidate_action is None or snapshot is None:
                return
            self._progress_target = self._candidate_action
            self._progress_measured_before = np.asarray(
                [snapshot.values[key] for key in YAM_SCALAR_KEYS],
                dtype=np.float64,
            )
            self._candidate_action = None

    def _check_dispatch_progress(self) -> None:
        with self._lock:
            snapshot = self._latest_observation
            if self._progress_target is None or self._progress_measured_before is None or snapshot is None:
                return
            measured_after = np.asarray(
                [snapshot.values[key] for key in YAM_SCALAR_KEYS],
                dtype=np.float64,
            )
            target = self._progress_target
            measured_before = self._progress_measured_before
            self._progress_target = None
            self._progress_measured_before = None
            try:
                self._progress_watchdog.observe(target, measured_before, measured_after)
            except RuntimeError:
                self._clear_scheduler_locked()
                self._failed.set()
                self._policy_active.clear()
                if self._global_shutdown_event is not None:
                    self._global_shutdown_event.set()
                raise

    def reset(self) -> None:
        self._queries.clear()
        self._progress_watchdog.reset()
        self._candidate_action = None
        self._last_dispatched_action = None
        self._progress_target = None
        self._progress_measured_before = None
        super().reset()


__all__ = [
    "YamCurrentRelativeR6DRemoteInferenceEngine",
    "YamCurrentRelativeR6DSettings",
]
