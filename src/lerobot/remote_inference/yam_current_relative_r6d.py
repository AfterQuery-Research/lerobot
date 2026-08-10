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

"""Inference-time bridge for the dual-UMI current-relative Rotation6D policy."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from lerobot.datasets.umi_current_relative import (
    LEFT_STATE_SLICE,
    RIGHT_STATE_SLICE,
    UMI_CURRENTREL_ACTION_NAMES,
    UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
    UMI_CURRENTREL_STATE_DIM,
    UMI_CURRENTREL_STATE_NAMES,
    decode_relative_pose,
    encode_relative_pose,
    invert_rigid_transform,
    relative_transform,
    validate_rigid_transform,
)

from .yam_umi_ee_bridge import (
    LEFT_GRIPPER_KEY,
    LEFT_JOINT_KEYS,
    RIGHT_GRIPPER_KEY,
    RIGHT_JOINT_KEYS,
    YAM_FLANGE_FRAME,
    YAM_FLANGE_TO_FINGERTIP,
    YAM_JOINT_LIMITS,
    YAM_SCALAR_KEYS,
    GripperMap,
    YamArmKinematics,
)

YAM_CURRENTREL_SCHEMA_ID = UMI_CURRENTREL_ONSET_V3_SCHEMA_ID
YAM_CURRENTREL_STATE_NAMES = UMI_CURRENTREL_STATE_NAMES
YAM_CURRENTREL_ACTION_NAMES = UMI_CURRENTREL_ACTION_NAMES
YAM_CURRENTREL_CAMERA_KEYS = ("umi1", "umi2")


class ArmKinematics(Protocol):
    def fk(self, joints_rad: np.ndarray) -> np.ndarray: ...

    def ik(self, target_pose: np.ndarray, seed_rad: np.ndarray, *, check: bool = True) -> np.ndarray: ...

    def residual(self, joints_rad: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]: ...


@dataclass(frozen=True)
class YamCurrentRelativeQuery:
    """Policy input plus the fixed query anchors required to decode its action chunk."""

    state: np.ndarray
    left_tcp: np.ndarray
    right_tcp: np.ndarray
    left_joints: np.ndarray
    right_joints: np.ndarray
    left_gripper: float
    right_gripper: float


@dataclass(frozen=True)
class DecodedActionChunk:
    actions: np.ndarray
    position_residual_m: np.ndarray
    orientation_residual_rad: np.ndarray


@dataclass(frozen=True)
class RateLimitedActionChunk:
    actions: np.ndarray
    dispatches_per_waypoint: np.ndarray


@dataclass
class YamJointProgressWatchdog:
    max_hold_steps: int = 90
    target_tolerance_rad: float = 2e-3
    min_progress_rad: float = 1e-4
    stalled_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_hold_steps <= 0:
            raise ValueError("max_hold_steps must be positive")
        if self.target_tolerance_rad <= 0 or self.min_progress_rad < 0:
            raise ValueError("progress tolerances are invalid")

    def observe(
        self,
        commanded: np.ndarray,
        measured_before: np.ndarray,
        measured_after: np.ndarray,
    ) -> None:
        """Require measured joint error to decrease after each dispatched target."""

        target = np.asarray(commanded, dtype=np.float64)
        before = np.asarray(measured_before, dtype=np.float64)
        after = np.asarray(measured_after, dtype=np.float64)
        expected_shape = (len(YAM_SCALAR_KEYS),)
        if target.shape != expected_shape or before.shape != expected_shape or after.shape != expected_shape:
            raise ValueError(f"progress watchdog expects three YAM vectors with shape {expected_shape}")
        if not np.isfinite(target).all() or not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError("progress watchdog received non-finite YAM positions")

        joint_indices = np.asarray([*range(6), *range(7, 13)])
        error_before = float(np.abs(target[joint_indices] - before[joint_indices]).max())
        error_after = float(np.abs(target[joint_indices] - after[joint_indices]).max())
        reached = error_after <= self.target_tolerance_rad
        progressed = error_after <= error_before - self.min_progress_rad
        if reached or progressed or error_before <= self.target_tolerance_rad:
            self.stalled_steps = 0
            return

        self.stalled_steps += 1
        if self.stalled_steps >= self.max_hold_steps:
            raise RuntimeError(
                "YAM joint tracking made no measurable progress for "
                f"{self.stalled_steps} dispatches (error {error_before:.4f} -> {error_after:.4f} rad)"
            )

    def reset(self) -> None:
        self.stalled_steps = 0


def rate_limit_action_chunk(
    actions: np.ndarray,
    initial_action: np.ndarray,
    *,
    max_joint_delta: float,
    max_gripper_delta: float,
    max_dispatches_per_waypoint: int,
) -> RateLimitedActionChunk:
    """Expand YAM waypoints into vector-preserving, per-dispatch bounded targets."""

    rows = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(initial_action, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != len(YAM_SCALAR_KEYS):
        raise ValueError(f"expected YAM action chunk shape (N, {len(YAM_SCALAR_KEYS)}), got {rows.shape}")
    if previous.shape != (len(YAM_SCALAR_KEYS),):
        raise ValueError(f"expected initial YAM action shape ({len(YAM_SCALAR_KEYS)},), got {previous.shape}")
    if not np.isfinite(rows).all() or not np.isfinite(previous).all():
        raise ValueError("YAM rate limiter received non-finite actions")
    if max_joint_delta <= 0 or max_gripper_delta <= 0 or max_dispatches_per_waypoint <= 0:
        raise ValueError("YAM rate limits and dispatch cap must be positive")

    joint_indices = np.asarray([*range(6), *range(7, 13)])
    gripper_indices = np.asarray([6, 13])
    expanded: list[np.ndarray] = []
    dispatch_counts = np.empty(len(rows), dtype=np.int64)
    for row_index, target in enumerate(rows):
        joint_distance = float(np.abs(target[joint_indices] - previous[joint_indices]).max())
        gripper_distance = float(np.abs(target[gripper_indices] - previous[gripper_indices]).max())
        joint_dispatches = int(np.ceil(max(0.0, joint_distance - 1e-9) / max_joint_delta))
        gripper_dispatches = int(np.ceil(max(0.0, gripper_distance - 1e-9) / max_gripper_delta))
        dispatches = max(1, joint_dispatches, gripper_dispatches)
        if dispatches > max_dispatches_per_waypoint:
            raise ValueError(
                f"current-relative waypoint {row_index + 1} requires {dispatches} dispatches; "
                f"limit is {max_dispatches_per_waypoint}"
            )
        delta = target - previous
        expanded.extend(previous + delta * (step / dispatches) for step in range(1, dispatches + 1))
        dispatch_counts[row_index] = dispatches
        previous = target

    return RateLimitedActionChunk(
        actions=np.ascontiguousarray(expanded, dtype=np.float32),
        dispatches_per_waypoint=dispatch_counts,
    )


def resolve_yam_urdf(path: str | Path | None = None) -> Path:
    """Resolve the YAM URDF from an explicit path or the installed i2rt package."""

    if path:
        resolved = Path(path).expanduser().resolve()
    else:
        spec = importlib.util.find_spec("i2rt")
        if spec is None or spec.origin is None:
            raise FileNotFoundError("i2rt is not installed; set --inference.urdf_path explicitly")
        resolved = Path(spec.origin).resolve().parent / "robot_models/arm/yam/yam.urdf"
    if not resolved.is_file():
        raise FileNotFoundError(f"YAM URDF does not exist: {resolved}")
    return resolved


def prepare_policy_image(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Validate an RGB camera frame and resize it to the training artifact dimensions."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"camera image must be HWC uint8 RGB, got shape={array.shape} dtype={array.dtype}")
    if array.shape[:2] != (height, width):
        interpolation = (
            cv2.INTER_AREA if array.shape[0] >= height and array.shape[1] >= width else cv2.INTER_LINEAR
        )
        array = cv2.resize(array, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(array, dtype=np.uint8)


@dataclass
class YamCurrentRelativeR6DAdapter:
    """Convert measured YAM joints and 20D model chunks without changing model semantics.

    The policy state is ``inverse(T_current) @ T_previous``. Every action row is
    independently anchored to the same query pose as
    ``inverse(T_query) @ T_future``. Returned rows are therefore never integrated
    recursively.
    """

    left: ArmKinematics
    right: ArmKinematics
    flange_to_tcp: np.ndarray = field(default_factory=lambda: YAM_FLANGE_TO_FINGERTIP.copy())
    gripper: GripperMap = field(default_factory=GripperMap)
    joint_limits: tuple[tuple[float, float], ...] = YAM_JOINT_LIMITS

    def __post_init__(self) -> None:
        self.flange_to_tcp = np.asarray(self.flange_to_tcp, dtype=np.float64)
        validate_rigid_transform(self.flange_to_tcp, name="flange_to_tcp")
        if len(self.joint_limits) != 6:
            raise ValueError("joint_limits must contain six lower/upper pairs")

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path | None = None,
        *,
        target_frame_name: str = YAM_FLANGE_FRAME,
        ik_iterations: int = 3,
        max_position_residual_m: float = 2e-3,
        max_orientation_residual_rad: float = np.deg2rad(1.0),
        flange_to_tcp: np.ndarray | None = None,
        gripper: GripperMap | None = None,
    ) -> YamCurrentRelativeR6DAdapter:
        resolved = resolve_yam_urdf(urdf_path)
        kinematics_kwargs = {
            "target_frame_name": target_frame_name,
            "iterations": ik_iterations,
            "max_position_residual": max_position_residual_m,
            "max_orientation_residual": max_orientation_residual_rad,
        }
        return cls(
            left=YamArmKinematics(str(resolved), **kinematics_kwargs),
            right=YamArmKinematics(str(resolved), **kinematics_kwargs),
            flange_to_tcp=(YAM_FLANGE_TO_FINGERTIP.copy() if flange_to_tcp is None else flange_to_tcp),
            gripper=gripper or GripperMap(),
        )

    @staticmethod
    def _joints_from_observation(
        observation: dict,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        missing = [key for key in YAM_SCALAR_KEYS if key not in observation]
        if missing:
            raise KeyError(f"observation is missing YAM scalars: {missing}")
        left = np.asarray([observation[key] for key in LEFT_JOINT_KEYS], dtype=np.float64)
        right = np.asarray([observation[key] for key in RIGHT_JOINT_KEYS], dtype=np.float64)
        values = np.concatenate((left, right))
        if not np.isfinite(values).all():
            raise ValueError("YAM observation contains non-finite joints")
        return (
            left,
            right,
            float(observation[LEFT_GRIPPER_KEY]),
            float(observation[RIGHT_GRIPPER_KEY]),
        )

    def _tcp_poses(self, observation: dict) -> tuple[np.ndarray, np.ndarray]:
        left, right, _, _ = self._joints_from_observation(observation)
        left_tcp = self.left.fk(left) @ self.flange_to_tcp
        right_tcp = self.right.fk(right) @ self.flange_to_tcp
        validate_rigid_transform(left_tcp, name="left_tcp")
        validate_rigid_transform(right_tcp, name="right_tcp")
        return left_tcp, right_tcp

    def build_query(self, observation: dict, previous_observation: dict | None) -> YamCurrentRelativeQuery:
        """Build one measured current-relative state and retain its decode anchors."""

        previous_observation = previous_observation or observation
        left_joints, right_joints, left_gripper, right_gripper = self._joints_from_observation(observation)
        left_tcp, right_tcp = self._tcp_poses(observation)
        previous_left_tcp, previous_right_tcp = self._tcp_poses(previous_observation)

        state = np.empty(UMI_CURRENTREL_STATE_DIM, dtype=np.float64)
        state[LEFT_STATE_SLICE.start : LEFT_STATE_SLICE.stop - 1] = encode_relative_pose(
            relative_transform(left_tcp, previous_left_tcp)
        )
        state[LEFT_STATE_SLICE.stop - 1] = self.gripper.yam_to_umi(left_gripper, left=True)
        state[RIGHT_STATE_SLICE.start : RIGHT_STATE_SLICE.stop - 1] = encode_relative_pose(
            relative_transform(right_tcp, previous_right_tcp)
        )
        state[RIGHT_STATE_SLICE.stop - 1] = self.gripper.yam_to_umi(right_gripper, left=False)
        if not np.isfinite(state).all():
            raise ValueError("current-relative policy state is non-finite")
        return YamCurrentRelativeQuery(
            state=np.ascontiguousarray(state, dtype=np.float32),
            left_tcp=left_tcp,
            right_tcp=right_tcp,
            left_joints=left_joints,
            right_joints=right_joints,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        )

    def decode_action_chunk(
        self,
        actions: np.ndarray,
        query: YamCurrentRelativeQuery,
    ) -> DecodedActionChunk:
        """Decode same-query-anchor 20D rows into absolute YAM joint/gripper targets."""

        rows = np.asarray(actions, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != UMI_CURRENTREL_STATE_DIM:
            raise ValueError(f"expected action chunk shape (N, 20), got {rows.shape}")
        if not np.isfinite(rows).all():
            raise ValueError("policy returned non-finite current-relative actions")

        decoded = np.empty((rows.shape[0], len(YAM_SCALAR_KEYS)), dtype=np.float64)
        residuals = np.empty((rows.shape[0], 2, 2), dtype=np.float64)
        seeds = [query.left_joints.copy(), query.right_joints.copy()]
        anchors = (query.left_tcp, query.right_tcp)
        solvers = (self.left, self.right)
        arm_slices = (LEFT_STATE_SLICE, RIGHT_STATE_SLICE)
        flange_to_tcp_inverse = invert_rigid_transform(self.flange_to_tcp)

        for row_index, row in enumerate(rows):
            solutions = []
            for arm_index, (arm_slice, anchor, solver) in enumerate(
                zip(arm_slices, anchors, solvers, strict=True)
            ):
                relative_tcp = decode_relative_pose(row[arm_slice.start : arm_slice.stop - 1])
                target_tcp = anchor @ relative_tcp
                target_flange = target_tcp @ flange_to_tcp_inverse
                solution = np.asarray(solver.ik(target_flange, seeds[arm_index], check=True))
                self._validate_joint_solution(solution, arm="left" if arm_index == 0 else "right")
                residuals[row_index, arm_index] = solver.residual(solution, target_flange)
                seeds[arm_index] = solution
                solutions.append(solution)

            decoded[row_index, :6] = solutions[0]
            decoded[row_index, 6] = self.gripper.umi_to_yam(row[LEFT_STATE_SLICE.stop - 1], left=True)
            decoded[row_index, 7:13] = solutions[1]
            decoded[row_index, 13] = self.gripper.umi_to_yam(row[RIGHT_STATE_SLICE.stop - 1], left=False)

        return DecodedActionChunk(
            actions=np.ascontiguousarray(decoded, dtype=np.float32),
            position_residual_m=np.ascontiguousarray(residuals[:, :, 0], dtype=np.float64),
            orientation_residual_rad=np.ascontiguousarray(residuals[:, :, 1], dtype=np.float64),
        )

    def _validate_joint_solution(self, solution: np.ndarray, *, arm: str) -> None:
        if solution.shape != (6,) or not np.isfinite(solution).all():
            raise ValueError(f"{arm} IK returned an invalid joint vector: {solution}")
        lower = np.asarray([pair[0] for pair in self.joint_limits])
        upper = np.asarray([pair[1] for pair in self.joint_limits])
        outside = np.flatnonzero((solution < lower - 1e-6) | (solution > upper + 1e-6))
        if outside.size:
            joints = ", ".join(str(int(index + 1)) for index in outside)
            raise ValueError(f"{arm} IK solution violates operational limits at joint(s) {joints}")


__all__ = [
    "DecodedActionChunk",
    "RateLimitedActionChunk",
    "YAM_CURRENTREL_ACTION_NAMES",
    "YAM_CURRENTREL_CAMERA_KEYS",
    "YAM_CURRENTREL_SCHEMA_ID",
    "YAM_CURRENTREL_STATE_NAMES",
    "YamJointProgressWatchdog",
    "YamCurrentRelativeQuery",
    "YamCurrentRelativeR6DAdapter",
    "prepare_policy_image",
    "rate_limit_action_chunk",
    "resolve_yam_urdf",
]
