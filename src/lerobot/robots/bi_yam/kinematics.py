#!/usr/bin/env python

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

"""Fail-closed adapter around the authoritative I2RT YAM kinematics.

The implementation deliberately contains no local FK, IK, pseudo-TCP, or tool
axis correction. I2RT's pinned ``YAM + LINEAR_4310`` MuJoCo model and its
``grasp_site`` are the sole kinematic definition. LeRobot only validates the
contract, converts the public six arm joints to I2RT's eight model joints, and
checks every IK result by running I2RT FK.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

I2RT_REVISION = "7ed46f4e4e316133a0c39aa6cf34a73d2718e850"
I2RT_GRIPPER_XML_SHA256 = "42d7ab67071d33c2fecf73a74f529ccb80337b4c81f4208d5a8ecd56cc89d7a2"
I2RT_GRASP_SITE = "grasp_site"

# Golden value maintained by I2RT's
# test_linear_4310_replacement_grasp_site_fk at the pinned revision above.
I2RT_YAM_LINEAR_4310_Q0_FK = np.array(
    [
        [-0.000003673, 0.000003673, 1.0, 0.330597263],
        [0.000005307, -1.0, 0.000003673, 0.000001793],
        [1.0, 0.000005307, 0.000003673, 0.173502620],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# The first six qpos ranges in I2RT's pinned YAM MJCF. These checks are in
# addition to, not a replacement for, any limits used internally by I2RT IK.
I2RT_YAM_JOINT_LIMITS = np.array(
    [
        [-2.61799, 3.14159],
        [-8.88178e-16, 3.66519],
        [0.0, 3.14159],
        [-1.69297, 1.5708],
        [-1.5708, 1.5708],
        [-2.0944, 2.0944],
    ],
    dtype=np.float64,
)

_ARM_DOF = 6
_MODEL_DOF = 8
_TRANSFORM_ATOL = 1e-8
_ROTATION_ATOL = 1e-6
_MODEL_CONTRACT_ATOL = 1e-8
_JAW_ZERO_ATOL = 1e-12
_JOINT_LIMIT_ATOL = 1e-8
_SOLVER_POSITION_TOLERANCE_M = 1e-4
_SOLVER_ORIENTATION_TOLERANCE_RAD = 1e-4
_DEFAULT_POSITION_TOLERANCE_M = 0.002
_DEFAULT_ORIENTATION_TOLERANCE_RAD = 0.001
_DEFAULT_MAX_JOINT_STEP_RAD = 0.10


class _I2RTKinematicsBackend(Protocol):
    def fk(self, q: np.ndarray, site_name: str | None = None) -> np.ndarray: ...

    def ik(
        self,
        target_pose: np.ndarray,
        site_name: str,
        init_q: np.ndarray | None = None,
        **kwargs: object,
    ) -> tuple[bool, np.ndarray]: ...


BackendFactory = Callable[[], _I2RTKinematicsBackend]


def _as_finite_array(value: object, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array with shape {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _validate_arm_joints(value: object, *, name: str) -> np.ndarray:
    joints = _as_finite_array(value, name=name, shape=(_ARM_DOF,))
    lower, upper = I2RT_YAM_JOINT_LIMITS.T
    below = joints < lower - _JOINT_LIMIT_ATOL
    above = joints > upper + _JOINT_LIMIT_ATOL
    if np.any(below | above):
        indices = np.flatnonzero(below | above).tolist()
        raise ValueError(f"{name} violates authoritative I2RT YAM joint limits at indices {indices}")
    # MuJoCo/IK can land a few ulps outside a boundary. Only values already
    # accepted by the numerical tolerance above are projected onto the limit.
    return np.clip(joints, lower, upper)


def _validate_transform(value: object, *, name: str) -> np.ndarray:
    transform = _as_finite_array(value, name=name, shape=(4, 4))
    if not np.allclose(transform[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=_TRANSFORM_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} rotation must have determinant +1")
    return transform


def _q6_to_i2rt_q8(q: np.ndarray) -> np.ndarray:
    """Append the two closed, equal jaw slides required by LINEAR_4310."""

    return np.concatenate((q, np.zeros(2, dtype=np.float64)))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_i2rt_backend() -> _I2RTKinematicsBackend:
    """Import optional I2RT only when kinematics are actually requested."""

    try:
        from i2rt.robot_models import GRIPPER_LINEAR_4310_PATH
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    except ImportError as exc:
        raise ImportError(
            f"i2rt is required for BiYAM kinematics; install the bi-yam extra pinned to i2rt@{I2RT_REVISION}"
        ) from exc

    actual_hash = _sha256(GRIPPER_LINEAR_4310_PATH)
    if actual_hash != I2RT_GRIPPER_XML_SHA256:
        raise RuntimeError(
            "Refusing to use an unrecognized I2RT LINEAR_4310 model: expected sha256 "
            f"{I2RT_GRIPPER_XML_SHA256}, got {actual_hash}. Install i2rt@{I2RT_REVISION}."
        )

    combined_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    return Kinematics(combined_path, I2RT_GRASP_SITE)


def _rotation_residual_rad(target: np.ndarray, achieved: np.ndarray) -> float:
    relative = target[:3, :3].T @ achieved[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


class I2RTYAMKinematics:
    """Six-joint YAM API backed exclusively by I2RT's eight-qpos model.

    ``ik`` requires an explicit seed and independently verifies every returned
    candidate with FK. I2RT may report ``success=False`` at a joint boundary
    even when its candidate is accurate; such a candidate is accepted only
    when it passes the same residual, joint-limit, and step gates as a reported
    success.
    """

    def __init__(
        self,
        *,
        position_tolerance_m: float = _DEFAULT_POSITION_TOLERANCE_M,
        orientation_tolerance_rad: float = _DEFAULT_ORIENTATION_TOLERANCE_RAD,
        max_joint_step_rad: float = _DEFAULT_MAX_JOINT_STEP_RAD,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be positive and finite")
        if not np.isfinite(orientation_tolerance_rad) or orientation_tolerance_rad <= 0:
            raise ValueError("orientation_tolerance_rad must be positive and finite")
        if not np.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0:
            raise ValueError("max_joint_step_rad must be positive and finite")

        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self._backend = (backend_factory or _make_i2rt_backend)()
        self._verify_model_contract()

    def fk(self, q: object) -> np.ndarray:
        """Return I2RT ``grasp_site`` pose for six YAM arm joints."""

        joints = _validate_arm_joints(q, name="q")
        pose = self._backend.fk(_q6_to_i2rt_q8(joints), site_name=I2RT_GRASP_SITE)
        return _validate_transform(pose, name="I2RT FK result")

    def ik(self, target_pose: object, seed_q: object) -> np.ndarray:
        """Solve with I2RT, then fail closed unless FK reproduces the target."""

        target = _validate_transform(target_pose, name="target_pose")
        seed = _validate_arm_joints(seed_q, name="seed_q")
        success, raw_solution = self._backend.ik(
            target,
            I2RT_GRASP_SITE,
            init_q=_q6_to_i2rt_q8(seed),
            pos_threshold=_SOLVER_POSITION_TOLERANCE_M,
            ori_threshold=_SOLVER_ORIENTATION_TOLERANCE_RAD,
        )
        solution_q8 = _as_finite_array(raw_solution, name="I2RT IK result", shape=(_MODEL_DOF,))
        if not np.allclose(solution_q8[_ARM_DOF:], 0.0, atol=_JAW_ZERO_ATOL, rtol=0.0):
            raise RuntimeError("I2RT IK changed LINEAR_4310 jaw slides during arm-only IK")
        solution = _validate_arm_joints(solution_q8[:_ARM_DOF], name="I2RT IK result arm joints")

        achieved = _validate_transform(
            self._backend.fk(_q6_to_i2rt_q8(solution), site_name=I2RT_GRASP_SITE),
            name="I2RT IK verification FK",
        )
        position_error = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
        orientation_error = _rotation_residual_rad(target, achieved)
        if position_error > self.position_tolerance_m or orientation_error > self.orientation_tolerance_rad:
            raise RuntimeError(
                "I2RT IK residual exceeds tolerance: "
                f"position={position_error:.9g} m (limit {self.position_tolerance_m:.9g}), "
                f"orientation={orientation_error:.9g} rad (limit {self.orientation_tolerance_rad:.9g}), "
                f"solver_success={success}"
            )
        max_joint_step = float(np.max(np.abs(solution - seed)))
        if max_joint_step > self.max_joint_step_rad:
            raise RuntimeError(
                "I2RT IK joint step exceeds tolerance: "
                f"step={max_joint_step:.9g} rad (limit {self.max_joint_step_rad:.9g}), "
                f"solver_success={success}"
            )
        return solution

    def _verify_model_contract(self) -> None:
        q0 = np.zeros(_MODEL_DOF, dtype=np.float64)
        actual = _validate_transform(
            self._backend.fk(q0, site_name=I2RT_GRASP_SITE), name="I2RT zero-pose FK"
        )
        if not np.allclose(actual, I2RT_YAM_LINEAR_4310_Q0_FK, atol=_MODEL_CONTRACT_ATOL, rtol=0.0):
            max_error = float(np.max(np.abs(actual - I2RT_YAM_LINEAR_4310_Q0_FK)))
            raise RuntimeError(
                "I2RT YAM+LINEAR_4310 grasp_site does not match the pinned model contract "
                f"for {I2RT_REVISION} (maximum q=0 FK error {max_error:.9g})"
            )
