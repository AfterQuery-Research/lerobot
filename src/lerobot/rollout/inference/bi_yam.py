#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Minimal robot-side action adapter for the new BiYAM policies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol

import numpy as np

from lerobot.robots.bi_yam.config_bi_yam import YAM_SCALAR_KEYS
from lerobot.robots.bi_yam.kinematics import I2RT_YAM_JOINT_LIMITS, I2RTYAMKinematics

from .umi_yam import decode_bimanual_action, encode_bimanual_action

logger = logging.getLogger(__name__)

BiYAMActionMode = Literal["joint", "ee"]

_ARM_FEATURES = ("x", "y", "z", "r0x", "r0y", "r0z", "r1x", "r1y", "r1z", "gripper")
EE_FEATURE_NAMES = tuple(f"{side}_{name}" for side in ("left", "right") for name in _ARM_FEATURES)

# The I2RT linear_4310 gripper defines a 96 mm full stroke. Treat the requested
# 85 mm policy opening as model-space 1.0 in both observation and action paths.
_YAM_GRIPPER_FULL_STROKE_M = 0.096
_POLICY_GRIPPER_MAX_WIDTH_M = 0.085
_POLICY_TO_HARDWARE_GRIPPER_SCALE = _POLICY_GRIPPER_MAX_WIDTH_M / _YAM_GRIPPER_FULL_STROKE_M


class _Kinematics(Protocol):
    def fk(self, q: object) -> np.ndarray: ...

    def ik(self, target_pose: object, seed_q: object) -> np.ndarray: ...


KinematicsFactory = Callable[[], _Kinematics]


@dataclass(frozen=True)
class BiYAMQueryAnchor:
    """Measured I2RT poses and joints frozen at one policy query."""

    left_pose: np.ndarray
    right_pose: np.ndarray
    left_q: np.ndarray
    right_q: np.ndarray


@dataclass(frozen=True)
class PreparedBiYAMObservation:
    state: np.ndarray
    anchor: BiYAMQueryAnchor | None


def _finite_vector(value: object, *, name: str, size: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector of length {size}") from exc
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def validate_joint_action(action: object, *, name: str = "joint action") -> np.ndarray:
    """Validate the exact 14-D BiYAM driver command without clipping it."""

    value = _finite_vector(action, name=name, size=len(YAM_SCALAR_KEYS))
    joints = np.concatenate((value[:6], value[7:13]))
    limits = np.concatenate((I2RT_YAM_JOINT_LIMITS, I2RT_YAM_JOINT_LIMITS))
    invalid = (joints < limits[:, 0]) | (joints > limits[:, 1])
    if invalid.any():
        raise ValueError(
            f"{name} violates I2RT YAM joint limits at indices {np.flatnonzero(invalid).tolist()}"
        )
    grippers = value[[6, 13]]
    if ((grippers < 0.0) | (grippers > 1.0)).any():
        raise ValueError(f"{name} grippers must be in [0, 1]")
    return value


class BiYAMActionAdapter:
    """Adapt either absolute joints or query-anchored UMI EE rows to BiYAM."""

    def __init__(
        self,
        mode: BiYAMActionMode,
        *,
        kinematics_factory: KinematicsFactory = I2RTYAMKinematics,
    ) -> None:
        if mode not in ("joint", "ee"):
            raise ValueError("BiYAM action mode must be 'joint' or 'ee'")
        self.mode = mode
        self._kinematics_factory = kinematics_factory
        self._left_kinematics: _Kinematics | None = None
        self._right_kinematics: _Kinematics | None = None
        self._kinematics_lock = Lock()

    @property
    def state_features(self) -> tuple[str, ...]:
        return tuple(YAM_SCALAR_KEYS) if self.mode == "joint" else EE_FEATURE_NAMES

    @property
    def action_features(self) -> tuple[str, ...]:
        return self.state_features

    def validate_hardware_features(
        self, state_features: tuple[str, ...], action_features: tuple[str, ...]
    ) -> None:
        expected = tuple(YAM_SCALAR_KEYS)
        if state_features != expected or action_features != expected:
            raise ValueError("BiYAM inference requires the exact 14-D YAM_SCALAR_KEYS hardware order")

    def start(self) -> None:
        """Verify and load the pinned I2RT model before the robot is armed."""

        if self.mode == "ee" and self._left_kinematics is None:
            self._left_kinematics = self._kinematics_factory()
            self._right_kinematics = self._kinematics_factory()

    def prepare_observation(self, current_state: object) -> PreparedBiYAMObservation:
        current = validate_joint_action(current_state, name="BiYAM observation state")
        if self.mode == "joint":
            return PreparedBiYAMObservation(current.astype(np.float32), None)

        self.start()
        assert self._left_kinematics is not None and self._right_kinematics is not None
        with self._kinematics_lock:
            left_pose = self._left_kinematics.fk(current[:6])
            right_pose = self._right_kinematics.fk(current[7:13])
        identity = np.eye(4, dtype=np.float64)
        model_state = encode_bimanual_action(
            left_delta_transform=identity,
            left_gripper=np.asarray(np.clip(current[6] / _POLICY_TO_HARDWARE_GRIPPER_SCALE, 0.0, 1.0)),
            right_delta_transform=identity,
            right_gripper=np.asarray(np.clip(current[13] / _POLICY_TO_HARDWARE_GRIPPER_SCALE, 0.0, 1.0)),
        )
        anchor = BiYAMQueryAnchor(
            left_pose=left_pose.copy(),
            right_pose=right_pose.copy(),
            left_q=current[:6].copy(),
            right_q=current[7:13].copy(),
        )
        return PreparedBiYAMObservation(model_state.astype(np.float32), anchor)

    def to_joint_action(self, action: object, anchor: BiYAMQueryAnchor | None) -> np.ndarray:
        if self.mode == "joint":
            if anchor is not None:
                raise ValueError("joint actions must not carry an EE query anchor")
            return validate_joint_action(action).astype(np.float32)
        if anchor is None:
            raise ValueError("EE action is missing its measured query anchor")

        decoded = decode_bimanual_action(action)
        model_action = np.asarray(action, dtype=np.float64)
        logger.info(
            "EE model row: left_xyz_m(tool)=%s left_r6d_rows=%s right_xyz_m(tool)=%s right_r6d_rows=%s",
            np.array2string(model_action[:3], precision=6, separator=",", suppress_small=False),
            np.array2string(model_action[3:9], precision=6, separator=",", suppress_small=False),
            np.array2string(model_action[10:13], precision=6, separator=",", suppress_small=False),
            np.array2string(model_action[13:19], precision=6, separator=",", suppress_small=False),
        )
        left_gripper = float(decoded.left_gripper)
        right_gripper = float(decoded.right_gripper)
        if not np.isfinite(left_gripper) or not np.isfinite(right_gripper):
            raise ValueError("EE action grippers must be finite")
        left_gripper = float(np.clip(left_gripper, 0.0, 1.0) * _POLICY_TO_HARDWARE_GRIPPER_SCALE)
        right_gripper = float(np.clip(right_gripper, 0.0, 1.0) * _POLICY_TO_HARDWARE_GRIPPER_SCALE)
        assert self._left_kinematics is not None and self._right_kinematics is not None
        with self._kinematics_lock:
            left_q = self._left_kinematics.ik(anchor.left_pose @ decoded.left_delta_transform, anchor.left_q)
            right_q = self._right_kinematics.ik(
                anchor.right_pose @ decoded.right_delta_transform, anchor.right_q
            )
        command = np.concatenate((left_q, [left_gripper], right_q, [right_gripper]))
        return validate_joint_action(command, name="IK joint action").astype(np.float32)
