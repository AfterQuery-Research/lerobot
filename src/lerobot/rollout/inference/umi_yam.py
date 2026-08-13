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

"""Versioned UMI-to-YAM end-effector action contract.

This module only defines the numeric action representation. Input transforms
must already describe the authoritative I2RT ``grasp_site`` frame; no tool-axis
correction (including an additional ``Ry(pi)``) is applied here.

For each arm and future target ``t + k``, the action transform is anchored at
the query frame, ``inv(T_t) @ T_(t+k)``. A bimanual action contains, in order,
``left xyz + left R6D rows + left gripper + right ...``. Rotation6D is the first
two *rows* of the rotation matrix, matching the Stanford UMI convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EE_ACTION_CONTRACT_VERSION = "umi_yam.ee.current_relative.r6d_rows.v1"
I2RT_GRASP_MODEL_REVISION = "7ed46f4e4e316133a0c39aa6cf34a73d2718e850"
ACTION_HORIZON = 24
ARM_ACTION_DIM = 10
ACTION_DIM = 2 * ARM_ACTION_DIM

_ROTATION_ATOL = 1e-6
_HOMOGENEOUS_ATOL = 1e-8
_R6D_MIN_NORM = 1e-8


@dataclass(frozen=True)
class DecodedBimanualAction:
    """Decoded bimanual action arrays, retaining any leading batch dimensions."""

    left_delta_transform: np.ndarray
    left_gripper: np.ndarray
    right_delta_transform: np.ndarray
    right_gripper: np.ndarray


def _as_finite_array(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numeric array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_rotation_matrices(rotation: object, *, name: str) -> np.ndarray:
    matrix = _as_finite_array(rotation, name=name)
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"{name} must have shape (..., 3, 3), got {matrix.shape}")

    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(matrix @ np.swapaxes(matrix, -1, -2), identity, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.allclose(np.linalg.det(matrix), 1.0, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(f"{name} must be a proper rotation with determinant +1")
    return matrix


def _validate_transforms(transform: object, *, name: str) -> np.ndarray:
    matrix = _as_finite_array(transform, name=name)
    if matrix.ndim < 2 or matrix.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must have shape (..., 4, 4), got {matrix.shape}")
    expected_bottom_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(matrix[..., 3, :], expected_bottom_row, atol=_HOMOGENEOUS_ATOL, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    _validate_rotation_matrices(matrix[..., :3, :3], name=f"{name} rotation")
    return matrix


def rotation_matrix_to_r6d_rows(rotation: object) -> np.ndarray:
    """Flatten the first two matrix rows as ``[..., 6]``.

    For example, a +90 degree rotation about Z becomes
    ``[0, -1, 0, 1, 0, 0]``. This deliberately differs from first-column R6D.
    """

    matrix = _validate_rotation_matrices(rotation, name="rotation")
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,)).copy()


def r6d_rows_to_rotation_matrix(r6d: object) -> np.ndarray:
    """Decode first-row R6D using Gram-Schmidt orthonormalization."""

    rows = _as_finite_array(r6d, name="r6d")
    if rows.ndim < 1 or rows.shape[-1] != 6:
        raise ValueError(f"r6d must have shape (..., 6), got {rows.shape}")
    rows = rows.reshape(rows.shape[:-1] + (2, 3))

    row0 = rows[..., 0, :]
    row0_norm = np.linalg.norm(row0, axis=-1, keepdims=True)
    if np.any(row0_norm <= _R6D_MIN_NORM):
        raise ValueError("r6d first row must be non-zero")
    row0 = row0 / row0_norm

    row1 = rows[..., 1, :] - np.sum(rows[..., 1, :] * row0, axis=-1, keepdims=True) * row0
    row1_norm = np.linalg.norm(row1, axis=-1, keepdims=True)
    if np.any(row1_norm <= _R6D_MIN_NORM):
        raise ValueError("r6d rows must not be parallel")
    row1 = row1 / row1_norm
    row2 = np.cross(row0, row1)
    return np.stack((row0, row1, row2), axis=-2)


def _validate_gripper(gripper: object, *, name: str, leading_shape: tuple[int, ...]) -> np.ndarray:
    value = _as_finite_array(gripper, name=name)
    if value.shape != leading_shape:
        raise ValueError(f"{name} must have shape {leading_shape}, got {value.shape}")
    return value


def encode_bimanual_action(
    *,
    left_delta_transform: object,
    left_gripper: object,
    right_delta_transform: object,
    right_gripper: object,
) -> np.ndarray:
    """Encode matching left/right relative transforms into 20-D actions."""

    left = _validate_transforms(left_delta_transform, name="left_delta_transform")
    right = _validate_transforms(right_delta_transform, name="right_delta_transform")
    if left.shape != right.shape:
        raise ValueError(
            f"left and right transforms must have matching shapes, got {left.shape} and {right.shape}"
        )

    leading_shape = left.shape[:-2]
    left_jaw = _validate_gripper(left_gripper, name="left_gripper", leading_shape=leading_shape)
    right_jaw = _validate_gripper(right_gripper, name="right_gripper", leading_shape=leading_shape)
    left_arm = np.concatenate(
        (left[..., :3, 3], rotation_matrix_to_r6d_rows(left[..., :3, :3]), left_jaw[..., None]), axis=-1
    )
    right_arm = np.concatenate(
        (right[..., :3, 3], rotation_matrix_to_r6d_rows(right[..., :3, :3]), right_jaw[..., None]), axis=-1
    )
    return np.concatenate((left_arm, right_arm), axis=-1)


def decode_bimanual_action(action: object) -> DecodedBimanualAction:
    """Decode 20-D actions without applying any robot/tool frame correction."""

    value = _as_finite_array(action, name="action")
    if value.ndim < 1 or value.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must have shape (..., {ACTION_DIM}), got {value.shape}")

    leading_shape = value.shape[:-1]
    left = value[..., :ARM_ACTION_DIM]
    right = value[..., ARM_ACTION_DIM:]
    left_transform = np.zeros(leading_shape + (4, 4), dtype=np.float64)
    right_transform = np.zeros_like(left_transform)
    left_transform[..., :3, :3] = r6d_rows_to_rotation_matrix(left[..., 3:9])
    right_transform[..., :3, :3] = r6d_rows_to_rotation_matrix(right[..., 3:9])
    left_transform[..., :3, 3] = left[..., :3]
    right_transform[..., :3, 3] = right[..., :3]
    left_transform[..., 3, 3] = 1.0
    right_transform[..., 3, 3] = 1.0
    return DecodedBimanualAction(
        left_delta_transform=left_transform,
        left_gripper=left[..., 9].copy(),
        right_delta_transform=right_transform,
        right_gripper=right[..., 9].copy(),
    )
