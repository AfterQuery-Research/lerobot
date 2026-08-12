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

import numpy as np
import pytest

from lerobot.datasets.umi_yam import (
    ACTION_DIM,
    ACTION_HORIZON,
    EE_ACTION_CONTRACT_VERSION,
    I2RT_GRASP_MODEL_REVISION,
    decode_bimanual_action,
    encode_bimanual_action,
    r6d_rows_to_rotation_matrix,
    rotation_matrix_to_r6d_rows,
)


def _transform(*, xyz=(0.0, 0.0, 0.0), rotation=None):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.eye(3) if rotation is None else rotation
    transform[:3, 3] = xyz
    return transform


def _rz(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_contract_is_explicitly_versioned_and_pinned_to_pr12_i2rt():
    assert EE_ACTION_CONTRACT_VERSION == "umi_yam.ee.current_relative.r6d_rows.v1"
    assert I2RT_GRASP_MODEL_REVISION == "7ed46f4e4e316133a0c39aa6cf34a73d2718e850"
    assert ACTION_HORIZON == 24
    assert ACTION_DIM == 20


def test_rotation6d_uses_first_two_rows_for_known_positive_rz():
    rotation = _rz(np.pi / 2)

    encoded = rotation_matrix_to_r6d_rows(rotation)

    np.testing.assert_allclose(encoded, [0.0, -1.0, 0.0, 1.0, 0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(r6d_rows_to_rotation_matrix(encoded), rotation, atol=1e-15)


@pytest.mark.parametrize("angle", [0.0, 0.17, -0.9, np.pi / 2, np.pi])
def test_rotation6d_row_roundtrip(angle):
    rotation = _rz(angle)
    np.testing.assert_allclose(
        r6d_rows_to_rotation_matrix(rotation_matrix_to_r6d_rows(rotation)), rotation, atol=1e-14
    )


def test_bimanual_roundtrip_does_not_apply_an_extra_axis_transform():
    left = _transform(xyz=(0.1, -0.2, 0.3), rotation=_rz(np.pi / 2))
    right = _transform(xyz=(-0.4, 0.5, -0.6), rotation=_rz(-0.7))

    encoded = encode_bimanual_action(
        left_delta_transform=left,
        left_gripper=np.array(0.25),
        right_delta_transform=right,
        right_gripper=np.array(0.75),
    )
    decoded = decode_bimanual_action(encoded)

    assert encoded.shape == (20,)
    np.testing.assert_allclose(decoded.left_delta_transform, left, atol=1e-14)
    np.testing.assert_allclose(decoded.right_delta_transform, right, atol=1e-14)
    assert decoded.left_gripper == pytest.approx(0.25)
    assert decoded.right_gripper == pytest.approx(0.75)


def test_invalid_rotations_and_degenerate_r6d_fail_closed():
    invalid_rotation = np.eye(3)
    invalid_rotation[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        rotation_matrix_to_r6d_rows(invalid_rotation)
    with pytest.raises(ValueError, match="parallel"):
        r6d_rows_to_rotation_matrix([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        decode_bimanual_action(np.full(20, np.nan))
