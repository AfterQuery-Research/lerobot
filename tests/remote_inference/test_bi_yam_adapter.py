# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import numpy as np
import pytest

from lerobot.datasets.umi_yam import decode_bimanual_action, encode_bimanual_action
from lerobot.robots.bi_yam.config_bi_yam import YAM_SCALAR_KEYS
from lerobot.rollout.inference.bi_yam import (
    EE_FEATURE_NAMES,
    BiYAMActionAdapter,
    validate_joint_action,
)


def _transform(xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = xyz
    return pose


class FakeKinematics:
    def __init__(self) -> None:
        self.ik_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fk(self, q: object) -> np.ndarray:
        joints = np.asarray(q)
        return _transform(joints[:3])

    def ik(self, target_pose: object, seed_q: object) -> np.ndarray:
        target, seed = np.asarray(target_pose), np.asarray(seed_q)
        self.ik_calls.append((target.copy(), seed.copy()))
        result = seed.copy()
        result[:3] = target[:3, 3]
        return result


def _ee_adapter() -> tuple[BiYAMActionAdapter, list[FakeKinematics]]:
    instances: list[FakeKinematics] = []

    def factory() -> FakeKinematics:
        instance = FakeKinematics()
        instances.append(instance)
        return instance

    return BiYAMActionAdapter("ee", kinematics_factory=factory), instances


def _state(
    left_xyz=(0.2, 0.3, 0.4),
    right_xyz=(0.5, 0.6, 0.7),
    left_gripper=0.25,
    right_gripper=0.75,
) -> np.ndarray:
    return np.array([*left_xyz, 0, 0, 0, left_gripper, *right_xyz, 0, 0, 0, right_gripper])


def test_ee_state_is_identity_and_action_uses_frozen_query_anchor_per_arm():
    adapter, kinematics = _ee_adapter()
    current = _state()

    prepared = adapter.prepare_observation(current)
    decoded_state = decode_bimanual_action(prepared.state)
    assert np.allclose(decoded_state.left_delta_transform, np.eye(4))
    assert np.allclose(decoded_state.right_delta_transform, np.eye(4))
    assert decoded_state.left_gripper == pytest.approx(0.25)
    assert decoded_state.right_gripper == pytest.approx(0.75)

    action = encode_bimanual_action(
        left_delta_transform=_transform((0.05, 0.0, 0.0)),
        left_gripper=np.asarray(0.4),
        right_delta_transform=_transform((0.0, 0.02, 0.0)),
        right_gripper=np.asarray(0.6),
    )
    command = adapter.to_joint_action(action, prepared.anchor)

    assert len(kinematics) == 2
    assert np.allclose(kinematics[0].ik_calls[0][0][:3, 3], [0.25, 0.3, 0.4])
    assert np.allclose(kinematics[1].ik_calls[0][0][:3, 3], [0.5, 0.62, 0.7])
    assert np.allclose(kinematics[0].ik_calls[0][1], current[:6])
    assert np.allclose(kinematics[1].ik_calls[0][1], current[7:13])
    assert np.allclose(command, [0.25, 0.3, 0.4, 0, 0, 0, 0.4, 0.5, 0.62, 0.7, 0, 0, 0, 0.6])


def test_joint_mode_is_strict_passthrough_with_hardware_feature_order():
    adapter = BiYAMActionAdapter("joint")
    adapter.validate_hardware_features(tuple(YAM_SCALAR_KEYS), tuple(YAM_SCALAR_KEYS))
    prepared = adapter.prepare_observation(_state())
    assert prepared.anchor is None
    assert adapter.state_features == tuple(YAM_SCALAR_KEYS)
    assert np.array_equal(adapter.to_joint_action(prepared.state, None), prepared.state)
    with pytest.raises(ValueError, match="exact 14-D"):
        adapter.validate_hardware_features(tuple(reversed(YAM_SCALAR_KEYS)), tuple(YAM_SCALAR_KEYS))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.__setitem__(0, np.nan), "finite"),
        (lambda value: value.__setitem__(0, 4.0), "joint limits"),
        (lambda value: value.__setitem__(6, 1.1), "grippers"),
    ],
)
def test_joint_action_rejects_invalid_commands(mutate, message):
    value = _state()
    mutate(value)
    with pytest.raises(ValueError, match=message):
        validate_joint_action(value)


def test_ee_action_rejects_degenerate_rotation_before_ik():
    adapter, kinematics = _ee_adapter()
    prepared = adapter.prepare_observation(_state())
    action = encode_bimanual_action(
        left_delta_transform=np.eye(4),
        left_gripper=np.asarray(0.5),
        right_delta_transform=np.eye(4),
        right_gripper=np.asarray(0.5),
    )
    action[3:9] = [1, 0, 0, 2, 0, 0]
    with pytest.raises(ValueError, match="parallel"):
        adapter.to_joint_action(action, prepared.anchor)
    assert all(not instance.ik_calls for instance in kinematics)


def test_ee_contract_feature_names_are_unambiguous_rows():
    assert len(EE_FEATURE_NAMES) == 20
    assert EE_FEATURE_NAMES[:10] == (
        "left_x",
        "left_y",
        "left_z",
        "left_r0x",
        "left_r0y",
        "left_r0z",
        "left_r1x",
        "left_r1y",
        "left_r1z",
        "left_gripper",
    )
