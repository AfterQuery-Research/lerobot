# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import lerobot.scripts.replay_yam_relative_filtered as replay_module
from lerobot.scripts.replay_yam_relative_filtered import (
    FILTERED_YAM_HOME_TCP,
    UMI_TO_YAM_BASIS,
    clamp_joints_to_operational_limits,
    filtered_tcp_targets,
)


def test_first_filtered_pose_always_maps_to_down_home():
    state = np.asarray(
        [
            [0.3, -0.2, 0.8, 0.1, 0.2, -0.1, -0.4, 0.7, 0.2, -0.2, 0.1, 0.3],
            [0.4, -0.1, 1.0, 0.2, 0.1, 0.0, -0.5, 0.8, 0.4, -0.1, 0.0, 0.2],
        ]
    )

    targets = filtered_tcp_targets(state)

    assert np.allclose(targets[0, 0], FILTERED_YAM_HOME_TCP)
    assert np.allclose(targets[0, 1], FILTERED_YAM_HOME_TCP)


def test_umi_translation_and_rotation_are_mapped_into_yam_basis():
    first_rotation = Rotation.from_euler("xyz", [0.2, -0.1, 0.3])
    relative_rotation = Rotation.from_euler("x", 0.25)
    second_rotation = first_rotation * relative_rotation
    state = np.zeros((2, 12), dtype=np.float64)
    state[0, :3] = [1.0, 2.0, 3.0]
    state[1, :3] = [1.01, 2.02, 3.03]
    state[0, 3:6] = first_rotation.as_rotvec()
    state[1, 3:6] = second_rotation.as_rotvec()
    state[:, 6:] = state[:, :6]

    targets = filtered_tcp_targets(state)

    assert np.allclose(targets[1, 0, :3, 3] - FILTERED_YAM_HOME_TCP[:3, 3], [0.03, -0.02, 0.01])
    expected_relative = UMI_TO_YAM_BASIS @ relative_rotation.as_matrix() @ UMI_TO_YAM_BASIS.T
    assert np.allclose(
        FILTERED_YAM_HOME_TCP[:3, :3].T @ targets[1, 0, :3, :3],
        expected_relative,
    )


def test_joint_limit_roundoff_is_clamped_but_real_violation_is_rejected():
    joints = np.zeros((2, 6), dtype=np.float64)
    joints[0, 1] = -4.2e-16
    joints[0, 2] = -3.0e-16

    clamped = clamp_joints_to_operational_limits(joints)

    assert clamped[0, 1] == 0.0
    assert clamped[0, 2] == 0.0

    joints[1, 1] = -1e-3
    with pytest.raises(ValueError, match="exceeds YAM operational joint limits"):
        clamp_joints_to_operational_limits(joints)


def test_hardware_replay_never_waits_for_input_while_armed(monkeypatch, tmp_path):
    events = []

    class FakeRobot:
        def __init__(self, _config):
            self.is_connected = False

        def connect(self):
            self.is_connected = True
            events.append("connect")

        def arm(self):
            events.append("arm")

        def reset_for_policy(self):
            events.append("reset")

        def send_action(self, _action):
            events.append("send")

        def disarm(self):
            events.append("disarm")

        def disconnect(self):
            self.is_connected = False
            events.append("disconnect")

    monkeypatch.setattr(replay_module, "BiYAMFollower", FakeRobot)
    monkeypatch.setattr(replay_module, "precise_sleep", lambda _duration: None)
    monkeypatch.setattr("builtins.input", lambda *_args: pytest.fail("hardware replay must not prompt"))
    args = SimpleNamespace(
        robot_id="test-yam",
        left_adapter_serial="LEFT",
        right_adapter_serial="RIGHT",
        speed=0.5,
        max_joint_delta=0.08,
        max_gripper_delta=0.05,
        telemetry_path=tmp_path / "control.jsonl",
        episode=54,
    )
    replay = SimpleNamespace(
        raw_actions=np.zeros((1, 14), dtype=np.float32),
        limited=SimpleNamespace(actions=np.zeros((2, 14), dtype=np.float32)),
        source_fps=30.0,
    )

    replay_module.replay_on_hardware(args, replay)

    assert events == ["connect", "arm", "reset", "send", "send", "disarm", "disconnect"]
