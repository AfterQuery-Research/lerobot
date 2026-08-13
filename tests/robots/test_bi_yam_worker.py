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

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_yam.worker import (
    _nearest_calibrated_gripper_turns,
    _rebase_calibrated_gripper_encoder,
)


class _FakeMotorChain:
    def __init__(self, raw_position: float, *, direction: float = 1.0) -> None:
        self.raw_position = raw_position
        self.motor_offset = np.asarray([0.0], dtype=np.float64)
        self.motor_direction = np.asarray([direction], dtype=np.float64)

    def read_states(self):
        position = (self.raw_position - self.motor_offset[0]) * self.motor_direction[0]
        return [SimpleNamespace(pos=position)]


class _FakeBackend:
    def __init__(
        self,
        raw_position: float,
        calibration_limits: tuple[float, float],
        *,
        direction: float = 1.0,
    ) -> None:
        self.motor_chain = _FakeMotorChain(raw_position, direction=direction)
        self.calibration_limits = calibration_limits

    def get_robot_info(self):
        return {"gripper_index": 0}

    def get_joint_pos(self):
        position = self.motor_chain.read_states()[0].pos
        closed, open_ = self.calibration_limits
        return np.asarray([(position - closed) / (open_ - closed)], dtype=np.float64)


def test_gripper_encoder_rebase_recovers_observed_left_wrap():
    limits = (6.5127412832837415, 1.2205310139620043)
    wrapped_normalized = 1.756433
    wrapped_position = limits[0] + wrapped_normalized * (limits[1] - limits[0])
    backend = _FakeBackend(wrapped_position, limits)

    assert _nearest_calibrated_gripper_turns(wrapped_position, limits) == 1
    _rebase_calibrated_gripper_encoder(backend, limits)

    assert backend.motor_chain.motor_offset[0] == pytest.approx(-2 * np.pi)
    assert backend.get_joint_pos()[0] == pytest.approx(0.569181, abs=1e-4)


def test_gripper_encoder_rebase_leaves_calibrated_branch_unchanged():
    limits = (6.5127412832837415, 1.2205310139620043)
    position = limits[0] + 0.57 * (limits[1] - limits[0])
    backend = _FakeBackend(position, limits)

    assert _nearest_calibrated_gripper_turns(position, limits) == 0
    _rebase_calibrated_gripper_encoder(backend, limits)

    assert backend.motor_chain.motor_offset[0] == 0.0
    assert backend.get_joint_pos()[0] == pytest.approx(0.57)


def test_gripper_encoder_rebase_rejects_multiple_revolutions():
    limits = (6.5127412832837415, 1.2205310139620043)
    position = float(np.mean(limits) - 2 * (2 * np.pi))
    backend = _FakeBackend(position, limits)

    with pytest.raises(RuntimeError, match="refusing to rebase gripper encoder by 2 revolutions"):
        _rebase_calibrated_gripper_encoder(backend, limits)
