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

import importlib.util
import multiprocessing as mp
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_yam import YAM_SCALAR_KEYS, BiYAMFollower, BiYAMFollowerConfig, YAMArmConfig
from lerobot.robots.bi_yam.worker import (
    ArmCommand,
    ArmControl,
    ArmState,
    ArmWorkerRuntime,
    ProcessArmWorker,
    make_i2rt_backend,
)
from lerobot.robots.utils import make_robot_from_config


class FakeBackend:
    def __init__(self) -> None:
        self.positions = np.zeros(7, dtype=np.float64)
        self.commands: list[np.ndarray] = []
        self.idle_calls = 0
        self.close_calls = 0

    def num_dofs(self) -> int:
        return 7

    def get_joint_pos(self) -> np.ndarray:
        return self.positions.copy()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self.positions = np.asarray(joint_pos, dtype=np.float64).copy()
        self.commands.append(self.positions.copy())

    def enter_gravity_comp_idle(self) -> None:
        self.idle_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def make_fake_backend(_config: YAMArmConfig) -> FakeBackend:
    return FakeBackend()


def make_slow_backend(_config: YAMArmConfig) -> FakeBackend:
    time.sleep(1.0)
    return FakeBackend()


class FakeWorker:
    def __init__(self, side: str, *, fail_start: bool = False) -> None:
        self.side = side
        self.positions = np.zeros(7, dtype=np.float64)
        self.is_alive = False
        self.armed = False
        self.idle = True
        self.fail_start = fail_start
        self.fail_send = False
        self.fault: str | None = None
        self.state_sequence = 0
        self.control_sequence = 0
        self.last_applied_sequence = -1
        self.last_applied_positions: tuple[float, ...] | None = None
        self.commands: list[ArmCommand] = []
        self.disarm_calls = 0
        self.close_calls = 0

    def _state(self) -> ArmState:
        self.state_sequence += 1
        return ArmState(
            sequence=self.state_sequence,
            timestamp_ns=time.monotonic_ns(),
            positions=tuple(float(value) for value in self.positions),
            ready=self.is_alive,
            armed=self.armed,
            idle=self.idle,
            control_sequence=self.control_sequence,
            last_applied_command_sequence=self.last_applied_sequence,
            last_applied_positions=self.last_applied_positions,
            fault=self.fault,
        )

    def start(self, timeout_s: float) -> ArmState:
        del timeout_s
        if self.fail_start:
            raise RuntimeError(f"{self.side} startup failed")
        self.is_alive = True
        return self._state()

    def arm(self, timeout_s: float) -> ArmState:
        del timeout_s
        self.control_sequence += 1
        self.armed = True
        self.idle = True
        self.fault = None
        return self._state()

    def disarm(self, timeout_s: float) -> ArmState:
        del timeout_s
        self.disarm_calls += 1
        self.control_sequence += 1
        self.armed = False
        self.idle = True
        return self._state()

    def latest_state(self, timeout_s: float) -> ArmState:
        del timeout_s
        if not self.is_alive:
            raise RuntimeError(f"{self.side} worker failed")
        return self._state()

    def send_command(self, command: ArmCommand) -> None:
        if self.fail_send:
            raise RuntimeError(f"{self.side} send failed")
        self.commands.append(command)
        self.positions = np.asarray(command.positions, dtype=np.float64)
        self.last_applied_sequence = command.sequence
        self.last_applied_positions = command.positions
        self.idle = False

    def wait_applied(self, sequence: int, timeout_s: float) -> ArmState:
        del timeout_s
        if self.last_applied_sequence != sequence:
            raise TimeoutError(f"{self.side} command was not applied")
        return self._state()

    def close(self, timeout_s: float) -> None:
        del timeout_s
        self.close_calls += 1
        self.armed = False
        self.idle = True
        self.is_alive = False


class FakeCamera:
    def __init__(self, height: int, width: int, *, fail_connect: bool = False) -> None:
        self.height = height
        self.width = width
        self.fail_connect = fail_connect
        self.is_connected = False
        self.disconnect_calls = 0

    def connect(self) -> None:
        self.is_connected = True
        if self.fail_connect:
            raise RuntimeError("camera startup failed")

    def async_read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


def make_config(tmp_path, **overrides) -> BiYAMFollowerConfig:
    values = {
        "calibration_dir": tmp_path,
        "command_lead_time_s": 0.0,
        "left_joint_limits": [(-1.0, 1.0)] * 6,
        "right_joint_limits": [(-1.0, 1.0)] * 6,
        "max_joint_delta": 0.2,
        "max_gripper_delta": 0.1,
    }
    values.update(overrides)
    return BiYAMFollowerConfig(**values)


def make_robot(tmp_path, *, config=None, cameras=None, worker_options=None):
    workers: dict[str, FakeWorker] = {}
    worker_options = worker_options or {}

    def worker_factory(side, _config):
        worker = FakeWorker(side, **worker_options.get(side, {}))
        workers[side] = worker
        return worker

    cameras = cameras or {}
    robot = BiYAMFollower(
        config or make_config(tmp_path),
        worker_factory=worker_factory,
        camera_factory=lambda _configs: cameras,
    )
    return robot, workers


def zero_action() -> dict[str, float]:
    return dict.fromkeys(YAM_SCALAR_KEYS, 0.0)


def test_config_registration_and_exact_feature_order(tmp_path):
    camera_config = SimpleNamespace(fps=30, width=640, height=360, use_rgb=True, use_depth=False)
    config = make_config(tmp_path, cameras={"top": camera_config})
    robot, _ = make_robot(tmp_path, config=config, cameras={"top": FakeCamera(360, 640)})

    assert config.type == "bi_yam_follower"
    assert list(robot.action_features) == list(YAM_SCALAR_KEYS)
    assert list(robot.observation_features) == [*YAM_SCALAR_KEYS, "top"]
    assert robot.observation_features["top"] == (360, 640, 3)


def test_standard_robot_factory_discovers_bi_yam_without_i2rt(tmp_path):
    robot = make_robot_from_config(make_config(tmp_path))

    assert isinstance(robot, BiYAMFollower)
    assert not robot.is_connected


def test_connect_stays_disarmed_and_observation_uses_ordered_state_and_camera(tmp_path):
    camera_config = SimpleNamespace(fps=30, width=4, height=3, use_rgb=True, use_depth=False)
    camera = FakeCamera(3, 4)
    config = make_config(tmp_path, cameras={"top": camera_config})
    robot, workers = make_robot(tmp_path, config=config, cameras={"top": camera})

    robot.connect()
    workers["left"].positions = np.arange(7, dtype=np.float64) / 10
    workers["right"].positions = np.arange(7, 14, dtype=np.float64) / 10
    observation = robot.get_observation()

    assert not robot.is_armed
    assert list(observation) == [*YAM_SCALAR_KEYS, "top"]
    assert [observation[key] for key in YAM_SCALAR_KEYS] == pytest.approx(np.arange(14) / 10)
    assert observation["top"].shape == (3, 4, 3)
    assert set(robot.state_metadata) == {"left", "right"}

    with pytest.raises(RuntimeError, match="arm it locally"):
        robot.send_action(zero_action())
    robot.disconnect()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda action: action.pop("left_joint_0.pos"), "missing"),
        (lambda action: action.update({"unexpected.pos": 0.0}), "extra"),
        (lambda action: action.update({"left_joint_0.pos": float("nan")}), "finite"),
        (lambda action: action.update({"left_joint_0.pos": object()}), "numeric"),
    ],
)
def test_send_action_rejects_invalid_schema_and_values(tmp_path, mutate, message):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    action = zero_action()
    mutate(action)

    with pytest.raises(ValueError, match=message):
        robot.send_action(action)
    assert workers["left"].commands == []
    assert workers["right"].commands == []
    assert not robot.is_armed
    robot.disconnect()


def test_send_action_clips_limits_and_delta_and_synchronizes_workers(tmp_path):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    action = dict.fromkeys(YAM_SCALAR_KEYS, 10.0)

    applied = robot.send_action(action)

    expected = [0.2] * 6 + [0.1] + [0.2] * 6 + [0.1]
    assert list(applied) == list(YAM_SCALAR_KEYS)
    assert list(applied.values()) == pytest.approx(expected)
    left_command = workers["left"].commands[-1]
    right_command = workers["right"].commands[-1]
    assert left_command.sequence == right_command.sequence
    assert left_command.execute_at_ns == right_command.execute_at_ns
    assert (*left_command.positions, *right_command.positions) == pytest.approx(expected)
    robot.disconnect()


def test_operational_limit_clipping_is_reflected_in_returned_action(tmp_path):
    config = make_config(tmp_path, max_joint_delta=10.0, max_gripper_delta=10.0)
    robot, _ = make_robot(tmp_path, config=config)
    robot.connect()
    robot.arm()

    applied = robot.send_action(dict.fromkeys(YAM_SCALAR_KEYS, 20.0))

    assert list(applied.values()) == pytest.approx([1.0] * 6 + [1.0] + [1.0] * 6 + [1.0])
    robot.disconnect()


def test_worker_runtime_uses_latest_command_and_stale_watchdog_enters_idle():
    backend = FakeBackend()
    runtime = ArmWorkerRuntime(backend, command_ttl_s=0.1)
    runtime.handle_control(ArmControl(sequence=1, operation="arm"))
    runtime.offer_command(ArmCommand(sequence=1, execute_at_ns=0, positions=(0.1,) * 7))
    runtime.offer_command(ArmCommand(sequence=2, execute_at_ns=0, positions=(0.2,) * 7))

    applied = runtime.tick(now_ns=1)
    stale = runtime.tick(now_ns=100_000_002)

    assert len(backend.commands) == 1
    assert backend.commands[0] == pytest.approx(np.full(7, 0.2))
    assert applied.last_applied_command_sequence == 2
    assert stale.fault == "command_ttl_expired"
    assert not stale.armed
    assert stale.idle
    assert backend.idle_calls >= 2
    runtime.close()
    assert backend.close_calls == 1


def test_process_worker_ipc_applies_command_and_reports_ttl_fault():
    worker = ProcessArmWorker(
        "left",
        YAMArmConfig(channel="unused", sim=True, command_ttl_s=0.05, worker_poll_interval_s=0.002),
        context=mp.get_context("spawn"),
        backend_factory=make_fake_backend,
    )
    try:
        assert worker.start(timeout_s=2.0).ready
        assert worker.arm(timeout_s=1.0).armed
        command = ArmCommand(sequence=7, execute_at_ns=time.monotonic_ns(), positions=(0.25,) * 7)
        worker.send_command(command)
        applied = worker.wait_applied(sequence=7, timeout_s=1.0)
        assert applied.last_applied_positions == command.positions

        deadline = time.monotonic() + 1.0
        while True:
            state = worker.latest_state(timeout_s=0.2)
            if state.fault == "command_ttl_expired":
                break
            if time.monotonic() >= deadline:
                pytest.fail("worker command TTL did not expire")
        assert not state.armed
        assert state.idle
    finally:
        worker.close(timeout_s=1.0)


def test_process_worker_startup_timeout_terminates_child():
    worker = ProcessArmWorker(
        "left",
        YAMArmConfig(channel="unused"),
        context=mp.get_context("spawn"),
        backend_factory=make_slow_backend,
    )

    with pytest.raises(TimeoutError, match="Timed out"):
        worker.start(timeout_s=0.05)
    assert not worker.is_alive


@pytest.mark.skipif(importlib.util.find_spec("i2rt") is None, reason="i2rt is optional")
def test_i2rt_sim_backend_matches_worker_contract():
    backend = make_i2rt_backend(YAMArmConfig(channel="unused", sim=True))
    runtime = ArmWorkerRuntime(backend, command_ttl_s=0.2)
    try:
        initial = runtime.snapshot(time.monotonic_ns())
        assert len(initial.positions) == 7
        runtime.handle_control(ArmControl(sequence=1, operation="arm"))
        command = ArmCommand(
            sequence=1,
            execute_at_ns=0,
            positions=initial.positions,
        )
        runtime.offer_command(command)
        applied = runtime.tick(time.monotonic_ns())
        assert applied.last_applied_command_sequence == 1
        assert applied.last_applied_positions == initial.positions
    finally:
        runtime.close()


def test_worker_failure_idles_peer_and_disconnect_still_cleans_up(tmp_path):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    workers["left"].is_alive = False

    with pytest.raises(RuntimeError, match="dead arm workers"):
        robot.get_observation()

    assert robot.is_connected
    assert not robot.is_armed
    assert workers["right"].disarm_calls >= 1
    robot.disconnect()
    assert workers["left"].close_calls == 1
    assert workers["right"].close_calls == 1


def test_local_rearm_clears_recoverable_worker_fault(tmp_path):
    robot, workers = make_robot(tmp_path)
    robot.connect()
    robot.arm()
    for worker in workers.values():
        worker.armed = False
        worker.idle = True
        worker.fault = "command_ttl_expired"

    robot.arm()

    assert robot.is_armed
    assert all(worker.armed and worker.fault is None for worker in workers.values())
    robot.disconnect()


def test_connect_failure_transactionally_closes_workers_and_cameras(tmp_path):
    first_camera = FakeCamera(3, 4)
    failing_camera = FakeCamera(3, 4, fail_connect=True)
    camera_configs = {
        "top": SimpleNamespace(fps=30, width=4, height=3),
        "left": SimpleNamespace(fps=30, width=4, height=3),
    }
    robot, workers = make_robot(
        tmp_path,
        config=make_config(tmp_path, cameras=camera_configs),
        cameras={"top": first_camera, "left": failing_camera},
    )

    with pytest.raises(RuntimeError, match="camera startup failed"):
        robot.connect()

    assert not robot.is_connected
    assert first_camera.disconnect_calls == 1
    assert failing_camera.disconnect_calls == 1
    assert workers["left"].close_calls == 1
    assert workers["right"].close_calls == 1


def test_second_worker_start_failure_closes_first_worker(tmp_path):
    robot, workers = make_robot(tmp_path, worker_options={"right": {"fail_start": True}})

    with pytest.raises(RuntimeError, match="right startup failed"):
        robot.connect()

    assert workers["left"].close_calls == 1
    assert workers["right"].close_calls == 1


def test_i2rt_factory_forwards_hardware_and_sim_configuration(monkeypatch):
    calls = []
    backend = FakeBackend()

    class FakeArmType:
        @classmethod
        def from_string_name(cls, value):
            return f"arm:{value}"

    class FakeGripperType:
        @classmethod
        def from_string_name(cls, value):
            return f"gripper:{value}"

    i2rt = types.ModuleType("i2rt")
    robots = types.ModuleType("i2rt.robots")
    get_robot = types.ModuleType("i2rt.robots.get_robot")
    utils = types.ModuleType("i2rt.robots.utils")
    i2rt.__path__ = []
    robots.__path__ = []
    get_robot.get_yam_robot = lambda **kwargs: calls.append(kwargs) or backend
    utils.ArmType = FakeArmType
    utils.GripperType = FakeGripperType
    monkeypatch.setitem(sys.modules, "i2rt", i2rt)
    monkeypatch.setitem(sys.modules, "i2rt.robots", robots)
    monkeypatch.setitem(sys.modules, "i2rt.robots.get_robot", get_robot)
    monkeypatch.setitem(sys.modules, "i2rt.robots.utils", utils)

    result = make_i2rt_backend(
        YAMArmConfig(
            channel="can7",
            arm_type="yam_pro",
            gripper_type="linear_4310",
            sim=True,
            enable_auto_recovery=True,
        )
    )

    assert result is backend
    assert calls == [
        {
            "channel": "can7",
            "arm_type": "arm:yam_pro",
            "gripper_type": "gripper:linear_4310",
            "zero_gravity_mode": True,
            "sim": True,
            "enable_auto_recovery": True,
        }
    ]
