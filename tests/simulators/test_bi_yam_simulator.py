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

from lerobot.robots.bi_yam import (
    YAM_SCALAR_KEYS,
    BiYAMSimulatorRobot,
    BiYAMSimulatorRobotConfig,
)
from lerobot.robots.utils import make_robot_from_config
from lerobot.simulators.bi_yam import (
    CAMERA_NAMES,
    BiYAMFaults,
    BiYAMSimulator,
    BiYAMSimulatorConfig,
    mujoco_backend_available,
)


@pytest.fixture
def config() -> BiYAMSimulatorConfig:
    return BiYAMSimulatorConfig(
        physics_hz=300,
        control_hz=100,
        observation_hz=25,
        camera_height=24,
        camera_width=32,
        seed=17,
    )


@pytest.fixture
def simulator(config: BiYAMSimulatorConfig):
    sim = BiYAMSimulator(config, backend="fallback")
    sim.start()
    yield sim
    sim.close()


def _motion_target(state: np.ndarray) -> np.ndarray:
    target = state.copy()
    target[0] += 0.35
    target[1] += 0.20
    target[6] = 0.8
    target[7] -= 0.30
    target[8] += 0.15
    target[13] = 0.2
    return target


def test_api_contract_lifecycle_and_cleanup(config: BiYAMSimulatorConfig) -> None:
    sim = BiYAMSimulator(config, backend="fallback")
    assert not sim.is_connected
    with pytest.raises(RuntimeError, match="not connected"):
        sim.get_state()

    sim.connect()
    assert sim.is_connected
    assert not sim.is_running
    with pytest.raises(RuntimeError, match="not running"):
        sim.apply_action(np.asarray(config.initial_state, dtype=np.float32))

    sim.start()
    state = sim.get_state()
    health = sim.get_health()
    assert state.shape == (14,)
    assert state.dtype == np.float32
    assert health.backend == "fallback"
    assert health.healthy
    assert health.safe_idle
    assert health.sim_time_ns == health.state_timestamp_ns == 0

    sim.close()
    sim.close()
    assert not sim.is_connected
    assert not sim.is_running


def test_context_manager_starts_and_closes(config: BiYAMSimulatorConfig) -> None:
    sim = BiYAMSimulator(config, backend="fallback")
    with sim as active:
        assert active is sim
        assert sim.is_running
    assert not sim.is_connected


def test_reset_is_deterministic(config: BiYAMSimulatorConfig) -> None:
    sim = BiYAMSimulator(config, backend="fallback")
    sim.start()
    initial_state = sim.get_state()
    initial_frames = sim.render_cameras()

    target = _motion_target(initial_state)
    for _ in range(8):
        sim.apply_action(target)
    assert not np.array_equal(sim.get_state(), initial_state)

    sim.reset()
    reset_frames = sim.render_cameras()
    np.testing.assert_array_equal(sim.get_state(), initial_state)
    for name in CAMERA_NAMES:
        np.testing.assert_array_equal(reset_frames[name], initial_frames[name])
    assert sim.get_health().sim_time_ns == 0

    sim.reset(seed=config.seed + 1)
    changed_seed_frames = sim.render_cameras()
    assert any(not np.array_equal(changed_seed_frames[name], initial_frames[name]) for name in CAMERA_NAMES)
    sim.close()


def test_apply_action_uses_stepped_position_dynamics(simulator: BiYAMSimulator) -> None:
    initial = simulator.get_state()
    target = _motion_target(initial)

    simulator.apply_action(target)
    first_step = simulator.get_state()
    assert first_step[0] > initial[0]
    assert first_step[7] < initial[7]
    assert not np.allclose(first_step, target)

    first_error = np.linalg.norm(target - first_step)
    for _ in range(20):
        simulator.apply_action(target)
    assert np.linalg.norm(target - simulator.get_state()) < first_error
    assert simulator.get_health().state_timestamp_ns > 0


@pytest.mark.parametrize(
    "action, message",
    [
        (np.zeros(13, dtype=np.float32), "shape"),
        (np.full(14, np.nan, dtype=np.float32), "finite"),
        (np.asarray((0.0, 1.2, 1.8, 0.0, 0.0, 0.0, 1.1) * 2, dtype=np.float32), "limits"),
    ],
)
def test_action_validation(simulator: BiYAMSimulator, action: np.ndarray, message: str) -> None:
    before = simulator.get_state()
    with pytest.raises(ValueError, match=message):
        simulator.apply_action(action)
    np.testing.assert_array_equal(simulator.get_state(), before)


def test_camera_schema_and_frames_are_copies(simulator: BiYAMSimulator, config: BiYAMSimulatorConfig) -> None:
    frames = simulator.render_cameras()
    assert tuple(frames) == CAMERA_NAMES
    assert simulator.get_health().camera_timestamps_ns.keys() == frames.keys()
    for frame in frames.values():
        assert frame.shape == (config.camera_height, config.camera_width, 3)
        assert frame.dtype == np.uint8
        assert frame.flags.c_contiguous

    frames["top"].fill(0)
    assert np.any(simulator.render_cameras()["top"])


def test_frozen_camera_and_stale_state_faults(simulator: BiYAMSimulator) -> None:
    initial_state = simulator.get_state()
    initial_frames = simulator.render_cameras()
    initial_health = simulator.get_health()
    simulator.set_faults(BiYAMFaults(frozen_cameras={"top"}, stale_state=True))

    target = _motion_target(initial_state)
    for _ in range(8):
        simulator.apply_action(target)
    fault_frames = simulator.render_cameras()
    fault_health = simulator.get_health()

    np.testing.assert_array_equal(simulator.get_state(), initial_state)
    np.testing.assert_array_equal(fault_frames["top"], initial_frames["top"])
    assert not np.array_equal(fault_frames["left"], initial_frames["left"])
    assert fault_health.state_timestamp_ns == initial_health.state_timestamp_ns
    assert fault_health.camera_timestamps_ns["top"] == initial_health.camera_timestamps_ns["top"]
    assert fault_health.camera_timestamps_ns["left"] > initial_health.camera_timestamps_ns["left"]

    simulator.clear_faults()
    simulator.apply_action(target)
    assert not np.array_equal(simulator.get_state(), initial_state)


@pytest.mark.parametrize(
    ("faults", "failed_slice", "moving_index"),
    [
        (BiYAMFaults(left_worker_failed=True), slice(0, 7), 7),
        (BiYAMFaults(right_worker_failed=True), slice(7, 14), 0),
    ],
)
def test_worker_failure_freezes_failed_side(
    simulator: BiYAMSimulator,
    faults: BiYAMFaults,
    failed_slice: slice,
    moving_index: int,
) -> None:
    initial = simulator.get_state()
    simulator.set_faults(faults)
    target = _motion_target(initial)
    for _ in range(10):
        simulator.apply_action(target)

    state = simulator.get_state()
    np.testing.assert_array_equal(state[failed_slice], initial[failed_slice])
    assert state[moving_index] != pytest.approx(initial[moving_index])
    health = simulator.get_health()
    assert health.left_worker_ok is not faults.left_worker_failed
    assert health.right_worker_ok is not faults.right_worker_failed
    assert not health.healthy


def test_delayed_commands_and_safe_idle(simulator: BiYAMSimulator) -> None:
    initial = simulator.get_state()
    target = _motion_target(initial)
    simulator.set_faults(BiYAMFaults(command_delay_steps=2))

    simulator.apply_action(target)
    np.testing.assert_array_equal(simulator.get_state(), initial)
    simulator.apply_action(target)
    np.testing.assert_array_equal(simulator.get_state(), initial)
    assert simulator.get_health().pending_commands == 2

    simulator.apply_action(target)
    assert not np.array_equal(simulator.get_state(), initial)
    simulator.safe_idle()
    health = simulator.get_health()
    assert health.pending_commands == 0
    assert health.safe_idle


def test_stuck_joint_does_not_block_other_joints(simulator: BiYAMSimulator) -> None:
    initial = simulator.get_state()
    target = _motion_target(initial)
    simulator.set_faults(BiYAMFaults(stuck_joints={0, 13}))
    for _ in range(10):
        simulator.apply_action(target)

    state = simulator.get_state()
    assert state[0] == pytest.approx(initial[0])
    assert state[13] == pytest.approx(initial[13])
    assert state[1] != pytest.approx(initial[1])
    assert state[7] != pytest.approx(initial[7])


def test_non_divisible_rates_have_deterministic_timestamp() -> None:
    config = BiYAMSimulatorConfig(
        physics_hz=250,
        control_hz=100,
        observation_hz=30,
        camera_height=8,
        camera_width=8,
    )
    with BiYAMSimulator(config, backend="fallback") as simulator:
        action = simulator.get_state()
        for _ in range(3):
            simulator.apply_action(action)
        assert simulator.get_health().sim_time_ns == 100_000_000


def test_fault_configuration_validation() -> None:
    with pytest.raises(ValueError, match="Unknown cameras"):
        BiYAMFaults(frozen_cameras={"wrist"})
    with pytest.raises(ValueError, match="cannot be negative"):
        BiYAMFaults(command_delay_steps=-1)
    with pytest.raises(ValueError, match="indices"):
        BiYAMFaults(stuck_joints={14})


def test_lerobot_adapter_matches_physical_schema_and_lifecycle(tmp_path) -> None:
    sim_config = BiYAMSimulatorConfig(
        physics_hz=300,
        control_hz=100,
        observation_hz=25,
        camera_height=12,
        camera_width=16,
    )
    config = BiYAMSimulatorRobotConfig(
        calibration_dir=tmp_path,
        backend="fallback",
        simulator=sim_config,
    )
    robot = make_robot_from_config(config)

    assert isinstance(robot, BiYAMSimulatorRobot)
    assert list(robot.action_features) == list(YAM_SCALAR_KEYS)
    assert list(robot.observation_features) == [*YAM_SCALAR_KEYS, *CAMERA_NAMES]
    robot.connect()
    assert not robot.is_armed
    robot.arm()

    observation = robot.get_observation()
    action = dict.fromkeys(YAM_SCALAR_KEYS, 10.0)
    applied = robot.send_action(action)

    assert list(observation) == [*YAM_SCALAR_KEYS, *CAMERA_NAMES]
    assert all(observation[name].shape == (12, 16, 3) for name in CAMERA_NAMES)
    assert max(abs(applied[key] - observation[key]) for key in YAM_SCALAR_KEYS) <= 0.100001
    robot.disconnect()
    assert not robot.is_connected


@pytest.mark.skipif(not mujoco_backend_available(), reason="MuJoCo and i2rt are optional")
def test_mujoco_backend_uses_one_world_and_position_dynamics() -> None:
    import mujoco

    config = BiYAMSimulatorConfig(
        physics_hz=600,
        control_hz=200,
        observation_hz=30,
        camera_height=32,
        camera_width=48,
        seed=5,
    )
    simulator = BiYAMSimulator(config, backend="mujoco")
    simulator.start()
    try:
        model = simulator._backend._model
        for name in ("left_base", "right_base", "table", "red_cube", "blue_cube", "green_cylinder"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        for name in CAMERA_NAMES:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "workspace") >= 0

        initial = simulator.get_state()
        initial_frames = simulator.render_cameras()
        target = _motion_target(initial)
        simulator.apply_action(target)
        first_step = simulator.get_state()
        assert not np.array_equal(first_step, initial)
        assert not np.allclose(first_step, target)

        frames = simulator.render_cameras()
        assert tuple(frames) == CAMERA_NAMES
        assert all(frame.shape == (32, 48, 3) for frame in frames.values())
        assert all(int(frame.max()) > int(frame.min()) for frame in frames.values())

        simulator.reset()
        np.testing.assert_array_equal(simulator.get_state(), initial)
        reset_frames = simulator.render_cameras()
        for name in CAMERA_NAMES:
            np.testing.assert_array_equal(reset_frames[name], initial_frames[name])
    finally:
        simulator.close()
    assert simulator._backend._model is None
