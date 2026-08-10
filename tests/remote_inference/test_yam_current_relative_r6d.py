# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_ACTION_NAMES,
    UMI_CURRENTREL_STATE_NAMES,
    encode_relative_pose,
)
from lerobot.remote_inference.backend import (
    DeterministicPolicyBackend,
    DeterministicPolicyBackendConfig,
)
from lerobot.remote_inference.schema import ImageEncoding, PolicyActionChunk
from lerobot.remote_inference.server import RemotePolicyServerConfig, create_grpc_server
from lerobot.remote_inference.yam_current_relative_r6d import (
    YamCurrentRelativeR6DAdapter,
    YamJointProgressWatchdog,
    prepare_policy_image,
    rate_limit_action_chunk,
)
from lerobot.remote_inference.yam_umi_ee_bridge import YAM_SCALAR_KEYS, GripperMap, IkResidualError
from lerobot.rollout.inference.remote import RemoteEngineSettings, _ObservationSnapshot
from lerobot.rollout.inference.yam_current_relative_r6d import (
    YamCurrentRelativeR6DRemoteInferenceEngine,
    YamCurrentRelativeR6DSettings,
)


class CartesianFakeKinematics:
    """A reversible test arm whose first three joints are Cartesian position."""

    def fk(self, joints_rad: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = np.asarray(joints_rad[:3], dtype=np.float64)
        return transform

    def ik(self, target_pose: np.ndarray, seed_rad: np.ndarray, *, check: bool = True) -> np.ndarray:
        del check
        solution = np.asarray(seed_rad, dtype=np.float64).copy()
        solution[:3] = target_pose[:3, 3]
        return solution

    def residual(self, joints_rad: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
        reached = self.fk(joints_rad)
        return float(np.linalg.norm(reached[:3, 3] - target_pose[:3, 3])), 0.0


class FailingKinematics(CartesianFakeKinematics):
    def ik(self, target_pose: np.ndarray, seed_rad: np.ndarray, *, check: bool = True) -> np.ndarray:
        del target_pose, seed_rad, check
        raise IkResidualError("test residual")


class FailOnCallKinematics(CartesianFakeKinematics):
    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def ik(self, target_pose: np.ndarray, seed_rad: np.ndarray, *, check: bool = True) -> np.ndarray:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise IkResidualError("test residual")
        return super().ik(target_pose, seed_rad, check=check)


def _adapter() -> YamCurrentRelativeR6DAdapter:
    return YamCurrentRelativeR6DAdapter(
        left=CartesianFakeKinematics(),
        right=CartesianFakeKinematics(),
        flange_to_tcp=np.eye(4),
        gripper=GripperMap(a_left=1.0, b_left=0.0, a_right=1.0, b_right=0.0),
    )


class ExecutionWindowAdapter:
    """Record how many policy rows reach strict IK."""

    def __init__(self) -> None:
        self.inner = _adapter()
        self.decode_lengths: list[int] = []

    def build_query(self, observation: dict, previous_observation: dict | None):
        return self.inner.build_query(observation, previous_observation)

    def decode_action_chunk(self, actions: np.ndarray, query):
        self.decode_lengths.append(len(actions))
        return self.inner.decode_action_chunk(actions, query)


def _observation(left_xyz=(0.1, 0.5, 0.8), right_xyz=(0.2, 0.6, 0.9)) -> dict:
    values = [*left_xyz, 0.0, 0.0, 0.0, 0.7, *right_xyz, 0.0, 0.0, 0.0, 0.8]
    return dict(zip(YAM_SCALAR_KEYS, values, strict=True))


def _identity_action_row() -> np.ndarray:
    pose = encode_relative_pose(np.eye(4))
    return np.asarray([*pose, 0.7, *pose, 0.8], dtype=np.float32)


def test_query_state_uses_inverse_current_times_previous():
    adapter = _adapter()
    previous = _observation(left_xyz=(0.09, 0.48, 0.77), right_xyz=(0.18, 0.57, 0.86))
    current = _observation()

    query = adapter.build_query(current, previous)

    assert query.state.shape == (20,)
    assert query.state.dtype == np.float32
    assert np.allclose(query.state[:3], [-0.01, -0.02, -0.03])
    assert np.allclose(query.state[3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert np.allclose(query.state[10:13], [-0.02, -0.03, -0.04])
    assert np.allclose(query.state[13:19], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_action_rows_are_each_composed_from_the_same_query_anchor():
    adapter = _adapter()
    query = adapter.build_query(_observation(), None)
    first = _identity_action_row()
    second = _identity_action_row()
    first[:3] = [0.01, 0.02, 0.03]
    second[:3] = [0.02, 0.04, 0.06]
    first[10:13] = [-0.01, 0.01, 0.02]
    second[10:13] = [-0.02, 0.02, 0.04]

    decoded = adapter.decode_action_chunk(np.stack((first, second)), query)

    assert np.allclose(decoded.actions[0, :3], [0.11, 0.52, 0.83])
    assert np.allclose(decoded.actions[1, :3], [0.12, 0.54, 0.86])
    assert np.allclose(decoded.actions[1, 7:10], [0.18, 0.62, 0.94])
    assert np.allclose(decoded.position_residual_m, 0.0)


def test_ik_failure_identifies_action_row_arm_and_relative_target():
    adapter = YamCurrentRelativeR6DAdapter(
        left=FailingKinematics(),
        right=CartesianFakeKinematics(),
        flange_to_tcp=np.eye(4),
        gripper=GripperMap(a_left=1.0, b_left=0.0, a_right=1.0, b_right=0.0),
    )
    row = _identity_action_row()
    row[:3] = [0.01, -0.02, 0.03]
    query = adapter.build_query(_observation(), None)

    with pytest.raises(
        IkResidualError,
        match=r"action row 1/1, left arm, relative TCP translation \[ 0.01,-0.02, 0.03\] m",
    ):
        adapter.decode_action_chunk(row[None, :], query)


def test_ik_failure_truncates_only_the_invalid_action_suffix(caplog):
    adapter = YamCurrentRelativeR6DAdapter(
        left=CartesianFakeKinematics(),
        right=FailOnCallKinematics(fail_on_call=3),
        flange_to_tcp=np.eye(4),
        gripper=GripperMap(a_left=1.0, b_left=0.0, a_right=1.0, b_right=0.0),
    )
    rows = np.stack([_identity_action_row()] * 4)
    query = adapter.build_query(_observation(), None)

    decoded = adapter.decode_action_chunk(rows, query)

    assert decoded.actions.shape == (2, 14)
    assert decoded.position_residual_m.shape == (2, 2)
    assert decoded.orientation_residual_rad.shape == (2, 2)
    assert "Truncating current-relative action chunk to 2 valid rows" in caplog.text


def test_current_relative_ik_uses_ten_refinement_iterations_by_default():
    assert YamCurrentRelativeR6DSettings().ik_iterations == 10


def test_prepare_policy_image_resizes_to_training_dimensions():
    source = np.zeros((720, 1280, 3), dtype=np.uint8)
    source[:, :, 1] = 127

    prepared = prepare_policy_image(source, width=800, height=600)

    assert prepared.shape == (600, 800, 3)
    assert prepared.dtype == np.uint8
    assert prepared.flags.c_contiguous
    assert np.all(prepared[:, :, 1] == 127)


def test_rate_limiter_expands_waypoints_without_changing_vector_direction():
    initial = np.zeros(14, dtype=np.float32)
    target = np.asarray([0.03] * 6 + [0.08] + [-0.03] * 6 + [0.08], dtype=np.float32)

    limited = rate_limit_action_chunk(
        target[None, :],
        initial,
        max_joint_delta=0.02,
        max_gripper_delta=0.05,
        max_dispatches_per_waypoint=4,
    )

    assert limited.dispatches_per_waypoint.tolist() == [2]
    assert np.allclose(limited.actions[0], target / 2)
    assert np.allclose(limited.actions[1], target)
    assert np.abs(np.diff(np.vstack((initial, limited.actions)), axis=0)[:, :6]).max() <= 0.02


def test_rate_limiter_rejects_waypoint_requiring_more_than_four_dispatches():
    target = np.zeros((1, 14), dtype=np.float32)
    target[0, 0] = 0.081

    with pytest.raises(ValueError, match="waypoint 1 requires 5 dispatches"):
        rate_limit_action_chunk(
            target,
            np.zeros(14, dtype=np.float32),
            max_joint_delta=0.02,
            max_gripper_delta=0.05,
            max_dispatches_per_waypoint=4,
        )


def test_progress_watchdog_counts_only_consecutive_stalled_dispatches():
    watchdog = YamJointProgressWatchdog(max_hold_steps=2, target_tolerance_rad=0.002)
    target = np.zeros(14, dtype=np.float32)
    target[0] = 0.02
    before = np.zeros(14, dtype=np.float32)

    watchdog.observe(target, before, np.asarray([0.005] + [0.0] * 13))
    assert watchdog.stalled_steps == 0

    watchdog.observe(target, before, before)
    assert watchdog.stalled_steps == 1
    with pytest.raises(RuntimeError, match="no measurable progress for 2 dispatches"):
        watchdog.observe(target, before, before)


def test_progress_watchdog_accepts_a_target_already_within_tolerance():
    watchdog = YamJointProgressWatchdog(max_hold_steps=1, target_tolerance_rad=0.002)
    target = np.zeros(14, dtype=np.float32)
    target[0] = 0.001

    watchdog.observe(target, np.zeros(14), np.zeros(14))

    assert watchdog.stalled_steps == 0


class FakeRobot:
    id = "dual-yam-test"
    robot_type = "bi_yam_follower"


class FakeRobotWrapper:
    def __init__(self) -> None:
        self.inner = FakeRobot()
        self.robot_type = self.inner.robot_type
        self.observation_features = {
            **dict.fromkeys(YAM_SCALAR_KEYS, float),
            "left": (720, 1280, 3),
            "right": (720, 1280, 3),
        }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.01)


def test_specialized_engine_runs_transport_and_returns_yam_actions():
    port = _free_port()
    backend = DeterministicPolicyBackend(
        DeterministicPolicyBackendConfig(
            model_id="test/currentrel-r6d",
            action_horizon=24,
            state_features=UMI_CURRENTREL_STATE_NAMES,
            action_features=UMI_CURRENTREL_ACTION_NAMES,
            camera_keys=("umi1", "umi2"),
        )
    )
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port), backend)
    server.start()
    settings = RemoteEngineSettings(
        server_address=f"127.0.0.1:{port}",
        schema_id="ignored-by-specialized-engine",
        requested_model_id="test/currentrel-r6d",
        client_instance_id="test-client",
        connect_timeout_s=1.0,
        inference_timeout_s=1.0,
        max_message_bytes=8 * 1024 * 1024,
        jpeg_quality=90,
        tls_root_cert_path=None,
        tls_client_cert_path=None,
        tls_client_key_path=None,
        tls_server_name_override=None,
        execution_horizon=15,
        image_encoding=ImageEncoding.JPEG,
        camera_calibration_sha256={},
    )
    adapter = ExecutionWindowAdapter()
    engine = YamCurrentRelativeR6DRemoteInferenceEngine(
        settings=settings,
        yam_settings=YamCurrentRelativeR6DSettings(),
        robot_wrapper=FakeRobotWrapper(),
        dataset_features={},
        ordered_action_keys=list(YAM_SCALAR_KEYS),
        task="Put all oranges in the bowl",
        fps=30.0,
        adapter=adapter,
    )
    observation = {
        **_observation(),
        "left": np.zeros((720, 1280, 3), dtype=np.uint8),
        "right": np.zeros((720, 1280, 3), dtype=np.uint8),
    }
    try:
        engine.start()
        engine.resume()
        engine.notify_observation(observation)
        _wait_for(lambda: engine.action_queue_depth == 15)
        action = engine.get_action(None)
        assert action is not None
        assert action.shape == (14,)
        assert np.allclose(action.numpy(), list(_observation().values()))
        assert adapter.decode_lengths == [24]
        assert not engine.failed
    finally:
        engine.stop()
        server.stop(grace=0).wait()


def test_specialized_engine_latency_aligns_model_rows_before_rate_limiting():
    settings = RemoteEngineSettings(
        server_address="127.0.0.1:1",
        schema_id="ignored-by-specialized-engine",
        requested_model_id="test/currentrel-r6d",
        client_instance_id="test-client",
        connect_timeout_s=1.0,
        inference_timeout_s=1.0,
        max_message_bytes=8 * 1024 * 1024,
        jpeg_quality=90,
        tls_root_cert_path=None,
        tls_client_cert_path=None,
        tls_client_key_path=None,
        tls_server_name_override=None,
        execution_horizon=15,
        image_encoding=ImageEncoding.JPEG,
        camera_calibration_sha256={},
    )
    adapter = _adapter()
    engine = YamCurrentRelativeR6DRemoteInferenceEngine(
        settings=settings,
        yam_settings=YamCurrentRelativeR6DSettings(),
        robot_wrapper=FakeRobotWrapper(),
        dataset_features={},
        ordered_action_keys=list(YAM_SCALAR_KEYS),
        task="Put all oranges in the bowl",
        fps=30.0,
        adapter=adapter,
    )
    observation = {
        **_observation(left_xyz=(0.0, 0.5, 0.8)),
        "left": np.zeros((720, 1280, 3), dtype=np.uint8),
        "right": np.zeros((720, 1280, 3), dtype=np.uint8),
    }
    snapshot = _ObservationSnapshot(
        sequence=7,
        capture_tick=3,
        capture_monotonic_ns=1,
        values=observation,
    )
    engine._queries[snapshot.sequence] = adapter.build_query(observation, None)
    rows = np.stack([_identity_action_row()] * 24)
    rows[:, 0] = np.arange(1, 25, dtype=np.float32) * 0.015
    chunk = PolicyActionChunk(
        observation_sequence=snapshot.sequence,
        first_action_tick=snapshot.capture_tick,
        actions=rows,
        model_fingerprint="test",
    )

    decoded = engine._prepare_chunk_for_execution(chunk, snapshot)
    engine._latest_observation = snapshot
    engine._last_dispatched_action = np.asarray(
        [observation[key] for key in YAM_SCALAR_KEYS], dtype=np.float64
    )
    executable = engine._future_actions_for_execution(decoded, elapsed_steps=5)

    assert decoded.actions.shape == (24, 14)
    assert executable.shape == (19, 14)
    assert np.allclose(executable[-1], decoded.actions[19])
    assert not np.allclose(executable[-1], decoded.actions[-1])
    dispatch_steps = np.diff(np.vstack((engine._last_dispatched_action, executable)), axis=0)
    assert np.abs(dispatch_steps[:, [*range(6), *range(7, 13)]]).max() <= 0.02 + 1e-7
