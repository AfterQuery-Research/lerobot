# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import socket
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("datasets", reason="rollout requires the dataset extra")

from lerobot.remote_inference.backend import (
    DeterministicPolicyBackend,
    DeterministicPolicyBackendConfig,
)
from lerobot.remote_inference.schema import ImageEncoding
from lerobot.remote_inference.server import RemotePolicyServerConfig, create_grpc_server
from lerobot.rollout.inference.remote import RemoteEngineSettings, RemoteInferenceEngine

STATE_FEATURES = ("joint_0.pos", "joint_1.pos")
CAMERA_KEYS = ("top",)


class CountingBackend(DeterministicPolicyBackend):
    def __init__(
        self,
        *,
        state_features: tuple[str, ...] = STATE_FEATURES,
        camera_keys: tuple[str, ...] = CAMERA_KEYS,
        action_horizon: int = 4,
    ) -> None:
        super().__init__(
            DeterministicPolicyBackendConfig(
                action_horizon=action_horizon,
                state_features=state_features,
                action_features=state_features,
                camera_keys=camera_keys,
            )
        )
        self.inference_count = 0

    def infer(self, observation):
        self.inference_count += 1
        return super().infer(observation)


class BlockingBackend(CountingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, observation):
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test backend was not released")
        return super().infer(observation)


class FakeRobot:
    id = "fake-robot"
    robot_type = "fake"
    observation_features = {
        "joint_0.pos": float,
        "joint_1.pos": float,
        "top": (3, 4, 3),
    }


class FakeRobotWrapper:
    def __init__(self) -> None:
        self.inner = FakeRobot()
        self.robot_type = self.inner.robot_type
        self.observation_features = self.inner.observation_features


DATASET_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": list(STATE_FEATURES),
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": list(STATE_FEATURES),
    },
    "observation.images.top": {
        "dtype": "image",
        "shape": (3, 4, 3),
        "names": ["height", "width", "channels"],
    },
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _settings(address: str) -> RemoteEngineSettings:
    return RemoteEngineSettings(
        server_address=address,
        schema_id="engine-test-v1",
        requested_model_id="",
        client_instance_id="test-client",
        connect_timeout_s=1.0,
        inference_timeout_s=0.3,
        max_message_bytes=1024 * 1024,
        jpeg_quality=95,
        tls_root_cert_path=None,
        tls_client_cert_path=None,
        tls_client_key_path=None,
        tls_server_name_override=None,
        prefetch_threshold=10,
        image_encoding=ImageEncoding.PNG,
        camera_calibration_sha256={},
    )


def _engine(address: str, shutdown_event: threading.Event | None = None) -> RemoteInferenceEngine:
    return RemoteInferenceEngine(
        settings=_settings(address),
        robot_wrapper=FakeRobotWrapper(),
        dataset_features=DATASET_FEATURES,
        ordered_action_keys=list(STATE_FEATURES),
        task="test task",
        fps=30.0,
        shutdown_event=shutdown_event,
    )


def _observation(value: float = 0.25) -> dict:
    return {
        "joint_0.pos": value,
        "joint_1.pos": value + 0.5,
        "top": np.full((3, 4, 3), 127, dtype=np.uint8),
    }


def _wait_for(predicate, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.01)


def test_remote_engine_runs_real_transport_without_duplicate_requests():
    port = _free_port()
    backend = CountingBackend()
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port), backend)
    server.start()
    engine = _engine(f"127.0.0.1:{port}")
    try:
        engine.start()
        engine.resume()
        engine.notify_observation(_observation())
        _wait_for(lambda: engine.action_queue_depth == 4)

        time.sleep(0.1)
        assert not engine.failed
        assert backend.inference_count == 1
        assert np.allclose(engine.get_action(None).numpy(), [0.25, 0.75])

        engine.notify_observation(_observation(0.5))
        _wait_for(lambda: backend.inference_count == 2)
        assert not engine.failed
    finally:
        engine.stop()
        server.stop(grace=0).wait()


def test_empty_queue_does_not_advance_execution_tick_or_discard_first_chunk():
    port = _free_port()
    backend = BlockingBackend()
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port), backend)
    server.start()
    engine = _engine(f"127.0.0.1:{port}")
    try:
        engine.start()
        engine.resume()
        engine.notify_observation(_observation())
        assert backend.started.wait(timeout=1.0)

        for _ in range(10):
            assert engine.get_action(None) is None

        backend.release.set()
        _wait_for(lambda: engine.action_queue_depth == 4)
        assert np.allclose(engine.get_action(None).numpy(), [0.25, 0.75])
        assert engine.action_queue_depth == 3
    finally:
        backend.release.set()
        engine.stop()
        server.stop(grace=0).wait()


def test_remote_engine_fails_closed_when_server_disappears():
    port = _free_port()
    backend = CountingBackend()
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port), backend)
    server.start()
    shutdown_event = threading.Event()
    engine = _engine(f"127.0.0.1:{port}", shutdown_event)
    try:
        engine.start()
        engine.resume()
        server.stop(grace=0).wait()
        engine.notify_observation(_observation())

        _wait_for(lambda: engine.failed)
        assert shutdown_event.is_set()
        assert engine.action_queue_depth == 0
    finally:
        engine.stop()


def test_full_rollout_controls_shared_world_simulator_over_grpc(monkeypatch, tmp_path):
    from lerobot.robots.bi_yam import YAM_SCALAR_KEYS, BiYAMSimulatorRobotConfig
    from lerobot.rollout import (
        BaseStrategyConfig,
        RemoteInferenceConfig,
        RolloutConfig,
        build_rollout_context,
        create_strategy,
    )
    from lerobot.simulators.bi_yam import CAMERA_NAMES, BiYAMSimulatorConfig

    port = _free_port()
    backend = CountingBackend(
        state_features=YAM_SCALAR_KEYS,
        camera_keys=CAMERA_NAMES,
        action_horizon=30,
    )
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port), backend)
    server.start()
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", "--inference.type=remote"])
    cfg = RolloutConfig(
        robot=BiYAMSimulatorRobotConfig(
            calibration_dir=tmp_path,
            backend="fallback",
            simulator=BiYAMSimulatorConfig(
                physics_hz=300,
                control_hz=100,
                observation_hz=20,
                camera_height=12,
                camera_width=16,
            ),
        ),
        strategy=BaseStrategyConfig(),
        inference=RemoteInferenceConfig(
            server_address=f"127.0.0.1:{port}",
            image_encoding="png",
            inference_timeout_s=1.0,
            prefetch_threshold=20,
        ),
        fps=20.0,
        duration=0.35,
        task="move the blocks",
        return_to_initial_position=False,
    )
    shutdown_event = threading.Event()
    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = create_strategy(cfg.strategy)
    try:
        assert ctx.policy.policy is None
        strategy.setup(ctx)
        assert ctx.hardware.robot_wrapper.inner.is_armed
        strategy.run(ctx)
        assert backend.inference_count > 0
        assert ctx.hardware.robot_wrapper.inner.simulator.get_health().sim_time_ns > 0
    finally:
        strategy.teardown(ctx)
        server.stop(grace=0).wait()
    assert not ctx.hardware.robot_wrapper.is_connected


def test_full_rollout_fails_closed_when_remote_server_stops(monkeypatch, tmp_path):
    from lerobot.robots.bi_yam import YAM_SCALAR_KEYS, BiYAMSimulatorRobotConfig
    from lerobot.rollout import (
        BaseStrategyConfig,
        RemoteInferenceConfig,
        RolloutConfig,
        build_rollout_context,
        create_strategy,
    )
    from lerobot.simulators.bi_yam import CAMERA_NAMES, BiYAMSimulatorConfig

    port = _free_port()
    backend = CountingBackend(
        state_features=YAM_SCALAR_KEYS,
        camera_keys=CAMERA_NAMES,
        action_horizon=30,
    )
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port), backend)
    server.start()
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", "--inference.type=remote"])
    cfg = RolloutConfig(
        robot=BiYAMSimulatorRobotConfig(
            calibration_dir=tmp_path,
            backend="fallback",
            simulator=BiYAMSimulatorConfig(camera_height=12, camera_width=16),
        ),
        strategy=BaseStrategyConfig(),
        inference=RemoteInferenceConfig(
            server_address=f"127.0.0.1:{port}",
            image_encoding="png",
            inference_timeout_s=0.5,
        ),
        fps=20.0,
        duration=2.0,
        task="move the blocks",
        return_to_initial_position=True,
    )
    shutdown_event = threading.Event()
    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = create_strategy(cfg.strategy)
    robot = ctx.hardware.robot_wrapper.inner
    try:
        strategy.setup(ctx)
        assert robot.is_armed
        server.stop(grace=0).wait()

        strategy.run(ctx)

        assert shutdown_event.is_set()
        assert ctx.policy.inference.failed
        sim_time_before_teardown = robot.simulator.get_health().sim_time_ns
    finally:
        strategy.teardown(ctx)
    assert not robot.is_connected
    assert not robot.is_armed
    assert robot.simulator.get_health().sim_time_ns == sim_time_before_teardown
