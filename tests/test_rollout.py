# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        ActionProbeStrategyConfig,
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    action_probe = ActionProbeStrategyConfig()
    assert action_probe.type == "action_probe"
    assert action_probe.reset_robot is False
    assert BaseStrategyConfig().type == "base"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import RemoteInferenceConfig, RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None

    remote = RemoteInferenceConfig()
    assert remote.type == "remote"
    assert remote.server_address == "127.0.0.1:8081"
    assert remote.execution_horizon == 15


def test_remote_rollout_config_does_not_require_local_policy(monkeypatch):
    from lerobot.rollout import RemoteInferenceConfig, RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", "--inference.type=remote"])
    cfg = RolloutConfig(robot=MockRobotConfig(), inference=RemoteInferenceConfig())

    assert cfg.policy is None
    assert cfg.device == "cpu"
    assert cfg.display_data is False


def test_current_relative_rollout_can_start_without_operator_confirmation(monkeypatch):
    from lerobot.rollout import (
        BaseStrategyConfig,
        RolloutConfig,
        YamCurrentRelativeR6DRemoteInferenceConfig,
    )
    from tests.mocks.mock_robot import MockRobotConfig

    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", "--inference.type=remote_yam_currentrel_r6d"])
    strategy = BaseStrategyConfig(require_operator_start=False)
    cfg = RolloutConfig(
        robot=MockRobotConfig(),
        strategy=strategy,
        inference=YamCurrentRelativeR6DRemoteInferenceConfig(),
        fps=30,
    )

    assert cfg.strategy.require_operator_start is False


def test_rollout_logging_creates_shared_timestamped_artifact_directory(tmp_path, monkeypatch):
    import draccus

    import lerobot.scripts.lerobot_rollout as rollout_script
    from lerobot.robots.bi_yam.config_bi_yam import BiYAMFollowerConfig, YAMArmConfig
    from lerobot.rollout import RemoteInferenceConfig, RolloutConfig

    robot = BiYAMFollowerConfig(
        id="test-dual-yam",
        left_arm_config=YAMArmConfig(sim=True),
        right_arm_config=YAMArmConfig(sim=True),
        max_joint_delta=0.15,
        max_gripper_delta=0.05,
    )
    cfg = RolloutConfig(
        robot=robot,
        inference=RemoteInferenceConfig(
            server_address="policy-server.local:8081",
            requested_model_id="test/model",
            execution_horizon=16,
        ),
        enable_logging=True,
        logging_dir=tmp_path,
        fps=20,
        duration=90,
        task="Put all oranges in the bowl",
    )
    configured = []
    monkeypatch.setattr(rollout_script, "init_logging", lambda *, log_file: configured.append(log_file))

    run_dir = rollout_script._configure_rollout_logging(cfg, timestamp="20260805T010203Z")
    config_path = run_dir / "resolved_config.yaml"
    resolved = draccus.parse(RolloutConfig, config_path, args=[])

    assert run_dir == tmp_path / "20260805T010203Z"
    assert run_dir.is_dir()
    assert configured == [run_dir / "rollout.log"]
    assert robot.control_telemetry_path == run_dir / "control.jsonl"
    assert resolved.enable_logging is True
    assert resolved.logging_dir == tmp_path
    assert resolved.fps == 20
    assert resolved.duration == 90
    assert resolved.task == "Put all oranges in the bowl"
    assert resolved.inference.type == "remote"
    assert resolved.inference.server_address == "policy-server.local:8081"
    assert resolved.inference.requested_model_id == "test/model"
    assert resolved.inference.execution_horizon == 16
    assert resolved.robot.type == "bi_yam_follower"
    assert resolved.robot.id == "test-dual-yam"
    assert resolved.robot.max_joint_delta == 0.15
    assert resolved.robot.max_gripper_delta == 0.05
    assert resolved.robot.control_telemetry_path == run_dir / "control.jsonl"


def test_rollout_logging_can_be_disabled(tmp_path, monkeypatch):
    import lerobot.scripts.lerobot_rollout as rollout_script

    robot = SimpleNamespace(control_telemetry_path=None)
    cfg = SimpleNamespace(enable_logging=False, logging_dir=tmp_path, robot=robot)
    configured = []
    monkeypatch.setattr(rollout_script, "init_logging", lambda *, log_file: configured.append(log_file))

    run_dir = rollout_script._configure_rollout_logging(cfg, timestamp="unused")

    assert run_dir is None
    assert configured == [None]
    assert robot.control_telemetry_path is None
    assert list(tmp_path.iterdir()) == []


def test_rollout_logging_flags_parse_with_draccus(tmp_path, monkeypatch):
    import draccus

    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig  # noqa: F401

    args = [
        "--robot.type=mock_robot",
        "--inference.type=remote",
        "--enable_logging=true",
        f"--logging_dir={tmp_path}",
    ]
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", *args])

    cfg = draccus.parse(RolloutConfig, args=args)

    assert cfg.enable_logging is True
    assert cfg.logging_dir == tmp_path


def test_rollout_logs_unexpected_failure_and_tears_down(monkeypatch, caplog):
    import logging

    import lerobot.scripts.lerobot_rollout as rollout_script

    ctx = object()
    strategy = MagicMock()
    strategy.run.side_effect = RuntimeError("synthetic rollout failure")
    cfg = SimpleNamespace(
        display_data=False,
        dataset=None,
        task="test",
        strategy=SimpleNamespace(type="base"),
        robot=SimpleNamespace(type="mock"),
        fps=30.0,
        duration=1.0,
    )
    monkeypatch.setattr(rollout_script, "_configure_rollout_logging", lambda cfg: None)
    monkeypatch.setattr(
        rollout_script,
        "ProcessSignalHandler",
        lambda **kwargs: SimpleNamespace(shutdown_event=object()),
    )
    monkeypatch.setattr(rollout_script, "build_rollout_context", lambda cfg, shutdown_event: ctx)
    monkeypatch.setattr(rollout_script, "create_strategy", lambda strategy_cfg: strategy)

    with (
        caplog.at_level(logging.ERROR, logger=rollout_script.__name__),
        pytest.raises(RuntimeError, match="synthetic rollout failure"),
    ):
        rollout_script.rollout.__wrapped__(cfg)

    assert "Rollout failed" in caplog.text
    assert "RuntimeError: synthetic rollout failure" in caplog.text
    strategy.teardown.assert_called_once_with(ctx)


def test_remote_rollout_config_rejects_local_policy_path(monkeypatch):
    from lerobot.rollout import RemoteInferenceConfig, RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    monkeypatch.setattr(
        sys,
        "argv",
        ["lerobot-rollout", "--inference.type=remote", "--policy.path=user/policy"],
    )
    with pytest.raises(ValueError, match="does not load --policy.path"):
        RolloutConfig(robot=MockRobotConfig(), inference=RemoteInferenceConfig())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"connect_timeout_s": 0}, "timeouts"),
        ({"execution_horizon": 0}, "execution_horizon"),
        ({"jpeg_quality": 101}, "jpeg_quality"),
        ({"image_encoding": "webp"}, "image_encoding"),
        ({"tls_client_cert_path": "cert.pem"}, "certificate and key"),
        (
            {"tls_client_cert_path": "cert.pem", "tls_client_key_path": "key.pem"},
            "TLS root certificate",
        ),
        ({"tls_server_name_override": "policy.internal"}, "TLS root certificate"),
    ],
)
def test_remote_inference_config_validation(kwargs, message):
    from lerobot.rollout import RemoteInferenceConfig

    with pytest.raises(ValueError, match=message):
        RemoteInferenceConfig(**kwargs)


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


def test_rollout_config_passes_policy_pretrained_revision(monkeypatch):
    from lerobot.configs import PreTrainedConfig, parser
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    captured = {}

    def fake_from_pretrained(cls, pretrained_name_or_path, **kwargs):
        captured["pretrained_name_or_path"] = pretrained_name_or_path
        captured.update(kwargs)
        return SimpleNamespace(device="cpu", pretrained_revision=kwargs["revision"])

    monkeypatch.setattr(parser, "get_yaml_overrides", lambda _: ["--pretrained_revision=yaml-sha"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["lerobot-rollout", "--policy.path=user/policy", "--policy.pretrained_revision=cli-sha"],
    )
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", classmethod(fake_from_pretrained))

    cfg = RolloutConfig(robot=MockRobotConfig())

    assert captured["pretrained_name_or_path"] == "user/policy"
    assert captured["revision"] == "cli-sha"
    assert captured["cli_overrides"] == [
        "--pretrained_revision=yaml-sha",
        "--pretrained_revision=cli-sha",
    ]
    assert cfg.policy.pretrained_path == "user/policy"
    assert cfg.policy.pretrained_revision == "cli-sha"


def test_load_pretrained_policy_passes_revision(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=False,
        pretrained_path="user/policy",
        pretrained_revision="policy-sha",
    )
    policy_class = MagicMock()
    loaded_policy = MagicMock()
    policy_class.from_pretrained.return_value = loaded_policy
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is loaded_policy
    policy_class.from_pretrained.assert_called_once_with(
        "user/policy",
        config=policy_config,
        revision="policy-sha",
    )


def test_load_pretrained_peft_policy_keeps_adapter_and_base_revisions_separate(monkeypatch):
    import lerobot.rollout.context as rollout_context

    policy_config = SimpleNamespace(
        type="mock",
        use_peft=True,
        pretrained_path="user/adapter",
        pretrained_revision="adapter-sha",
    )
    policy_class = MagicMock()
    base_policy = MagicMock()
    policy_class.from_pretrained.return_value = base_policy
    monkeypatch.setattr(rollout_context, "get_policy_class", lambda _: policy_class)

    peft_config = SimpleNamespace(
        base_model_name_or_path="user/base-policy",
        revision="base-sha",
    )
    peft_config_from_pretrained = MagicMock(return_value=peft_config)
    adapted_policy = MagicMock()
    peft_model_from_pretrained = MagicMock(return_value=adapted_policy)
    require_package = MagicMock()
    monkeypatch.setattr(rollout_context, "require_package", require_package)
    monkeypatch.setattr(
        rollout_context,
        "PeftConfig",
        SimpleNamespace(from_pretrained=peft_config_from_pretrained),
        raising=False,
    )
    monkeypatch.setattr(
        rollout_context,
        "PeftModel",
        SimpleNamespace(from_pretrained=peft_model_from_pretrained),
        raising=False,
    )

    policy = rollout_context._load_pretrained_policy(policy_config)

    assert policy is adapted_policy
    require_package.assert_called_once_with("peft", extra="peft")
    peft_config_from_pretrained.assert_called_once_with("user/adapter", revision="adapter-sha")
    policy_class.from_pretrained.assert_called_once_with(
        pretrained_name_or_path="user/base-policy",
        config=policy_config,
        revision="base-sha",
    )
    peft_model_from_pretrained.assert_called_once_with(
        base_policy,
        "user/adapter",
        config=peft_config,
        revision="adapter-sha",
    )


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


def test_base_strategy_refreshes_hold_while_waiting_for_operator(monkeypatch):
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies.base import BaseStrategy

    release_input = threading.Event()
    hold_phases = []

    class HoldingRobot:
        def hold_position(self, *, phase):
            hold_phases.append(phase)
            if len(hold_phases) >= 2:
                release_input.set()

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: release_input.wait(timeout=1.0))
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(shutdown_event=threading.Event()),
        hardware=SimpleNamespace(robot_wrapper=HoldingRobot()),
    )

    started = BaseStrategy(BaseStrategyConfig())._wait_for_operator_start(ctx, control_interval=0.001)

    assert started
    assert hold_phases == ["operator_hold", "operator_hold", "operator_hold"]


def test_base_strategy_holds_until_async_action_is_ready(monkeypatch):
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.strategies.base import BaseStrategy

    shutdown_event = threading.Event()
    robot = MagicMock()
    robot.get_observation.return_value = {}
    engine = MagicMock()
    interpolator = MagicMock()
    interpolator.get_control_interval.return_value = 0.001
    cfg = SimpleNamespace(fps=30.0, duration=60.0, use_torch_compile=False)
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=cfg, shutdown_event=shutdown_event),
        hardware=SimpleNamespace(robot_wrapper=robot),
        policy=SimpleNamespace(inference=engine),
        processors=SimpleNamespace(),
    )
    strategy = BaseStrategy(BaseStrategyConfig())
    strategy._engine = engine
    strategy._interpolator = interpolator
    monkeypatch.setattr(strategy, "_process_observation_and_notify", lambda _processors, _obs: {})
    monkeypatch.setattr(strategy, "_log_telemetry", lambda *_args: None)

    def no_action(*_args, **_kwargs):
        shutdown_event.set()
        return None

    monkeypatch.setattr("lerobot.rollout.strategies.base.send_next_action", no_action)

    strategy.run(ctx)

    robot.hold_position.assert_called_once_with(phase="inference_hold")
    engine.resume.assert_called_once_with()


def test_strategy_uses_distinct_policy_start_and_end_resets():
    from lerobot.rollout import BaseStrategyConfig
    from lerobot.rollout.context import HardwareContext
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.rollout.strategies.base import BaseStrategy

    events = []
    reset_position = {"joint.pos": 0.25}
    end_position = {"joint.pos": 0.0}

    class FakeEngine:
        def reset(self):
            events.append("engine.reset")

        def start(self):
            events.append("engine.start")

    class ResettableRobot:
        def arm(self):
            events.append("robot.arm")

        def reset_for_policy(self):
            events.append("robot.reset_for_policy")
            return reset_position

        def reset_after_policy(self):
            events.append("robot.reset_after_policy")
            return end_position

    hardware = HardwareContext(
        robot_wrapper=ThreadSafeRobot(ResettableRobot()),
        teleop=None,
        initial_position={"joint.pos": -0.5},
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=SimpleNamespace(interpolation_multiplier=1)),
        policy=SimpleNamespace(inference=FakeEngine()),
        hardware=hardware,
    )
    strategy = BaseStrategy(BaseStrategyConfig())

    strategy._init_engine(ctx)

    assert events == ["engine.reset", "engine.start", "robot.arm", "robot.reset_for_policy"]
    assert hardware.initial_position == reset_position

    strategy._return_to_initial_position(hardware)
    assert events[-1] == "robot.reset_after_policy"


@pytest.mark.parametrize("reset_robot", [False, True])
def test_action_probe_only_prepares_robot_when_explicitly_requested(tmp_path, reset_robot):
    from lerobot.rollout import ActionProbeStrategyConfig
    from lerobot.rollout.context import HardwareContext
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.rollout.strategies.action_probe import ActionProbeStrategy

    events = []
    reset_position = {"joint.pos": 0.25}

    class FakeEngine:
        def reset(self):
            events.append("engine.reset")

        def start(self):
            events.append("engine.start")

    class ResettableRobot:
        def arm(self):
            events.append("robot.arm")

        def disarm(self):
            events.append("robot.disarm")

        def reset_for_policy(self):
            events.append("robot.reset")
            return reset_position

    hardware = HardwareContext(
        robot_wrapper=ThreadSafeRobot(ResettableRobot()),
        teleop=None,
        initial_position={"joint.pos": -0.5},
    )
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(cfg=SimpleNamespace(interpolation_multiplier=1)),
        policy=SimpleNamespace(inference=FakeEngine()),
        hardware=hardware,
    )
    strategy = ActionProbeStrategy(
        ActionProbeStrategyConfig(
            action_log_path=tmp_path / "actions.jsonl",
            reset_robot=reset_robot,
        )
    )

    try:
        strategy.setup(ctx)
    finally:
        strategy._close_action_file()

    expected = ["engine.reset", "engine.start"]
    if reset_robot:
        expected.extend(["robot.arm", "robot.reset"])
        assert hardware.initial_position == reset_position
    else:
        assert hardware.initial_position == {"joint.pos": -0.5}
    expected.append("robot.disarm")
    assert events == expected


def test_send_next_action_can_discard_without_hardware_dispatch():
    from lerobot.rollout.strategies import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    interpolator = ActionInterpolator()
    interpolator.add(torch.tensor([0.25, -0.5]))
    robot_wrapper = MagicMock()
    engine = MagicMock()
    ctx = SimpleNamespace(
        policy=SimpleNamespace(inference=engine),
        data=SimpleNamespace(dataset_features={}, ordered_action_keys=["left.pos", "right.pos"]),
        processors=SimpleNamespace(robot_action_processor=lambda pair: pair[0]),
        hardware=SimpleNamespace(robot_wrapper=robot_wrapper),
    )

    action = send_next_action({}, {}, ctx, interpolator, execute=False)

    assert action == {"left.pos": pytest.approx(0.25), "right.pos": pytest.approx(-0.5)}
    robot_wrapper.send_action.assert_not_called()
    engine.notify_action_sent.assert_not_called()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        ActionProbeStrategy,
        ActionProbeStrategyConfig,
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(ActionProbeStrategyConfig()), ActionProbeStrategy)
    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


def test_create_inference_engine_remote():
    from lerobot.rollout import RemoteInferenceConfig, create_inference_engine
    from lerobot.rollout.inference.remote import RemoteInferenceEngine

    robot_wrapper = MagicMock(robot_type="mock")
    robot_wrapper.inner.id = "mock-id"
    robot_wrapper.observation_features = {"top": (3, 4, 3)}
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint_0.pos", "joint_1.pos"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["joint_0.pos", "joint_1.pos"],
        },
        "observation.images.top": {
            "dtype": "image",
            "shape": (3, 4, 3),
            "names": ["height", "width", "channels"],
        },
    }
    engine = create_inference_engine(
        RemoteInferenceConfig(image_encoding="png"),
        policy=None,
        preprocessor=None,
        postprocessor=None,
        robot_wrapper=robot_wrapper,
        hw_features={},
        dataset_features=features,
        ordered_action_keys=["joint_0.pos", "joint_1.pos"],
        task="test",
        fps=30.0,
        device="cpu",
    )

    assert isinstance(engine, RemoteInferenceEngine)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}
