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

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.remote_inference.schema import ModelManifest, PolicyActionChunk
from lerobot.scripts import lerobot_policy_probe as probe_module


class FakeCamera:
    def __init__(self, value: int, *, height: int, width: int):
        self.value = value
        self.height = height
        self.width = width
        self.connected = False
        self.disconnect_count = 0

    def connect(self) -> None:
        self.connected = True

    def async_read(self) -> np.ndarray:
        assert self.connected
        return np.full((self.height, self.width, 3), self.value, dtype=np.uint8)

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1


class FakeClient:
    instances = []

    def __init__(self, config):
        self.config = config
        self.observation = None
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.closed = True

    def connect(self, embodiment, **_kwargs):
        model = ModelManifest(
            model_id="allenai/MolmoAct2-BimanualYAM",
            revision="test",
            policy_type="molmoact2",
            norm_tag="yam_dual_molmoact2",
            action_horizon=30,
            action_dim=14,
            state_features=probe_module.YAM_FEATURES,
            action_features=probe_module.YAM_FEATURES,
            camera_keys=tuple(probe_module.AFTERQUERY_CAMERA_SERIALS),
            fingerprint="fake-fingerprint",
        )
        model.assert_compatible(embodiment)
        return SimpleNamespace(model=model)

    def infer(self, observation):
        self.observation = observation
        return PolicyActionChunk(
            observation_sequence=0,
            first_action_tick=0,
            actions=np.full((30, 14), 0.25, dtype=np.float32),
            model_fingerprint="fake-fingerprint",
            server_compute_ns=1_500_000,
        )


def _config(tmp_path):
    return probe_module.PolicyProbeConfig(
        server_address="127.0.0.1:8081",
        task="pick up the object",
        state=[0.0] * 14,
        state_source="unit-test",
        output_dir=tmp_path,
    )


def test_probe_captures_once_and_discards_actions(tmp_path):
    FakeClient.instances.clear()
    configs = probe_module._default_cameras()
    cameras = {
        name: FakeCamera(index, height=int(config.height), width=int(config.width))
        for index, (name, config) in enumerate(configs.items())
    }

    result = probe_module.run_policy_probe(
        _config(tmp_path),
        camera_factory=lambda _configs: cameras,
        client_factory=FakeClient,
    )

    assert result.action_shape == (30, 14)
    assert result.actions_discarded is True
    assert result.action_min == result.action_max == 0.25
    assert all(camera.disconnect_count == 1 for camera in cameras.values())
    assert FakeClient.instances[0].closed is True
    assert tuple(frame.key for frame in FakeClient.instances[0].observation.images) == (
        "top",
        "left",
        "right",
    )
    report = json.loads((result.evidence_dir / "report.json").read_text())
    assert report["result"]["actions_discarded"] is True
    assert not any("actions" in path.name for path in result.evidence_dir.iterdir())
    assert all((result.evidence_dir / f"{name}.png").is_file() for name in cameras)


def test_probe_camera_defaults_match_afterquery_robot_preset():
    from lerobot.robots.bi_yam.config_bi_yam import AfterQueryDualYAMConfig

    probe_cameras = probe_module._default_cameras()
    robot_cameras = AfterQueryDualYAMConfig().cameras
    assert tuple(probe_cameras) == tuple(robot_cameras)
    for name in probe_cameras:
        probe_camera = probe_cameras[name]
        robot_camera = robot_cameras[name]
        assert (
            probe_camera.serial_number_or_name,
            probe_camera.width,
            probe_camera.height,
            probe_camera.fps,
        ) == (
            robot_camera.serial_number_or_name,
            robot_camera.width,
            robot_camera.height,
            robot_camera.fps,
        )


def test_probe_refuses_missing_or_unlabelled_state_before_camera_creation(tmp_path):
    cfg = _config(tmp_path)
    cfg.state = []
    created = False

    def camera_factory(_configs):
        nonlocal created
        created = True
        return {}

    with pytest.raises(ValueError, match="exactly 14"):
        probe_module.run_policy_probe(cfg, camera_factory=camera_factory, client_factory=FakeClient)
    assert created is False

    cfg = _config(tmp_path)
    cfg.state_source = ""
    with pytest.raises(ValueError, match="state_source"):
        probe_module.run_policy_probe(cfg, camera_factory=camera_factory, client_factory=FakeClient)
    assert created is False


def test_probe_module_has_no_robot_control_dependency():
    source = inspect.getsource(probe_module)
    forbidden = ("lerobot.robots", "i2rt", "send_action", ".arm(", "get_yam_robot")
    assert all(token not in source for token in forbidden)
