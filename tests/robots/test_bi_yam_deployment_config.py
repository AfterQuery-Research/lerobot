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

from pathlib import Path

import draccus

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots.bi_yam import BiYAMFollowerConfig  # noqa: F401
from lerobot.rollout import RemoteInferenceConfig, RolloutConfig

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "deployment" / "afterquery_dual_yam_client.yaml"
)


def test_afterquery_dual_yam_config_parses_with_verified_hardware_mapping():
    config = draccus.parse(RolloutConfig, config_path=CONFIG_PATH, args=[])

    assert isinstance(config.robot, BiYAMFollowerConfig)
    assert config.robot.left_arm_config.channel == "can_yam_new"
    assert config.robot.right_arm_config.channel == "can_yam_old"
    assert config.robot.left_arm_config.gripper_limits_override is None
    assert config.robot.right_arm_config.gripper_limits_override is None
    assert not config.robot.left_arm_config.allow_gripper_calibration
    assert not config.robot.right_arm_config.allow_gripper_calibration

    assert list(config.robot.cameras) == ["top", "left", "right"]
    assert {name: camera.serial_number_or_name for name, camera in config.robot.cameras.items()} == {
        "top": "262422074066",
        "left": "323622270338",
        "right": "323622270243",
    }
    assert all(camera.type == "intelrealsense" for camera in config.robot.cameras.values())
    assert all(
        (camera.width, camera.height, camera.fps) == (640, 360, 30)
        for camera in config.robot.cameras.values()
    )

    assert isinstance(config.inference, RemoteInferenceConfig)
    assert config.inference.server_address == "127.0.0.1:8081"
    assert config.inference.requested_model_id == "allenai/MolmoAct2-BimanualYAM"
    assert config.fps == 30
    assert config.duration == 5.0
    assert not config.return_to_initial_position
