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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.simulators.bi_yam import CAMERA_NAMES, BiYAMSimulator, BiYAMSimulatorConfig
from lerobot.simulators.bi_yam.config import action_bounds
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..config import RobotConfig
from ..robot import Robot
from .config_bi_yam import YAM_SCALAR_KEYS


@RobotConfig.register_subclass("bi_yam_simulator")
@dataclass(kw_only=True)
class BiYAMSimulatorRobotConfig(RobotConfig):
    """LeRobot adapter for the shared-world dual-YAM simulator."""

    id: str | None = "bi_yam_simulator"
    backend: Literal["auto", "mujoco", "fallback"] = "auto"
    simulator: BiYAMSimulatorConfig = field(default_factory=BiYAMSimulatorConfig)
    max_joint_delta: float = 0.1
    max_gripper_delta: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_joint_delta <= 0 or self.max_gripper_delta <= 0:
            raise ValueError("simulator action delta limits must be positive")


class BiYAMSimulatorRobot(Robot):
    """Robot-shaped facade with the same policy contract as the physical dual YAM."""

    config_class = BiYAMSimulatorRobotConfig
    name = "bi_yam_simulator"

    def __init__(
        self,
        config: BiYAMSimulatorRobotConfig,
        *,
        simulator: BiYAMSimulator | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self._simulator = simulator or BiYAMSimulator(config.simulator, backend=config.backend)
        self._armed = False

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        camera_shape = (
            self.config.simulator.camera_height,
            self.config.simulator.camera_width,
            3,
        )
        return {
            **dict.fromkeys(YAM_SCALAR_KEYS, float),
            **dict.fromkeys(CAMERA_NAMES, camera_shape),
        }

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(YAM_SCALAR_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._simulator.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def simulator(self) -> BiYAMSimulator:
        return self._simulator

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.__class__.__name__} is already connected.")
        self._simulator.start()
        self._armed = False

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def arm(self) -> None:
        self._require_connected()
        self._require_healthy()
        self._armed = True

    def disarm(self) -> None:
        self._require_connected()
        self._simulator.safe_idle()
        self._armed = False

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        self._require_healthy()
        state = self._simulator.get_state()
        frames = self._simulator.render_cameras()
        return {
            **{key: float(value) for key, value in zip(YAM_SCALAR_KEYS, state, strict=True)},
            **frames,
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_connected()
        self._require_healthy()
        if not self._armed:
            raise RuntimeError("BiYAM simulator is in safe idle; arm it locally before sending actions")
        if tuple(action) != YAM_SCALAR_KEYS:
            raise ValueError("Action keys and order must match the dual-YAM policy schema")
        try:
            requested = np.asarray([float(action[key]) for key in YAM_SCALAR_KEYS], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            self.disarm()
            raise ValueError("Action values must be numeric scalars") from exc
        if not np.isfinite(requested).all():
            self.disarm()
            raise ValueError("Action values must all be finite")

        lower, upper = action_bounds()
        bounded = np.clip(requested, lower, upper)
        present = self._simulator.get_state()
        deltas = np.asarray(
            [
                *([self.config.max_joint_delta] * 6),
                self.config.max_gripper_delta,
                *([self.config.max_joint_delta] * 6),
                self.config.max_gripper_delta,
            ],
            dtype=np.float32,
        )
        applied = np.clip(bounded, present - deltas, present + deltas)
        try:
            self._simulator.apply_action(applied)
        except Exception:
            self.disarm()
            raise
        return {key: float(value) for key, value in zip(YAM_SCALAR_KEYS, applied, strict=True)}

    def disconnect(self) -> None:
        self._require_connected()
        try:
            self.disarm()
        finally:
            self._simulator.close()
            self._armed = False

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )

    def _require_healthy(self) -> None:
        health = self._simulator.get_health()
        if not health.healthy:
            self._simulator.safe_idle()
            self._armed = False
            raise RuntimeError("BiYAM simulator backend is unhealthy")
