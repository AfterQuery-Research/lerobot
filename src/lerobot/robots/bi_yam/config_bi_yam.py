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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

YAM_SCALAR_KEYS = (
    *(f"left_joint_{index}.pos" for index in range(6)),
    "left_gripper.pos",
    *(f"right_joint_{index}.pos" for index in range(6)),
    "right_gripper.pos",
)

BI_YAM_POLICY_START_POSITION = (*([0.0] * 6), 1.0, *([0.0] * 6), 1.0)


def _default_joint_limits() -> list[tuple[float, float]]:
    # Operational limits match the YAM model limits without i2rt's hardware-level buffer.
    return [
        (-2.61799, 3.14159),
        (0.0, 3.66519),
        (0.0, 3.14159),
        (-1.69297, 1.5708),
        (-1.5708, 1.5708),
        (-2.0944, 2.0944),
    ]


def _validate_limits(name: str, limits: list[tuple[float, float]], expected: int) -> None:
    if len(limits) != expected:
        raise ValueError(f"{name} must contain {expected} (lower, upper) pairs")
    for index, pair in enumerate(limits):
        if len(pair) != 2:
            raise ValueError(f"{name}[{index}] must contain exactly two values")
        lower, upper = pair
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"{name}[{index}] must be a finite, increasing interval")


def _validate_raw_gripper_limits(name: str, limits: tuple[float, float]) -> None:
    if len(limits) != 2:
        raise ValueError(f"{name} must contain [closed, open]")
    closed, open_ = limits
    if not math.isfinite(closed) or not math.isfinite(open_) or closed == open_:
        raise ValueError(f"{name} must contain two distinct finite values")


@dataclass(frozen=True)
class YAMGripperCalibration:
    """Raw i2rt gripper endpoints persisted in LeRobot's calibration directory."""

    gripper_limits: tuple[float, float]

    def __post_init__(self) -> None:
        _validate_raw_gripper_limits("gripper_limits", self.gripper_limits)


@dataclass(kw_only=True)
class YAMArmConfig:
    # Prefer adapter_serial for portable physical setups. channel remains available for
    # simulations and advanced installations with externally managed interface names.
    adapter_serial: str | None = None
    channel: str | None = None
    arm_type: str = "yam"
    gripper_type: str = "linear_4310"
    # Raw i2rt [closed, open] endpoints. Supplying these skips moving calibration.
    gripper_limits_override: tuple[float, float] | None = None
    allow_gripper_calibration: bool = False
    sim: bool = False
    command_ttl_s: float = 1.0
    worker_poll_interval_s: float = 0.004
    enable_auto_recovery: bool = False

    def __post_init__(self) -> None:
        if self.adapter_serial is not None:
            self.adapter_serial = self.adapter_serial.strip()
            if not self.adapter_serial:
                raise ValueError("adapter_serial must not be empty")
        if self.channel is not None:
            self.channel = self.channel.strip()
            if not self.channel:
                raise ValueError("channel must not be empty")
        if self.adapter_serial is not None and self.channel is not None:
            raise ValueError("adapter_serial and channel are mutually exclusive")
        if not math.isfinite(self.command_ttl_s) or self.command_ttl_s <= 0:
            raise ValueError("command_ttl_s must be positive")
        if not math.isfinite(self.worker_poll_interval_s) or self.worker_poll_interval_s <= 0:
            raise ValueError("worker_poll_interval_s must be positive")
        if self.gripper_limits_override is not None:
            _validate_raw_gripper_limits("gripper_limits_override", self.gripper_limits_override)
            if self.allow_gripper_calibration:
                raise ValueError(
                    "gripper_limits_override and allow_gripper_calibration cannot be set together"
                )

    @property
    def has_fixed_gripper_calibration(self) -> bool:
        return (
            self.sim
            or self.gripper_type in {"no_gripper", "yam_teaching_handle"}
            or self.gripper_limits_override is not None
        )

    def validate_hardware_startup(self) -> None:
        if not self.sim and self.adapter_serial is None and self.channel is None:
            raise RuntimeError("set adapter_serial (recommended) or channel for this physical YAM arm")
        if not self.has_fixed_gripper_calibration and not self.allow_gripper_calibration:
            identity = self.adapter_serial or self.channel or "unconfigured arm"
            raise RuntimeError(
                f"Refusing to open {identity}: gripper {self.gripper_type!r} needs raw "
                "[closed, open] limits. Set gripper_limits_override, or explicitly set "
                "allow_gripper_calibration=true for a supervised calibration that moves the gripper."
            )


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass(kw_only=True)
class BiYAMFollowerConfig(RobotConfig):
    id: str | None = None
    left_arm_config: YAMArmConfig = field(default_factory=YAMArmConfig)
    right_arm_config: YAMArmConfig = field(default_factory=YAMArmConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Set only while running lerobot-calibrate for one arm. Normal rollout leaves this unset.
    calibration_side: Literal["left", "right"] | None = None

    startup_timeout_s: float = 15.0
    state_timeout_s: float = 0.25
    command_ack_timeout_s: float = 0.25
    command_lead_time_s: float = 0.01
    shutdown_timeout_s: float = 2.0

    left_joint_limits: list[tuple[float, float]] = field(default_factory=_default_joint_limits)
    right_joint_limits: list[tuple[float, float]] = field(default_factory=_default_joint_limits)
    gripper_limits: tuple[float, float] = (0.0, 1.0)
    gripper_state_tolerance: float = 0.15
    max_joint_delta: float = 0.1
    max_gripper_delta: float = 0.03
    control_telemetry_path: Path | None = None
    control_telemetry_console_interval_s: float = 1.0

    # A configured pose is reached before policy control and between policy episodes.
    policy_start_position: tuple[float, ...] | None = BI_YAM_POLICY_START_POSITION
    policy_reset_step_size: float = 0.01
    policy_reset_max_steps: int = 100
    policy_reset_fps: float = 30.0
    policy_reset_tolerance: float = 0.035
    policy_reset_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.id is None or not self.id.strip():
            raise ValueError("--robot.id is required for a physical bi_yam_follower")
        self.id = self.id.strip()
        if self.calibration_side not in (None, "left", "right"):
            raise ValueError("calibration_side must be left, right, or unset")
        required_sides = (self.calibration_side,) if self.calibration_side is not None else ("left", "right")
        for side in required_sides:
            arm_config = getattr(self, f"{side}_arm_config")
            if not arm_config.sim and arm_config.adapter_serial is None and arm_config.channel is None:
                raise ValueError(
                    f"--robot.{side}_arm_config.adapter_serial is required for a physical YAM arm "
                    f"(or set --robot.{side}_arm_config.channel for an externally named interface)"
                )
        if (
            self.left_arm_config.adapter_serial is not None
            and self.right_arm_config.adapter_serial is not None
            and self.left_arm_config.adapter_serial.casefold()
            == self.right_arm_config.adapter_serial.casefold()
        ):
            raise ValueError("left and right YAM arms must use different adapter_serial values")
        if (
            self.left_arm_config.channel is not None
            and self.left_arm_config.channel == self.right_arm_config.channel
        ):
            raise ValueError("left and right YAM arms must use different channel values")
        for name in (
            "startup_timeout_s",
            "state_timeout_s",
            "command_ack_timeout_s",
            "shutdown_timeout_s",
            "max_joint_delta",
            "max_gripper_delta",
            "policy_reset_step_size",
            "policy_reset_fps",
            "policy_reset_tolerance",
            "policy_reset_timeout_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.command_lead_time_s) or self.command_lead_time_s < 0:
            raise ValueError("command_lead_time_s must be non-negative")
        if not math.isfinite(self.gripper_state_tolerance) or self.gripper_state_tolerance < 0:
            raise ValueError("gripper_state_tolerance must be non-negative")
        if (
            not math.isfinite(self.control_telemetry_console_interval_s)
            or self.control_telemetry_console_interval_s < 0
        ):
            raise ValueError("control_telemetry_console_interval_s must be non-negative")
        if self.policy_reset_max_steps <= 0:
            raise ValueError("policy_reset_max_steps must be positive")

        _validate_limits("left_joint_limits", self.left_joint_limits, 6)
        _validate_limits("right_joint_limits", self.right_joint_limits, 6)
        _validate_limits("gripper_limits", [self.gripper_limits], 1)

        if self.policy_start_position is not None:
            if len(self.policy_start_position) != len(YAM_SCALAR_KEYS):
                raise ValueError(f"policy_start_position must contain {len(YAM_SCALAR_KEYS)} values")
            try:
                values = tuple(float(value) for value in self.policy_start_position)
            except (TypeError, ValueError) as exc:
                raise ValueError("policy_start_position values must be numeric") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError("policy_start_position values must be finite")

            limits = [
                *self.left_joint_limits,
                self.gripper_limits,
                *self.right_joint_limits,
                self.gripper_limits,
            ]
            out_of_bounds = [
                key
                for key, value, (lower, upper) in zip(YAM_SCALAR_KEYS, values, limits, strict=True)
                if value < lower or value > upper
            ]
            if out_of_bounds:
                raise ValueError(
                    "policy_start_position is outside operational limits for " + ", ".join(out_of_bounds)
                )
            self.policy_start_position = values
