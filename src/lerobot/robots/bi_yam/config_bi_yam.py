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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

YAM_SCALAR_KEYS = (
    *(f"left_joint_{index}.pos" for index in range(6)),
    "left_gripper.pos",
    *(f"right_joint_{index}.pos" for index in range(6)),
    "right_gripper.pos",
)


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


@dataclass(kw_only=True)
class YAMArmConfig:
    channel: str
    arm_type: str = "yam"
    gripper_type: str = "linear_4310"
    # Raw i2rt [closed, open] endpoints. Supplying these skips moving calibration.
    gripper_limits_override: tuple[float, float] | None = None
    allow_gripper_calibration: bool = False
    sim: bool = False
    command_ttl_s: float = 0.2
    worker_poll_interval_s: float = 0.004
    enable_auto_recovery: bool = False

    def __post_init__(self) -> None:
        if not self.channel:
            raise ValueError("channel must not be empty")
        if not math.isfinite(self.command_ttl_s) or self.command_ttl_s <= 0:
            raise ValueError("command_ttl_s must be positive")
        if not math.isfinite(self.worker_poll_interval_s) or self.worker_poll_interval_s <= 0:
            raise ValueError("worker_poll_interval_s must be positive")
        if self.gripper_limits_override is not None:
            if len(self.gripper_limits_override) != 2:
                raise ValueError("gripper_limits_override must contain [closed, open]")
            closed, open_ = self.gripper_limits_override
            if not math.isfinite(closed) or not math.isfinite(open_) or closed == open_:
                raise ValueError("gripper_limits_override must contain two distinct finite values")
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
        if not self.has_fixed_gripper_calibration and not self.allow_gripper_calibration:
            raise RuntimeError(
                f"Refusing to open {self.channel}: gripper {self.gripper_type!r} needs raw "
                "[closed, open] limits. Set gripper_limits_override, or explicitly set "
                "allow_gripper_calibration=true for a supervised calibration that moves the gripper."
            )


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass(kw_only=True)
class BiYAMFollowerConfig(RobotConfig):
    id: str | None = "bi_yam_follower"
    left_arm_config: YAMArmConfig = field(default_factory=lambda: YAMArmConfig(channel="can0"))
    right_arm_config: YAMArmConfig = field(default_factory=lambda: YAMArmConfig(channel="can1"))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    startup_timeout_s: float = 15.0
    state_timeout_s: float = 0.25
    command_ack_timeout_s: float = 0.25
    command_lead_time_s: float = 0.01
    shutdown_timeout_s: float = 2.0

    left_joint_limits: list[tuple[float, float]] = field(default_factory=_default_joint_limits)
    right_joint_limits: list[tuple[float, float]] = field(default_factory=_default_joint_limits)
    gripper_limits: tuple[float, float] = (0.0, 1.0)
    max_joint_delta: float = 0.1
    max_gripper_delta: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "startup_timeout_s",
            "state_timeout_s",
            "command_ack_timeout_s",
            "shutdown_timeout_s",
            "max_joint_delta",
            "max_gripper_delta",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.command_lead_time_s) or self.command_lead_time_s < 0:
            raise ValueError("command_lead_time_s must be non-negative")

        _validate_limits("left_joint_limits", self.left_joint_limits, 6)
        _validate_limits("right_joint_limits", self.right_joint_limits, 6)
        _validate_limits("gripper_limits", [self.gripper_limits], 1)
