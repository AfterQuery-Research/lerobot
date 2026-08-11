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

from dataclasses import dataclass

import numpy as np

CAMERA_NAMES = ("top", "left", "right")
STATE_SIZE = 14

# Command-space limits from i2rt's YAM MJCF. Grippers use i2rt's normalized [0, 1] convention.
YAM_JOINT_LIMITS = (
    (-2.61799, 3.14159),
    (0.0, 3.66519),
    (0.0, 3.14159),
    (-1.69297, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)
DEFAULT_ARM_STATE = (0.0, 1.2, 1.8, 0.0, 0.0, 0.0, 0.5)
DEFAULT_INITIAL_STATE = DEFAULT_ARM_STATE + DEFAULT_ARM_STATE


@dataclass(frozen=True)
class BiYAMSimulatorConfig:
    """Configuration for both deterministic and MuJoCo bimanual YAM backends.

    Simulation is deliberately step-driven. Each action advances one observation
    period, making tests reproducible and leaving real-time pacing to the caller.
    """

    physics_hz: int = 600
    control_hz: int = 200
    observation_hz: int = 30
    camera_height: int = 480
    camera_width: int = 640
    seed: int = 0
    headless: bool = True
    initial_state: tuple[float, ...] = DEFAULT_INITIAL_STATE
    left_base_position: tuple[float, float, float] = (-0.32, 0.0, 0.4)
    right_base_position: tuple[float, float, float] = (0.32, 0.0, 0.4)
    left_base_quaternion: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, 0.7071068)
    right_base_quaternion: tuple[float, float, float, float] = (0.7071068, 0.0, 0.0, -0.7071068)
    fallback_position_gain: float = 80.0
    fallback_damping: float = 18.0
    mujoco_arm_kp: float = 80.0
    mujoco_gripper_kp: float = 200.0

    def __post_init__(self) -> None:
        for name in ("physics_hz", "control_hz", "observation_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.control_hz > self.physics_hz:
            raise ValueError("control_hz cannot exceed physics_hz")
        if self.camera_height <= 0 or self.camera_width <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.fallback_position_gain <= 0 or self.fallback_damping < 0:
            raise ValueError("fallback dynamics gains must be non-negative, with a positive position gain")
        if self.mujoco_arm_kp <= 0 or self.mujoco_gripper_kp <= 0:
            raise ValueError("MuJoCo actuator gains must be positive")

        initial_state = np.asarray(self.initial_state, dtype=np.float32)
        if initial_state.shape != (STATE_SIZE,):
            raise ValueError(f"initial_state must have shape ({STATE_SIZE},)")
        if not np.isfinite(initial_state).all():
            raise ValueError("initial_state must contain only finite values")
        lower, upper = action_bounds()
        if np.any(initial_state < lower) or np.any(initial_state > upper):
            raise ValueError("initial_state exceeds the YAM joint or gripper limits")


def action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return fresh lower and upper command-space bounds in MolmoAct2 ordering."""

    arm_lower = tuple(limit[0] for limit in YAM_JOINT_LIMITS) + (0.0,)
    arm_upper = tuple(limit[1] for limit in YAM_JOINT_LIMITS) + (1.0,)
    return (
        np.asarray(arm_lower + arm_lower, dtype=np.float32),
        np.asarray(arm_upper + arm_upper, dtype=np.float32),
    )
