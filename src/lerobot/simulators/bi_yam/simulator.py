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

"""Deterministic bimanual YAM simulation backends.

These backends validate the software contract, control scheduling, camera
schema, and common fault handling without robot hardware. They do not validate
CAN timing, motor calibration, gravity-compensation safety, physical E-stops,
camera calibration, or transfer of a vision policy from synthetic to real images.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from lerobot.simulators.bi_yam.config import CAMERA_NAMES, STATE_SIZE, BiYAMSimulatorConfig, action_bounds


@dataclass(frozen=True)
class BiYAMFaults:
    """Per-simulator deterministic fault settings."""

    frozen_cameras: frozenset[str] = frozenset()
    stale_state: bool = False
    left_worker_failed: bool = False
    right_worker_failed: bool = False
    command_delay_steps: int = 0
    stuck_joints: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        frozen_cameras = frozenset(self.frozen_cameras)
        stuck_joints = frozenset(self.stuck_joints)
        unknown_cameras = frozen_cameras.difference(CAMERA_NAMES)
        if unknown_cameras:
            raise ValueError(f"Unknown cameras: {sorted(unknown_cameras)}")
        if self.command_delay_steps < 0:
            raise ValueError("command_delay_steps cannot be negative")
        if any(index < 0 or index >= STATE_SIZE for index in stuck_joints):
            raise ValueError(f"stuck_joints indices must be in [0, {STATE_SIZE})")
        object.__setattr__(self, "frozen_cameras", frozen_cameras)
        object.__setattr__(self, "stuck_joints", stuck_joints)


@dataclass(frozen=True)
class BiYAMHealth:
    """Health and simulation-time snapshot returned to a robot adapter."""

    backend: str
    connected: bool
    running: bool
    safe_idle: bool
    sim_time_ns: int
    state_timestamp_ns: int
    camera_timestamps_ns: dict[str, int]
    left_worker_ok: bool
    right_worker_ok: bool
    pending_commands: int

    @property
    def healthy(self) -> bool:
        return self.connected and self.running and self.left_worker_ok and self.right_worker_ok


class _StepDrivenBiYAMBackend(ABC):
    """Shared deterministic lifecycle, scheduling, and fault behavior."""

    backend_name: str

    def __init__(self, config: BiYAMSimulatorConfig) -> None:
        self.config = config
        self._connected = False
        self._running = False
        self._safe_idle = True
        self._seed = config.seed
        self._faults = BiYAMFaults()
        self._pending_commands: deque[tuple[int, np.ndarray]] = deque()
        self._command_target = np.asarray(config.initial_state, dtype=np.float32).copy()
        self._sampled_state = self._command_target.copy()
        self._stuck_positions: dict[int, float] = {}
        self._failed_worker_positions: dict[int, float] = {}
        self._camera_cache: dict[str, np.ndarray] = {}
        self._camera_timestamps_ns: dict[str, int] = {}
        self._state_timestamp_ns = 0
        self._physics_steps = 0
        self._observation_steps = 0
        self._physics_step_budget = 0
        self._control_step_budget = config.physics_hz

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def faults(self) -> BiYAMFaults:
        return self._faults

    def connect(self) -> None:
        """Allocate backend resources and reset to the configured initial state."""

        if self._connected:
            return
        self._connected = True
        try:
            self._reset(self._seed)
        except Exception:
            self._connected = False
            self._close_backend()
            raise

    def start(self) -> None:
        """Connect if necessary and allow actions to advance the simulation."""

        if not self._connected:
            self.connect()
        self._running = True

    def reset(self, seed: int | None = None) -> None:
        """Reset state, clocks, queued commands, objects, and rendered-frame caches."""

        self._require_connected()
        if seed is not None:
            self._seed = seed
        self._reset(self._seed)

    def get_state(self) -> np.ndarray:
        """Return left arm, left gripper, right arm, right gripper as float32[14]."""

        self._require_connected()
        return self._sampled_state.copy()

    def apply_action(self, action: np.ndarray) -> None:
        """Queue a validated target and advance exactly one observation period."""

        self._require_running()
        validated = self._validate_action(action)
        due_step = self._observation_steps + self._faults.command_delay_steps
        self._pending_commands.append((due_step, validated))
        self._activate_due_commands()
        self._safe_idle = False
        self._advance_one_observation()

    def render_cameras(self) -> dict[str, np.ndarray]:
        """Render RGB uint8 images for the fixed top, left, and right cameras."""

        self._require_connected()
        frames: dict[str, np.ndarray] = {}
        for camera_name in CAMERA_NAMES:
            frozen = camera_name in self._faults.frozen_cameras
            if not frozen or camera_name not in self._camera_cache:
                frame = np.asarray(self._render_camera(camera_name), dtype=np.uint8)
                expected_shape = (self.config.camera_height, self.config.camera_width, 3)
                if frame.shape != expected_shape:
                    raise RuntimeError(
                        f"Backend returned {camera_name!r} frame shape {frame.shape}; expected {expected_shape}"
                    )
                self._camera_cache[camera_name] = np.ascontiguousarray(frame)
                self._camera_timestamps_ns[camera_name] = self._sim_time_ns
            frames[camera_name] = self._camera_cache[camera_name].copy()
        return frames

    def set_faults(self, faults: BiYAMFaults) -> None:
        """Replace this instance's active deterministic fault settings."""

        self._require_connected()
        actual_state = self._read_state()
        self._stuck_positions = {
            index: self._stuck_positions.get(index, float(actual_state[index]))
            for index in faults.stuck_joints
        }
        failed_indices: list[int] = []
        if faults.left_worker_failed:
            failed_indices.extend(range(7))
        if faults.right_worker_failed:
            failed_indices.extend(range(7, STATE_SIZE))
        self._failed_worker_positions = {
            index: self._failed_worker_positions.get(index, float(actual_state[index]))
            for index in failed_indices
        }
        self._faults = faults

    def clear_faults(self) -> None:
        self.set_faults(BiYAMFaults())

    def safe_idle(self) -> None:
        """Discard delayed commands and hold the current simulated position."""

        self._require_connected()
        self._pending_commands.clear()
        self._command_target = self._read_state()
        self._set_control_target(self._command_target)
        self._safe_idle = True

    def get_health(self) -> BiYAMHealth:
        return BiYAMHealth(
            backend=self.backend_name,
            connected=self._connected,
            running=self._running,
            safe_idle=self._safe_idle,
            sim_time_ns=self._sim_time_ns,
            state_timestamp_ns=self._state_timestamp_ns,
            camera_timestamps_ns=dict(self._camera_timestamps_ns),
            left_worker_ok=not self._faults.left_worker_failed,
            right_worker_ok=not self._faults.right_worker_failed,
            pending_commands=len(self._pending_commands),
        )

    def close(self) -> None:
        """Release rendering and physics resources. Safe to call repeatedly."""

        if self._connected:
            self.safe_idle()
        self._running = False
        self._connected = False
        self._close_backend()

    def __enter__(self) -> _StepDrivenBiYAMBackend:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def _sim_time_ns(self) -> int:
        return round(self._physics_steps * 1_000_000_000 / self.config.physics_hz)

    def _reset(self, seed: int) -> None:
        self._pending_commands.clear()
        self._stuck_positions.clear()
        self._failed_worker_positions.clear()
        self._camera_cache.clear()
        self._camera_timestamps_ns.clear()
        self._faults = BiYAMFaults()
        self._physics_steps = 0
        self._observation_steps = 0
        self._physics_step_budget = 0
        self._control_step_budget = self.config.physics_hz
        self._command_target = np.asarray(self.config.initial_state, dtype=np.float32).copy()
        self._reset_backend(seed, self._command_target)
        self._set_control_target(self._command_target)
        self._sampled_state = self._read_state()
        self._state_timestamp_ns = 0
        self._safe_idle = True

    def _activate_due_commands(self) -> None:
        while self._pending_commands and self._pending_commands[0][0] <= self._observation_steps:
            _, self._command_target = self._pending_commands.popleft()

    def _advance_one_observation(self) -> None:
        self._physics_step_budget += self.config.physics_hz
        physics_steps = self._physics_step_budget // self.config.observation_hz
        self._physics_step_budget %= self.config.observation_hz

        for _ in range(physics_steps):
            self._control_step_budget += self.config.control_hz
            if self._control_step_budget >= self.config.physics_hz:
                self._control_step_budget -= self.config.physics_hz
                self._set_control_target(self._effective_target())
            self._step_physics(self._effective_stuck_positions())
            self._physics_steps += 1

        self._observation_steps += 1
        if not self._faults.stale_state:
            self._sampled_state = self._read_state()
            self._state_timestamp_ns = self._sim_time_ns

    def _effective_target(self) -> np.ndarray:
        target = self._command_target.copy()
        if self._faults.left_worker_failed:
            target[:7] = [self._failed_worker_positions[index] for index in range(7)]
        if self._faults.right_worker_failed:
            target[7:] = [self._failed_worker_positions[index] for index in range(7, STATE_SIZE)]
        for index, position in self._stuck_positions.items():
            target[index] = position
        return target

    def _effective_stuck_positions(self) -> dict[int, float]:
        return self._failed_worker_positions | self._stuck_positions

    @staticmethod
    def _validate_action(action: np.ndarray) -> np.ndarray:
        validated = np.asarray(action, dtype=np.float32)
        if validated.shape != (STATE_SIZE,):
            raise ValueError(f"action must have shape ({STATE_SIZE},), got {validated.shape}")
        if not np.isfinite(validated).all():
            raise ValueError("action must contain only finite values")
        lower, upper = action_bounds()
        if np.any(validated < lower) or np.any(validated > upper):
            invalid = np.flatnonzero((validated < lower) | (validated > upper)).tolist()
            raise ValueError(f"action exceeds YAM joint or gripper limits at indices {invalid}")
        return validated.copy()

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Simulator is not connected")

    def _require_running(self) -> None:
        self._require_connected()
        if not self._running:
            raise RuntimeError("Simulator is not running")

    @abstractmethod
    def _reset_backend(self, seed: int, initial_state: np.ndarray) -> None: ...

    @abstractmethod
    def _read_state(self) -> np.ndarray: ...

    @abstractmethod
    def _set_control_target(self, target: np.ndarray) -> None: ...

    @abstractmethod
    def _step_physics(self, stuck_positions: dict[int, float]) -> None: ...

    @abstractmethod
    def _render_camera(self, camera_name: str) -> np.ndarray: ...

    @abstractmethod
    def _close_backend(self) -> None: ...


class DeterministicBiYAMBackend(_StepDrivenBiYAMBackend):
    """Dependency-light deterministic dynamics and synthetic RGB backend."""

    backend_name = "fallback"

    def __init__(self, config: BiYAMSimulatorConfig) -> None:
        super().__init__(config)
        self._positions = np.asarray(config.initial_state, dtype=np.float64).copy()
        self._velocities = np.zeros(STATE_SIZE, dtype=np.float64)
        self._target = self._positions.copy()
        self._camera_offsets = np.zeros((len(CAMERA_NAMES), 3), dtype=np.uint16)

    def _reset_backend(self, seed: int, initial_state: np.ndarray) -> None:
        self._positions = initial_state.astype(np.float64, copy=True)
        self._velocities = np.zeros(STATE_SIZE, dtype=np.float64)
        self._target = self._positions.copy()
        rng = np.random.default_rng(seed)
        self._camera_offsets = rng.integers(0, 256, size=(len(CAMERA_NAMES), 3), dtype=np.uint16)

    def _read_state(self) -> np.ndarray:
        return self._positions.astype(np.float32)

    def _set_control_target(self, target: np.ndarray) -> None:
        self._target = target.astype(np.float64, copy=True)

    def _step_physics(self, stuck_positions: dict[int, float]) -> None:
        dt = 1.0 / self.config.physics_hz
        acceleration = (
            self.config.fallback_position_gain * (self._target - self._positions)
            - self.config.fallback_damping * self._velocities
        )
        self._velocities += acceleration * dt
        self._positions += self._velocities * dt
        lower, upper = action_bounds()
        self._positions = np.clip(self._positions, lower, upper)
        for index, position in stuck_positions.items():
            self._positions[index] = position
            self._velocities[index] = 0.0

    def _render_camera(self, camera_name: str) -> np.ndarray:
        camera_index = CAMERA_NAMES.index(camera_name)
        height, width = self.config.camera_height, self.config.camera_width
        x = np.arange(width, dtype=np.int32)[None, :]
        y = np.arange(height, dtype=np.int32)[:, None]
        pose_code = int(np.rint(np.dot(self._positions, np.arange(1, STATE_SIZE + 1)) * 23.0))
        offsets = self._camera_offsets[camera_index]
        frame = np.empty((height, width, 3), dtype=np.uint8)
        frame[..., 0] = (x + pose_code + int(offsets[0])) % 256
        frame[..., 1] = (y + 2 * pose_code + int(offsets[1])) % 256
        frame[..., 2] = ((x // 2 + y // 2) + 3 * pose_code + int(offsets[2])) % 256
        return frame

    def _close_backend(self) -> None:
        return None


def mujoco_backend_available() -> bool:
    """Return whether both optional packages required by the full backend are importable."""

    return importlib.util.find_spec("mujoco") is not None and importlib.util.find_spec("i2rt") is not None


class BiYAMSimulator:
    """Small stable facade selecting the full or fallback bimanual backend."""

    def __init__(
        self,
        config: BiYAMSimulatorConfig | None = None,
        backend: Literal["auto", "mujoco", "fallback"] = "auto",
    ) -> None:
        self.config = config or BiYAMSimulatorConfig()
        if backend not in {"auto", "mujoco", "fallback"}:
            raise ValueError(f"Unknown backend {backend!r}")
        selected_backend = "mujoco" if backend == "auto" and mujoco_backend_available() else backend
        if selected_backend == "auto":
            selected_backend = "fallback"
        if selected_backend == "mujoco":
            if not mujoco_backend_available():
                raise ModuleNotFoundError("The MuJoCo backend requires both 'mujoco' and 'i2rt'")
            from lerobot.simulators.bi_yam.mujoco_backend import MujocoBiYAMBackend

            self._backend: _StepDrivenBiYAMBackend = MujocoBiYAMBackend(self.config)
        else:
            self._backend = DeterministicBiYAMBackend(self.config)

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    @property
    def is_connected(self) -> bool:
        return self._backend.is_connected

    @property
    def is_running(self) -> bool:
        return self._backend.is_running

    @property
    def faults(self) -> BiYAMFaults:
        return self._backend.faults

    def connect(self) -> None:
        self._backend.connect()

    def start(self) -> None:
        self._backend.start()

    def reset(self, seed: int | None = None) -> None:
        self._backend.reset(seed)

    def get_state(self) -> np.ndarray:
        return self._backend.get_state()

    def apply_action(self, action: np.ndarray) -> None:
        self._backend.apply_action(action)

    def render_cameras(self) -> dict[str, np.ndarray]:
        return self._backend.render_cameras()

    def set_faults(self, faults: BiYAMFaults) -> None:
        self._backend.set_faults(faults)

    def clear_faults(self) -> None:
        self._backend.clear_faults()

    def safe_idle(self) -> None:
        self._backend.safe_idle()

    def get_health(self) -> BiYAMHealth:
        return self._backend.get_health()

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> BiYAMSimulator:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
