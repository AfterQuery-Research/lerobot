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

import json
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import draccus
import numpy as np

from lerobot.cameras import Camera, make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.can import (
    CANInterfaceInfo,
    can_interface_readiness_error,
    discover_can_interfaces,
    resolve_can_interface,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_bi_yam import (
    YAM_SCALAR_KEYS,
    BiYAMFollowerConfig,
    YAMArmConfig,
    YAMGripperCalibration,
)
from .worker import ArmCommand, ArmState, ArmWorker, ProcessArmWorker

logger = logging.getLogger(__name__)

WorkerFactory = Callable[[str, YAMArmConfig], ArmWorker]
CameraFactory = Callable[[dict[str, Any]], dict[str, Camera]]
CANDiscovery = Callable[[], list[CANInterfaceInfo]]


class BiYAMFollower(Robot):
    """Dual YAM follower with process-isolated i2rt ownership and local arming."""

    config_class = BiYAMFollowerConfig
    name = "bi_yam_follower"

    def __init__(
        self,
        config: BiYAMFollowerConfig,
        *,
        worker_factory: WorkerFactory | None = None,
        camera_factory: CameraFactory | None = None,
        can_discovery: CANDiscovery = discover_can_interfaces,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ):
        self._yam_calibration: dict[str, YAMGripperCalibration] = {}
        super().__init__(config)
        self.config = config
        self._arm_configs = self._resolve_arm_configs()
        self._worker_factory = worker_factory or (lambda side, cfg: ProcessArmWorker(side, cfg))
        self._can_discovery = can_discovery
        self._monotonic_ns = monotonic_ns
        self.cameras = (camera_factory or make_cameras_from_configs)(config.cameras)
        self._workers: dict[str, ArmWorker] = {}
        self._states: dict[str, ArmState] = {}
        self._connected = False
        self._armed = False
        self._command_sequence = 0
        self._control_telemetry_file: TextIO | None = None
        self._last_control_summary_ns = 0

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = dict.fromkeys(YAM_SCALAR_KEYS, float)
        for name, camera_config in self.config.cameras.items():
            if getattr(camera_config, "use_rgb", True):
                features[name] = (camera_config.height, camera_config.width, 3)
            if getattr(camera_config, "use_depth", False):
                features[f"{name}_depth"] = (camera_config.height, camera_config.width, 1)
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(YAM_SCALAR_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return all(arm_config.has_fixed_gripper_calibration for arm_config in self._arm_configs.values())

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def state_metadata(self) -> dict[str, dict[str, int]]:
        return {
            side: {"sequence": state.sequence, "timestamp_ns": state.timestamp_ns}
            for side, state in self._states.items()
        }

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self.__class__.__name__} is already connected.")

        arm_configs = self._startup_arm_configs()
        for side, arm_config in arm_configs.items():
            try:
                arm_config.validate_hardware_startup()
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Unsafe {side} arm configuration: {exc} No usable {side} calibration was loaded "
                    f"from {self.calibration_fpath}. Run `lerobot-calibrate --robot.type={self.config.type} "
                    f"--robot.id={self.id} --robot.calibration_side={side} "
                    f"--robot.{side}_arm_config.allow_gripper_calibration=true` under supervision."
                ) from exc

        arm_configs = self._resolve_hardware_channels(arm_configs)

        self._workers = {}
        try:
            for side, arm_config in arm_configs.items():
                self._workers[side] = self._worker_factory(side, arm_config)
            for side, worker in self._workers.items():
                self._states[side] = worker.start(self.config.startup_timeout_s)
            if self.config.calibration_side is None:
                for camera in self.cameras.values():
                    camera.connect()
            self._open_control_telemetry()
        except Exception:
            self._close_control_telemetry()
            self._cleanup_resources(list(self.cameras.values()))
            self._workers.clear()
            self._states.clear()
            raise

        self._connected = True
        self._armed = False
        logger.info("%s connected in safe idle", self)

    def calibrate(self) -> None:
        self._require_connected()
        side = self.config.calibration_side
        if side is None:
            raise RuntimeError(
                "Set calibration_side to left or right and calibrate one YAM gripper at a time"
            )
        state = self._states.get(side)
        if state is None or state.gripper_limits is None:
            raise RuntimeError(f"{side} arm did not report calibrated gripper limits")

        calibration = YAMGripperCalibration(gripper_limits=state.gripper_limits)
        self._yam_calibration[side] = calibration
        self._save_calibration()
        original = self.config.left_arm_config if side == "left" else self.config.right_arm_config
        self._arm_configs[side] = replace(
            original,
            gripper_limits_override=calibration.gripper_limits,
            allow_gripper_calibration=False,
        )
        logger.info("Saved %s YAM gripper calibration to %s", side, self.calibration_fpath)

    def configure(self) -> None:
        return

    def arm(self) -> None:
        """Locally enable command acceptance; policy action dictionaries cannot arm the robot."""
        self._require_rollout_mode()
        self._require_connected()
        states = self._refresh_states(allow_fault=True)
        self._validate_operational_state(states)
        try:
            for side, worker in self._workers.items():
                self._states[side] = worker.arm(self.config.command_ack_timeout_s)
        except Exception:
            self._safe_idle_workers()
            raise
        self._armed = True

    def disarm(self) -> None:
        self._require_resources()
        self._safe_idle_workers()

    def get_observation(self) -> RobotObservation:
        self._require_rollout_mode()
        self._require_connected()
        states = self._refresh_states()
        observation: RobotObservation = {}
        positions = self._control_positions(states)
        for key, value in zip(YAM_SCALAR_KEYS, positions, strict=True):
            observation[key] = value

        try:
            for name, camera in self.cameras.items():
                config = self.config.cameras[name]
                if getattr(config, "use_rgb", True):
                    observation[name] = camera.async_read()
                if getattr(config, "use_depth", False):
                    observation[f"{name}_depth"] = camera.async_read_depth()
        except Exception:
            self._safe_idle_workers()
            raise
        return observation

    def send_action(self, action: RobotAction) -> RobotAction:
        self._require_rollout_mode()
        self._require_connected()
        if not self._armed:
            raise RuntimeError("BiYAM is in safe idle; arm it locally before sending actions")

        states = self._refresh_commandable_states()
        try:
            requested = self._validate_action(action)
        except Exception:
            self._safe_idle_workers()
            raise
        present = self._control_positions(states)
        bounded = np.clip(requested, *self._operational_bounds())
        applied = self._apply_safety_limits(requested, present)
        applied_action = self._dispatch_positions(applied)
        measured = self._control_positions(self._states)
        self._record_control_telemetry(
            phase="policy",
            requested=requested,
            bounded=bounded,
            applied=applied,
            measured_before=present,
            measured_after=measured,
        )
        return applied_action

    def reset_for_policy(self) -> RobotAction | None:
        """Move to the configured policy start pose before autonomous control."""
        configured_target = self.config.policy_start_position
        if configured_target is None:
            return None

        self._require_rollout_mode()
        self._require_connected()
        if not self._armed:
            raise RuntimeError("BiYAM must be armed before resetting for policy control")

        target = np.asarray(configured_target, dtype=np.float64)
        target_action = {key: float(value) for key, value in zip(YAM_SCALAR_KEYS, target, strict=True)}
        started_at = time.monotonic()
        control_interval_s = 1.0 / self.config.policy_reset_fps
        logger.info("Moving BiYAM to its configured policy start position")

        try:
            states = self._refresh_commandable_states()
            start = self._control_positions(states)
            max_error = float(np.max(np.abs(target - start)))
            if max_error <= self.config.policy_reset_tolerance:
                logger.info("BiYAM is already at its policy start position (max error %.4f)", max_error)
                return target_action

            trajectory_steps = min(
                max(int(np.ceil(max_error / self.config.policy_reset_step_size)), 1),
                self.config.policy_reset_max_steps,
            )
            logger.info("Executing policy reset trajectory (%d steps)", trajectory_steps)
            for waypoint in np.linspace(start, target, trajectory_steps + 1)[1:]:
                loop_started_at = time.perf_counter()
                states = self._refresh_commandable_states()
                present = self._control_positions(states)
                max_error = float(np.max(np.abs(target - present)))
                if time.monotonic() - started_at >= self.config.policy_reset_timeout_s:
                    raise TimeoutError(
                        "BiYAM did not reach its policy start position within "
                        f"{self.config.policy_reset_timeout_s:.1f}s (max error {max_error:.4f})"
                    )
                self._dispatch_positions(waypoint)
                measured = self._control_positions(self._states)
                self._record_control_telemetry(
                    phase="reset",
                    requested=waypoint,
                    bounded=waypoint,
                    applied=waypoint,
                    measured_before=present,
                    measured_after=measured,
                )
                time.sleep(max(0.0, control_interval_s - (time.perf_counter() - loop_started_at)))

            # Keep the final absolute target active until measured state settles.
            while True:
                loop_started_at = time.perf_counter()
                states = self._refresh_commandable_states()
                present = self._control_positions(states)
                max_error = float(np.max(np.abs(target - present)))
                if max_error <= self.config.policy_reset_tolerance:
                    logger.info("BiYAM reached its policy start position (max error %.4f)", max_error)
                    return target_action
                if time.monotonic() - started_at >= self.config.policy_reset_timeout_s:
                    raise TimeoutError(
                        "BiYAM did not reach its policy start position within "
                        f"{self.config.policy_reset_timeout_s:.1f}s (max error {max_error:.4f})"
                    )
                self._dispatch_positions(target)
                measured = self._control_positions(self._states)
                self._record_control_telemetry(
                    phase="reset_hold",
                    requested=target,
                    bounded=target,
                    applied=target,
                    measured_before=present,
                    measured_after=measured,
                )
                time.sleep(max(0.0, control_interval_s - (time.perf_counter() - loop_started_at)))
        except (Exception, KeyboardInterrupt):
            self._safe_idle_workers()
            raise

    def _dispatch_positions(self, applied: np.ndarray) -> RobotAction:
        self._command_sequence += 1
        command_sequence = self._command_sequence
        execute_at_ns = self._monotonic_ns() + int(self.config.command_lead_time_s * 1e9)
        commands = {
            "left": ArmCommand(command_sequence, execute_at_ns, tuple(applied[:7])),
            "right": ArmCommand(command_sequence, execute_at_ns, tuple(applied[7:])),
        }

        try:
            for side, worker in self._workers.items():
                worker.send_command(commands[side])
            for side, worker in self._workers.items():
                state = worker.wait_applied(command_sequence, self.config.command_ack_timeout_s)
                if state.last_applied_positions != commands[side].positions:
                    raise RuntimeError(f"{side} arm acknowledged different positions")
                self._states[side] = state
        except Exception:
            self._safe_idle_workers()
            raise

        return {key: float(value) for key, value in zip(YAM_SCALAR_KEYS, applied, strict=True)}

    def disconnect(self) -> None:
        self._require_resources()
        errors = self._cleanup_resources(list(self.cameras.values()))
        self._close_control_telemetry()
        self._connected = False
        self._armed = False
        self._states.clear()
        self._workers.clear()
        if errors:
            raise RuntimeError("BiYAM cleanup failed: " + "; ".join(errors))
        logger.info("%s disconnected", self)

    def _open_control_telemetry(self) -> None:
        path = self.config.control_telemetry_path
        if path is None or self.config.calibration_side is not None:
            return
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._control_telemetry_file = path.open("w", encoding="utf-8", buffering=1)
        logger.info("Writing YAM control telemetry to %s", path)

    def _close_control_telemetry(self) -> None:
        if self._control_telemetry_file is not None:
            with suppress(OSError):
                self._control_telemetry_file.close()
            self._control_telemetry_file = None

    def _record_control_telemetry(
        self,
        *,
        phase: str,
        requested: np.ndarray,
        bounded: np.ndarray,
        applied: np.ndarray,
        measured_before: np.ndarray,
        measured_after: np.ndarray,
    ) -> None:
        bound_clipped = [
            key
            for key, raw, limited in zip(YAM_SCALAR_KEYS, requested, bounded, strict=True)
            if not np.isclose(raw, limited, rtol=0.0, atol=1e-9)
        ]
        delta_clipped = [
            key
            for key, limited, dispatched in zip(YAM_SCALAR_KEYS, bounded, applied, strict=True)
            if not np.isclose(limited, dispatched, rtol=0.0, atol=1e-9)
        ]
        requested_error = requested - measured_after
        applied_error = applied - measured_after
        timestamp_ns = self._monotonic_ns()
        record = {
            "sequence": self._command_sequence,
            "phase": phase,
            "monotonic_ns": timestamp_ns,
            "names": list(YAM_SCALAR_KEYS),
            "requested": requested.tolist(),
            "bounded": bounded.tolist(),
            "applied": applied.tolist(),
            "measured_before": measured_before.tolist(),
            "measured_after": measured_after.tolist(),
            "requested_tracking_error": requested_error.tolist(),
            "applied_tracking_error": applied_error.tolist(),
            "bound_clipped": bound_clipped,
            "delta_clipped": delta_clipped,
        }
        if self._control_telemetry_file is not None:
            try:
                self._control_telemetry_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            except OSError as exc:
                logger.warning("Disabling YAM control telemetry after write failure: %s", exc)
                self._close_control_telemetry()

        interval_ns = int(self.config.control_telemetry_console_interval_s * 1e9)
        if interval_ns == 0 or timestamp_ns - self._last_control_summary_ns >= interval_ns:
            logger.info(
                "YAM control phase=%s seq=%d max_applied_tracking_error=%.4f "
                "bound_clipped=%s delta_clipped=%s",
                phase,
                self._command_sequence,
                float(np.max(np.abs(applied_error))),
                bound_clipped,
                delta_clipped,
            )
            self._last_control_summary_ns = timestamp_ns

    def _require_resources(self) -> None:
        if not self._connected and not self._workers:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )

    def _require_rollout_mode(self) -> None:
        if self.config.calibration_side is not None:
            raise RuntimeError("Policy control is disabled while calibrating a YAM gripper")

    def _resolve_arm_configs(self) -> dict[str, YAMArmConfig]:
        resolved = {
            "left": self.config.left_arm_config,
            "right": self.config.right_arm_config,
        }
        for side, calibration in self._yam_calibration.items():
            arm_config = resolved[side]
            recalibrating = self.config.calibration_side == side and arm_config.allow_gripper_calibration
            if arm_config.gripper_limits_override is None and not recalibrating:
                resolved[side] = replace(
                    arm_config,
                    gripper_limits_override=calibration.gripper_limits,
                )
        return resolved

    def _startup_arm_configs(self) -> dict[str, YAMArmConfig]:
        calibration_side = self.config.calibration_side
        if calibration_side is None:
            moving_sides = [
                side
                for side, arm_config in self._arm_configs.items()
                if arm_config.allow_gripper_calibration and not arm_config.has_fixed_gripper_calibration
            ]
            if moving_sides:
                raise RuntimeError(
                    "Moving gripper calibration is only allowed through single-arm calibration mode; "
                    f"set calibration_side for {moving_sides[0]}"
                )
            return dict(self._arm_configs)

        arm_config = self._arm_configs[calibration_side]
        if not arm_config.allow_gripper_calibration:
            raise RuntimeError(
                f"Calibrating {calibration_side} requires "
                f"{calibration_side}_arm_config.allow_gripper_calibration=true"
            )
        return {calibration_side: arm_config}

    def _resolve_hardware_channels(self, arm_configs: dict[str, YAMArmConfig]) -> dict[str, YAMArmConfig]:
        if not any(config.adapter_serial is not None for config in arm_configs.values()):
            return dict(arm_configs)

        interfaces = self._can_discovery()
        resolved: dict[str, YAMArmConfig] = {}
        readiness_errors = []
        for side, arm_config in arm_configs.items():
            if arm_config.adapter_serial is None:
                resolved[side] = arm_config
                continue

            interface = resolve_can_interface(arm_config.adapter_serial, interfaces=interfaces)
            readiness_error = can_interface_readiness_error(
                interface,
                bitrate=1_000_000,
                use_fd=False,
            )
            if readiness_error is not None:
                readiness_errors.append(
                    f"{side} adapter {arm_config.adapter_serial} resolved to {interface.name}: "
                    f"{readiness_error}"
                )
            logger.info(
                "Resolved %s YAM CAN adapter %s to %s",
                side,
                arm_config.adapter_serial,
                interface.name,
            )
            resolved[side] = replace(arm_config, adapter_serial=None, channel=interface.name)

        channels = [config.channel for config in resolved.values() if config.channel is not None]
        if len(channels) != len(set(channels)):
            raise RuntimeError("Left and right YAM adapters resolved to the same SocketCAN interface")
        if readiness_errors:
            setup_flags = " ".join(
                f"--{side}_adapter_serial={config.adapter_serial}"
                for side, config in (
                    ("left", self.config.left_arm_config),
                    ("right", self.config.right_arm_config),
                )
                if config.adapter_serial is not None
            )
            raise RuntimeError(
                "SocketCAN is not ready:\n- "
                + "\n- ".join(readiness_errors)
                + f"\nRun `lerobot-setup-can {setup_flags}` before connecting the robot."
            )
        return resolved

    def _load_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as calibration_file, draccus.config_type("json"):
            calibration = draccus.load(dict[str, YAMGripperCalibration], calibration_file)
        unknown_sides = set(calibration) - {"left", "right"}
        if unknown_sides:
            raise ValueError(f"Unknown YAM calibration sides: {sorted(unknown_sides)}")
        self._yam_calibration = calibration

    def _save_calibration(self, fpath: Path | None = None) -> None:
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as calibration_file, draccus.config_type("json"):
            draccus.dump(self._yam_calibration, calibration_file, indent=4)

    def _require_connected(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        dead_sides = [side for side, worker in self._workers.items() if not worker.is_alive]
        disconnected_cameras = (
            []
            if self.config.calibration_side is not None
            else [name for name, camera in self.cameras.items() if not camera.is_connected]
        )
        if dead_sides or disconnected_cameras:
            self._safe_idle_workers()
            details = []
            if dead_sides:
                details.append(f"dead arm workers: {dead_sides}")
            if disconnected_cameras:
                details.append(f"disconnected cameras: {disconnected_cameras}")
            raise RuntimeError("BiYAM connection became unhealthy (" + ", ".join(details) + ")")

    def _refresh_states(self, *, allow_fault: bool = False) -> dict[str, ArmState]:
        try:
            for side, worker in self._workers.items():
                state = worker.latest_state(self.config.state_timeout_s)
                if not state.ready:
                    raise RuntimeError(f"{side} arm is not ready: {state.fault}")
                if state.fault is not None and not allow_fault:
                    raise RuntimeError(f"{side} arm fault: {state.fault}")
                now_ns = self._monotonic_ns()
                age_ns = now_ns - state.timestamp_ns
                if age_ns < 0 or age_ns > int(self.config.state_timeout_s * 1e9):
                    raise RuntimeError(f"{side} arm state is stale")
                self._states[side] = state
        except Exception:
            self._safe_idle_workers()
            raise
        return dict(self._states)

    def _validate_operational_state(self, states: dict[str, ArmState]) -> None:
        positions = np.asarray((*states["left"].positions, *states["right"].positions), dtype=np.float64)
        lower, upper = self._operational_bounds()
        lower[[6, 13]] -= self.config.gripper_state_tolerance
        upper[[6, 13]] += self.config.gripper_state_tolerance
        if not np.isfinite(positions).all() or np.any(positions < lower) or np.any(positions > upper):
            raise RuntimeError("Cannot arm while measured state is outside operational limits")

    def _control_positions(self, states: dict[str, ArmState]) -> np.ndarray:
        positions = np.asarray((*states["left"].positions, *states["right"].positions), dtype=np.float64)
        gripper_lower, gripper_upper = self.config.gripper_limits
        positions[[6, 13]] = np.clip(positions[[6, 13]], gripper_lower, gripper_upper)
        return positions

    def _refresh_commandable_states(self) -> dict[str, ArmState]:
        states = self._refresh_states()
        if not all(state.armed for state in states.values()):
            self._safe_idle_workers()
            raise RuntimeError("An arm worker left the armed state")
        try:
            self._validate_operational_state(states)
        except Exception:
            self._safe_idle_workers()
            raise
        return states

    def _validate_action(self, action: RobotAction) -> np.ndarray:
        expected = set(YAM_SCALAR_KEYS)
        received = set(action)
        if received != expected or len(action) != len(YAM_SCALAR_KEYS):
            missing = sorted(expected - received)
            extra = sorted(received - expected)
            raise ValueError(f"Action keys must match the BiYAM schema; missing={missing}, extra={extra}")
        try:
            values = np.asarray([float(action[key]) for key in YAM_SCALAR_KEYS], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Action values must be numeric scalars") from exc
        if not np.isfinite(values).all():
            raise ValueError("Action values must all be finite")
        return values

    def _apply_safety_limits(self, requested: np.ndarray, present: np.ndarray) -> np.ndarray:
        lower, upper = self._operational_bounds()
        bounded = np.clip(requested, lower, upper)
        delta = np.asarray(
            [
                *([self.config.max_joint_delta] * 6),
                self.config.max_gripper_delta,
                *([self.config.max_joint_delta] * 6),
                self.config.max_gripper_delta,
            ],
            dtype=np.float64,
        )
        return np.clip(bounded, present - delta, present + delta)

    def _operational_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        limits = [
            *self.config.left_joint_limits,
            self.config.gripper_limits,
            *self.config.right_joint_limits,
            self.config.gripper_limits,
        ]
        return (
            np.asarray([lower for lower, _ in limits], dtype=np.float64),
            np.asarray([upper for _, upper in limits], dtype=np.float64),
        )

    def _safe_idle_workers(self) -> None:
        self._armed = False
        for worker in self._workers.values():
            if worker.is_alive:
                with suppress(Exception):
                    worker.disarm(self.config.command_ack_timeout_s)

    def _cleanup_resources(self, cameras: list[Camera]) -> list[str]:
        errors: list[str] = []
        self._safe_idle_workers()
        for camera in cameras:
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:
                errors.append(f"camera: {exc}")
        for side, worker in self._workers.items():
            try:
                worker.close(self.config.shutdown_timeout_s)
            except Exception as exc:
                errors.append(f"{side} arm: {exc}")
        return errors
