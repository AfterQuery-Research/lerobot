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

import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Protocol

import numpy as np

from .config_bi_yam import YAMArmConfig

logger = logging.getLogger(__name__)

_GRIPPER_ENCODER_PERIOD_RAD = 2 * np.pi
_GRIPPER_REBASE_TIMEOUT_S = 0.25


class ArmBackend(Protocol):
    def num_dofs(self) -> int: ...

    def get_joint_pos(self) -> np.ndarray: ...

    def get_robot_info(self) -> dict[str, Any]: ...

    def command_joint_pos(self, joint_pos: np.ndarray) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ArmCommand:
    sequence: int
    execute_at_ns: int
    positions: tuple[float, ...]


@dataclass(frozen=True)
class ArmControl:
    sequence: int
    operation: str


@dataclass(frozen=True)
class ArmState:
    sequence: int
    timestamp_ns: int
    positions: tuple[float, ...]
    ready: bool
    armed: bool
    idle: bool
    control_sequence: int
    last_applied_command_sequence: int
    last_applied_positions: tuple[float, ...] | None
    gripper_limits: tuple[float, float] | None = None
    fault: str | None = None


class ArmWorker(Protocol):
    @property
    def is_alive(self) -> bool: ...

    def start(self, timeout_s: float) -> ArmState: ...

    def arm(self, timeout_s: float) -> ArmState: ...

    def disarm(self, timeout_s: float) -> ArmState: ...

    def latest_state(self, timeout_s: float) -> ArmState: ...

    def send_command(self, command: ArmCommand) -> None: ...

    def wait_applied(self, sequence: int, timeout_s: float) -> ArmState: ...

    def close(self, timeout_s: float) -> None: ...


class _I2RTHardwareBackend:
    """Add deterministic thread shutdown around i2rt's hardware backend."""

    def __init__(
        self,
        backend: ArmBackend,
        motor_chain: Any,
        control_threads: tuple[threading.Thread, ...],
        *,
        shutdown_timeout_s: float = 1.0,
    ) -> None:
        self._backend = backend
        self._motor_chain = motor_chain
        self._control_threads = control_threads
        self._shutdown_timeout_s = shutdown_timeout_s
        self._closed = False

    def num_dofs(self) -> int:
        return self._backend.num_dofs()

    def get_joint_pos(self) -> np.ndarray:
        return self._backend.get_joint_pos()

    def get_robot_info(self) -> dict[str, Any]:
        return self._backend.get_robot_info()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._backend.command_joint_pos(joint_pos)

    def enter_gravity_comp_idle(self) -> None:
        self._backend.enter_gravity_comp_idle()  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        deadline = time.monotonic() + self._shutdown_timeout_s

        # i2rt 26ffe68 closes SocketCAN before its motor thread exits. Stop the
        # producer threads explicitly so they cannot touch an invalid fd.
        stop_event = getattr(self._backend, "_stop_event", None)
        server_thread = getattr(self._backend, "_server_thread", None)
        if stop_event is None or not isinstance(server_thread, threading.Thread):
            self._backend.close()
            return

        stop_event.set()
        self._join_thread(server_thread, deadline)
        self._motor_chain.running = False
        for thread in self._control_threads:
            self._join_thread(thread, deadline)
        self._backend.close()

    @staticmethod
    def _join_thread(thread: threading.Thread, deadline: float) -> None:
        if thread is threading.current_thread():
            raise RuntimeError("i2rt shutdown attempted to join the current thread")
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if thread.is_alive():
            raise RuntimeError(f"Timed out stopping i2rt thread {thread.name!r}")


def _nearest_calibrated_gripper_turns(
    current_position: float,
    calibration_limits: tuple[float, float],
) -> int:
    """Return the 2π branch shift that puts a motor reading nearest its calibrated stroke."""

    limits = np.asarray(calibration_limits, dtype=np.float64)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] == limits[1]:
        raise ValueError("gripper calibration limits must be two distinct finite values")
    if not np.isfinite(current_position):
        raise ValueError("gripper motor position must be finite")
    if abs(float(limits[1] - limits[0])) >= _GRIPPER_ENCODER_PERIOD_RAD:
        raise ValueError("gripper calibration stroke must be shorter than one encoder revolution")

    calibration_center = float(np.mean(limits))
    return int(np.rint((calibration_center - current_position) / _GRIPPER_ENCODER_PERIOD_RAD))


def _rebase_calibrated_gripper_encoder(
    backend: ArmBackend,
    calibration_limits: tuple[float, float] | None,
) -> None:
    """Align a single-turn gripper encoder with its persisted calibrated angle branch."""

    if calibration_limits is None:
        return
    info = backend.get_robot_info()
    gripper_index = info.get("gripper_index")
    if gripper_index is None:
        return

    motor_chain = getattr(backend, "motor_chain", None)
    if motor_chain is None:
        raise RuntimeError("i2rt gripper backend does not expose its motor chain")
    motor_states = motor_chain.read_states()
    current_position = float(motor_states[gripper_index].pos)
    turns = _nearest_calibrated_gripper_turns(current_position, calibration_limits)
    if turns == 0:
        return
    if abs(turns) != 1:
        raise RuntimeError(
            f"refusing to rebase gripper encoder by {turns} revolutions from {current_position:.6f} rad"
        )

    direction = float(motor_chain.motor_direction[gripper_index])
    if not np.isfinite(direction) or direction == 0.0:
        raise RuntimeError("i2rt gripper motor direction must be finite and nonzero")
    old_offset = float(motor_chain.motor_offset[gripper_index])
    offset_delta = -turns * _GRIPPER_ENCODER_PERIOD_RAD / direction
    motor_chain.motor_offset[gripper_index] = old_offset + offset_delta

    closed, open_ = calibration_limits
    corrected_position = current_position + turns * _GRIPPER_ENCODER_PERIOD_RAD
    expected_normalized = (corrected_position - closed) / (open_ - closed)
    deadline = time.monotonic() + _GRIPPER_REBASE_TIMEOUT_S
    actual_normalized = float(backend.get_joint_pos()[gripper_index])
    while time.monotonic() < deadline and not np.isclose(
        actual_normalized, expected_normalized, atol=1e-3, rtol=0.0
    ):
        time.sleep(0.005)
        actual_normalized = float(backend.get_joint_pos()[gripper_index])
    if not np.isfinite(actual_normalized) or not np.isclose(
        actual_normalized, expected_normalized, atol=1e-3, rtol=0.0
    ):
        motor_chain.motor_offset[gripper_index] = old_offset
        raise RuntimeError(
            "i2rt gripper encoder branch correction did not reach the expected normalized position"
        )

    logger.info(
        "Rebased calibrated gripper encoder by %+d turn (raw %.6f -> %.6f rad, normalized %.6f)",
        turns,
        current_position,
        corrected_position,
        actual_normalized,
    )


def _find_motor_control_threads(
    motor_chain: Any,
    threads_before_startup: set[threading.Thread],
) -> tuple[threading.Thread, ...]:
    threads = threading.enumerate()
    owned = [
        thread
        for thread in threads
        if getattr(getattr(thread, "_target", None), "__self__", None) is motor_chain
    ]
    if owned:
        return tuple(owned)
    return tuple(
        thread
        for thread in threads
        if thread not in threads_before_startup and "_set_torques_and_update_state" in thread.name
    )


def make_i2rt_backend(config: YAMArmConfig) -> ArmBackend:
    config.validate_hardware_startup()
    if not config.sim and config.channel is None:
        raise RuntimeError(
            "Physical YAM adapter_serial must be resolved to a SocketCAN channel by BiYAMFollower"
        )
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
    except ImportError as exc:
        raise ImportError("i2rt is required to use the bi_yam_follower robot") from exc

    arm_type = ArmType.from_string_name(config.arm_type)
    gripper_type = GripperType.from_string_name(config.gripper_type)
    gripper_limits_override = (
        None
        if config.gripper_limits_override is None
        else np.asarray(config.gripper_limits_override, dtype=np.float64)
    )

    threads_before_startup = set(threading.enumerate())
    backend = get_yam_robot(
        channel=config.channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        zero_gravity_mode=True,
        gripper_limits_override=gripper_limits_override,
        sim=config.sim,
        enable_auto_recovery=config.enable_auto_recovery,
    )
    if not config.sim and config.gripper_type == "linear_4310":
        try:
            _rebase_calibrated_gripper_encoder(backend, config.gripper_limits_override)
        except Exception:
            backend.close()
            raise
    motor_chain = getattr(backend, "motor_chain", None)
    if motor_chain is None:
        return backend
    return _I2RTHardwareBackend(
        backend,
        motor_chain,
        _find_motor_control_threads(motor_chain, threads_before_startup),
    )


def _enter_safe_idle(backend: ArmBackend) -> None:
    for method_name in ("enter_gravity_comp_idle", "enable_gravity_comp", "zero_torque_mode"):
        method = getattr(backend, method_name, None)
        if method is not None:
            method()
            return
    raise RuntimeError("i2rt backend does not expose a safe idle operation")


def _read_gripper_limits(backend: ArmBackend) -> tuple[float, float] | None:
    info = backend.get_robot_info()
    raw_limits = info.get("gripper_limits")
    if raw_limits is None:
        return None
    limits = np.asarray(raw_limits, dtype=np.float64)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] == limits[1]:
        raise RuntimeError("i2rt returned invalid gripper limits")
    return (float(limits[0]), float(limits[1]))


class ArmWorkerRuntime:
    """State machine executed inside exactly one arm-owning process."""

    def __init__(self, backend: ArmBackend, command_ttl_s: float):
        self._backend = backend
        self._command_ttl_ns = int(command_ttl_s * 1e9)
        self._state_sequence = 0
        self._control_sequence = -1
        self._last_applied_sequence = -1
        self._last_applied_positions: tuple[float, ...] | None = None
        self._last_command_ns: int | None = None
        self._pending_command: ArmCommand | None = None
        self._armed = False
        self._idle = True
        self._fault: str | None = None
        self._closed = False

        if self._backend.num_dofs() != 7:
            raise ValueError(
                f"BiYAM requires seven controllable values per arm, got {self._backend.num_dofs()}"
            )
        self._gripper_limits = _read_gripper_limits(self._backend)
        _enter_safe_idle(self._backend)

    def handle_control(self, control: ArmControl) -> None:
        if control.operation == "arm":
            self._armed = True
            self._idle = True
            self._last_command_ns = None
            self._pending_command = None
            self._fault = None
        elif control.operation == "idle":
            self._pending_command = None
            self._armed = False
            self._enter_idle()
        else:
            raise ValueError(f"Unknown arm worker operation: {control.operation}")
        self._control_sequence = control.sequence

    def offer_command(self, command: ArmCommand) -> None:
        if self._armed and command.sequence > self._last_applied_sequence:
            self._pending_command = command

    def tick(self, now_ns: int) -> ArmState:
        command = self._pending_command
        if command is not None and self._armed and now_ns >= command.execute_at_ns:
            self._pending_command = None
            try:
                positions = np.asarray(command.positions, dtype=np.float64)
                if positions.shape != (7,) or not np.isfinite(positions).all():
                    raise ValueError("command must contain seven finite positions")
                self._backend.command_joint_pos(positions)
            except Exception as exc:
                self._fault = f"invalid_or_failed_command: {type(exc).__name__}: {exc}"
                self._armed = False
                self._enter_idle()
            else:
                self._last_applied_sequence = command.sequence
                self._last_applied_positions = tuple(float(value) for value in positions)
                self._last_command_ns = now_ns
                self._idle = False

        if (
            self._armed
            and self._last_command_ns is not None
            and now_ns - self._last_command_ns > self._command_ttl_ns
        ):
            self._fault = "command_ttl_expired"
            self._armed = False
            self._pending_command = None
            self._enter_idle()

        return self.snapshot(now_ns)

    def snapshot(self, now_ns: int) -> ArmState:
        positions = np.asarray(self._backend.get_joint_pos(), dtype=np.float64)
        if positions.shape != (7,) or not np.isfinite(positions).all():
            raise RuntimeError("i2rt returned an invalid seven-value joint state")
        self._state_sequence += 1
        return ArmState(
            sequence=self._state_sequence,
            timestamp_ns=now_ns,
            positions=tuple(float(value) for value in positions),
            ready=True,
            armed=self._armed,
            idle=self._idle,
            control_sequence=self._control_sequence,
            last_applied_command_sequence=self._last_applied_sequence,
            last_applied_positions=self._last_applied_positions,
            gripper_limits=self._gripper_limits,
            fault=self._fault,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._armed = False
            self._pending_command = None
            self._enter_idle()
        finally:
            self._backend.close()

    def _enter_idle(self) -> None:
        _enter_safe_idle(self._backend)
        self._idle = True


def _replace_queue_item(target_queue: Any, item: Any) -> None:
    while True:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                time.sleep(0)


def _drain_latest(source_queue: Any) -> Any | None:
    latest = None
    while True:
        try:
            latest = source_queue.get_nowait()
        except queue.Empty:
            return latest


def _arm_worker_entry(
    config: YAMArmConfig,
    command_queue: Any,
    control_queue: Any,
    state_queue: Any,
    stop_event: Any,
    backend_factory: Callable[[YAMArmConfig], ArmBackend],
) -> None:
    backend: ArmBackend | None = None
    runtime: ArmWorkerRuntime | None = None
    try:
        backend = backend_factory(config)
        runtime = ArmWorkerRuntime(backend, config.command_ttl_s)
        _replace_queue_item(state_queue, runtime.snapshot(time.monotonic_ns()))

        while not stop_event.is_set():
            while True:
                try:
                    runtime.handle_control(control_queue.get_nowait())
                except queue.Empty:
                    break

            command = _drain_latest(command_queue)
            if command is not None:
                runtime.offer_command(command)

            _replace_queue_item(state_queue, runtime.tick(time.monotonic_ns()))
            stop_event.wait(config.worker_poll_interval_s)
    except BaseException as exc:
        state = ArmState(
            sequence=0,
            timestamp_ns=time.monotonic_ns(),
            positions=(),
            ready=False,
            armed=False,
            idle=False,
            control_sequence=-1,
            last_applied_command_sequence=-1,
            last_applied_positions=None,
            fault=f"{type(exc).__name__}: {exc}",
        )
        _replace_queue_item(state_queue, state)
    finally:
        if runtime is not None:
            with suppress(Exception):
                runtime.close()
        elif backend is not None:
            with suppress(Exception):
                _enter_safe_idle(backend)
            with suppress(Exception):
                backend.close()


class ProcessArmWorker:
    def __init__(
        self,
        side: str,
        config: YAMArmConfig,
        *,
        context: BaseContext | None = None,
        backend_factory: Callable[[YAMArmConfig], ArmBackend] = make_i2rt_backend,
    ):
        self.side = side
        self.config = config
        self._context = context or mp.get_context("spawn")
        self._command_queue = self._context.Queue(maxsize=1)
        self._control_queue = self._context.Queue(maxsize=8)
        self._state_queue = self._context.Queue(maxsize=1)
        self._stop_event = self._context.Event()
        self._process = self._context.Process(
            name=f"bi-yam-{side}",
            target=_arm_worker_entry,
            args=(
                config,
                self._command_queue,
                self._control_queue,
                self._state_queue,
                self._stop_event,
                backend_factory,
            ),
            daemon=False,
        )
        self._latest_state: ArmState | None = None
        self._control_sequence = 0
        self._started = False
        self._closed = False

    @property
    def is_alive(self) -> bool:
        return self._started and not self._closed and self._process.is_alive()

    def start(self, timeout_s: float) -> ArmState:
        if self._started:
            raise RuntimeError(f"{self.side} arm worker has already been started")
        self._started = True
        self._process.start()
        try:
            state = self._wait_for(lambda item: item.ready or item.fault is not None, timeout_s)
        except Exception:
            self.close(timeout_s=min(timeout_s, 1.0))
            raise
        if not state.ready:
            self.close(timeout_s=min(timeout_s, 1.0))
            raise RuntimeError(f"{self.side} arm worker failed to start: {state.fault}")
        return state

    def arm(self, timeout_s: float) -> ArmState:
        control = self._send_control("arm", timeout_s)
        state = self._wait_for(
            lambda item: item.control_sequence >= control.sequence and item.armed,
            timeout_s,
        )
        if state.fault is not None:
            raise RuntimeError(f"{self.side} arm failed to arm: {state.fault}")
        return state

    def disarm(self, timeout_s: float) -> ArmState:
        control = self._send_control("idle", timeout_s)
        return self._wait_for(
            lambda item: item.control_sequence >= control.sequence and not item.armed and item.idle,
            timeout_s,
        )

    def latest_state(self, timeout_s: float) -> ArmState:
        return self._read_state(timeout_s)

    def send_command(self, command: ArmCommand) -> None:
        if not self.is_alive:
            raise RuntimeError(f"{self.side} arm worker is not running")
        _replace_queue_item(self._command_queue, command)

    def wait_applied(self, sequence: int, timeout_s: float) -> ArmState:
        state = self._wait_for(
            lambda item: item.last_applied_command_sequence >= sequence or item.fault is not None,
            timeout_s,
        )
        if state.fault is not None:
            raise RuntimeError(f"{self.side} arm command failed: {state.fault}")
        if state.last_applied_command_sequence != sequence:
            raise RuntimeError(
                f"{self.side} arm acknowledged command {state.last_applied_command_sequence}, expected {sequence}"
            )
        return state

    def close(self, timeout_s: float) -> None:
        if self._closed:
            return
        try:
            if self._started and self._process.is_alive():
                with suppress(Exception):
                    self.disarm(timeout_s=min(timeout_s, 0.5))
                self._stop_event.set()
                self._process.join(timeout=timeout_s)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=timeout_s)
            elif self._started:
                self._process.join(timeout=timeout_s)
        finally:
            self._closed = True

    def _send_control(self, operation: str, timeout_s: float) -> ArmControl:
        if not self.is_alive:
            raise RuntimeError(f"{self.side} arm worker is not running")
        self._control_sequence += 1
        control = ArmControl(self._control_sequence, operation)
        self._control_queue.put(control, timeout=timeout_s)
        return control

    def _wait_for(self, predicate: Callable[[ArmState], bool], timeout_s: float) -> ArmState:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for {self.side} arm worker")
            state = self._read_state(remaining)
            if predicate(state):
                return state
            if not self.is_alive:
                raise RuntimeError(f"{self.side} arm worker exited: {state.fault or 'no error reported'}")

    def _read_state(self, timeout_s: float) -> ArmState:
        try:
            state = self._state_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            if not self.is_alive:
                raise RuntimeError(f"{self.side} arm worker exited") from exc
            raise TimeoutError(f"Timed out reading {self.side} arm state") from exc

        while True:
            try:
                state = self._state_queue.get_nowait()
            except queue.Empty:
                break
        self._latest_state = state
        return state
