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

import multiprocessing as mp
import queue
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Protocol

import numpy as np

from .config_bi_yam import YAMArmConfig


class ArmBackend(Protocol):
    def num_dofs(self) -> int: ...

    def get_joint_pos(self) -> np.ndarray: ...

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


def make_i2rt_backend(config: YAMArmConfig) -> ArmBackend:
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType
    except ImportError as exc:
        raise ImportError("i2rt is required to use the bi_yam_follower robot") from exc

    return get_yam_robot(
        channel=config.channel,
        arm_type=ArmType.from_string_name(config.arm_type),
        gripper_type=GripperType.from_string_name(config.gripper_type),
        zero_gravity_mode=True,
        sim=config.sim,
        enable_auto_recovery=config.enable_auto_recovery,
    )


def _enter_safe_idle(backend: ArmBackend) -> None:
    for method_name in ("enter_gravity_comp_idle", "enable_gravity_comp", "zero_torque_mode"):
        method = getattr(backend, method_name, None)
        if method is not None:
            method()
            return
    raise RuntimeError("i2rt backend does not expose a safe idle operation")


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
