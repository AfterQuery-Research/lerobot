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

"""Policy action inspection without action dispatch."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import TextIO

from lerobot.utils.robot_utils import precise_sleep

from ..configs import ActionProbeStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy, send_next_action

logger = logging.getLogger(__name__)


class ActionProbeStrategy(RolloutStrategy):
    """Persist policy outputs without executing them."""

    config: ActionProbeStrategyConfig

    def __init__(self, config: ActionProbeStrategyConfig) -> None:
        super().__init__(config)
        self._action_file: TextIO | None = None
        self._action_sequence = 0
        self._last_console_log_ns = 0

    def setup(self, ctx: RolloutContext) -> None:
        robot = ctx.hardware.robot_wrapper.inner
        arm = getattr(robot, "arm", None)
        disarm = getattr(robot, "disarm", None)
        if self.config.reset_robot and callable(arm) and not callable(disarm):
            raise RuntimeError("Reset-enabled action probe requires disarm() when the robot exposes arm()")

        path = Path(self.config.action_log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._action_file = path.open("w", encoding="utf-8", buffering=1)
        logger.warning("ACTION PROBE MODE: policy actions will be logged but never sent to the robot")
        try:
            self._init_engine(ctx, prepare_robot=self.config.reset_robot)
            if callable(disarm):
                disarm()
            if self.config.reset_robot:
                logger.info("Robot disarmed after supervised reset; beginning action probe")
            else:
                logger.info("Robot remains in safe idle; beginning observation-only action probe")
        except BaseException:
            self._close_action_file()
            raise

    def run(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("Action probe control loop started")
        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()
            if cfg.duration > 0 and (loop_start - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            observation = robot.get_observation()
            processed_observation = self._process_observation_and_notify(ctx.processors, observation)
            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action = send_next_action(
                processed_observation,
                observation,
                ctx,
                interpolator,
                execute=False,
            )
            if action is not None:
                self._record_action(action, observation, engine)
            self._log_telemetry(processed_observation, action, ctx.runtime)

            elapsed = time.perf_counter() - loop_start
            if (sleep_time := control_interval - elapsed) > 0:
                precise_sleep(sleep_time)
        engine.pause()

    def teardown(self, ctx: RolloutContext) -> None:
        try:
            self._teardown_hardware(ctx.hardware, return_to_initial_position=False)
        finally:
            self._close_action_file()
        logger.info("Action probe teardown complete (%d actions discarded)", self._action_sequence)

    def _record_action(self, action: dict, observation: dict, engine) -> None:
        values = [float(value) for value in action.values()]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Policy returned a non-finite action during action probe")

        measured = {
            key: float(observation[key])
            for key in action
            if key in observation and isinstance(observation[key], (int, float))
        }
        tracking_delta = {key: float(value) - measured[key] for key, value in action.items() if key in measured}
        record = {
            "sequence": self._action_sequence,
            "monotonic_ns": time.monotonic_ns(),
            "action": {key: float(value) for key, value in action.items()},
            "measured_position": measured,
            "action_minus_measured": tracking_delta,
            "action_queue_depth": int(getattr(engine, "action_queue_depth", 0)),
            "executed": False,
        }
        if self._action_file is not None:
            self._action_file.write(json.dumps(record, separators=(",", ":")) + "\n")

        now_ns = record["monotonic_ns"]
        interval_ns = int(self.config.console_log_interval_s * 1e9)
        if interval_ns == 0 or now_ns - self._last_console_log_ns >= interval_ns:
            max_delta = max((abs(value) for value in tracking_delta.values()), default=0.0)
            logger.info(
                "Discarded policy action seq=%d range=[%.4f, %.4f] max_delta=%.4f queue=%d",
                self._action_sequence,
                min(values),
                max(values),
                max_delta,
                record["action_queue_depth"],
            )
            self._last_console_log_ns = now_ns
        self._action_sequence += 1

    def _close_action_file(self) -> None:
        if self._action_file is not None:
            self._action_file.close()
            self._action_file = None
