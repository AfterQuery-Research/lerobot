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

"""Ubuntu-local camera preview for robot rollouts."""

from __future__ import annotations

import atexit
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

import cv2
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation

logger = logging.getLogger(__name__)

_CAMERA_ORDER = ("top", "left", "right")
_PANEL_WIDTH = 480
_PANEL_HEIGHT = 360
_HEADER_HEIGHT = 88
_CANVAS_WIDTH = _PANEL_WIDTH * len(_CAMERA_ORDER)
_CANVAS_HEIGHT = _HEADER_HEIGHT + _PANEL_HEIGHT
_PREVIEW_FPS = 15
_STOP = object()


def _display_unavailable_reason(
    *,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform != "linux":
        return "local visualization is supported only on Linux"
    if any(env.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return "the rollout was launched through SSH"
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return "no graphical display is available"
    if which("gst-launch-1.0") is None:
        return "gst-launch-1.0 is not installed"
    return None


def _as_rgb_uint8(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected an image with 1, 3, or 4 channels, got shape {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        if image.size and float(np.nanmax(image)) <= 1.0 and float(np.nanmin(image)) >= 0.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _is_image_value(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) in (2, 3, 4)


def _select_camera_views(observation: RobotObservation) -> list[tuple[str, Any | None]]:
    images: dict[str, Any] = {}
    for key, value in observation.items():
        name = str(key).rsplit(".", 1)[-1]
        if name.endswith("_depth") or not _is_image_value(value):
            continue
        images.setdefault(name, value)

    selected: list[tuple[str, Any | None]] = []
    used: set[str] = set()
    for name in _CAMERA_ORDER:
        if name in images:
            selected.append((name, images[name]))
            used.add(name)
        else:
            selected.append((name, None))

    extras = ((name, value) for name, value in images.items() if name not in used)
    for index, (_, value) in enumerate(selected):
        if value is not None:
            continue
        try:
            extra_name, extra_value = next(extras)
        except StopIteration:
            break
        selected[index] = (extra_name, extra_value)
    return selected


def _fit_panel(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(_PANEL_WIDTH / width, _PANEL_HEIGHT / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    panel = np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), 18, dtype=np.uint8)
    x = (_PANEL_WIDTH - resized_width) // 2
    y = (_PANEL_HEIGHT - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def _truncate_text(text: str, max_width: int, *, scale: float, thickness: int) -> str:
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text
    suffix = "..."
    while (
        text and cv2.getTextSize(text + suffix, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_width
    ):
        text = text[:-1]
    return text + suffix


def _compose_preview(observation: RobotObservation, task: str) -> np.ndarray:
    canvas = np.full((_CANVAS_HEIGHT, _CANVAS_WIDTH, 3), 12, dtype=np.uint8)
    cv2.putText(
        canvas,
        "LeRobot rollout",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    task_text = _truncate_text(
        f"Task: {task or '(not specified)'}", _CANVAS_WIDTH - 40, scale=0.8, thickness=2
    )
    cv2.putText(
        canvas,
        task_text,
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (102, 224, 180),
        2,
        cv2.LINE_AA,
    )

    for index, (name, value) in enumerate(_select_camera_views(observation)):
        x = index * _PANEL_WIDTH
        if value is None:
            panel = np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), 18, dtype=np.uint8)
            cv2.putText(
                panel,
                f"Waiting for {name} camera",
                (64, _PANEL_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
        else:
            panel = _fit_panel(_as_rgb_uint8(value))
        cv2.rectangle(panel, (0, 0), (116, 36), (0, 0, 0), thickness=-1)
        cv2.putText(
            panel,
            name.title(),
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        canvas[_HEADER_HEIGHT:, x : x + _PANEL_WIDTH] = panel
    return np.ascontiguousarray(canvas)


class _LocalPreview:
    def __init__(
        self,
        *,
        task: str,
        gst_launcher: str,
        process_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self._task = task
        self._gst_launcher = gst_launcher
        self._process_factory = process_factory
        self._process: Any | None = None
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._last_submit = 0.0
        self._failure_logged = False

    def start(self) -> None:
        command = [
            self._gst_launcher,
            "-q",
            "fdsrc",
            "fd=0",
            "!",
            "rawvideoparse",
            f"width={_CANVAS_WIDTH}",
            f"height={_CANVAS_HEIGHT}",
            "format=rgb",
            f"framerate={_PREVIEW_FPS}/1",
            "!",
            "videoconvert",
            "!",
            "autovideosink",
            "sync=false",
        ]
        self._process = self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self._process.stdin is None:
            raise RuntimeError("GStreamer preview did not expose an input pipe")
        self._thread = threading.Thread(target=self._render_loop, name="LocalRolloutPreview", daemon=True)
        self._thread.start()

    def submit(self, observation: RobotObservation) -> None:
        if self._stop_event.is_set() or self._process is None:
            return
        returncode = self._process.poll()
        if returncode is not None:
            if not self._failure_logged:
                logger.warning(
                    "Local visualization exited with status %s; rollout will continue without it", returncode
                )
                self._failure_logged = True
            self._stop_event.set()
            return
        now = time.monotonic()
        if now - self._last_submit < 1.0 / _PREVIEW_FPS:
            return
        self._last_submit = now
        views = dict(_select_camera_views(observation))
        if not any(value is not None for value in views.values()):
            return
        try:
            self._queue.put_nowait(views)
        except queue.Full:
            with suppress(queue.Empty):
                self._queue.get_nowait()
            self._queue.put_nowait(views)

    def _render_loop(self) -> None:
        assert self._process is not None and self._process.stdin is not None
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                frame = _compose_preview(item, self._task)
                self._process.stdin.write(frame.tobytes())
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError, cv2.error) as exc:
                if not self._failure_logged and not self._stop_event.is_set():
                    logger.warning("Local visualization stopped; rollout will continue without it: %s", exc)
                    self._failure_logged = True
                self._stop_event.set()
                return

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            with suppress(queue.Empty):
                self._queue.get_nowait()
            self._queue.put_nowait(_STOP)

        if self._thread is not None:
            self._thread.join(timeout=1.0)
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if process is not None and process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        self._thread = None
        self._process = None


_preview: _LocalPreview | None = None
_atexit_registered = False


def init_local_visualization(*, task: str = "") -> bool:
    """Start a local preview when launched from the Ubuntu desktop."""
    global _atexit_registered, _preview
    if _preview is not None:
        return True
    reason = _display_unavailable_reason()
    if reason is not None:
        logger.info("Local visualization disabled: %s; rollout will continue headlessly", reason)
        return False

    gst_launcher = shutil.which("gst-launch-1.0")
    assert gst_launcher is not None
    preview = _LocalPreview(task=task, gst_launcher=gst_launcher)
    try:
        preview.start()
    except (OSError, RuntimeError) as exc:
        logger.warning("Could not start local visualization; rollout will continue headlessly: %s", exc)
        preview.close()
        return False
    _preview = preview
    if not _atexit_registered:
        atexit.register(shutdown_local_visualization)
        _atexit_registered = True
    logger.info("Local visualization started with top, left, and right camera views")
    return True


def log_local_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    del action, compress_images
    if _preview is not None and observation is not None:
        _preview.submit(observation)


def shutdown_local_visualization() -> None:
    global _preview
    if _preview is None:
        return
    preview, _preview = _preview, None
    preview.close()
    logger.info("Local visualization stopped")
