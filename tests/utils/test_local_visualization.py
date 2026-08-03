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

import time

import numpy as np

from lerobot.utils import local_visualization as lv


def _camera_observation():
    return {
        "right": np.full((360, 640, 3), (0, 0, 255), dtype=np.uint8),
        "top": np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8),
        "left": np.full((360, 640, 3), (0, 255, 0), dtype=np.uint8),
        "left_joint_0.pos": 0.0,
    }


def test_display_detection_rejects_ssh_even_with_forwarded_display():
    reason = lv._display_unavailable_reason(
        env={"SSH_CONNECTION": "client server", "DISPLAY": ":0"},
        platform="linux",
        which=lambda _: "/usr/bin/gst-launch-1.0",
    )

    assert reason == "the rollout was launched through SSH"


def test_display_detection_accepts_local_linux_desktop():
    reason = lv._display_unavailable_reason(
        env={"WAYLAND_DISPLAY": "wayland-0"},
        platform="linux",
        which=lambda _: "/usr/bin/gst-launch-1.0",
    )

    assert reason is None


def test_compose_preview_orders_three_views_and_adds_header():
    preview = lv._compose_preview(_camera_observation(), "Pick up the oranges")

    assert preview.shape == (lv._CANVAS_HEIGHT, lv._CANVAS_WIDTH, 3)
    assert np.array_equal(preview[lv._HEADER_HEIGHT + 200, 240], (255, 0, 0))
    assert np.array_equal(preview[lv._HEADER_HEIGHT + 180, 720], (0, 255, 0))
    assert np.array_equal(preview[lv._HEADER_HEIGHT + 180, 1200], (0, 0, 255))
    assert np.any(preview[: lv._HEADER_HEIGHT] != 12)


class _Sink:
    def __init__(self):
        self.bytes_written = 0
        self.closed = False

    def write(self, value):
        self.bytes_written += len(value)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _Process:
    def __init__(self):
        self.stdin = _Sink()
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout):
        del timeout
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_preview_renders_latest_frame_and_stops_subprocess():
    process = _Process()
    commands = []

    def process_factory(command, **kwargs):
        commands.append((command, kwargs))
        return process

    preview = lv._LocalPreview(
        task="Pick up the oranges",
        gst_launcher="/usr/bin/gst-launch-1.0",
        process_factory=process_factory,
    )
    preview.start()
    preview.submit(_camera_observation())
    deadline = time.monotonic() + 2
    while process.stdin.bytes_written == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    preview.close()

    assert process.stdin.bytes_written == lv._CANVAS_WIDTH * lv._CANVAS_HEIGHT * 3
    assert "rawvideoparse" in commands[0][0]
    assert process.terminated
    assert process.stdin.closed


def test_headless_initialization_is_a_noop(monkeypatch):
    lv.shutdown_local_visualization()
    monkeypatch.setattr(lv, "_display_unavailable_reason", lambda: "no graphical display is available")

    assert not lv.init_local_visualization(task="Pick up the oranges")
    assert lv._preview is None
