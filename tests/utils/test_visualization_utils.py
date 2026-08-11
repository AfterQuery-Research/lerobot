#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for the backend-agnostic visualization dispatch.

These exercise the display-mode routing/validation only; they need neither ``rerun`` nor
``foxglove`` installed since the unknown-mode branch raises before touching any backend. Backend
behavior is covered in ``test_rerun_visualization.py`` and ``test_foxglove_visualization.py``.
"""

import pytest

from lerobot.utils import visualization_utils as vu


def test_visualization_modes():
    assert vu.VISUALIZATION_MODES == ("rerun", "foxglove", "local")


def test_local_visualization_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(vu, "init_local_visualization", lambda **kwargs: calls.append(("init", kwargs)))
    monkeypatch.setattr(vu, "log_local_data", lambda **kwargs: calls.append(("log", kwargs)))
    monkeypatch.setattr(vu, "shutdown_local_visualization", lambda: calls.append(("shutdown", {})))

    observation = {"top": object()}
    action = {"joint.pos": 0.5}
    vu.init_visualization("local", task="Pick up the orange")
    vu.log_visualization_data("local", observation=observation, action=action)
    vu.shutdown_visualization("local")

    assert calls == [
        ("init", {"task": "Pick up the orange"}),
        ("log", {"observation": observation, "action": action, "compress_images": False}),
        ("shutdown", {}),
    ]


@pytest.mark.parametrize("func", ["init_visualization", "log_visualization_data", "shutdown_visualization"])
def test_dispatch_rejects_unknown_mode(func):
    with pytest.raises(ValueError, match="Unknown display_mode"):
        getattr(vu, func)("bogus")
