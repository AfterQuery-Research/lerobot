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

import torch

from lerobot.remote_inference.backend import (
    LeRobotPolicyBackendConfig,
    _clamp_normalized_action_chunk,
)


def test_normalized_action_clamp_is_opt_in():
    config = LeRobotPolicyBackendConfig(pretrained_name_or_path="test/model")

    assert config.clamp_normalized_actions is False


def test_normalized_action_clamp_bounds_only_finite_outliers():
    chunk = torch.tensor([[[-1.2, -1.0, 0.5, 1.0, 1.5, float("nan")]]])

    clamped, count = _clamp_normalized_action_chunk(chunk)

    assert count == 2
    torch.testing.assert_close(
        clamped[..., :5],
        torch.tensor([[[-1.0, -1.0, 0.5, 1.0, 1.0]]]),
    )
    assert torch.isnan(clamped[..., 5]).all()
    assert chunk[0, 0, 0] == -1.2
    assert chunk[0, 0, 4] == 1.5
