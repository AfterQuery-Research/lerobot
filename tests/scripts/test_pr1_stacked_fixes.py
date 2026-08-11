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

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs import parser  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.rollout.context import _local_preprocessor_overrides  # noqa: E402
from lerobot.scripts.lerobot_train import _drop_inapplicable_normalizer_overrides  # noqa: E402


def test_molmoact2_drops_only_inapplicable_normalizer_overrides():
    preprocessor_overrides = {
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {"stats": {"action": {}}},
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {"stats": {"action": {}}},
        "absolute_actions_processor": {"enabled": True},
    }

    _drop_inapplicable_normalizer_overrides("molmoact2", preprocessor_overrides, postprocessor_overrides)

    assert preprocessor_overrides == {"device_processor": {"device": "cuda"}}
    assert postprocessor_overrides == {"absolute_actions_processor": {"enabled": True}}


def test_non_molmo_pipeline_keeps_generic_normalizer_overrides():
    preprocessor_overrides = {"normalizer_processor": {"stats": {}}}
    postprocessor_overrides = {"unnormalizer_processor": {"stats": {}}}

    _drop_inapplicable_normalizer_overrides("pi05", preprocessor_overrides, postprocessor_overrides)

    assert "normalizer_processor" in preprocessor_overrides
    assert "unnormalizer_processor" in postprocessor_overrides


def test_empty_cli_rename_map_does_not_override_checkpoint_map():
    assert _local_preprocessor_overrides("cuda", {}) == {"device_processor": {"device": "cuda"}}


def test_explicit_cli_rename_map_is_forwarded():
    rename_map = {"observation.images.left": "observation.images.left_wrist_0_rgb"}

    assert _local_preprocessor_overrides("cuda", rename_map) == {
        "device_processor": {"device": "cuda"},
        "rename_observations_processor": {"rename_map": rename_map},
    }


def test_resume_rejects_a_second_policy_path(monkeypatch):
    cfg = object.__new__(TrainPipelineConfig)
    cfg.resume = True

    def get_path_arg(name: str):
        return "org/base-policy" if name == "policy" else None

    monkeypatch.setattr(parser, "get_path_arg", get_path_arg)

    with pytest.raises(ValueError, match="--policy.path cannot be combined with --resume=true"):
        cfg._resolve_pretrained_from_cli()
