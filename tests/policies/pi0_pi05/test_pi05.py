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

"""Test script to verify PI0.5 (pi05) support in PI0 policy"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.policies.factory import make_policy_config, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05 import (  # noqa: E402
    PI05Config,
    PI05Policy,
    make_pi05_pre_post_processors,  # noqa: E402
)
from lerobot.scripts.lerobot_train import _clear_runtime_processor_load_paths  # noqa: E402
from lerobot.utils.random_utils import set_seed
from tests.utils import require_cuda, require_hf_token  # noqa: E402


def test_pretrained_processor_uses_local_pin_but_saves_portable_configs(tmp_path):
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": "unpinned/saved-tokenizer"},
                    }
                ],
            }
        )
    )
    (tmp_path / "policy_postprocessor.json").write_text(
        json.dumps({"name": "policy_postprocessor", "steps": []})
    )
    local_snapshot = "/cache/models--google--paligemma-3b-pt-224/snapshots/pinned"
    config = PI05Config(tokenizer_load_path=local_snapshot)
    training_config = TrainPipelineConfig(dataset=DatasetConfig(repo_id="test/dataset"), policy=config)
    policy = SimpleNamespace(config=config)

    with patch("lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained") as load:
        preprocessor, _ = make_pre_post_processors(config, pretrained_path=str(tmp_path))

    load.assert_called_once_with(local_snapshot)
    tokenizer_step = preprocessor.steps[0]
    assert tokenizer_step.tokenizer_name == "google/paligemma-3b-pt-224"
    assert tokenizer_step.tokenizer_revision == "35e4f46485b4d07967e7e9935bc3786aad50687c"
    assert tokenizer_step.tokenizer_load_path == local_snapshot
    assert tokenizer_step.get_config() == {
        "max_length": 512,
        "task_key": "task",
        "padding_side": "right",
        "padding": "max_length",
        "truncation": True,
        "tokenizer_name": "google/paligemma-3b-pt-224",
        "tokenizer_revision": "35e4f46485b4d07967e7e9935bc3786aad50687c",
    }

    preprocessor.save_pretrained(tmp_path)
    _clear_runtime_processor_load_paths(config, policy.config)
    config._save_pretrained(tmp_path)
    training_config._save_pretrained(tmp_path)

    policy_payload = json.loads((tmp_path / "config.json").read_text())
    train_payload = json.loads((tmp_path / "train_config.json").read_text())
    processor_payload = json.loads((tmp_path / "policy_preprocessor.json").read_text())
    assert policy_payload["tokenizer_load_path"] is None
    assert train_payload["policy"]["tokenizer_load_path"] is None
    assert "tokenizer_load_path" not in processor_payload["steps"][0]["config"]

    resume_config = TrainPipelineConfig.from_pretrained(
        tmp_path, cli_args=["--policy.tokenizer_load_path=/cache/resolved-on-resume"]
    )
    assert resume_config.policy.tokenizer_load_path == "/cache/resolved-on-resume"

    portable_config = PreTrainedConfig.from_pretrained(tmp_path)
    with patch("lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained") as load:
        portable_preprocessor, _ = make_pre_post_processors(portable_config, pretrained_path=str(tmp_path))

    load.assert_called_once_with(
        "google/paligemma-3b-pt-224",
        revision="35e4f46485b4d07967e7e9935bc3786aad50687c",
    )
    assert portable_preprocessor.steps[0].tokenizer_load_path is None


def test_preprocess_images_preserves_configured_slots_when_leading_camera_is_missing():
    policy = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(policy)
    policy.register_parameter("device_anchor", torch.nn.Parameter(torch.zeros(())))
    policy.config = SimpleNamespace(
        image_features={
            "observation.images.base": object(),
            "observation.images.left": object(),
            "observation.images.right": object(),
        },
        image_resolution=(2, 2),
    )
    batch = {
        "observation.images.left": torch.full((1, 3, 2, 2), 0.25),
        "observation.images.right": torch.full((1, 3, 2, 2), 0.75),
    }

    images, masks = policy._preprocess_images(batch)

    assert len(images) == 3
    torch.testing.assert_close(images[0], torch.full_like(images[0], -1.0))
    torch.testing.assert_close(images[1], torch.full_like(images[1], -0.5))
    torch.testing.assert_close(images[2], torch.full_like(images[2], 0.5))
    assert [mask.tolist() for mask in masks] == [[False], [True], [True]]


@require_cuda
@require_hf_token
def test_policy_instantiation():
    # Create config
    set_seed(42)
    config = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    # Set up input_features and output_features in the config
    from lerobot.configs.types import FeatureType, PolicyFeature

    config.input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(14,),
        ),
        "observation.images.base_0_rgb": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224),
        ),
    }

    config.output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(7,),
        ),
    }

    assert config.tokenizer_max_length == 200, (
        f"Expected tokenizer_max_length=200 for pi05, got {config.tokenizer_max_length}"
    )

    # Create dummy dataset stats
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "min": torch.zeros(14),
            "max": torch.ones(14),
            "q01": torch.zeros(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "min": torch.zeros(7),
            "max": torch.ones(7),
            "q01": torch.zeros(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }

    # Instantiate policy
    policy = PI05Policy(config)
    # Test forward pass with dummy data
    batch_size = 1
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    device = config.device
    batch = {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, config.chunk_size, 7, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),  # Use rand for [0,1] range
        "task": ["Pick up the object"] * batch_size,
    }
    batch = preprocessor(batch)
    try:
        loss, loss_dict = policy.forward(batch)
        print(f"Forward pass successful. Loss: {loss_dict['loss']:.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise
    try:
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
            print(f"Action: {action}")
        print(f"Action prediction successful. Action shape: {action.shape}")
    except Exception as e:
        print(f"Action prediction failed: {e}")
        raise

    # Verify pi05 model components exist
    # Check that time_mlp layers exist (for AdaRMS conditioning)
    assert hasattr(policy.model, "time_mlp_in"), "Missing time_mlp_in layer for pi05"
    assert hasattr(policy.model, "time_mlp_out"), "Missing time_mlp_out layer for pi05"

    # Check that action_time_mlp layers don't exist (pi0 only)
    assert not hasattr(policy.model, "action_time_mlp_in"), "action_time_mlp_in should not exist in pi05 mode"
    assert not hasattr(policy.model, "action_time_mlp_out"), (
        "action_time_mlp_out should not exist in pi05 mode"
    )

    # Check that state_proj doesn't exist in pi05 mode
    assert not hasattr(policy.model, "state_proj"), "state_proj should not exist in pi05 mode"

    # Check AdaRMS configuration in the underlying model
    adarms_config = policy.model.paligemma_with_expert.paligemma.config.text_config.use_adarms
    assert adarms_config == False, f"PaliGemma should not use AdaRMS, got {adarms_config}"  # noqa: E712

    adarms_expert_config = policy.model.paligemma_with_expert.gemma_expert.config.use_adarms
    assert adarms_expert_config == True, (  # noqa: E712
        f"Action expert should use AdaRMS in pi05, got {adarms_expert_config}"
    )


@require_cuda
@require_hf_token
def test_config_creation():
    """Test policy config creation through factory."""
    try:
        config = make_policy_config(
            policy_type="pi0",
            max_action_dim=7,
            max_state_dim=14,
        )
        print("Config created successfully through factory")
        print(f"  Config type: {type(config).__name__}")
        print(f"  PaliGemma variant: {config.paligemma_variant}")
        print(f"  Action expert variant: {config.action_expert_variant}")
    except Exception as e:
        print(f"Config creation failed: {e}")
        raise
