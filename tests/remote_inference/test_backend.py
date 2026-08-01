# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from lerobot.configs import FeatureType
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.remote_inference.backend import LeRobotPolicyBackend
from lerobot.utils.constants import ACTION, OBS_STATE


def test_original_molmoact2_schema_is_derived_from_norm_metadata(monkeypatch):
    state_names = ["left_joint_0.pos", "left_gripper.pos"]
    action_names = ["left_joint_0.pos", "left_gripper.pos"]
    metadata = {
        "state_stats": {"names": state_names},
        "action_stats": {"names": action_names},
        "camera_keys": ["observation.images.top", "observation.images.left"],
        "normalize_gripper": False,
    }
    stats = {OBS_STATE: {"q01": [0.0, 0.0]}, ACTION: {"q01": [0.0, 0.0]}}

    from lerobot.policies.molmoact2 import processor_molmoact2

    monkeypatch.setattr(
        processor_molmoact2,
        "_load_hf_norm_stats_for_tag",
        lambda *args, **kwargs: (stats, metadata),
    )
    config = MolmoAct2Config(checkpoint_path="test/model", norm_tag="test")

    resolved_stats = LeRobotPolicyBackend._configure_original_molmoact2(config)

    assert resolved_stats is stats
    assert config.dataset_feature_names == {OBS_STATE: state_names, ACTION: action_names}
    assert config.input_features[OBS_STATE].shape == (2,)
    assert config.input_features[OBS_STATE].type is FeatureType.STATE
    assert config.output_features[ACTION].shape == (2,)
    assert config.output_features[ACTION].type is FeatureType.ACTION
    assert config.image_keys == ["observation.images.top", "observation.images.left"]
    assert all(config.input_features[key].type is FeatureType.VISUAL for key in config.image_keys)
