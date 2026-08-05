# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

from lerobot.configs import FeatureType
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.remote_inference import backend as backend_module
from lerobot.remote_inference.backend import LeRobotPolicyBackend, LeRobotPolicyBackendConfig
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


def test_saved_policy_loads_from_backend_path(monkeypatch, tmp_path):
    loaded = {}

    class FakePolicy:
        @classmethod
        def from_pretrained(cls, pretrained_name_or_path, **kwargs):
            loaded["path"] = pretrained_name_or_path
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    policy_config = SimpleNamespace(type="fake", pretrained_path=None, pretrained_revision=None)

    monkeypatch.setattr(LeRobotPolicyBackend, "_load_policy_config", lambda self, config: policy_config)
    monkeypatch.setattr(LeRobotPolicyBackend, "_build_manifest", lambda self: object())
    monkeypatch.setattr(backend_module, "get_policy_class", lambda policy_type: FakePolicy)
    monkeypatch.setattr(
        backend_module, "make_pre_post_processors", lambda *args, **kwargs: (object(), object())
    )

    checkpoint = tmp_path / "pretrained_model"
    LeRobotPolicyBackend(LeRobotPolicyBackendConfig(pretrained_name_or_path=str(checkpoint), device="cpu"))

    assert loaded["path"] == str(checkpoint)
