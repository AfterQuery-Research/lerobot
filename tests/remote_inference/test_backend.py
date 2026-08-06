# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.processor.rename_processor import RenameObservationsProcessorStep
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


class _SavedStateStats:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim

    def state_dict(self):
        return {f"{OBS_STATE}.q01": torch.zeros(self.state_dim)}


def _pi05_backend(rename_map: dict[str, str]) -> LeRobotPolicyBackend:
    joint_names = [
        *(f"left_joint_{index}.pos" for index in range(6)),
        "left_gripper.pos",
        *(f"right_joint_{index}.pos" for index in range(6)),
        "right_gripper.pos",
    ]
    backend = LeRobotPolicyBackend.__new__(LeRobotPolicyBackend)
    backend._config = LeRobotPolicyBackendConfig(pretrained_name_or_path="test/pi05-yam", device="cpu")
    backend._policy_config = SimpleNamespace(
        type="pi05",
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
            "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.left_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
            "observation.images.right_wrist_0_rgb": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        dataset_feature_names=None,
        action_feature_names=joint_names,
        n_action_steps=30,
        chunk_size=30,
        norm_tag=None,
    )
    backend._preprocessor = SimpleNamespace(
        steps=[RenameObservationsProcessorStep(rename_map=rename_map), _SavedStateStats(14)]
    )
    return backend


def test_pi05_manifest_uses_raw_processor_state_and_camera_contract():
    backend = _pi05_backend(
        {
            "observation.images.top": "observation.images.base_0_rgb",
            "observation.images.left": "observation.images.left_wrist_0_rgb",
            "observation.images.right": "observation.images.right_wrist_0_rgb",
        }
    )

    manifest = backend._build_manifest()

    assert len(manifest.state_features) == 14
    assert manifest.state_features == manifest.action_features
    assert manifest.action_dim == 14
    assert manifest.action_horizon == 30
    assert manifest.camera_keys == ("top", "left", "right")
    assert backend._configured_image_size("top") == (224, 224)


def test_pi05_manifest_omits_unmapped_optional_camera():
    backend = _pi05_backend(
        {
            "observation.images.left": "observation.images.left_wrist_0_rgb",
            "observation.images.right": "observation.images.right_wrist_0_rgb",
        }
    )

    manifest = backend._build_manifest()

    assert manifest.camera_keys == ("left", "right")
