from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from examples.umi_yam import validate_currentrel_onset_v3_training_run as validator

FULL_STAT_NAMES = (
    "count",
    "max",
    "mean",
    "min",
    "q01",
    "q10",
    "q50",
    "q90",
    "q99",
    "std",
)
VECTOR_FEATURE_DIMS = {
    "action": 20,
    "observation.state": 20,
    "umi.tcp_and_gripper": 16,
}
SCALAR_FEATURES = ("episode_index", "frame_index", "index", "task_index", "timestamp")
IMAGE_FEATURES = ("observation.images.umi1", "observation.images.umi2")


def _valid_normalizer_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for feature, dim in VECTOR_FEATURE_DIMS.items():
        for stat_name in FULL_STAT_NAMES:
            shape = (1,) if stat_name == "count" else (dim,)
            tensors[f"{feature}.{stat_name}"] = torch.zeros(shape, dtype=torch.float32)
    for feature in SCALAR_FEATURES:
        for stat_name in FULL_STAT_NAMES:
            tensors[f"{feature}.{stat_name}"] = torch.zeros(1, dtype=torch.float32)

    tensors["action.mask"] = torch.tensor([1.0] * 9 + [0.0] + [1.0] * 9 + [0.0])
    tensors["observation.state.mask"] = tensors["action.mask"].clone()
    for feature in IMAGE_FEATURES:
        tensors[f"{feature}.mean"] = torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32)
        tensors[f"{feature}.std"] = torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32)

    tensors["action.count"] = torch.tensor([validator.VALID_TRAIN_ACTION_ROWS], dtype=torch.float32)
    for feature in (*SCALAR_FEATURES, "observation.state", "umi.tcp_and_gripper"):
        tensors[f"{feature}.count"] = torch.tensor([validator.TRAIN_FRAMES], dtype=torch.float32)

    assert len(tensors) == 86
    return tensors


def _save_normalizer(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    save_file(tensors, path)


def _valid_dataset_stats(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, list[float]]]:
    return {
        feature: {stat_name: tensors[f"{feature}.{stat_name}"].tolist() for stat_name in FULL_STAT_NAMES}
        for feature in (*VECTOR_FEATURE_DIMS, *SCALAR_FEATURES)
    }


def _validate(path: Path, *, dataset_stats: dict | None = None) -> dict:
    tensors = _valid_normalizer_tensors()
    return validator._validate_normalizer(
        path,
        train_frames=validator.TRAIN_FRAMES,
        valid_train_action_rows=validator.VALID_TRAIN_ACTION_ROWS,
        dataset_stats=dataset_stats or _valid_dataset_stats(tensors),
    )


def _valid_policy_config(model_root: Path) -> dict:
    return {
        "type": "molmoact2",
        "chunk_size": 24,
        "n_action_steps": 24,
        "n_obs_steps": 1,
        "action_mode": "continuous",
        "inference_action_mode": "continuous",
        "num_flow_timesteps": 8,
        "expected_max_action_dim": 32,
        "mask_action_dim_padding": True,
        "normalize_gripper": False,
        "model_dtype": "bfloat16",
        "llm_residual_dropout": 0.1,
        "train_mode_vlm": "lora",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "freeze_embedding": True,
        "gradient_checkpointing": True,
        "optimizer_lr": 5e-5,
        "optimizer_vit_lr": 5e-6,
        "optimizer_connector_lr": 5e-6,
        "optimizer_action_expert_lr": 5e-5,
        "scheduler_warmup_steps": 600,
        "scheduler_decay_steps": 12_000,
        "scheduler_decay_lr": 1e-6,
        "checkpoint_path": str(model_root),
        "checkpoint_revision": validator.MODEL_REVISION,
        "image_keys": ["observation.images.umi1", "observation.images.umi2"],
        "input_features": {"observation.state": {"shape": [validator.STATE_DIM]}},
        "output_features": {"action": {"shape": [validator.STATE_DIM]}},
    }


def test_policy_config_binds_frozen_llm_residual_dropout(tmp_path: Path) -> None:
    policy = _valid_policy_config(tmp_path)
    validator._assert_policy_config(policy, tmp_path)

    policy["llm_residual_dropout"] = 0.0
    with pytest.raises(ValueError, match="unexpected policy llm_residual_dropout"):
        validator._assert_policy_config(policy, tmp_path)


@pytest.mark.parametrize(
    "recipe_name",
    ["train_currentrel_onset_v3_smoke.sbatch", "train_currentrel_onset_v3_full.sbatch"],
)
def test_onset_v3_recipe_sets_frozen_llm_residual_dropout(recipe_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    recipe = (repo_root / "examples/umi_yam" / recipe_name).read_text(encoding="utf-8")
    assert recipe.count("--policy.llm_residual_dropout=0.1") == 1


def test_normalizer_accepts_exact_86_tensor_imagenet_schema(tmp_path: Path) -> None:
    path = tmp_path / "normalizer.safetensors"
    _save_normalizer(path, _valid_normalizer_tensors())

    report = _validate(path)

    assert report["tensor_count"] == 86
    assert report["action_count"] == validator.VALID_TRAIN_ACTION_ROWS
    assert report["state_count"] == validator.TRAIN_FRAMES
    assert report["helper_count"] == validator.TRAIN_FRAMES
    assert report["matches_dataset_stats"] is True


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_normalizer_rejects_missing_or_extra_tensor(tmp_path: Path, mutation: str) -> None:
    tensors = _valid_normalizer_tensors()
    if mutation == "missing":
        del tensors["timestamp.q99"]
    else:
        tensors["unexpected.mean"] = torch.zeros(1, dtype=torch.float32)
    path = tmp_path / f"normalizer-{mutation}.safetensors"
    _save_normalizer(path, tensors)

    with pytest.raises(ValueError, match="normalization tensor schema mismatch"):
        _validate(path)


@pytest.mark.parametrize(
    ("key", "wrong_tensor", "error"),
    [
        ("action.q99", torch.zeros(19), "unexpected normalization tensor shape"),
        (
            "observation.state.std",
            torch.zeros(20, dtype=torch.float64),
            "unexpected normalization tensor dtype",
        ),
        (
            "observation.images.umi1.mean",
            torch.zeros((3, 1, 1)),
            "unexpected ImageNet mean tensor",
        ),
    ],
)
def test_normalizer_rejects_wrong_tensor(
    tmp_path: Path,
    key: str,
    wrong_tensor: torch.Tensor,
    error: str,
) -> None:
    tensors = _valid_normalizer_tensors()
    tensors[key] = wrong_tensor
    path = tmp_path / "normalizer-wrong.safetensors"
    _save_normalizer(path, tensors)

    with pytest.raises(ValueError, match=error):
        _validate(path)


def test_normalizer_rejects_value_different_from_dataset_stats(tmp_path: Path) -> None:
    tensors = _valid_normalizer_tensors()
    path = tmp_path / "normalizer-wrong-value.safetensors"
    _save_normalizer(path, tensors)
    dataset_stats = _valid_dataset_stats(tensors)
    dataset_stats["observation.state"]["q01"][0] = 1.0

    with pytest.raises(ValueError, match="differ from dataset stats"):
        _validate(path, dataset_stats=dataset_stats)
