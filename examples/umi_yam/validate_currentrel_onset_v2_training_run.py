#!/usr/bin/env python

"""Fail-closed smoke/full-run checks for the onset-aligned UMI training recipe.

This validator intentionally does not decide deployment eligibility.  It verifies
that training used the frozen onset-v2 artifact and exact predeclared optimizer,
split, augmentation, checkpoint, and evaluation settings.  The independent
model->resolver->IK gate remains responsible for deployment eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors import safe_open

SCHEMA_ID = "dual-lidar-umi-currentrel-r6d-onset-v2"
DATASET_REPO_ID = "brandonyang/dual-lidar-umi-currentrel-r6d-onset-v2"
DATASET_BASENAME = "dual-lidar-umi-currentrel-r6d-onset-v2"
SEMANTICS_REL = Path("meta/umi_current_relative_r6d_onset_v2.json")
ARTIFACT_MANIFEST_REL = Path("meta/artifact_manifest.sha256")
ARTIFACT_METADATA_REL = Path("meta/artifact_manifest.json")
MODEL_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"

TRAIN_EPISODES = list(range(52))
VALIDATION_EPISODES = [52, 53]
TRAIN_FRAMES = 47_618
VALIDATION_FRAMES = 1_884
TOTAL_FRAMES = 49_502
CHECKPOINT_STEPS = list(range(500, 4_001, 500))
TRAIN_LOG_STEPS = list(range(10, 4_001, 10))
ACTION_HORIZON = 24
STATE_DIM = 20
PER_RANK_BATCH_SIZE = 4
WORLD_SIZE = 8
MODEL_TENSOR_COUNT = 1_899
MODEL_SAFETENSORS_BYTES = 11_483_634_784
OPTIMIZER_TENSOR_COUNT = 3_558
OPTIMIZER_SAFETENSORS_BYTES = 3_487_466_024

COLOR_JITTER = {
    "brightness": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"brightness": [0.8, 1.2]},
    },
    "contrast": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"contrast": [0.8, 1.2]},
    },
    "saturation": {
        "weight": 1.0,
        "type": "ColorJitter",
        "kwargs": {"saturation": [0.8, 1.2]},
    },
}

SELECTION_POLICY = {
    "candidate_checkpoint_steps": CHECKPOINT_STEPS,
    "eligibility_source": "independent frozen model->resolver->strict-IK multiseed gate",
    "selection_rule": (
        "Consider only checkpoints marked deployment/advancement eligible by the complete "
        "predeclared gate. Among eligible checkpoints select the lowest full onset-v2 holdout "
        "loss reported by this run; ties at the logged precision select the earlier step."
    ),
    "no_seed_or_checkpoint_cherry_picking": True,
    "no_eligible_checkpoint_behavior": "do not advance any checkpoint to hardware",
}

TRAIN_METRIC_RE = re.compile(
    r"\bstep:(\d+)\b.*?\bloss:([^\s]+).*?\bgrdn:([^\s]+).*?\baction_flow_loss:([^\s]+)"
)
EVAL_METRIC_RE = re.compile(r"\bstep (\d+): eval_loss=([^\s]+)")
FATAL_TEXT = (
    "traceback (most recent call last)",
    "cuda out of memory",
    "nccl error",
    "childfailederror",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _validate_artifact_checksum_manifest(root: Path, expected_manifest_sha: str) -> dict[str, str]:
    """Verify every artifact byte named by the frozen checksum manifest."""

    checksum_path = root / ARTIFACT_MANIFEST_REL
    if _sha256(checksum_path) != expected_manifest_sha:
        raise ValueError("dataset artifact checksum-manifest SHA-256 mismatch")

    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest_text, separator, relative = line.partition("  ")
        if not separator:
            raise ValueError(f"malformed artifact checksum line: {line!r}")
        digest = _require_sha256(digest_text, name=f"artifact checksum for {relative!r}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe artifact checksum path: {relative!r}")
        if relative in entries:
            raise ValueError(f"duplicate artifact checksum path: {relative!r}")
        artifact_path = root / relative_path
        try:
            artifact_path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"artifact checksum path is missing or escapes the dataset: {relative!r}"
            ) from exc
        if _sha256(artifact_path) != digest:
            raise ValueError(f"artifact checksum mismatch for {relative!r}")
        entries[relative] = digest
    if not entries:
        raise ValueError("artifact checksum manifest is empty")
    return entries


def _finite(value: str, *, name: str) -> float:
    try:
        parsed = float(value.rstrip(","))
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {name}: {value!r}")
    return parsed


def _dataset_frame_counts(root: Path) -> tuple[int, int]:
    def count(indices: list[int]) -> int:
        return sum(
            pq.ParquetFile(root / "data/chunk-000" / f"file-{index:03d}.parquet").metadata.num_rows
            for index in indices
        )

    return count(TRAIN_EPISODES), count(VALIDATION_EPISODES)


def validate_dataset_identity(root: Path, manifest_sha256: str) -> dict[str, Any]:
    root = root.resolve(strict=True)
    expected_manifest_sha = _require_sha256(manifest_sha256, name="dataset manifest SHA-256")
    if root.name != DATASET_BASENAME:
        raise ValueError(f"dataset basename must be {DATASET_BASENAME!r}, got {root.name!r}")

    semantics_path = root / SEMANTICS_REL
    checksum_entries = _validate_artifact_checksum_manifest(root, expected_manifest_sha)

    semantics = _load_json(semantics_path)
    split = _load_json(root / "meta/split_manifest.json")
    info = _load_json(root / "meta/info.json")
    stats = _load_json(root / "meta/stats.json")
    artifact_manifest = _load_json(root / ARTIFACT_METADATA_REL)
    declared_hashes = artifact_manifest.get("generated_file_sha256")
    if not isinstance(declared_hashes, dict):
        raise ValueError("artifact manifest generated_file_sha256 is missing or malformed")
    if set(checksum_entries) != {*declared_hashes, ARTIFACT_METADATA_REL.as_posix()}:
        raise ValueError("artifact JSON and checksum manifest path sets disagree")
    for relative, digest in declared_hashes.items():
        if checksum_entries.get(relative) != digest:
            raise ValueError(f"artifact JSON and checksum manifest disagree for {relative!r}")

    if semantics.get("schema_id") != SCHEMA_ID:
        raise ValueError(f"unexpected onset-v2 schema: {semantics.get('schema_id')!r}")
    if semantics.get("dataset_name") != DATASET_BASENAME:
        raise ValueError(f"unexpected onset-v2 dataset name: {semantics.get('dataset_name')!r}")
    if semantics.get("repository") != DATASET_REPO_ID:
        raise ValueError(f"unexpected onset-v2 repo id: {semantics.get('repository')!r}")
    if int(semantics.get("fps", -1)) != 30:
        raise ValueError("onset-v2 semantics must declare fps=30")
    horizon = semantics.get("action_horizon", semantics.get("horizon", -1))
    if int(horizon) != ACTION_HORIZON:
        raise ValueError("onset-v2 semantics must declare a 24-row action horizon")
    if split.get("train_episodes") != TRAIN_EPISODES:
        raise ValueError("onset-v2 split must keep train episodes 0..51")
    if split.get("validation_episodes") != VALIDATION_EPISODES:
        raise ValueError("onset-v2 split must keep holdout episodes 52 and 53")
    if set(TRAIN_EPISODES) & set(VALIDATION_EPISODES):
        raise AssertionError("internal train/validation split overlap")
    if info.get("total_episodes") != 54:
        raise ValueError("onset-v2 artifact must retain all 54 episodes")
    for feature, width in (
        ("observation.state", STATE_DIM),
        ("action", STATE_DIM),
        ("umi.tcp_and_gripper", 16),
    ):
        shape = info.get("features", {}).get(feature, {}).get("shape")
        if shape != [width]:
            raise ValueError(f"unexpected {feature} shape: {shape!r}")

    train_frames, validation_frames = _dataset_frame_counts(root)
    total_frames = train_frames + validation_frames
    if (train_frames, validation_frames, total_frames) != (
        TRAIN_FRAMES,
        VALIDATION_FRAMES,
        TOTAL_FRAMES,
    ):
        raise ValueError(
            "onset-v2 frame counts changed from the locked 47,618 train / 1,884 holdout / 49,502 total"
        )
    if int(info.get("total_frames", -1)) != total_frames:
        raise ValueError("info.json total_frames disagrees with episode parquet row counts")
    expected_counts = {
        "observation.state": train_frames,
        "action": train_frames * ACTION_HORIZON,
        "umi.tcp_and_gripper": train_frames,
    }
    for feature, expected in expected_counts.items():
        if stats.get(feature, {}).get("count") != [expected]:
            raise ValueError(f"unexpected train-only stats count for {feature}")
        for stat_name, values in stats[feature].items():
            if stat_name == "count":
                continue
            if not values or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"non-finite or empty stats for {feature}/{stat_name}")

    videos = root / "videos"
    if videos.is_symlink() or not videos.is_dir():
        raise ValueError("onset-v2 videos must be regular files copied into the artifact")
    video_paths = sorted(videos.rglob("*.mp4"))
    if len(video_paths) != 108 or any(path.is_symlink() or not path.is_file() for path in video_paths):
        raise ValueError("onset-v2 artifact must contain 108 regular MP4 files")
    video_keys = [(Path("videos") / path.relative_to(videos)).as_posix() for path in video_paths]
    declared_video_keys = sorted(key for key in declared_hashes if key.startswith("videos/"))
    if video_keys != declared_video_keys:
        raise ValueError("artifact manifest video entries disagree with copied MP4 files")
    expected_video_metadata = {
        "handling": "byte-for-byte source copies included in generated_file_sha256",
        "file_count": len(video_paths),
        "total_bytes": sum(path.stat().st_size for path in video_paths),
    }
    if artifact_manifest.get("videos") != expected_video_metadata:
        raise ValueError("artifact manifest copied-video metadata mismatch")

    return {
        "root": str(root),
        "schema_id": SCHEMA_ID,
        "repo_id": DATASET_REPO_ID,
        "artifact_manifest_sha256": expected_manifest_sha,
        "semantics_sha256": _sha256(semantics_path),
        "train_episodes": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "train_frames": train_frames,
        "validation_frames": validation_frames,
        "total_frames": total_frames,
        "video_file_count": len(video_paths),
        "video_total_bytes": expected_video_metadata["total_bytes"],
    }


def _parse_training_metrics(log_path: Path, *, full: bool) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    for marker in FATAL_TEXT:
        if marker in lowered:
            raise ValueError(f"fatal training marker found in log: {marker}")

    train_rows: dict[int, dict[str, float]] = {}
    for step_text, loss_text, grad_text, action_loss_text in TRAIN_METRIC_RE.findall(text):
        step = int(step_text)
        if step in train_rows:
            raise ValueError(f"duplicate training metric for step {step}")
        train_rows[step] = {
            "loss": _finite(loss_text, name=f"step-{step} loss"),
            "grad_norm": _finite(grad_text, name=f"step-{step} grad norm"),
            "action_flow_loss": _finite(action_loss_text, name=f"step-{step} action flow loss"),
        }

    expected_train_steps = TRAIN_LOG_STEPS if full else list(range(1, 6))
    if sorted(train_rows) != expected_train_steps:
        raise ValueError(
            f"training metric steps mismatch: expected {expected_train_steps[:3]}..."
            f"{expected_train_steps[-3:]}, got {sorted(train_rows)[:3]}...{sorted(train_rows)[-3:]}"
        )

    eval_rows: dict[int, float] = {}
    for step_text, loss_text in EVAL_METRIC_RE.findall(text):
        step = int(step_text)
        if step in eval_rows:
            raise ValueError(f"duplicate holdout metric for step {step}")
        eval_rows[step] = _finite(loss_text, name=f"step-{step} full holdout loss")
    expected_eval_steps = CHECKPOINT_STEPS if full else []
    if sorted(eval_rows) != expected_eval_steps:
        raise ValueError(
            f"holdout metric steps mismatch: expected {expected_eval_steps}, got {sorted(eval_rows)}"
        )

    return {
        "train_metrics": {str(step): train_rows[step] for step in sorted(train_rows)},
        "full_holdout_eval_loss": {str(step): eval_rows[step] for step in sorted(eval_rows)},
        "full_holdout_eval_loss_precision": "four decimal places in the immutable training log",
    }


def _safetensors_inventory(path: Path, *, expected_count: int, expected_bytes: int) -> dict[str, Any]:
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"unexpected byte size for {path}: {size} != {expected_bytes}")
    with safe_open(path, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
    if len(keys) != expected_count:
        raise ValueError(f"unexpected tensor count for {path}: {len(keys)} != {expected_count}")

    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"truncated safetensors header length: {path}")
        header_size = int.from_bytes(header_size_bytes, byteorder="little", signed=False)
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
    return {
        "bytes": size,
        "tensor_count": len(keys),
        "header_sha256": hashlib.sha256(header_size_bytes + header).hexdigest(),
    }


def _assert_policy_config(policy: dict[str, Any], model_root: Path) -> None:
    expected_scalars = {
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
        "scheduler_warmup_steps": 200,
        "scheduler_decay_steps": 4_000,
        "scheduler_decay_lr": 1e-6,
        "checkpoint_path": str(model_root),
        "checkpoint_revision": MODEL_REVISION,
    }
    for key, expected in expected_scalars.items():
        if policy.get(key) != expected:
            raise ValueError(f"unexpected policy {key}: {policy.get(key)!r} != {expected!r}")
    if policy.get("image_keys") != ["observation.images.umi1", "observation.images.umi2"]:
        raise ValueError("checkpoint camera order is not umi1, umi2")
    if policy.get("input_features", {}).get("observation.state", {}).get("shape") != [STATE_DIM]:
        raise ValueError("checkpoint state dimension is not 20")
    if policy.get("output_features", {}).get("action", {}).get("shape") != [STATE_DIM]:
        raise ValueError("checkpoint action dimension is not 20")


def _validate_normalizer(path: Path, *, train_frames: int) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
        if len(keys) != 102:
            raise ValueError(f"unexpected normalization tensor count in {path}")
        action_count = float(stream.get_tensor("action.count").item())
        state_count = float(stream.get_tensor("observation.state.count").item())
        helper_count = float(stream.get_tensor("umi.tcp_and_gripper.count").item())
        expected_mask = torch.tensor([1.0] * 9 + [0.0] + [1.0] * 9 + [0.0])
        for feature in ("action", "observation.state"):
            actual_mask = stream.get_tensor(f"{feature}.mask").cpu()
            if not torch.equal(actual_mask, expected_mask):
                raise ValueError(f"unexpected pose/gripper normalization mask for {feature}")
        for key in keys:
            tensor = stream.get_tensor(key)
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite normalization tensor: {path.name}/{key}")
    if action_count != train_frames * ACTION_HORIZON:
        raise ValueError("checkpoint action stats do not contain train_frames*24 rows")
    if state_count != train_frames or helper_count != train_frames:
        raise ValueError("checkpoint state/helper stats do not contain train-only rows")
    return {
        "tensor_count": len(keys),
        "action_count": int(action_count),
        "state_count": int(state_count),
        "helper_count": int(helper_count),
        "sha256": _sha256(path),
    }


def _validate_checkpoint(
    checkpoint: Path,
    *,
    step: int,
    run_root: Path,
    dataset_root: Path,
    model_root: Path,
    train_frames: int,
) -> dict[str, Any]:
    pretrained = checkpoint / "pretrained_model"
    training_state = checkpoint / "training_state"
    train_config = _load_json(pretrained / "train_config.json")
    policy_config = _load_json(pretrained / "config.json")
    _assert_policy_config(policy_config, model_root)
    _assert_policy_config(train_config.get("policy", {}), model_root)

    expected_train_scalars = {
        "output_dir": str(run_root),
        "job_name": "molmoact2-umi-currentrel-r6d-onset-v2-4k",
        "seed": 1000,
        "num_workers": 8,
        "batch_size": PER_RANK_BATCH_SIZE,
        "steps": 4_000,
        "env_eval_freq": 0,
        "log_freq": 10,
        "eval_steps": 500,
        "max_eval_samples": 0,
        "save_checkpoint": True,
        "save_freq": 500,
        "save_checkpoint_to_hub": False,
    }
    for key, expected in expected_train_scalars.items():
        if train_config.get(key) != expected:
            raise ValueError(f"checkpoint {step}: unexpected train config {key}")
    dataset_config = train_config.get("dataset", {})
    if dataset_config.get("repo_id") != DATASET_REPO_ID:
        raise ValueError(f"checkpoint {step}: wrong dataset repo id")
    if dataset_config.get("root") != str(dataset_root):
        raise ValueError(f"checkpoint {step}: wrong dataset root")
    if dataset_config.get("eval_split") != 2 / 54:
        raise ValueError(f"checkpoint {step}: wrong holdout fraction")
    image_transforms = dataset_config.get("image_transforms", {})
    if image_transforms.get("enable") is not True or image_transforms.get("tfs") != COLOR_JITTER:
        raise ValueError(f"checkpoint {step}: ColorJitter contract mismatch")

    training_step = _load_json(training_state / "training_step.json")
    if training_step != {
        "step": step,
        "num_processes": WORLD_SIZE,
        "batch_size": PER_RANK_BATCH_SIZE,
    }:
        raise ValueError(f"checkpoint {step}: training-state world/batch/step mismatch")
    scheduler = _load_json(training_state / "scheduler_state.json")
    if scheduler.get("last_epoch") != step or scheduler.get("_step_count") != step + 1:
        raise ValueError(f"checkpoint {step}: scheduler state mismatch")

    model_inventory = _safetensors_inventory(
        pretrained / "model.safetensors",
        expected_count=MODEL_TENSOR_COUNT,
        expected_bytes=MODEL_SAFETENSORS_BYTES,
    )
    optimizer_inventory = _safetensors_inventory(
        training_state / "optimizer_state.safetensors",
        expected_count=OPTIMIZER_TENSOR_COUNT,
        expected_bytes=OPTIMIZER_SAFETENSORS_BYTES,
    )
    normalizer = _validate_normalizer(
        pretrained / "policy_preprocessor_step_3_molmoact2_masked_normalizer.safetensors",
        train_frames=train_frames,
    )
    unnormalizer = _validate_normalizer(
        pretrained / "policy_postprocessor_step_1_molmoact2_masked_unnormalizer.safetensors",
        train_frames=train_frames,
    )
    for required in (
        pretrained / "policy_preprocessor.json",
        pretrained / "policy_postprocessor.json",
        training_state / "rng_state.safetensors",
        training_state / "optimizer_param_groups.json",
    ):
        if not required.is_file() or required.stat().st_size == 0:
            raise ValueError(f"checkpoint {step}: missing/empty artifact {required.name}")

    return {
        "step": step,
        "path": str(checkpoint),
        "model": model_inventory,
        "optimizer": optimizer_inventory,
        "normalizer": normalizer,
        "unnormalizer": unnormalizer,
        "train_config_sha256": _sha256(pretrained / "train_config.json"),
        "policy_config_sha256": _sha256(pretrained / "config.json"),
    }


def validate_full_run(
    run_root: Path,
    log_path: Path,
    dataset: dict[str, Any],
    model_root: Path,
) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    dataset_root = Path(dataset["root"])
    checkpoints_root = run_root / "checkpoints"
    expected_names = {f"{step:06d}" for step in CHECKPOINT_STEPS}
    actual_names = {path.name for path in checkpoints_root.iterdir() if path.is_dir() and path.name != "last"}
    if actual_names != expected_names:
        raise ValueError(f"checkpoint directory set mismatch: {sorted(actual_names)}")
    last = checkpoints_root / "last"
    if not last.is_symlink() or last.readlink() != Path("004000"):
        raise ValueError("checkpoints/last must be the relative symlink 004000")

    metrics = _parse_training_metrics(log_path, full=True)
    artifacts = [
        _validate_checkpoint(
            checkpoints_root / f"{step:06d}",
            step=step,
            run_root=run_root,
            dataset_root=dataset_root,
            model_root=model_root,
            train_frames=int(dataset["train_frames"]),
        )
        for step in CHECKPOINT_STEPS
    ]
    return {"metrics": metrics, "checkpoints": artifacts}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _common_report(args: argparse.Namespace, dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": "umi-yam-onset-v2-training-validation-v1",
        "status": "pass",
        "hardware_control_performed": False,
        "job_id": str(args.job_id),
        "dataset": dataset,
        "code_manifest_sha256": _require_sha256(args.code_manifest_sha256, name="code manifest SHA-256"),
        "base_model": {
            "root": str(args.model_root.resolve(strict=True)),
            "revision": MODEL_REVISION,
        },
    }


def command_smoke(args: argparse.Namespace) -> None:
    dataset = validate_dataset_identity(args.dataset_root, args.dataset_manifest_sha256)
    metrics = _parse_training_metrics(args.log, full=False)
    report = _common_report(args, dataset)
    report.update(
        {
            "kind": "one-gpu-five-step-smoke",
            "world_size": 1,
            "per_rank_batch_size": PER_RANK_BATCH_SIZE,
            "metrics": metrics,
        }
    )
    _write_report(args.report, report)
    print(json.dumps(report, sort_keys=True))


def command_verify_smoke(args: argparse.Namespace) -> None:
    report = _load_json(args.report)
    expected = {
        "status": "pass",
        "kind": "one-gpu-five-step-smoke",
        "job_id": str(args.job_id),
        "code_manifest_sha256": _require_sha256(args.code_manifest_sha256, name="code manifest SHA-256"),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"smoke report {key} mismatch")
    manifest_sha = _require_sha256(args.dataset_manifest_sha256, name="dataset manifest SHA-256")
    if report.get("dataset", {}).get("artifact_manifest_sha256") != manifest_sha:
        raise ValueError("smoke report dataset artifact mismatch")
    print(json.dumps({"smoke_report_verified": True, "job_id": str(args.job_id)}, sort_keys=True))


def command_full(args: argparse.Namespace) -> None:
    dataset = validate_dataset_identity(args.dataset_root, args.dataset_manifest_sha256)
    full = validate_full_run(args.run_root, args.log, dataset, args.model_root.resolve(strict=True))
    report = _common_report(args, dataset)
    report.update(
        {
            "kind": "eight-gpu-four-thousand-step-full",
            "world_size": WORLD_SIZE,
            "per_rank_batch_size": PER_RANK_BATCH_SIZE,
            "global_batch_size": WORLD_SIZE * PER_RANK_BATCH_SIZE,
            "full_run": full,
            "selection_policy": SELECTION_POLICY,
        }
    )
    _write_report(args.report, report)
    print(json.dumps(report, sort_keys=True))


def command_select(args: argparse.Namespace) -> None:
    """Apply the frozen gate-first/loss-second checkpoint rule without inference."""

    training = _load_json(args.training_report)
    gate = _load_json(args.gate_aggregate)
    if training.get("status") != "pass" or training.get("kind") != "eight-gpu-four-thousand-step-full":
        raise ValueError("selection requires a passing onset-v2 full-training validation report")
    if training.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("training report does not contain the frozen selection policy")
    if training.get("dataset", {}).get("schema_id") != SCHEMA_ID:
        raise ValueError("training report does not bind the onset-v2 dataset")

    for key in (
        "immutable_plan_hash_verified",
        "all_child_hashes_and_resolver_outputs_verified",
        "all_child_ik_and_behavior_summaries_recomputed",
        "comparison_execution_passed",
    ):
        if gate.get(key) is not True:
            raise ValueError(f"gate aggregate did not verify {key}")
    if gate.get("automatic_checkpoint_selection_performed") is not False:
        raise ValueError("gate aggregate must not have selected a checkpoint internally")
    if gate.get("overall_checkpoint_selection_performed") is not False:
        raise ValueError("gate aggregate must leave overall selection to this frozen rule")

    summaries = gate.get("checkpoint_summaries")
    if not isinstance(summaries, list):
        raise ValueError("gate aggregate is missing checkpoint_summaries")
    by_step = {int(summary["checkpoint_step"]): summary for summary in summaries}
    if sorted(by_step) != CHECKPOINT_STEPS or len(summaries) != len(CHECKPOINT_STEPS):
        raise ValueError("gate aggregate must cover each of the eight checkpoints exactly once")

    eligible_steps: list[int] = []
    eligible_labels: list[str] = []
    for step in CHECKPOINT_STEPS:
        summary = by_step[step]
        advancement = summary.get("advancement_eligible") is True
        deployment = summary.get("deployment_gate_eligible") is True
        if advancement != deployment:
            raise ValueError(f"checkpoint {step}: advancement/deployment eligibility disagree")
        if advancement:
            if summary.get("all_10_strict_execution_prefix_pass") is not True:
                raise ValueError(f"checkpoint {step}: eligible without all ten strict prefix passes")
            if summary.get("pooled_idle_drift_diagnostic_beats_identity_hold") is not True:
                raise ValueError(f"checkpoint {step}: eligible without the frozen behavior criterion")
            eligible_steps.append(step)
            eligible_labels.append(str(summary.get("checkpoint_label")))
    if gate.get("advancement_eligible_checkpoints") != eligible_labels:
        raise ValueError("gate aggregate eligible list disagrees with checkpoint summaries")

    losses_raw = training.get("full_run", {}).get("metrics", {}).get("full_holdout_eval_loss", {})
    losses = {int(step): float(value) for step, value in losses_raw.items()}
    if sorted(losses) != CHECKPOINT_STEPS or not all(math.isfinite(value) for value in losses.values()):
        raise ValueError("training report must contain eight finite full-holdout losses")
    selected_step = min(eligible_steps, key=lambda step: (losses[step], step)) if eligible_steps else None
    report = {
        "schema_id": "umi-yam-onset-v2-checkpoint-selection-v1",
        "status": "selected" if selected_step is not None else "no-eligible-checkpoint",
        "hardware_control_performed": False,
        "selection_policy": SELECTION_POLICY,
        "training_report": str(args.training_report.resolve(strict=True)),
        "training_report_sha256": _sha256(args.training_report),
        "gate_aggregate": str(args.gate_aggregate.resolve(strict=True)),
        "gate_aggregate_sha256": _sha256(args.gate_aggregate),
        "eligible_checkpoint_steps": eligible_steps,
        "full_holdout_eval_loss": {str(step): losses[step] for step in CHECKPOINT_STEPS},
        "selected_checkpoint_step": selected_step,
    }
    _write_report(args.report, report)
    print(json.dumps(report, sort_keys=True))


def _add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--code-manifest-sha256", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    _add_identity_args(smoke)
    smoke.add_argument("--log", type=Path, required=True)
    smoke.add_argument("--report", type=Path, required=True)
    smoke.set_defaults(func=command_smoke)

    verify_smoke = subparsers.add_parser("verify-smoke")
    verify_smoke.add_argument("--report", type=Path, required=True)
    verify_smoke.add_argument("--job-id", required=True)
    verify_smoke.add_argument("--dataset-manifest-sha256", required=True)
    verify_smoke.add_argument("--code-manifest-sha256", required=True)
    verify_smoke.set_defaults(func=command_verify_smoke)

    full = subparsers.add_parser("full")
    _add_identity_args(full)
    full.add_argument("--run-root", type=Path, required=True)
    full.add_argument("--log", type=Path, required=True)
    full.add_argument("--report", type=Path, required=True)
    full.set_defaults(func=command_full)

    select = subparsers.add_parser("select")
    select.add_argument("--training-report", type=Path, required=True)
    select.add_argument("--gate-aggregate", type=Path, required=True)
    select.add_argument("--report", type=Path, required=True)
    select.set_defaults(func=command_select)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
