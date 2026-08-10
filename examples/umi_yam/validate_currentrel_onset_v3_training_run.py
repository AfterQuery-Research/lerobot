#!/usr/bin/env python

"""Fail-closed smoke/full-run and checkpoint-selection checks for onset-v3.

Training validation binds the frozen dataset, recipe, and gate plan.  Selection
then verifies a complete independent model->resolver->strict-IK aggregate before
using full-holdout loss to choose among offline-eligible checkpoints.  Because
the interior YAM start has not yet been verified physically, selection can never
declare a checkpoint hardware-deployment ready.
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

SCHEMA_ID = "dual-lidar-umi-currentrel-r6d-onset-v3"
DATASET_REPO_ID = "brandonyang/dual-lidar-umi-currentrel-r6d-onset-v3"
DATASET_BASENAME = "dual-lidar-umi-currentrel-r6d-onset-v3"
SEMANTICS_REL = Path("meta/umi_current_relative_r6d_onset_v3.json")
ARTIFACT_MANIFEST_REL = Path("meta/artifact_manifest.sha256")
ARTIFACT_METADATA_REL = Path("meta/artifact_manifest.json")
GATE_PLAN_SCHEMA_ID = "umi-yam-onset-v3-checkpoint-gate-plan-v1"
GATE_AGGREGATE_SCHEMA_ID = "umi-yam-onset-v3-checkpoint-gate-aggregate-v1"
MODEL_REVISION = "e432d85f6e039edca44afb93c262f3084ab72a9c"

TRAIN_EPISODES = list(range(52))
VALIDATION_EPISODES = [52, 53]
TRAIN_FRAMES = 47_237
VALIDATION_FRAMES = 1_884
TOTAL_FRAMES = 49_121
VALID_TRAIN_ACTION_ROWS = 1_118_088
VALIDATION_ACTION_ROWS = 44_616
VALID_ALL_ACTION_ROWS = 1_162_704
PADDED_ACTION_ROWS_PER_EPISODE = 300
CHECKPOINT_STEPS = list(range(1_000, 12_001, 1_000))
TRAIN_LOG_STEPS = list(range(10, 12_001, 10))
ACTION_HORIZON = 24
STATE_DIM = 20
PER_RANK_BATCH_SIZE = 4
WORLD_SIZE = 7
GLOBAL_BATCH_SIZE = WORLD_SIZE * PER_RANK_BATCH_SIZE
FULL_SAMPLE_PRESENTATIONS = 12_000 * GLOBAL_BATCH_SIZE
EFFECTIVE_TRAIN_FRAME_EPOCHS = FULL_SAMPLE_PRESENTATIONS / TRAIN_FRAMES
MODEL_TENSOR_COUNT = 1_899
MODEL_SAFETENSORS_BYTES = 11_483_634_784
OPTIMIZER_TENSOR_COUNT = 3_558
OPTIMIZER_SAFETENSORS_BYTES = 3_487_466_024
GATE_HOLDOUT_EPISODES = [52, 53]
GATE_SEEDS = [0, 1, 2, 3, 4]
GATE_EXECUTION_HORIZON = 15
GATE_NUM_FLOW_STEPS = 10
GATE_CELLS_PER_CHECKPOINT = 10
GATE_MINIMUM_IDENTITY_IMPROVEMENT = 0.10
GATE_START_DRIVER_TARGET = [
    0.0,
    0.05,
    0.05,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.05,
    0.05,
    0.0,
    0.0,
    0.0,
    1.0,
]
GATE_MOTION_FIELDS = [
    "predicted_endpoint_translation_m_by_cell",
    "target_endpoint_translation_m_by_cell",
    "predicted_max_translation_m_by_cell",
    "target_max_translation_m_by_cell",
    "predicted_endpoint_rotation_rad_by_cell",
    "target_endpoint_rotation_rad_by_cell",
    "endpoint_translation_direction_cosine_by_cell",
]
TEMPORAL_GATE_SCHEMA_ID = "umi-yam-onset-v3-temporal-teacher-forced-gate-contract-v1"
TEMPORAL_GATE_SCOPE = {
    "holdout_episodes": GATE_HOLDOUT_EPISODES,
    "seeds": GATE_SEEDS,
    "num_flow_steps": GATE_NUM_FLOW_STEPS,
    "action_horizon_rows": 24,
    "execution_horizon_rows": GATE_EXECUTION_HORIZON,
    "query_stride_source_frames": GATE_EXECUTION_HORIZON,
    "teacher_forced_source_images": True,
    "teacher_forced_source_policy_state": True,
    "predicted_yam_joints_carried_continuously_across_queries": True,
    "query_anchor_from_carried_yam_fk": True,
    "initial_arm_joints_rad": GATE_START_DRIVER_TARGET[:6],
    "original_idle_reset_prefix_included": False,
    "original_idle_reset_prefix_role": ("separate diagnostic only; it is not an ACTIVE runtime phase"),
}
TEMPORAL_TASK_CRITICAL_WINDOWS = [
    {
        "episode": 52,
        "exported_frame_count": 974,
        "source_onset_frame": 27,
        "final_contact_source_frame": 826,
        "final_contact_exported_frame": 799,
        "required_through_exported_frame_inclusive": 823,
        "last_required_query_exported_frame": 810,
        "diagnostic_tail_exported_frames_inclusive": [824, 973],
    },
    {
        "episode": 53,
        "exported_frame_count": 910,
        "source_onset_frame": 26,
        "final_contact_source_frame": 784,
        "final_contact_exported_frame": 758,
        "required_through_exported_frame_inclusive": 782,
        "last_required_query_exported_frame": 780,
        "diagnostic_tail_exported_frames_inclusive": [783, 909],
    },
]
TEMPORAL_BEHAVIOR_GATE = {
    "metric": (
        "for each critical executed arm-row: 0.5*(translation_error_m/0.01 + "
        "SO3_geodesic_error_rad/radians(10)); compare raw current-relative UMI prediction "
        "with the dataset target"
    ),
    "translation_scale_m": 0.01,
    "rotation_scale_deg": 10.0,
    "identity_baseline": "zero translation and identity rotation independently at every query",
    "pooling": (
        "exactly 10 fixed seed-by-episode cells x 2 arms x every critical executed row; "
        "16050 arm-rows per checkpoint; no seed, episode, arm, query, or row exclusion"
    ),
    "critical_arm_row_count_per_checkpoint": 16_050,
    "minimum_relative_improvement_over_identity": GATE_MINIMUM_IDENTITY_IMPROVEMENT,
    "relative_improvement_equation": (
        "1 - pooled_model_normalized_pose_error / pooled_identity_normalized_pose_error"
    ),
    "all_strict_rows_still_required": True,
    "tail_excluded_from_behavior_eligibility": True,
    "motion_amplitude_and_phase_report_required": True,
    "binary_eligibility_not_checkpoint_score": True,
}
TEMPORAL_PER_ROW_FIELDS = [
    "episode",
    "seed",
    "query_exported_frame",
    "chunk_row",
    "executed_exported_frame",
    "window_label",
    "arm",
    "model_input_state_float32_sha256",
    "model_input_image_rgb_sha256",
    "raw_prediction_float32_sha256",
    "resolved_target_base_tcp_xyz_rotvec",
    "ik_continuity_seed_rad",
    "ik_joint_target_rad",
    "ik_converged",
    "fk_position_residual_m",
    "fk_orientation_residual_rad",
    "operational_joint_limits_pass",
    "required_dispatches_under_ideal_measured_feedback",
    "maximum_simulated_command_delta_rad",
    "row_pass",
    "predicted_gripper",
    "ground_truth_gripper",
]
RUNTIME_PHASE_CONTRACT = {
    "RESET_HOLD": (
        "policy is not queried and policy actions are not consumed; hold the measured safe target "
        "while reset/start gates are checked"
    ),
    "START": (
        "an explicit operator START edge is required after physical start, collision, "
        "camera-viewpoint, gripper-calibration, and object-scene gates pass"
    ),
    "ACTIVE15_REQUERY": (
        "query from fresh measured state and images, execute at most rows 1..15 with strict "
        "IK/rate/progress checks, then requery from fresh measured FK"
    ),
    "STOP_HOLD_DISARM": (
        "on operator stop, stale or malformed input, nonfinite output, strict IK failure, "
        "rate/progress failure, timeout, or transport fault: consume no further policy rows, "
        "hold safely, then disarm according to the hardware procedure"
    ),
    "best_effort_execution_allowed": False,
    "original_idle_reset_prefix_role": (
        "offline diagnostic only; never use it to loosen ACTIVE safety gates"
    ),
}

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

NORMALIZER_FULL_STAT_NAMES = (
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
NORMALIZER_VECTOR_FEATURE_DIMS = {
    "action": STATE_DIM,
    "observation.state": STATE_DIM,
    "umi.tcp_and_gripper": 16,
}
NORMALIZER_SCALAR_FEATURES = (
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
)
NORMALIZER_IMAGE_FEATURES = (
    "observation.images.umi1",
    "observation.images.umi2",
)
IMAGENET_NORMALIZATION_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],
    "std": [[[0.229]], [[0.224]], [[0.225]]],
}


def _normalizer_tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for feature, dim in NORMALIZER_VECTOR_FEATURE_DIMS.items():
        for stat_name in NORMALIZER_FULL_STAT_NAMES:
            shapes[f"{feature}.{stat_name}"] = (1,) if stat_name == "count" else (dim,)
    for feature in NORMALIZER_SCALAR_FEATURES:
        for stat_name in NORMALIZER_FULL_STAT_NAMES:
            shapes[f"{feature}.{stat_name}"] = (1,)
    for feature in ("action", "observation.state"):
        shapes[f"{feature}.mask"] = (STATE_DIM,)
    for feature in NORMALIZER_IMAGE_FEATURES:
        for stat_name in IMAGENET_NORMALIZATION_STATS:
            shapes[f"{feature}.{stat_name}"] = (3, 1, 1)
    return shapes


NORMALIZER_TENSOR_SHAPES = _normalizer_tensor_shapes()

SELECTION_POLICY = {
    "candidate_checkpoint_steps": CHECKPOINT_STEPS,
    "gate_plan_schema_id": GATE_PLAN_SCHEMA_ID,
    "holdout_episodes": GATE_HOLDOUT_EPISODES,
    "seeds": GATE_SEEDS,
    "execution_horizon_rows": GATE_EXECUTION_HORIZON,
    "minimum_relative_improvement_over_identity": GATE_MINIMUM_IDENTITY_IMPROVEMENT,
    "temporal_minimum_relative_improvement_over_identity": GATE_MINIMUM_IDENTITY_IMPROVEMENT,
    "eligibility_source": (
        "complete independent frozen model->resolver->strict-IK gate at the predeclared "
        "interior YAM start anchor"
    ),
    "selection_rule": (
        "Require all 10 first-15 onset cells to pass strict IK and pooled physical SE(3) error "
        "to improve on identity by at least 10%. Run the fixed temporal teacher-forced grid on "
        "every onset-eligible checkpoint; require every critical row to pass strict IK/rate and "
        "its pooled raw SE(3) error to improve on identity by at least 10%, with no tolerance or "
        "row relaxation. Among checkpoints passing both gates, select the lowest full onset-v3 "
        "holdout loss; ties at logged "
        "precision select the earlier step. Neither gate score may override full-holdout loss."
    ),
    "preselection_temporal_gate_schema_id": TEMPORAL_GATE_SCHEMA_ID,
    "temporal_gate_required_for_every_onset_eligible_checkpoint": True,
    "temporal_gate_is_binary_eligibility_not_score": True,
    "no_temporal_threshold_or_tolerance_relaxation": True,
    "hardware_start_verified": False,
    "hardware_deployment_blocked_until_supervised_start_verification": True,
    "no_seed_or_checkpoint_cherry_picking": True,
    "no_eligible_checkpoint_behavior": "do not advance any checkpoint",
}

TRAIN_METRIC_RE = re.compile(
    r"\bstep:(\d+(?:\.\d+)?[KMBTQ]?)\b.*?\bloss:([^\s]+).*?\bgrdn:([^\s]+).*?"
    r"\baction_flow_loss:([^\s]+)"
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


def validate_gate_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Bind training and selection to one exact, structurally checked gate plan."""

    path = path.resolve(strict=True)
    expected_sha = _require_sha256(expected_sha256, name="gate plan SHA-256")
    if _sha256(path) != expected_sha:
        raise ValueError("checkpoint-gate plan SHA-256 mismatch")
    plan = _load_json(path)
    if plan.get("schema_id") != GATE_PLAN_SCHEMA_ID:
        raise ValueError("unexpected checkpoint-gate plan schema")
    dataset = plan.get("dataset", {})
    if dataset != {
        "schema_id": SCHEMA_ID,
        "repo_id": DATASET_REPO_ID,
        "holdout_episodes": GATE_HOLDOUT_EPISODES,
        "query_index_within_exported_episode": 0,
    }:
        raise ValueError("checkpoint-gate dataset/query contract mismatch")
    if plan.get("candidate_checkpoint_steps") != CHECKPOINT_STEPS:
        raise ValueError("checkpoint-gate candidate steps changed")
    inference = plan.get("inference", {})
    if inference != {
        "seeds": GATE_SEEDS,
        "num_flow_steps": GATE_NUM_FLOW_STEPS,
        "one_fresh_query_per_cell": True,
        "expected_cells_per_checkpoint": GATE_CELLS_PER_CHECKPOINT,
    }:
        raise ValueError("checkpoint-gate inference grid mismatch")
    execution = plan.get("execution", {})
    expected_execution = {
        "horizon_rows": GATE_EXECUTION_HORIZON,
        "full_24_row_execution_performed": False,
        "adapter_max_joint_delta_rad_per_call": 0.02,
        "driver_max_joint_delta_rad_per_call": 0.1,
        "max_progress_hold_steps": 90,
        "strict_ik_required_for_every_arm_row": True,
        "required_strict_cells_per_checkpoint": GATE_CELLS_PER_CHECKPOINT,
    }
    if execution != expected_execution:
        raise ValueError("checkpoint-gate execution contract mismatch")
    start = plan.get("yam_start_anchor", {})
    if start.get("absolute_driver_target") != GATE_START_DRIVER_TARGET:
        raise ValueError("checkpoint gate did not bind the interior 14-D YAM start target")
    if start.get("left_arm_joints_rad") != GATE_START_DRIVER_TARGET[:6]:
        raise ValueError("checkpoint-gate left start joints changed")
    if start.get("right_arm_joints_rad") != GATE_START_DRIVER_TARGET[7:13]:
        raise ValueError("checkpoint-gate right start joints changed")
    if start.get("grippers_normalized") != [1.0, 1.0]:
        raise ValueError("checkpoint-gate start grippers must both be open")
    if start.get("hardware_start_verified") is not False:
        raise ValueError("interior start must remain hardware-unverified until supervised confirmation")
    behavior = plan.get("behavior_gate", {})
    if behavior.get("translation_scale_m") != 0.01:
        raise ValueError("checkpoint-gate translation normalization scale changed")
    if behavior.get("rotation_scale_deg") != 10.0:
        raise ValueError("checkpoint-gate rotation normalization scale changed")
    if behavior.get("minimum_relative_improvement_over_identity") != (GATE_MINIMUM_IDENTITY_IMPROVEMENT):
        raise ValueError("checkpoint-gate identity improvement margin changed")
    if behavior.get("per_arm_motion_report_fields") != GATE_MOTION_FIELDS:
        raise ValueError("checkpoint-gate per-arm motion diagnostics changed")
    temporal = plan.get("temporal_teacher_forced_gate", {})
    if temporal.get("schema_id") != TEMPORAL_GATE_SCHEMA_ID:
        raise ValueError("temporal teacher-forced gate schema changed")
    if temporal.get("implementation_status") != (
        "predeclared fail-closed contract; runnable producer and independent aggregate are "
        "required before supervised start verification"
    ):
        raise ValueError("temporal gate must remain explicitly fail-closed until implemented")
    if temporal.get("execution_order") != (
        "run on every checkpoint that passes the first-stage onset gate before final checkpoint selection"
    ):
        raise ValueError("temporal gate execution order changed")
    if temporal.get("checkpoint_selection_effect") != (
        "temporal pass is an additional eligibility requirement; after all fixed cells finish, "
        "select the lowest full-holdout loss among checkpoints passing both gates"
    ):
        raise ValueError("temporal gate must remain a binary eligibility gate")
    if temporal.get("failure_behavior") != (
        "exclude a failing checkpoint from final eligibility; if none pass both gates, block "
        "advancement and diagnose or retrain"
    ):
        raise ValueError("temporal-gate failure behavior changed")
    if temporal.get("scope") != TEMPORAL_GATE_SCOPE:
        raise ValueError("temporal teacher-forced replay scope changed")
    if temporal.get("task_critical_windows") != TEMPORAL_TASK_CRITICAL_WINDOWS:
        raise ValueError("temporal teacher-forced critical windows changed")
    if temporal.get("critical_window_rule") != (
        "from exported onset through the last source contact anchor plus 24 frames, clipped to "
        "the episode; query at 0,15,... and execute rows 1..15"
    ):
        raise ValueError("temporal teacher-forced critical-window rule changed")
    if temporal.get("behavior_gate") != TEMPORAL_BEHAVIOR_GATE:
        raise ValueError("temporal teacher-forced pooled behavior gate changed")
    expected_temporal_recomputation = {
        "independent_of_child_boolean_summaries": True,
        "resolver_output_recomputed_bit_exactly": True,
        "fk_position_and_orientation_residuals_recomputed": True,
        "operational_joint_limits_recomputed": True,
        "joint_rate_and_progress_dispatches_recomputed": True,
        "adapter_max_joint_delta_rad_per_call": 0.02,
        "driver_max_joint_delta_rad_per_call": 0.1,
        "max_progress_hold_steps": 90,
        "all_rows_through_required_window_must_pass_for_every_episode_and_seed": True,
        "best_bounded_continuation_after_failure_is_diagnostic_only": True,
        "tail_pass_is_reported_separately_and_is_not_an_eligibility_substitute": True,
    }
    if temporal.get("strict_recomputation") != expected_temporal_recomputation:
        raise ValueError("temporal teacher-forced independent recomputation contract changed")
    if temporal.get("required_per_row_record_fields") != TEMPORAL_PER_ROW_FIELDS:
        raise ValueError("temporal teacher-forced per-row evidence schema changed")
    if temporal.get("required_reports") != {
        "per_row_records_saved": True,
        "critical_and_tail_summaries_separate": True,
        "gripper_range_and_event_timing_diagnostics": True,
        "motion_amplitude_and_phase_diagnostics": True,
        "first_strict_failure_saved": True,
        "checkpoint_dataset_code_i2rt_and_runner_sha256_bound": True,
    }:
        raise ValueError("temporal teacher-forced report requirements changed")
    if temporal.get("claims") != {
        "autonomous_closed_loop_rollout": False,
        "visual_compounding_tested": False,
        "objects_or_contact_simulated": False,
        "grasp_success_proven": False,
        "collision_clearance_proven": False,
        "hardware_control_performed": False,
        "hardware_deployment_ready": False,
    }:
        raise ValueError("temporal teacher-forced scope limitations changed")
    if plan.get("runtime_phase_contract") != RUNTIME_PHASE_CONTRACT:
        raise ValueError("RESET/HOLD -> START -> ACTIVE15/requery -> STOP/HOLD/DISARM contract changed")
    eligibility = plan.get("eligibility", {})
    for required_true in (
        "all_10_first_15_strict_execution_cells_required",
        "behavior_margin_required",
        "hardware_start_verified_required_for_hardware_deployment",
        "full_holdout_loss_used_only_after_gate_eligibility",
        "temporal_teacher_forced_gate_runs_on_every_onset_eligible_checkpoint_before_selection",
        "temporal_teacher_forced_gate_is_additional_eligibility_not_a_score",
        "temporal_full_critical_behavior_margin_required",
        "final_selection_uses_loss_only_among_checkpoints_passing_both_gates",
        "no_temporal_threshold_or_tolerance_relaxation",
        "no_seed_or_checkpoint_cherry_picking",
    ):
        if eligibility.get(required_true) is not True:
            raise ValueError(f"checkpoint-gate eligibility flag {required_true!r} must be true")
    if eligibility.get("hardware_start_verified_required_for_offline_checkpoint_eligibility") is not False:
        raise ValueError("offline selection must remain distinct from hardware-start verification")
    audit = plan.get("ground_truth_start_anchor_audit", {})
    if audit != {
        "episode_count": 54,
        "first_15_arm_row_pass_count": 1_620,
        "first_15_arm_row_count": 1_620,
        "full_24_arm_row_pass_count": 2_592,
        "full_24_arm_row_count": 2_592,
        "maximum_required_dispatches_per_waypoint": 4,
    }:
        raise ValueError("interior-start ground-truth audit contract mismatch")
    q0 = plan.get("negative_q0_provenance", {})
    if q0.get("first_15_arm_row_pass_count") != 1_614 or q0.get("first_15_arm_row_count") != 1_620:
        raise ValueError("q0 negative provenance changed")
    if q0.get("audit_report_sha256") != ("6286f864b68db796d63e22bd03e537986596f14938d6c6d0dc97bdb82a3bf105"):
        raise ValueError("q0 negative-provenance report digest changed")
    return {
        "path": str(path),
        "sha256": expected_sha,
        "schema_id": GATE_PLAN_SCHEMA_ID,
        "holdout_episodes": GATE_HOLDOUT_EPISODES,
        "seeds": GATE_SEEDS,
        "execution_horizon_rows": GATE_EXECUTION_HORIZON,
        "start_absolute_driver_target": GATE_START_DRIVER_TARGET,
        "hardware_start_verified": False,
        "minimum_relative_improvement_over_identity": GATE_MINIMUM_IDENTITY_IMPROVEMENT,
        "temporal_teacher_forced_gate_schema_id": TEMPORAL_GATE_SCHEMA_ID,
        "runtime_phase_contract": RUNTIME_PHASE_CONTRACT,
    }


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


def _format_logged_step(step: int) -> str:
    """Mirror the trainer's precision-zero compact display used in metric lines."""

    value = float(step)
    for suffix in ("", "K", "M", "B", "T", "Q"):
        if abs(value) < 1_000.0:
            return f"{value:.0f}{suffix}"
        value /= 1_000.0
    raise AssertionError(f"training step is outside the supported display range: {step}")


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
        raise ValueError(f"unexpected onset-v3 schema: {semantics.get('schema_id')!r}")
    if semantics.get("dataset_name") != DATASET_BASENAME:
        raise ValueError(f"unexpected onset-v3 dataset name: {semantics.get('dataset_name')!r}")
    if semantics.get("repository") != DATASET_REPO_ID:
        raise ValueError(f"unexpected onset-v3 repo id: {semantics.get('repository')!r}")
    if int(semantics.get("fps", -1)) != 30:
        raise ValueError("onset-v3 semantics must declare fps=30")
    horizon = semantics.get("action_horizon", semantics.get("horizon", -1))
    if int(horizon) != ACTION_HORIZON:
        raise ValueError("onset-v3 semantics must declare a 24-row action horizon")
    if split.get("train_episodes") != TRAIN_EPISODES:
        raise ValueError("onset-v3 split must keep train episodes 0..51")
    if split.get("validation_episodes") != VALIDATION_EPISODES:
        raise ValueError("onset-v3 split must keep holdout episodes 52 and 53")
    if set(TRAIN_EPISODES) & set(VALIDATION_EPISODES):
        raise AssertionError("internal train/validation split overlap")
    if info.get("total_episodes") != 54:
        raise ValueError("onset-v3 artifact must retain all 54 episodes")
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
            "onset-v3 frame counts changed from the locked 47,237 train / 1,884 holdout / 49,121 total"
        )
    if int(info.get("total_frames", -1)) != total_frames:
        raise ValueError("info.json total_frames disagrees with episode parquet row counts")
    expected_valid_action_rows = {
        "train": train_frames * ACTION_HORIZON - len(TRAIN_EPISODES) * PADDED_ACTION_ROWS_PER_EPISODE,
        "validation": validation_frames * ACTION_HORIZON
        - len(VALIDATION_EPISODES) * PADDED_ACTION_ROWS_PER_EPISODE,
        "all": total_frames * ACTION_HORIZON
        - (len(TRAIN_EPISODES) + len(VALIDATION_EPISODES)) * PADDED_ACTION_ROWS_PER_EPISODE,
    }
    if expected_valid_action_rows != {
        "train": VALID_TRAIN_ACTION_ROWS,
        "validation": VALIDATION_ACTION_ROWS,
        "all": VALID_ALL_ACTION_ROWS,
    }:
        raise AssertionError("locked valid-action row arithmetic is internally inconsistent")
    expected_counts = {
        "observation.state": train_frames,
        "action": VALID_TRAIN_ACTION_ROWS,
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
        raise ValueError("onset-v3 videos must be regular files copied into the artifact")
    video_paths = sorted(videos.rglob("*.mp4"))
    if len(video_paths) != 108 or any(path.is_symlink() or not path.is_file() for path in video_paths):
        raise ValueError("onset-v3 artifact must contain 108 regular MP4 files")
    video_keys = [(Path("videos") / path.relative_to(videos)).as_posix() for path in video_paths]
    declared_video_keys = sorted(key for key in declared_hashes if key.startswith("videos/"))
    if video_keys != declared_video_keys:
        raise ValueError("artifact manifest video entries disagree with copied MP4 files")
    expected_video_metadata = {
        "handling": "byte-for-byte source copies included in generated_file_sha256",
        "file_count": len(video_paths),
        "total_bytes": sum(path.stat().st_size for path in video_paths),
        "all_source_sha256_equal": True,
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
        "padded_inclusive_train_action_rows": train_frames * ACTION_HORIZON,
        "valid_train_action_rows": VALID_TRAIN_ACTION_ROWS,
        "valid_validation_action_rows": VALIDATION_ACTION_ROWS,
        "valid_all_action_rows": VALID_ALL_ACTION_ROWS,
        "padded_action_rows_per_episode": PADDED_ACTION_ROWS_PER_EPISODE,
        "video_file_count": len(video_paths),
        "video_total_bytes": expected_video_metadata["total_bytes"],
    }


def _parse_training_metrics(log_path: Path, *, full: bool) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    for marker in FATAL_TEXT:
        if marker in lowered:
            raise ValueError(f"fatal training marker found in log: {marker}")

    expected_train_steps = TRAIN_LOG_STEPS if full else list(range(1, 6))
    matches = TRAIN_METRIC_RE.findall(text)
    if len(matches) != len(expected_train_steps):
        raise ValueError(
            f"training metric row count mismatch: expected {len(expected_train_steps)}, got {len(matches)}"
        )
    train_rows: dict[int, dict[str, float]] = {}
    for step, (displayed_step, loss_text, grad_text, action_loss_text) in zip(
        expected_train_steps, matches, strict=True
    ):
        expected_display = _format_logged_step(step)
        if displayed_step != expected_display:
            raise ValueError(
                f"training metric order/step mismatch: expected {expected_display!r} for step "
                f"{step}, got {displayed_step!r}"
            )
        train_rows[step] = {
            "loss": _finite(loss_text, name=f"step-{step} loss"),
            "grad_norm": _finite(grad_text, name=f"step-{step} grad norm"),
            "action_flow_loss": _finite(action_loss_text, name=f"step-{step} action flow loss"),
        }

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


def _validate_normalizer(
    path: Path,
    *,
    train_frames: int,
    valid_train_action_rows: int,
    dataset_stats: dict[str, Any],
) -> dict[str, Any]:
    expected_dataset_features = set(NORMALIZER_VECTOR_FEATURE_DIMS) | set(NORMALIZER_SCALAR_FEATURES)
    if set(dataset_stats) != expected_dataset_features:
        raise ValueError(
            "dataset normalization-stat feature set mismatch: "
            f"got={sorted(dataset_stats)}, expected={sorted(expected_dataset_features)}"
        )
    with safe_open(path, framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
        actual_keys = set(keys)
        expected_keys = set(NORMALIZER_TENSOR_SHAPES)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            raise ValueError(
                f"normalization tensor schema mismatch in {path}: missing={missing}, extra={extra}"
            )
        for key, expected_shape in NORMALIZER_TENSOR_SHAPES.items():
            tensor = stream.get_tensor(key)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"unexpected normalization tensor shape for {path.name}/{key}: "
                    f"{tuple(tensor.shape)} != {expected_shape}"
                )
            if tensor.dtype != torch.float32:
                raise ValueError(
                    f"unexpected normalization tensor dtype for {path.name}/{key}: "
                    f"{tensor.dtype} != {torch.float32}"
                )
        action_count = float(stream.get_tensor("action.count").item())
        state_count = float(stream.get_tensor("observation.state.count").item())
        helper_count = float(stream.get_tensor("umi.tcp_and_gripper.count").item())
        for feature in NORMALIZER_SCALAR_FEATURES:
            count = float(stream.get_tensor(f"{feature}.count").item())
            if count != train_frames:
                raise ValueError(f"checkpoint {feature} stats do not contain train-only rows")
        expected_mask = torch.tensor([1.0] * 9 + [0.0] + [1.0] * 9 + [0.0], dtype=torch.float32)
        for feature in ("action", "observation.state"):
            actual_mask = stream.get_tensor(f"{feature}.mask").cpu()
            if not torch.equal(actual_mask, expected_mask):
                raise ValueError(f"unexpected pose/gripper normalization mask for {feature}")
        for feature in NORMALIZER_IMAGE_FEATURES:
            for stat_name, expected_values in IMAGENET_NORMALIZATION_STATS.items():
                actual = stream.get_tensor(f"{feature}.{stat_name}").cpu()
                expected = torch.tensor(expected_values, dtype=torch.float32)
                if not torch.equal(actual, expected):
                    raise ValueError(f"unexpected ImageNet {stat_name} tensor for {feature}")
        for feature in sorted(expected_dataset_features):
            feature_stats = dataset_stats.get(feature)
            if not isinstance(feature_stats, dict) or set(feature_stats) != set(NORMALIZER_FULL_STAT_NAMES):
                raise ValueError(f"dataset normalization-stat schema mismatch for {feature}")
            for stat_name in NORMALIZER_FULL_STAT_NAMES:
                key = f"{feature}.{stat_name}"
                expected = torch.tensor(feature_stats[stat_name], dtype=torch.float32)
                actual = stream.get_tensor(key).cpu()
                if not torch.equal(actual, expected):
                    raise ValueError(f"checkpoint normalization values differ from dataset stats for {key}")
        for key in keys:
            tensor = stream.get_tensor(key)
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise ValueError(f"non-finite normalization tensor: {path.name}/{key}")
    if action_count != valid_train_action_rows:
        raise ValueError("checkpoint action stats include padded terminal rows or omit valid rows")
    if state_count != train_frames or helper_count != train_frames:
        raise ValueError("checkpoint state/helper stats do not contain train-only rows")
    return {
        "tensor_count": len(keys),
        "action_count": int(action_count),
        "state_count": int(state_count),
        "helper_count": int(helper_count),
        "matches_dataset_stats": True,
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
        "job_name": "molmoact2-umi-currentrel-r6d-onset-v3-12k",
        "seed": 1000,
        "num_workers": 8,
        "batch_size": PER_RANK_BATCH_SIZE,
        "steps": 12_000,
        "env_eval_freq": 0,
        "log_freq": 10,
        "eval_steps": 1_000,
        "max_eval_samples": 0,
        "save_checkpoint": True,
        "save_freq": 1_000,
        "save_checkpoint_to_hub": False,
    }
    for key, expected in expected_train_scalars.items():
        if train_config.get(key) != expected:
            raise ValueError(f"checkpoint {step}: unexpected train config {key}")
    wandb_config = train_config.get("wandb", {})
    expected_wandb = {
        "enable": True,
        "entity": "aq-robotics",
        "project": "molmoact2-yam-oranges",
        "disable_artifact": True,
    }
    if not isinstance(wandb_config, dict):
        raise ValueError(f"checkpoint {step}: malformed Weights & Biases config")
    for key, expected in expected_wandb.items():
        if wandb_config.get(key) != expected:
            raise ValueError(f"checkpoint {step}: unexpected Weights & Biases {key}")
    dataset_config = train_config.get("dataset", {})
    if dataset_config.get("repo_id") != DATASET_REPO_ID:
        raise ValueError(f"checkpoint {step}: wrong dataset repo id")
    if dataset_config.get("root") != str(dataset_root):
        raise ValueError(f"checkpoint {step}: wrong dataset root")
    if dataset_config.get("eval_split") != 2 / 54:
        raise ValueError(f"checkpoint {step}: wrong holdout fraction")
    if dataset_config.get("use_imagenet_stats") is not True:
        raise ValueError(f"checkpoint {step}: ImageNet statistics must be enabled")
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
    dataset_stats = _load_json(dataset_root / "meta/stats.json")
    normalizer = _validate_normalizer(
        pretrained / "policy_preprocessor_step_3_molmoact2_masked_normalizer.safetensors",
        train_frames=train_frames,
        valid_train_action_rows=VALID_TRAIN_ACTION_ROWS,
        dataset_stats=dataset_stats,
    )
    unnormalizer = _validate_normalizer(
        pretrained / "policy_postprocessor_step_1_molmoact2_masked_unnormalizer.safetensors",
        train_frames=train_frames,
        valid_train_action_rows=VALID_TRAIN_ACTION_ROWS,
        dataset_stats=dataset_stats,
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
    if not last.is_symlink() or last.readlink() != Path("012000"):
        raise ValueError("checkpoints/last must be the relative symlink 012000")

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


def _common_report(
    args: argparse.Namespace,
    dataset: dict[str, Any],
    gate_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_id": "umi-yam-onset-v3-training-validation-v1",
        "status": "pass",
        "hardware_control_performed": False,
        "job_id": str(args.job_id),
        "dataset": dataset,
        "checkpoint_gate_plan": gate_plan,
        "code_manifest_sha256": _require_sha256(args.code_manifest_sha256, name="code manifest SHA-256"),
        "base_model": {
            "root": str(args.model_root.resolve(strict=True)),
            "revision": MODEL_REVISION,
        },
    }


def command_smoke(args: argparse.Namespace) -> None:
    dataset = validate_dataset_identity(args.dataset_root, args.dataset_manifest_sha256)
    gate_plan = validate_gate_plan(args.gate_plan, args.gate_plan_sha256)
    metrics = _parse_training_metrics(args.log, full=False)
    report = _common_report(args, dataset, gate_plan)
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
    gate_plan_sha = _require_sha256(args.gate_plan_sha256, name="gate plan SHA-256")
    if report.get("checkpoint_gate_plan", {}).get("sha256") != gate_plan_sha:
        raise ValueError("smoke report checkpoint-gate plan mismatch")
    print(json.dumps({"smoke_report_verified": True, "job_id": str(args.job_id)}, sort_keys=True))


def command_full(args: argparse.Namespace) -> None:
    dataset = validate_dataset_identity(args.dataset_root, args.dataset_manifest_sha256)
    gate_plan = validate_gate_plan(args.gate_plan, args.gate_plan_sha256)
    full = validate_full_run(args.run_root, args.log, dataset, args.model_root.resolve(strict=True))
    report = _common_report(args, dataset, gate_plan)
    report.update(
        {
            "kind": "seven-gpu-twelve-thousand-step-full",
            "world_size": WORLD_SIZE,
            "per_rank_batch_size": PER_RANK_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "sample_presentations": FULL_SAMPLE_PRESENTATIONS,
            "effective_train_frame_epochs": EFFECTIVE_TRAIN_FRAME_EPOCHS,
            "schedule_provenance": (
                "v1 optimizer, 12000-update, warmup/decay, and 1000-step checkpoint/eval cadence; "
                "seven-GPU global batch 28 intentionally changes sample exposure"
            ),
            "full_run": full,
            "selection_policy": SELECTION_POLICY,
        }
    )
    _write_report(args.report, report)
    print(json.dumps(report, sort_keys=True))


def _finite_number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{name} is outside its valid range")
    return parsed


def _validate_per_arm_motion_report(summary: dict[str, Any], *, step: int) -> None:
    motion = summary.get("per_arm_motion")
    if not isinstance(motion, dict) or set(motion) != {"left", "right"}:
        raise ValueError(f"checkpoint {step}: per-arm motion report must contain left and right")
    for arm in ("left", "right"):
        arm_report = motion[arm]
        if not isinstance(arm_report, dict) or set(arm_report) != set(GATE_MOTION_FIELDS):
            raise ValueError(f"checkpoint {step}/{arm}: motion diagnostic fields changed")
        for field in GATE_MOTION_FIELDS:
            values = arm_report[field]
            if not isinstance(values, list) or len(values) != GATE_CELLS_PER_CHECKPOINT:
                raise ValueError(f"checkpoint {step}/{arm}/{field}: expected exactly 10 cells")
            for index, value in enumerate(values):
                name = f"checkpoint {step}/{arm}/{field}[{index}]"
                if field == "endpoint_translation_direction_cosine_by_cell" and value is None:
                    continue
                parsed = _finite_number(value, name=name)
                if field == "endpoint_translation_direction_cosine_by_cell":
                    if not -1.0 <= parsed <= 1.0:
                        raise ValueError(f"{name} must be in [-1,1] or null")
                elif parsed < 0.0:
                    raise ValueError(f"{name} must be non-negative")


def _validate_gate_aggregate(
    gate: dict[str, Any],
    *,
    gate_plan_sha256: str,
    dataset_manifest_sha256: str,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    if gate.get("schema_id") != GATE_AGGREGATE_SCHEMA_ID:
        raise ValueError("unexpected onset-v3 gate aggregate schema")
    exact_values = {
        "gate_plan_sha256": gate_plan_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "candidate_checkpoint_steps": CHECKPOINT_STEPS,
        "holdout_episodes": GATE_HOLDOUT_EPISODES,
        "seeds": GATE_SEEDS,
        "execution_horizon_rows": GATE_EXECUTION_HORIZON,
        "num_flow_steps": GATE_NUM_FLOW_STEPS,
        "yam_start_absolute_driver_target": GATE_START_DRIVER_TARGET,
        "hardware_start_verified": False,
        "full_24_row_execution_performed": False,
    }
    for key, expected in exact_values.items():
        if gate.get(key) != expected:
            raise ValueError(f"gate aggregate {key!r} does not match the frozen plan")
    expected_cell_order = [
        {"seed": seed, "episode": episode} for seed in GATE_SEEDS for episode in GATE_HOLDOUT_EPISODES
    ]
    if gate.get("cell_order") != expected_cell_order:
        raise ValueError("gate aggregate cell order/coverage changed")
    for key in (
        "all_predeclared_cells_evaluated",
        "all_child_hashes_and_resolver_outputs_verified",
        "all_child_ik_and_behavior_summaries_recomputed",
    ):
        if gate.get(key) is not True:
            raise ValueError(f"gate aggregate did not verify {key}")
    if gate.get("checkpoint_selection_performed") is not False:
        raise ValueError("gate aggregate must leave checkpoint selection to full-holdout loss")

    summaries = gate.get("checkpoint_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(CHECKPOINT_STEPS):
        raise ValueError(f"gate aggregate must contain exactly {len(CHECKPOINT_STEPS)} checkpoint summaries")
    if any(not isinstance(summary, dict) or "checkpoint_step" not in summary for summary in summaries):
        raise ValueError("gate aggregate contains a malformed checkpoint summary")
    by_step = {int(summary["checkpoint_step"]): summary for summary in summaries}
    if sorted(by_step) != CHECKPOINT_STEPS or len(by_step) != len(summaries):
        raise ValueError("gate aggregate must cover each checkpoint exactly once")

    eligible_steps: list[int] = []
    for step in CHECKPOINT_STEPS:
        summary = by_step[step]
        if summary.get("evaluated_cell_count") != GATE_CELLS_PER_CHECKPOINT:
            raise ValueError(f"checkpoint {step}: did not evaluate all ten predeclared cells")
        strict_count = summary.get("strict_first_15_cell_pass_count")
        strict_pass = strict_count == GATE_CELLS_PER_CHECKPOINT
        if summary.get("all_10_first_15_strict_execution_cells_pass") is not strict_pass:
            raise ValueError(f"checkpoint {step}: strict execution summary is inconsistent")

        model_error = _finite_number(
            summary.get("pooled_model_normalized_pose_error"),
            name=f"checkpoint {step} pooled model pose error",
            minimum=0.0,
        )
        identity_error = _finite_number(
            summary.get("pooled_identity_normalized_pose_error"),
            name=f"checkpoint {step} pooled identity pose error",
            minimum=0.0,
        )
        if identity_error <= 0.0:
            raise ValueError(f"checkpoint {step}: identity pose error must be positive")
        improvement = 1.0 - model_error / identity_error
        stored_improvement = _finite_number(
            summary.get("relative_improvement_over_identity"),
            name=f"checkpoint {step} identity improvement",
        )
        if not math.isclose(stored_improvement, improvement, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"checkpoint {step}: identity-improvement arithmetic mismatch")
        margin_pass = improvement >= GATE_MINIMUM_IDENTITY_IMPROVEMENT
        if summary.get("physical_pose_error_margin_pass") is not margin_pass:
            raise ValueError(f"checkpoint {step}: physical pose-error margin flag mismatch")
        _validate_per_arm_motion_report(summary, step=step)

        eligible = strict_pass and margin_pass
        if summary.get("offline_advancement_eligible") is not eligible:
            raise ValueError(f"checkpoint {step}: offline eligibility flag mismatch")
        if summary.get("hardware_deployment_eligible") is not False:
            raise ValueError(f"checkpoint {step}: unverified start cannot be hardware eligible")
        if eligible:
            eligible_steps.append(step)
    if gate.get("offline_advancement_eligible_checkpoint_steps") != eligible_steps:
        raise ValueError("gate aggregate eligible checkpoint list is inconsistent")
    return eligible_steps, by_step


def command_select(args: argparse.Namespace) -> None:
    """Validate stage one and fail closed until every onset-eligible temporal cell exists."""

    gate_plan = validate_gate_plan(args.gate_plan, args.gate_plan_sha256)
    training = _load_json(args.training_report)
    gate = _load_json(args.gate_aggregate)
    if training.get("status") != "pass" or training.get("kind") != "seven-gpu-twelve-thousand-step-full":
        raise ValueError("selection requires a passing onset-v3 full-training validation report")
    if training.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("training report does not contain the frozen selection policy")
    dataset = training.get("dataset", {})
    if dataset.get("schema_id") != SCHEMA_ID:
        raise ValueError("training report does not bind the onset-v3 dataset")
    if training.get("checkpoint_gate_plan") != gate_plan:
        raise ValueError("training report and selection use different checkpoint-gate plans")

    dataset_manifest_sha = _require_sha256(
        str(dataset.get("artifact_manifest_sha256", "")),
        name="training-report dataset manifest SHA-256",
    )
    eligible_steps, _summaries = _validate_gate_aggregate(
        gate,
        gate_plan_sha256=gate_plan["sha256"],
        dataset_manifest_sha256=dataset_manifest_sha,
    )
    losses_raw = training.get("full_run", {}).get("metrics", {}).get("full_holdout_eval_loss", {})
    losses = {int(step): float(value) for step, value in losses_raw.items()}
    if sorted(losses) != CHECKPOINT_STEPS or not all(math.isfinite(value) for value in losses.values()):
        raise ValueError(f"training report must contain {len(CHECKPOINT_STEPS)} finite full-holdout losses")
    loss_ranked_onset_steps = sorted(eligible_steps, key=lambda step: (losses[step], step))
    report = {
        "schema_id": "umi-yam-onset-v3-checkpoint-selection-v1",
        "status": (
            "pending-temporal-teacher-forced-gate" if eligible_steps else "no-offline-eligible-checkpoint"
        ),
        "hardware_control_performed": False,
        "hardware_start_verified": False,
        "hardware_deployment_ready": False,
        "selection_policy": SELECTION_POLICY,
        "training_report": str(args.training_report.resolve(strict=True)),
        "training_report_sha256": _sha256(args.training_report),
        "gate_plan": gate_plan,
        "gate_aggregate": str(args.gate_aggregate.resolve(strict=True)),
        "gate_aggregate_sha256": _sha256(args.gate_aggregate),
        "offline_eligible_checkpoint_steps": eligible_steps,
        "onset_gate_eligible_checkpoint_steps": eligible_steps,
        "loss_ranked_onset_gate_eligible_checkpoint_steps": loss_ranked_onset_steps,
        "full_holdout_eval_loss": {str(step): losses[step] for step in CHECKPOINT_STEPS},
        "selected_checkpoint_step": None,
        "selected_checkpoint_gate_summary": None,
        "final_selection_performed": False,
        "temporal_gate_required_checkpoint_steps": eligible_steps,
        "temporal_teacher_forced_gate_passed": False,
        "supervised_start_verification_ready": False,
        "next_required_gate": (
            (
                "run the predeclared temporal teacher-forced gate on every onset-eligible "
                "checkpoint; all fixed-seed rows through each holdout's final contact plus 24 "
                "frames must pass independently recomputed strict IK/rate checks, then select "
                "lowest full-holdout loss only among checkpoints passing both gates"
            )
            if eligible_steps
            else "retrain or revise the frozen experiment; no checkpoint passed the onset gate"
        ),
    }
    _write_report(args.report, report)
    print(json.dumps(report, sort_keys=True))


def _add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--code-manifest-sha256", required=True)
    parser.add_argument("--gate-plan", type=Path, required=True)
    parser.add_argument("--gate-plan-sha256", required=True)
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
    verify_smoke.add_argument("--gate-plan-sha256", required=True)
    verify_smoke.set_defaults(func=command_verify_smoke)

    full = subparsers.add_parser("full")
    _add_identity_args(full)
    full.add_argument("--run-root", type=Path, required=True)
    full.add_argument("--log", type=Path, required=True)
    full.add_argument("--report", type=Path, required=True)
    full.set_defaults(func=command_full)

    select = subparsers.add_parser("select")
    select.add_argument("--training-report", type=Path, required=True)
    select.add_argument("--gate-plan", type=Path, required=True)
    select.add_argument("--gate-plan-sha256", required=True)
    select.add_argument("--gate-aggregate", type=Path, required=True)
    select.add_argument("--report", type=Path, required=True)
    select.set_defaults(func=command_select)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
