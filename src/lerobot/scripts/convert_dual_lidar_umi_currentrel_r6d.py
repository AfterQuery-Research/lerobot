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
"""Build the versioned dual-LiDAR UMI current-relative Rotation6D artifact.

This converter is intentionally observation-only.  The curated source omits an
``action`` column, and targets are derived from future measured/optimized TCP
observations using homogeneous SE(3) composition.  No rotvec subtraction and no
scene-derived transform is used.

The regular per-frame ``action`` column stores the one-step target for schema
inspection.  Training uses :class:`UmiCurrentRelativeR6dDataset`, which ignores
that convenience row and constructs all 24 query-anchored future rows after the
sampler selects a query frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_HELPER_DIM,
    UMI_CURRENTREL_HORIZON,
    UMI_CURRENTREL_METADATA_PATH,
    UMI_CURRENTREL_SCHEMA_ID,
    UMI_CURRENTREL_SPLIT_PATH,
    UMI_CURRENTREL_STATE_DIM,
    UMI_TCP_WINDOW_KEY,
    matrix_to_rotation_6d,
    pack_tcp_and_gripper,
    validate_rigid_transform,
)

SOURCE_REPO_ID = "brandonyang/dual-lidar-umi-filtered"
SOURCE_REVISION = "ad1db459b264ffacffce5c1fcee9d517bf795926"
UPSTREAM_CAPTURE_REVISION = "f54456b93896ea0f9c6c81b8bc54c96a67828b0b"
OUTPUT_DATASET_NAME = "dual-lidar-umi-currentrel-r6d-v1"
TASK = "Put all oranges in the bowl"

EXPECTED_EPISODES = 54
EXPECTED_SOURCE_FRAMES = 52_067
TRAIN_EPISODES = list(range(52))
VALIDATION_EPISODES = [52, 53]

# Seven optimized trajectories contain a large, non-physical oscillation only
# after the task is complete. Keep every episode, but retain each prefix through
# its final material gripper transition plus one full 24-frame policy horizon.
# Values are retained lengths, not final frame indices.
TAIL_RETAINED_LENGTHS = {2: 627, 5: 689, 19: 816, 25: 904, 30: 1_144, 34: 939, 47: 707}
EXPECTED_TRAIN_FRAMES = 48_997
EXPECTED_VALIDATION_FRAMES = 1_937
EXPECTED_OUTPUT_FRAMES = EXPECTED_TRAIN_FRAMES + EXPECTED_VALIDATION_FRAMES

GRIPPER_WIDTH_NORMALIZER_MM = 125.0
GRIPPER_OUTLIER_THRESHOLD_MM = 126.0

# The pinned filtered dataset stores the collaborator-confirmed jaw-centre EE
# pose, not a raw LiDAR-body pose.  Keep the identity explicit in the artifact
# provenance so a historical LiDAR lever arm cannot be applied a second time.
STORED_POSE_T_TCP = np.eye(4, dtype=np.float64)

POSE_R6D_NAMES = [
    "relative_x_m",
    "relative_y_m",
    "relative_z_m",
    "relative_r6d_col0_x",
    "relative_r6d_col0_y",
    "relative_r6d_col0_z",
    "relative_r6d_col1_x",
    "relative_r6d_col1_y",
    "relative_r6d_col1_z",
]
STATE_NAMES = (
    [f"left_previous_{name}" for name in POSE_R6D_NAMES]
    + ["left_current_gripper"]
    + [f"right_previous_{name}" for name in POSE_R6D_NAMES]
    + ["right_current_gripper"]
)
ACTION_NAMES = (
    [f"left_future_{name}" for name in POSE_R6D_NAMES]
    + ["left_future_gripper"]
    + [f"right_future_{name}" for name in POSE_R6D_NAMES]
    + ["right_future_gripper"]
)
HELPER_NAMES = [
    "left_tcp_x_m",
    "left_tcp_y_m",
    "left_tcp_z_m",
    "left_tcp_qx",
    "left_tcp_qy",
    "left_tcp_qz",
    "left_tcp_qw",
    "left_gripper",
] + [
    "right_tcp_x_m",
    "right_tcp_y_m",
    "right_tcp_z_m",
    "right_tcp_qx",
    "right_tcp_qy",
    "right_tcp_qz",
    "right_tcp_qw",
    "right_gripper",
]

QUANTILES = {"q01": 1, "q10": 10, "q50": 50, "q90": 90, "q99": 99}


def _fixed_size_float_array(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"expected a two-dimensional width-{width} array, got {array.shape}")
    return pa.FixedSizeListArray.from_arrays(pa.array(array.reshape(-1), type=pa.float32()), width)


def _stats_block(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"statistics input must be a finite nonempty matrix, got {array.shape}")
    result: dict[str, list[float] | list[int]] = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }
    for name, percentile in QUANTILES.items():
        result[name] = np.percentile(array, percentile, axis=0).tolist()
    return result


def _clean_gripper_width(width_mm: np.ndarray) -> np.ndarray:
    """Repair impossible AprilTag outliers, then apply the physical 125 mm range."""

    values = np.asarray(width_mm, dtype=np.float64).copy()
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("gripper width must be a finite nonempty vector")
    bad = values > GRIPPER_OUTLIER_THRESHOLD_MM
    if bad.any():
        good_indices = np.flatnonzero(~bad)
        if good_indices.size < 2:
            raise ValueError("not enough valid gripper samples to interpolate outliers")
        bad_indices = np.flatnonzero(bad)
        values[bad] = np.interp(bad_indices, good_indices, values[~bad])
    return np.clip(values / GRIPPER_WIDTH_NORMALIZER_MM, 0.0, 1.0)


def stored_pose_track_to_tcp(stored_pose: np.ndarray) -> np.ndarray:
    """Decode the stored jaw-centre pose track without an additional lever arm."""

    values = np.asarray(stored_pose, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError(f"stored pose track must have finite shape (N, 6), got {values.shape}")
    validate_rigid_transform(STORED_POSE_T_TCP, name="STORED_POSE_T_TCP")
    stored = np.broadcast_to(np.eye(4, dtype=np.float64), (values.shape[0], 4, 4)).copy()
    stored[:, :3, :3] = Rotation.from_rotvec(values[:, 3:]).as_matrix()
    stored[:, :3, 3] = values[:, :3]
    tcp = stored @ STORED_POSE_T_TCP
    for index in (0, values.shape[0] - 1):
        validate_rigid_transform(tcp[index], name=f"tcp[{index}]")
    return tcp


def build_episode_arrays(table: pa.Table) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return state, one-step action, helper, and full target chunks for one episode."""

    if "action" in table.column_names:
        raise ValueError("the filtered observation-only source must not contain an action column")
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    if state.ndim != 2 or state.shape[1] != 12:
        raise ValueError(f"source observation.state must have shape (N, 12), got {state.shape}")
    left_tcp = stored_pose_track_to_tcp(state[:, :6])
    right_tcp = stored_pose_track_to_tcp(state[:, 6:])
    tcp = np.stack((left_tcp, right_tcp), axis=1)
    gripper = np.stack(
        (
            _clean_gripper_width(table["observation.gripper_width.umi1"].to_numpy()),
            _clean_gripper_width(table["observation.gripper_width.umi2"].to_numpy()),
        ),
        axis=-1,
    )

    frame_count = state.shape[0]
    state20 = np.empty((frame_count, UMI_CURRENTREL_STATE_DIM), dtype=np.float64)
    chunks = np.empty((frame_count, UMI_CURRENTREL_HORIZON, UMI_CURRENTREL_STATE_DIM), dtype=np.float64)

    def encode_batch(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
        anchor_rotation = anchor[:, :3, :3]
        target_rotation = target[:, :3, :3]
        relative_rotation = np.einsum("nji,njk->nik", anchor_rotation, target_rotation)
        displacement = target[:, :3, 3] - anchor[:, :3, 3]
        relative_translation = np.einsum("nji,nj->ni", anchor_rotation, displacement)
        return np.concatenate((relative_translation, matrix_to_rotation_6d(relative_rotation)), axis=-1)

    current_indices = np.arange(frame_count)
    previous_indices = np.maximum(current_indices - 1, 0)
    for arm, start in ((0, 0), (1, 10)):
        state20[:, start : start + 9] = encode_batch(tcp[:, arm], tcp[previous_indices, arm])
        state20[:, start + 9] = gripper[:, arm]
        state20[0, start : start + 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for row, offset in enumerate(range(1, UMI_CURRENTREL_HORIZON + 1)):
        future_indices = np.minimum(current_indices + offset, frame_count - 1)
        for arm, start in ((0, 0), (1, 10)):
            chunks[:, row, start : start + 9] = encode_batch(tcp[:, arm], tcp[future_indices, arm])
            chunks[:, row, start + 9] = gripper[future_indices, arm]
            chunks[-1, row, start : start + 9] = [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]

    state20 = state20.astype(np.float32)
    chunks = chunks.astype(np.float32)
    action20 = chunks[:, 0].copy()
    helper = pack_tcp_and_gripper(tcp, gripper).astype(np.float32)

    if state20.shape != (frame_count, UMI_CURRENTREL_STATE_DIM):
        raise RuntimeError("state shape invariant failed")
    if chunks.shape != (frame_count, UMI_CURRENTREL_HORIZON, UMI_CURRENTREL_STATE_DIM):
        raise RuntimeError("action chunk shape invariant failed")
    if helper.shape != (frame_count, UMI_CURRENTREL_HELPER_DIM):
        raise RuntimeError("helper shape invariant failed")
    if not (np.isfinite(state20).all() and np.isfinite(chunks).all() and np.isfinite(helper).all()):
        raise RuntimeError("converted episode contains non-finite values")
    return state20, action20, helper, chunks


def _rewrite_episode_metadata(
    source_path: Path,
    output_path: Path,
    *,
    state_stats: dict,
    action_stats: dict,
    helper_stats: dict,
    scalar_stats: dict[str, dict],
    frame_count: int,
    dataset_from_index: int,
) -> None:
    table = pq.read_table(source_path)
    drop_prefixes = (
        "stats/observation.state/",
        "stats/action/",
        "stats/observation.gripper_width.umi1/",
        "stats/observation.gripper_width.umi2/",
        f"stats/{UMI_TCP_WINDOW_KEY}/",
        "stats/timestamp/",
        "stats/frame_index/",
        "stats/episode_index/",
        "stats/index/",
        "stats/task_index/",
    )
    columns = {
        name: table[name]
        for name in table.column_names
        if not any(name.startswith(prefix) for prefix in drop_prefixes)
    }
    if table.num_rows != 1:
        raise ValueError(f"episode metadata must contain exactly one row, got {table.num_rows}")
    columns["tasks"] = pa.array([[TASK]], type=pa.list_(pa.string()))
    columns["length"] = pa.array([frame_count], type=pa.int64())
    columns["dataset_from_index"] = pa.array([dataset_from_index], type=pa.int64())
    columns["dataset_to_index"] = pa.array([dataset_from_index + frame_count], type=pa.int64())
    for key in tuple(columns):
        if key.startswith("videos/") and key.endswith("/to_timestamp"):
            columns[key] = pa.array([frame_count / 30.0], type=pa.float64())
    for feature, feature_stats in (
        ("observation.state", state_stats),
        ("action", action_stats),
        (UMI_TCP_WINDOW_KEY, helper_stats),
        *scalar_stats.items(),
    ):
        for statistic, value in feature_stats.items():
            value_type = pa.list_(pa.int64()) if statistic == "count" else pa.list_(pa.float64())
            columns[f"stats/{feature}/{statistic}"] = pa.array([value], type=value_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), output_path)


def _semantic_metadata() -> dict:
    return {
        "schema_id": UMI_CURRENTREL_SCHEMA_ID,
        "schema_version": 1,
        "dataset_name": OUTPUT_DATASET_NAME,
        "fps": 30,
        "action_horizon": UMI_CURRENTREL_HORIZON,
        "source": {
            "repository": SOURCE_REPO_ID,
            "revision": SOURCE_REVISION,
            "episodes": EXPECTED_EPISODES,
            "frames": EXPECTED_SOURCE_FRAMES,
            "action_column_required": False,
            "state_semantics_from_source_metadata": (
                "continuous optimizer achieved YAM FK pose mapped back into the original 12-D "
                "UMI Cartesian observation.state convention"
            ),
            "filtering_declared_upstream_revision": "main (not pinned by the filtered repository)",
            "upstream_capture_revision_used_for_coordinate_provenance": UPSTREAM_CAPTURE_REVISION,
        },
        "arm_and_camera_order": {
            "umi1": "left",
            "umi2": "right",
            "images": ["observation.images.umi1", "observation.images.umi2"],
        },
        "stored_pose": {
            "translation_units": "metres",
            "rotation_storage": "rotation vector",
            "rotation_units": "radians",
            "rotation_convention": "active right-handed rotation",
            "matrix_meaning": "T_episode_jaw_tcp",
            "frame": "collaborator-confirmed jaw-centre end-effector pose in UMI axes",
        },
        "stored_pose_to_tcp": {
            "symbol": "T_stored_pose_tcp",
            "direction": "jaw-centre TCP coordinates to the stored jaw-centre pose frame",
            "application": "T_episode_tcp = T_episode_stored_pose @ T_stored_pose_tcp",
            "matrix": STORED_POSE_T_TCP.tolist(),
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "identity": True,
            "evidence": {
                "filtered_dataset_revision": SOURCE_REVISION,
                "filtered_dataset_state_semantics": (
                    "continuous optimizer achieved YAM FK pose mapped back into the original "
                    "12-D UMI Cartesian observation.state convention"
                ),
                "hardware_collaborator_confirmation": (
                    "the stored 6-DoF block is already the end-effector/jaw-centre pose"
                ),
            },
            "historical_capture_warning": (
                "do not apply the historical raw-LiDAR lever arm [-0.10479, 0, 0.22244] m "
                "to this filtered EE-pose artifact; that would double-transform the TCP"
            ),
            "scene_fitted": False,
        },
        "rotation6d": {
            "convention": "first two columns of active rotation matrix, concatenated column-major",
            "layout": ["R00", "R10", "R20", "R01", "R11", "R21"],
            "decoder": "Gram-Schmidt columns; third=cross(first,second)",
            "identity": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
        "state_semantics": "history_i(t) = inverse(T_tcp_i(t)) @ T_tcp_i(t-1)",
        "action_semantics": "action_i(t,k) = inverse(T_tcp_i(t)) @ T_tcp_i(t+k), k=1..24",
        "padding_semantics": "supervise_clamped_future_rows",
        "terminal_padding": "clamp t+k to final episode frame; all rows retain query-time T_tcp_i(t)",
        "gripper": {
            "input_units": "millimetres",
            "normalization": "clip(width_mm / 125.0, 0, 1)",
            "measurement": "per-frame ArUco jaw-tag separation",
            "normalizer_mm": GRIPPER_WIDTH_NORMALIZER_MM,
            "larger_means": "open",
            "outlier_repair": "values >126 mm linearly interpolated from temporal neighbours",
            "runtime_mapping": (
                "for each physical UMI device, measure tag-centre separation at fully closed "
                "and fully open; convert model_value*125 mm affinely between those endpoints "
                "to the calibrated BiYAM 0..1 motor command"
            ),
            "runtime_state_mapping": (
                "map measured BiYAM 0..1 through the same endpoints, then divide the resulting "
                "tag-centre separation by 125 for the model state"
            ),
            "runtime_mapping_requires_verified_endpoint_calibration": True,
            "runtime_endpoint_schema_version": 2,
            "runtime_endpoint_assignments": {"umi1": "left", "umi2": "right"},
            "runtime_endpoint_fields": [
                "closed_width_mm",
                "open_width_mm",
                "verified",
                "dataset_device",
                "assigned_arm",
                "device_id",
                "evidence_uri",
                "evidence_sha256",
                "detector_config_id",
                "detector_config_sha256",
                "fisheye_calibration_id",
                "fisheye_calibration_sha256",
                "geometry_config_id",
                "geometry_config_sha256",
            ],
            "runtime_driver_semantics": "0=closed, 1=open",
            "mapping_warning": (
                "ArUco tag-centre separation/125 is not numerically identical to the YAM motor "
                "endpoint fraction even when both embodiments use the same gripper mechanism"
            ),
            "do_not_fit_demo_extrema": (
                "observed minima include oranges held between the jaws and are not closed endpoints"
            ),
            "scene_fitted": False,
        },
        "tail_cleanup": {
            "all_training_episodes_retained": True,
            "retained_lengths_by_output_episode": {
                str(index): length for index, length in TAIL_RETAINED_LENGTHS.items()
            },
            "removed_training_frames": 1_133,
            "training_frames_after_cleanup": EXPECTED_TRAIN_FRAMES,
            "validation_frames": EXPECTED_VALIDATION_FRAMES,
            "output_frames": EXPECTED_OUTPUT_FRAMES,
            "rule": (
                "remove optimizer-only post-task suffix after final material gripper transition, "
                "while retaining one complete 24-frame policy horizon"
            ),
            "video_handling": "full source videos remain linked; metadata exposes only each retained prefix",
        },
        "dual_arm_common_frame": None,
        "dual_arm_common_frame_note": (
            "v1 deliberately carries no invented inter-gripper transform; "
            "runtime controller handles collision"
        ),
        "statistics": {
            "scope": "training episodes 0..51 only",
            "state_samples": "one 20-D row per training frame",
            "action_samples": "all 24 query-anchored 20-D rows per training frame",
            "fresh": True,
        },
    }


def convert(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    info_path = source_root / "meta/info.json"
    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    if source_info["total_episodes"] != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} source episodes")
    if source_info["total_frames"] != EXPECTED_SOURCE_FRAMES:
        raise ValueError(f"expected {EXPECTED_SOURCE_FRAMES} source frames")
    if "action" in source_info["features"]:
        raise ValueError("filtered source must be observation-only")
    if source_info["features"]["observation.state"]["shape"] != [12]:
        raise ValueError("filtered source must retain the 12-D Cartesian observation schema")

    data_files = sorted((source_root / "data").rglob("*.parquet"))
    episode_files = sorted((source_root / "meta/episodes").rglob("*.parquet"))
    if len(data_files) != EXPECTED_EPISODES or len(episode_files) != EXPECTED_EPISODES:
        raise ValueError(
            f"expected one data and metadata file per episode, got {len(data_files)}/{len(episode_files)}"
        )

    (output_root / "data/chunk-000").mkdir(parents=True)
    (output_root / "meta/episodes/chunk-000").mkdir(parents=True)
    (output_root / "videos").symlink_to(source_root / "videos", target_is_directory=True)

    train_states: list[np.ndarray] = []
    train_actions: list[np.ndarray] = []
    train_helpers: list[np.ndarray] = []
    all_scalars: dict[str, list[np.ndarray]] = {
        key: [] for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    total_source_frames = 0
    total_output_frames = 0
    for data_path, episode_path in zip(data_files, episode_files, strict=True):
        table = pq.read_table(data_path)
        episode_index = int(table["episode_index"][0].as_py())
        expected_index = int(data_path.stem.split("-")[-1])
        if episode_index != expected_index:
            raise ValueError(f"episode/file mismatch: {episode_index} vs {data_path}")
        source_frame_count = len(table)
        total_source_frames += source_frame_count
        retained_length = TAIL_RETAINED_LENGTHS.get(episode_index, source_frame_count)
        if not 1 <= retained_length <= source_frame_count:
            raise ValueError(
                f"invalid retained length {retained_length} for episode {episode_index} "
                f"with {source_frame_count} source frames"
            )
        table = table.slice(0, retained_length)
        state20, action20, helper, chunks = build_episode_arrays(table)
        frame_count = len(table)
        dataset_from_index = total_output_frames
        contiguous_index = np.arange(dataset_from_index, dataset_from_index + frame_count, dtype=np.int64)
        total_output_frames += frame_count

        scalar_values = {
            "timestamp": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64).reshape(-1, 1),
            "frame_index": np.asarray(table["frame_index"].to_numpy(), dtype=np.float64).reshape(-1, 1),
            "episode_index": np.asarray(table["episode_index"].to_numpy(), dtype=np.float64).reshape(-1, 1),
            "index": contiguous_index.astype(np.float64).reshape(-1, 1),
            "task_index": np.asarray(table["task_index"].to_numpy(), dtype=np.float64).reshape(-1, 1),
        }
        for key, values in scalar_values.items():
            all_scalars[key].append(values)

        converted = pa.table(
            {
                "observation.state": _fixed_size_float_array(state20, UMI_CURRENTREL_STATE_DIM),
                "action": _fixed_size_float_array(action20, UMI_CURRENTREL_STATE_DIM),
                UMI_TCP_WINDOW_KEY: _fixed_size_float_array(helper, UMI_CURRENTREL_HELPER_DIM),
                "timestamp": table["timestamp"],
                "frame_index": table["frame_index"],
                "episode_index": table["episode_index"],
                "index": pa.array(contiguous_index, type=pa.int64()),
                "task_index": table["task_index"],
            }
        )
        output_data_path = output_root / "data/chunk-000" / data_path.name
        pq.write_table(converted, output_data_path)

        episode_state_stats = _stats_block(state20)
        episode_action_stats = _stats_block(chunks.reshape(-1, UMI_CURRENTREL_STATE_DIM))
        episode_helper_stats = _stats_block(helper)
        _rewrite_episode_metadata(
            episode_path,
            output_root / "meta/episodes/chunk-000" / episode_path.name,
            state_stats=episode_state_stats,
            action_stats=episode_action_stats,
            helper_stats=episode_helper_stats,
            scalar_stats={key: _stats_block(values) for key, values in scalar_values.items()},
            frame_count=frame_count,
            dataset_from_index=dataset_from_index,
        )

        if episode_index in TRAIN_EPISODES:
            train_states.append(state20)
            train_actions.append(chunks.reshape(-1, UMI_CURRENTREL_STATE_DIM))
            train_helpers.append(helper)
        if episode_index % 10 == 0:
            trim_note = f" (trimmed from {source_frame_count})" if frame_count != source_frame_count else ""
            print(f"converted episode {episode_index:02d}: {frame_count} frames{trim_note}", flush=True)

    if total_source_frames != EXPECTED_SOURCE_FRAMES:
        raise RuntimeError(f"read {total_source_frames} source frames, expected {EXPECTED_SOURCE_FRAMES}")
    if total_output_frames != EXPECTED_OUTPUT_FRAMES:
        raise RuntimeError(f"converted {total_output_frames} frames, expected {EXPECTED_OUTPUT_FRAMES}")

    task_frame = pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task"))
    task_frame.to_parquet(output_root / "meta/tasks.parquet")

    features = source_info["features"]
    features.pop("observation.gripper_width.umi1")
    features.pop("observation.gripper_width.umi2")
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [UMI_CURRENTREL_STATE_DIM],
        "names": STATE_NAMES,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [UMI_CURRENTREL_STATE_DIM],
        "names": ACTION_NAMES,
    }
    features[UMI_TCP_WINDOW_KEY] = {
        "dtype": "float32",
        "shape": [UMI_CURRENTREL_HELPER_DIM],
        "names": HELPER_NAMES,
    }
    source_info["splits"] = {"train": "0:52", "validation": "52:54"}
    source_info["robot_type"] = "bimanual UMI grippers mounted on YAM"
    source_info["total_frames"] = EXPECTED_OUTPUT_FRAMES
    (output_root / "meta/info.json").write_text(json.dumps(source_info, indent=2) + "\n", encoding="utf-8")

    source_stats = json.loads((source_root / "meta/stats.json").read_text(encoding="utf-8"))
    source_stats.pop("observation.gripper_width.umi1", None)
    source_stats.pop("observation.gripper_width.umi2", None)
    source_stats.pop("action", None)
    source_stats["observation.state"] = _stats_block(np.concatenate(train_states, axis=0))
    source_stats["action"] = _stats_block(np.concatenate(train_actions, axis=0))
    source_stats[UMI_TCP_WINDOW_KEY] = _stats_block(np.concatenate(train_helpers, axis=0))
    for key, values in all_scalars.items():
        source_stats[key] = _stats_block(np.concatenate(values, axis=0))
    (output_root / "meta/stats.json").write_text(json.dumps(source_stats, indent=2) + "\n", encoding="utf-8")

    split_manifest = {
        "schema_id": UMI_CURRENTREL_SCHEMA_ID,
        "split_unit": "episode",
        "source_revision": SOURCE_REVISION,
        "train_episodes": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "selection": "deterministic contiguous split requested by user; no frame-level randomization",
        "tail_retained_lengths": {str(index): length for index, length in TAIL_RETAINED_LENGTHS.items()},
        "train_frames": EXPECTED_TRAIN_FRAMES,
        "validation_frames": EXPECTED_VALIDATION_FRAMES,
    }
    (output_root / UMI_CURRENTREL_SPLIT_PATH).write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / UMI_CURRENTREL_METADATA_PATH).write_text(
        json.dumps(_semantic_metadata(), indent=2) + "\n", encoding="utf-8"
    )

    filtering_path = source_root / "meta/filtering.json"
    if filtering_path.exists():
        shutil.copy2(filtering_path, output_root / "meta/source_filtering.json")
    readme = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
---

# {OUTPUT_DATASET_NAME}

Query-anchored jaw-centre TCP targets derived from `{SOURCE_REPO_ID}` at pinned revision
`{SOURCE_REVISION}`. Images are linked to that source snapshot, not duplicated.

For every query frame `t`, all 24 target rows use
`inverse(T_tcp(t)) @ T_tcp(t+k)` for `k=1..24`. Pose rotation uses the first two columns of
the active rotation matrix as Rotation6D. See `meta/{UMI_CURRENTREL_METADATA_PATH.name}` for
the complete transform direction, units, provenance, feature order, and statistics scope.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {output_root}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert(args.source_root, args.output_root)


if __name__ == "__main__":
    main()
