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
"""Build the immutable onset-v3 dual-LiDAR UMI current-relative artifact.

V3 preserves v2's scene-free, sustained-motion onset and all SE(3), camera,
split, and gripper contracts. It adds three audited *training-only* source
suffix cuts and excludes every beyond-episode k=1..24 target from action
statistics and MolmoAct2 supervision. It does not smooth or retime trajectories.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_HELPER_DIM,
    UMI_CURRENTREL_HORIZON,
    UMI_CURRENTREL_ONSET_V3_METADATA_PATH,
    UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
    UMI_CURRENTREL_SPLIT_PATH,
    UMI_CURRENTREL_STATE_DIM,
    UMI_TCP_WINDOW_KEY,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d import (
    ACTION_NAMES,
    EXPECTED_EPISODES,
    EXPECTED_SOURCE_FRAMES,
    HELPER_NAMES,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    STATE_NAMES,
    TAIL_RETAINED_LENGTHS,
    TASK,
    TRAIN_EPISODES,
    VALIDATION_EPISODES,
    _clean_gripper_width,
    _fixed_size_float_array,
    _stats_block,
    build_episode_arrays,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v2 import (
    EXPECTED_ONSET_TRAIN_FRAMES as EXPECTED_V2_TRAIN_FRAMES,
    EXPECTED_ONSET_VALIDATION_FRAMES as EXPECTED_V2_VALIDATION_FRAMES,
    FPS,
    find_motion_onset_in_table,
    semantic_metadata as _v2_semantic_metadata,
)

OUTPUT_DATASET_NAME = "dual-lidar-umi-currentrel-r6d-onset-v3"
OUTPUT_REPO_ID = f"brandonyang/{OUTPUT_DATASET_NAME}"

# These are source-frame end-exclusive indices, before onset reindexing. They
# remove an isolated optimizer/video mismatch suffix cluster from training only.
SOURCE_SUFFIX_END_FRAME_EXCLUSIVE = {0: 1_126, 40: 792, 46: 791}
SUFFIX_LAST_CONTACT_FRAME = {0: 1_095, 40: 758, 46: 757}
SUFFIX_CONTACT_CLEARANCE_FRAMES = {0: 30, 40: 33, 46: 33}
EXPECTED_V3_RETAINED_LENGTHS = {0: 1_088, 40: 765, 46: 756}

EXPECTED_TRAIN_FRAMES = 47_237
EXPECTED_VALIDATION_FRAMES = 1_884
EXPECTED_OUTPUT_FRAMES = EXPECTED_TRAIN_FRAMES + EXPECTED_VALIDATION_FRAMES
EXPECTED_REMOVED_FROM_V2_TRAIN_FRAMES = 381

# With action rows k=1..24, an episode contributes 1+...+24=300 padded
# positions. All 54 retained episodes are longer than the 24-row horizon.
PADDED_ACTION_POSITIONS_PER_EPISODE = UMI_CURRENTREL_HORIZON * (UMI_CURRENTREL_HORIZON + 1) // 2
EXPECTED_VALID_TRAIN_ACTION_POSITIONS = 1_118_088
EXPECTED_VALID_VALIDATION_ACTION_POSITIONS = 44_616
EXPECTED_VALID_ACTION_POSITIONS = 1_162_704

# Bind conversion to the exact downloaded source snapshot used for v1/v2.
EXPECTED_SOURCE_INFO_SHA256 = "252cb2f9503383716e19f89f199036a08ebd0c6f20be6ccf80d0bc32221ff3f8"
EXPECTED_SOURCE_FILTERING_SHA256 = "f4cc85914f889ffb5b9cf82622cff7051c9f18683dc260416e184206ca430f5e"
EXPECTED_VIDEO_FILES = EXPECTED_EPISODES * 2
EXPECTED_VIDEO_BYTES = 704_765_539
EXPECTED_OPTIMIZER_CONTACT_ANCHORS = 883


@dataclass(frozen=True)
class OnsetV3Episode:
    """One converted episode plus source-index and padding provenance."""

    episode_index: int
    source_frame_count: int
    v2_tail_end_frame_exclusive: int
    tail_end_frame_exclusive: int
    onset_frame: int
    dataset_from_index: int
    table: pa.Table
    state: np.ndarray
    action: np.ndarray
    helper: np.ndarray
    chunks: np.ndarray
    action_is_pad: np.ndarray
    gripper_event_frame_counts: tuple[int, int]
    last_gripper_event_frames: tuple[int | None, int | None]
    scalar_values: dict[str, np.ndarray]

    @property
    def frame_count(self) -> int:
        return len(self.table)

    @property
    def valid_action_rows(self) -> np.ndarray:
        return self.chunks[~self.action_is_pad]

    @property
    def valid_action_position_count(self) -> int:
        return int((~self.action_is_pad).sum())


class TrainOnlyStatsAccumulator:
    """Accumulate train-only stats while removing padded future targets."""

    def __init__(self, *, expected_action_count: int | None = EXPECTED_VALID_TRAIN_ACTION_POSITIONS) -> None:
        self.expected_action_count = expected_action_count
        self._values: dict[str, list[np.ndarray]] = {
            "observation.state": [],
            "action": [],
            UMI_TCP_WINDOW_KEY: [],
            "timestamp": [],
            "frame_index": [],
            "episode_index": [],
            "index": [],
            "task_index": [],
        }

    def add(self, episode: OnsetV3Episode) -> None:
        if episode.episode_index not in TRAIN_EPISODES:
            return
        self._values["observation.state"].append(episode.state)
        self._values["action"].append(episode.valid_action_rows)
        self._values[UMI_TCP_WINDOW_KEY].append(episode.helper)
        for key, values in episode.scalar_values.items():
            self._values[key].append(values)

    def finalize(self) -> dict[str, dict[str, list[float] | list[int]]]:
        empty = [key for key, values in self._values.items() if not values]
        if empty:
            raise RuntimeError(f"no training values accumulated for {empty}")
        stats = {key: _stats_block(np.concatenate(values, axis=0)) for key, values in self._values.items()}
        if self.expected_action_count is not None and stats["action"]["count"] != [
            self.expected_action_count
        ]:
            raise RuntimeError(
                "valid training action-stat count changed: "
                f"got {stats['action']['count']}, expected {[self.expected_action_count]}"
            )
        return stats


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_padding_mask(frame_count: int) -> np.ndarray:
    """Return the exact (query, k) mask for beyond-end k=1..24 rows."""

    if frame_count <= UMI_CURRENTREL_HORIZON:
        raise ValueError(f"episode needs more than {UMI_CURRENTREL_HORIZON} frames, got {frame_count}")
    query = np.arange(frame_count, dtype=np.int64)[:, None]
    offsets = np.arange(1, UMI_CURRENTREL_HORIZON + 1, dtype=np.int64)[None, :]
    mask = query + offsets >= frame_count
    if mask.dtype != np.bool_ or mask.shape != (frame_count, UMI_CURRENTREL_HORIZON):
        raise RuntimeError(f"action padding mask shape/dtype invariant failed: {mask.shape}/{mask.dtype}")
    if int(mask.sum()) != PADDED_ACTION_POSITIONS_PER_EPISODE:
        raise RuntimeError(
            f"expected {PADDED_ACTION_POSITIONS_PER_EPISODE} padded targets, got {int(mask.sum())}"
        )
    return mask


def _gripper_event_mask(signal: np.ndarray) -> np.ndarray:
    """Mirror the locked task-independent event detector used by replay audits."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("gripper signal must be a finite nonempty vector")
    active_edges = np.flatnonzero(np.abs(np.diff(values)) * FPS >= 0.04)
    event_mask = np.zeros(len(values), dtype=bool)
    if active_edges.size == 0:
        return event_mask
    max_gap = max(1, int(round(0.12 * FPS)))
    edge_groups: list[tuple[int, int]] = []
    start = previous = int(active_edges[0])
    for edge_value in active_edges[1:]:
        edge = int(edge_value)
        if edge - previous > max_gap:
            edge_groups.append((start, previous + 1))
            start = edge
        previous = edge
    edge_groups.append((start, previous + 1))
    padding = int(round(0.10 * FPS))
    for start, end in edge_groups:
        if abs(float(values[end] - values[start])) < 0.08:
            continue
        event_mask[max(0, start - padding) : min(len(values) - 1, end + padding) + 1] = True
    return event_mask


def _validate_preserved_gripper_events(
    table: pa.Table,
    *,
    tail_end_frame_exclusive: int,
) -> tuple[tuple[int, int], tuple[int | None, int | None]]:
    counts: list[int] = []
    last_frames: list[int | None] = []
    for feature in ("observation.gripper_width.umi1", "observation.gripper_width.umi2"):
        signal = _clean_gripper_width(table[feature].to_numpy())
        event_frames = np.flatnonzero(_gripper_event_mask(signal))
        if event_frames.size and int(event_frames[-1]) >= tail_end_frame_exclusive:
            raise ValueError(
                f"v3 suffix cut {tail_end_frame_exclusive} removes {feature} event frame "
                f"{int(event_frames[-1])}"
            )
        counts.append(int(event_frames.size))
        last_frames.append(int(event_frames[-1]) if event_frames.size else None)
    return (counts[0], counts[1]), (last_frames[0], last_frames[1])


def _v2_tail_end(episode_index: int, source_frame_count: int) -> int:
    end = TAIL_RETAINED_LENGTHS.get(episode_index, source_frame_count)
    if not 1 <= end <= source_frame_count:
        raise ValueError(
            f"invalid v2 tail end {end} for episode {episode_index} with {source_frame_count} frames"
        )
    return end


def _v3_tail_end(episode_index: int, source_frame_count: int) -> tuple[int, int]:
    v2_end = _v2_tail_end(episode_index, source_frame_count)
    if episode_index not in SOURCE_SUFFIX_END_FRAME_EXCLUSIVE:
        return v2_end, v2_end
    if episode_index not in TRAIN_EPISODES:
        raise ValueError(f"v3 suffix cut unexpectedly targets holdout episode {episode_index}")
    end = SOURCE_SUFFIX_END_FRAME_EXCLUSIVE[episode_index]
    if not 1 <= end < v2_end:
        raise ValueError(
            f"v3 source-exclusive end {end} must precede v2 end {v2_end} for episode {episode_index}"
        )
    return v2_end, end


def prepare_onset_v3_episode(
    table: pa.Table,
    *,
    episode_index: int,
    dataset_from_index: int,
) -> OnsetV3Episode:
    """Apply v2 onset plus the locked v3 suffix cuts and reindex one episode."""

    source_frame_count = len(table)
    if source_frame_count == 0:
        raise ValueError(f"episode {episode_index} is empty")
    source_episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    if not np.all(source_episode_indices == episode_index):
        raise ValueError(f"episode_index column does not uniformly equal {episode_index}")

    v2_tail_end, tail_end = _v3_tail_end(episode_index, source_frame_count)
    gripper_event_counts, last_gripper_event_frames = _validate_preserved_gripper_events(
        table,
        tail_end_frame_exclusive=tail_end,
    )
    retained_source = table.slice(0, tail_end)
    onset_frame = find_motion_onset_in_table(retained_source)
    trimmed = retained_source.slice(onset_frame, tail_end - onset_frame)
    state, action, helper, chunks = build_episode_arrays(trimmed)
    frame_count = len(trimmed)
    padding = action_padding_mask(frame_count)

    local_frame_index = np.arange(frame_count, dtype=np.int64)
    local_timestamp = local_frame_index.astype(np.float64) / FPS
    contiguous_index = np.arange(dataset_from_index, dataset_from_index + frame_count, dtype=np.int64)
    task_index = np.asarray(trimmed["task_index"].to_numpy(), dtype=np.int64)
    if not np.all(task_index == task_index[0]):
        raise ValueError(f"episode {episode_index} has multiple task indices")

    scalar_values = {
        "timestamp": local_timestamp.reshape(-1, 1),
        "frame_index": local_frame_index.astype(np.float64).reshape(-1, 1),
        "episode_index": source_episode_indices[onset_frame:tail_end].astype(np.float64).reshape(-1, 1),
        "index": contiguous_index.astype(np.float64).reshape(-1, 1),
        "task_index": task_index.astype(np.float64).reshape(-1, 1),
    }
    converted = pa.table(
        {
            "observation.state": _fixed_size_float_array(state, UMI_CURRENTREL_STATE_DIM),
            "action": _fixed_size_float_array(action, UMI_CURRENTREL_STATE_DIM),
            UMI_TCP_WINDOW_KEY: _fixed_size_float_array(helper, UMI_CURRENTREL_HELPER_DIM),
            "timestamp": pa.array(local_timestamp, type=table.schema.field("timestamp").type),
            "frame_index": pa.array(local_frame_index, type=table.schema.field("frame_index").type),
            "episode_index": pa.array(
                np.full(frame_count, episode_index), type=table.schema.field("episode_index").type
            ),
            "index": pa.array(contiguous_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=table.schema.field("task_index").type),
        }
    )
    episode = OnsetV3Episode(
        episode_index=episode_index,
        source_frame_count=source_frame_count,
        v2_tail_end_frame_exclusive=v2_tail_end,
        tail_end_frame_exclusive=tail_end,
        onset_frame=onset_frame,
        dataset_from_index=dataset_from_index,
        table=converted,
        state=state,
        action=action,
        helper=helper,
        chunks=chunks,
        action_is_pad=padding,
        gripper_event_frame_counts=gripper_event_counts,
        last_gripper_event_frames=last_gripper_event_frames,
        scalar_values=scalar_values,
    )
    expected_length = EXPECTED_V3_RETAINED_LENGTHS.get(episode_index)
    if expected_length is not None and episode.frame_count != expected_length:
        raise RuntimeError(
            f"episode {episode_index} retained {episode.frame_count} frames, expected {expected_length}"
        )
    return episode


def rewrite_episode_metadata(
    source_path: Path,
    output_path: Path,
    *,
    episode: OnsetV3Episode,
) -> None:
    """Rewrite one metadata row for the exact retained video/table interval."""

    source = pq.read_table(source_path)
    if source.num_rows != 1:
        raise ValueError(f"episode metadata must contain exactly one row, got {source.num_rows}")
    metadata_episode = int(source["episode_index"][0].as_py())
    if metadata_episode != episode.episode_index:
        raise ValueError(f"episode metadata/data mismatch: {metadata_episode} vs {episode.episode_index}")
    columns = {name: source[name] for name in source.column_names if not name.startswith("stats/")}
    columns["tasks"] = pa.array([[TASK]], type=pa.list_(pa.string()))
    columns["length"] = pa.array([episode.frame_count], type=pa.int64())
    columns["dataset_from_index"] = pa.array([episode.dataset_from_index], type=pa.int64())
    columns["dataset_to_index"] = pa.array(
        [episode.dataset_from_index + episode.frame_count], type=pa.int64()
    )

    video_from_keys = [
        key for key in columns if key.startswith("videos/") and key.endswith("/from_timestamp")
    ]
    for from_key in video_from_keys:
        to_key = f"{from_key.removesuffix('/from_timestamp')}/to_timestamp"
        if to_key not in columns:
            raise ValueError(f"video metadata has {from_key} without {to_key}")
        source_from = float(source[from_key][0].as_py())
        source_to = float(source[to_key][0].as_py())
        retained_from = source_from + episode.onset_frame / FPS
        retained_to = retained_from + episode.frame_count / FPS
        expected_to = source_from + episode.tail_end_frame_exclusive / FPS
        if not np.isclose(retained_to, expected_to, atol=1e-9, rtol=0.0):
            raise RuntimeError("retained video interval/source-index invariant failed")
        if retained_to > source_to + 1e-6:
            raise ValueError(
                f"retained video interval [{retained_from}, {retained_to}] exceeds "
                f"source interval [{source_from}, {source_to}]"
            )
        columns[from_key] = pa.array([retained_from], type=source.schema.field(from_key).type)
        columns[to_key] = pa.array([retained_to], type=source.schema.field(to_key).type)

    episode_stats = {
        "observation.state": _stats_block(episode.state),
        "action": _stats_block(episode.valid_action_rows),
        UMI_TCP_WINDOW_KEY: _stats_block(episode.helper),
        **{key: _stats_block(values) for key, values in episode.scalar_values.items()},
    }
    for feature, feature_stats in episode_stats.items():
        for statistic, value in feature_stats.items():
            value_type = pa.list_(pa.int64()) if statistic == "count" else pa.list_(pa.float64())
            columns[f"stats/{feature}/{statistic}"] = pa.array([value], type=value_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), output_path)


def semantic_metadata(
    *,
    onset_frames: dict[int, int],
    retained_lengths: dict[int, int],
) -> dict:
    """Return the complete v3 contract and immutable cleanup provenance."""

    metadata = copy.deepcopy(
        _v2_semantic_metadata(
            onset_frames=onset_frames,
            retained_lengths=retained_lengths,
            train_frames=EXPECTED_TRAIN_FRAMES,
            validation_frames=EXPECTED_VALIDATION_FRAMES,
        )
    )
    metadata["schema_id"] = UMI_CURRENTREL_ONSET_V3_SCHEMA_ID
    metadata["schema_version"] = 3
    metadata["dataset_name"] = OUTPUT_DATASET_NAME
    metadata["repository"] = OUTPUT_REPO_ID
    metadata["onset_alignment"]["removed_prefix_frames"] = {
        "train": 48_997 - EXPECTED_V2_TRAIN_FRAMES,
        "validation": 1_937 - EXPECTED_V2_VALIDATION_FRAMES,
        "total": (48_997 + 1_937) - (EXPECTED_V2_TRAIN_FRAMES + EXPECTED_V2_VALIDATION_FRAMES),
    }
    metadata["onset_alignment"]["retained_length_by_episode"] = {
        str(key): value for key, value in retained_lengths.items()
    }
    metadata["v3_suffix_cleanup"] = {
        "scope": "training episodes only; holdouts 52 and 53 are unchanged from v2",
        "source_end_index_semantics": "end-exclusive source episode frame index before onset reindexing",
        "source_end_frame_exclusive_by_episode": {
            str(key): value for key, value in SOURCE_SUFFIX_END_FRAME_EXCLUSIVE.items()
        },
        "v3_retained_length_by_episode": {
            str(key): value for key, value in EXPECTED_V3_RETAINED_LENGTHS.items()
        },
        "last_preserved_contact_source_frame_by_episode": {
            str(key): value for key, value in SUFFIX_LAST_CONTACT_FRAME.items()
        },
        "frames_after_last_preserved_contact_by_episode": {
            str(key): value for key, value in SUFFIX_CONTACT_CLEARANCE_FRAMES.items()
        },
        "removed_training_frames_relative_to_v2": EXPECTED_REMOVED_FROM_V2_TRAIN_FRAMES,
        "selection_provenance": (
            "isolated optimizer/video suffix-mismatch audit on episodes 0, 40, and 46; "
            "the cuts preserve all audited optimizer contact anchors and gripper events"
        ),
        "optimizer_contact_validation": (
            f"all {EXPECTED_OPTIMIZER_CONTACT_ANCHORS} pinned filtering.json anchors remain before "
            "their final source-exclusive end and are declared strictly feasible"
        ),
        "gripper_event_validation": (
            "both normalized source gripper tracks are checked with the locked task-independent "
            "0.04/s speed, 0.08 excursion, 0.12 s bridge, and 0.10 s padding detector; "
            "no detected event frame is removed"
        ),
        "terminal_note": (
            "the retained endpoints remain in task-complete withdrawal rather than an invented settled hold; "
            "beyond-end targets are clamped for storage but excluded from supervision"
        ),
        "global_smoothing_applied": False,
        "retiming_applied": False,
        "scene_free_onset_unchanged": True,
    }
    metadata["tail_cleanup"].update(
        {
            "training_frames_after_v2_onset_alignment": EXPECTED_V2_TRAIN_FRAMES,
            "validation_frames_after_v2_onset_alignment": EXPECTED_V2_VALIDATION_FRAMES,
            "training_frames_after_v3_suffix_cleanup": EXPECTED_TRAIN_FRAMES,
            "validation_frames_after_v3_suffix_cleanup": EXPECTED_VALIDATION_FRAMES,
            "output_frames_after_v3_suffix_cleanup": EXPECTED_OUTPUT_FRAMES,
        }
    )
    metadata["terminal_padding"] = {
        "action_rows": "k=1..24, each anchored at the sampled query T_tcp(t)",
        "storage": "clamp t+k to the final retained episode frame",
        "mask_rule": "action_is_pad[t,k-1] = (t+k >= retained_episode_length)",
        "supervision": "every beyond-end row is excluded from the MolmoAct2 action loss",
        "processor_mapping": "action_is_pad -> action_horizon_is_pad",
        "padded_positions_per_episode": PADDED_ACTION_POSITIONS_PER_EPISODE,
        "valid_action_positions": {
            "train": EXPECTED_VALID_TRAIN_ACTION_POSITIONS,
            "validation": EXPECTED_VALID_VALIDATION_ACTION_POSITIONS,
            "all": EXPECTED_VALID_ACTION_POSITIONS,
        },
    }
    metadata["statistics"] = {
        "scope": "training episodes 0..51 after v2 onset alignment and v3 suffix cleanup",
        "excluded_episodes": VALIDATION_EPISODES,
        "state_samples": f"{EXPECTED_TRAIN_FRAMES} retained training rows",
        "action_samples": (
            f"{EXPECTED_VALID_TRAIN_ACTION_POSITIONS} non-padding k=1..24 target rows; "
            "all beyond-end clamped rows excluded"
        ),
        "image_statistics": "omitted because full copied videos include unexposed frames",
        "fresh": True,
    }
    metadata["deployment_start_pose_audit"] = {
        "part_of_training_tensors": False,
        "candidate_symmetric_arm_joints_rad": [0.0, 0.05, 0.05, 0.0, 0.0, 0.0],
        "candidate_gripper_command": 1.0,
        "offline_result": "all 54 onset-local frame-0 GT chunks pass strict first-15 and full-24 IK",
        "hardware_start_verified": False,
        "requirement": (
            "verify physical safety and a UMI-homologous jaw/wrist-camera viewpoint before deployment; "
            "do not bake this absolute joint candidate into current-relative training targets"
        ),
    }
    return metadata


def write_artifact_manifest(
    output_root: Path,
    source_root: Path,
    *,
    valid_train_actions: int,
    valid_validation_actions: int,
    gripper_event_frames: int,
) -> None:
    """Hash every generated table, sidecar, README, and regular copied video."""

    manifest_path = output_root / "meta/artifact_manifest.json"
    checksum_path = output_root / "meta/artifact_manifest.sha256"
    excluded = {manifest_path, checksum_path}
    generated_paths = sorted(
        path
        for root in (output_root / "data", output_root / "meta", output_root / "videos")
        for path in root.rglob("*")
        if path.is_file() and path not in excluded
    )
    generated_paths.append(output_root / "README.md")
    generated_hashes = {path.relative_to(output_root).as_posix(): _sha256(path) for path in generated_paths}
    video_paths = sorted((output_root / "videos").rglob("*.mp4"))
    video_bytes = sum(path.stat().st_size for path in video_paths)
    if len(video_paths) != EXPECTED_VIDEO_FILES or video_bytes != EXPECTED_VIDEO_BYTES:
        raise ValueError(
            f"expected {EXPECTED_VIDEO_FILES}/{EXPECTED_VIDEO_BYTES} copied video files/bytes, "
            f"got {len(video_paths)}/{video_bytes}"
        )
    if any(path.is_symlink() or not path.is_file() for path in video_paths):
        raise ValueError("artifact videos must be regular copied files, not symlinks")
    for output_video in video_paths:
        relative = output_video.relative_to(output_root / "videos")
        source_video = source_root / "videos" / relative
        if not source_video.is_file() or _sha256(output_video) != _sha256(source_video):
            raise ValueError(f"copied video does not match source bytes: {relative}")

    manifest = {
        "schema_id": UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
        "dataset_repository": OUTPUT_REPO_ID,
        "source": {
            "repository": SOURCE_REPO_ID,
            "revision": SOURCE_REVISION,
            "root": str(source_root.resolve(strict=True)),
            "info_sha256": _sha256(source_root / "meta/info.json"),
            "filtering_sha256": _sha256(source_root / "meta/filtering.json"),
        },
        "videos": {
            "handling": "byte-for-byte source copies included in generated_file_sha256",
            "file_count": len(video_paths),
            "total_bytes": video_bytes,
            "all_source_sha256_equal": True,
        },
        "split": {
            "train_episodes": TRAIN_EPISODES,
            "validation_episodes": VALIDATION_EPISODES,
            "train_frames": EXPECTED_TRAIN_FRAMES,
            "validation_frames": EXPECTED_VALIDATION_FRAMES,
        },
        "action_supervision": {
            "horizon_offsets": list(range(1, UMI_CURRENTREL_HORIZON + 1)),
            "padded_positions_per_episode": PADDED_ACTION_POSITIONS_PER_EPISODE,
            "valid_train_positions": valid_train_actions,
            "valid_validation_positions": valid_validation_actions,
        },
        "preserved_provenance": {
            "optimizer_contact_anchors": EXPECTED_OPTIMIZER_CONTACT_ANCHORS,
            "detected_gripper_event_frames": gripper_event_frames,
        },
        "generated_file_sha256": generated_hashes,
        "checksum_file": checksum_path.relative_to(output_root).as_posix(),
        "checksum_scope": "all generated data/meta/video files and README except the checksum file itself",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksummed_paths = [*generated_paths, manifest_path]
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output_root).as_posix()}\n"
            for path in sorted(checksummed_paths)
        ),
        encoding="utf-8",
    )


def _validate_source_root(source_root: Path) -> tuple[dict, list[Path], list[Path]]:
    info_path = source_root / "meta/info.json"
    filtering_path = source_root / "meta/filtering.json"
    if _sha256(info_path) != EXPECTED_SOURCE_INFO_SHA256:
        raise ValueError("source meta/info.json does not match the pinned training revision")
    if _sha256(filtering_path) != EXPECTED_SOURCE_FILTERING_SHA256:
        raise ValueError("source meta/filtering.json does not match the pinned training revision")
    source_info = json.loads(info_path.read_text(encoding="utf-8"))
    filtering = json.loads(filtering_path.read_text(encoding="utf-8"))
    if source_info.get("total_episodes") != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} source episodes")
    if source_info.get("total_frames") != EXPECTED_SOURCE_FRAMES:
        raise ValueError(f"expected {EXPECTED_SOURCE_FRAMES} source frames")
    if "action" in source_info.get("features", {}):
        raise ValueError("filtered source must be observation-only")
    if source_info["features"]["observation.state"]["shape"] != [12]:
        raise ValueError("filtered source must retain the 12-D Cartesian observation schema")

    optimizer_results = filtering.get("optimizer_results")
    if not isinstance(optimizer_results, list) or len(optimizer_results) != EXPECTED_EPISODES:
        raise ValueError("source filtering metadata must contain 54 optimizer results")
    contact_count = 0
    seen_results: set[int] = set()
    for result in optimizer_results:
        episode_index = int(result["output_episode_index"])
        source_frames = int(result["frames"])
        if episode_index in seen_results:
            raise ValueError(f"duplicate optimizer result for episode {episode_index}")
        seen_results.add(episode_index)
        _, tail_end = _v3_tail_end(episode_index, source_frames)
        contact_events = result["retarget"]["contact_events"]
        anchors = contact_events["anchors"]
        declared_count = int(contact_events["total"])
        if len(anchors) != declared_count or not bool(contact_events["all_strictly_feasible"]):
            raise ValueError(f"invalid optimizer contact provenance for episode {episode_index}")
        if any(not bool(anchor["feasible"]) for anchor in anchors):
            raise ValueError(f"non-feasible optimizer contact anchor in episode {episode_index}")
        frames = [int(anchor["frame"]) for anchor in anchors]
        if any(frame >= tail_end for frame in frames):
            raise ValueError(f"v3 suffix cut removes an optimizer contact in episode {episode_index}")
        if (
            episode_index in SUFFIX_LAST_CONTACT_FRAME
            and max(frames) != SUFFIX_LAST_CONTACT_FRAME[episode_index]
        ):
            raise ValueError(f"last contact frame changed for v3-cut episode {episode_index}")
        contact_count += declared_count
    if seen_results != set(range(EXPECTED_EPISODES)):
        raise ValueError("optimizer results do not cover every source episode exactly once")
    if contact_count != EXPECTED_OPTIMIZER_CONTACT_ANCHORS:
        raise ValueError(
            f"expected {EXPECTED_OPTIMIZER_CONTACT_ANCHORS} optimizer contacts, got {contact_count}"
        )

    data_files = sorted((source_root / "data").rglob("*.parquet"))
    episode_files = sorted((source_root / "meta/episodes").rglob("*.parquet"))
    if len(data_files) != EXPECTED_EPISODES or len(episode_files) != EXPECTED_EPISODES:
        raise ValueError(
            f"expected one data and metadata file per episode, got {len(data_files)}/{len(episode_files)}"
        )
    video_paths = sorted((source_root / "videos").rglob("*.mp4"))
    if len(video_paths) != EXPECTED_VIDEO_FILES:
        raise ValueError(f"expected {EXPECTED_VIDEO_FILES} source videos, got {len(video_paths)}")
    if sum(path.stat().st_size for path in video_paths) != EXPECTED_VIDEO_BYTES:
        raise ValueError("source video bytes do not match the pinned training revision")
    return source_info, data_files, episode_files


def convert(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    source_info, data_files, episode_files = _validate_source_root(source_root)

    (output_root / "data/chunk-000").mkdir(parents=True)
    (output_root / "meta/episodes/chunk-000").mkdir(parents=True)
    shutil.copytree(source_root / "videos", output_root / "videos", symlinks=False)

    statistics = TrainOnlyStatsAccumulator()
    onset_frames: dict[int, int] = {}
    retained_lengths: dict[int, int] = {}
    tail_ends: dict[int, int] = {}
    total_source_frames = 0
    total_output_frames = 0
    train_frames = 0
    validation_frames = 0
    valid_train_actions = 0
    valid_validation_actions = 0
    gripper_event_frames = 0
    for data_path, episode_path in zip(data_files, episode_files, strict=True):
        source_table = pq.read_table(data_path)
        episode_index = int(source_table["episode_index"][0].as_py())
        expected_index = int(data_path.stem.split("-")[-1])
        if episode_index != expected_index:
            raise ValueError(f"episode/file mismatch: {episode_index} vs {data_path}")
        total_source_frames += len(source_table)
        episode = prepare_onset_v3_episode(
            source_table,
            episode_index=episode_index,
            dataset_from_index=total_output_frames,
        )
        total_output_frames += episode.frame_count
        onset_frames[episode_index] = episode.onset_frame
        retained_lengths[episode_index] = episode.frame_count
        tail_ends[episode_index] = episode.tail_end_frame_exclusive
        if episode_index in TRAIN_EPISODES:
            train_frames += episode.frame_count
            valid_train_actions += episode.valid_action_position_count
        elif episode_index in VALIDATION_EPISODES:
            validation_frames += episode.frame_count
            valid_validation_actions += episode.valid_action_position_count
        else:
            raise ValueError(f"episode {episode_index} is outside the locked split")
        statistics.add(episode)
        gripper_event_frames += sum(episode.gripper_event_frame_counts)

        pq.write_table(episode.table, output_root / "data/chunk-000" / data_path.name)
        rewrite_episode_metadata(
            episode_path,
            output_root / "meta/episodes/chunk-000" / episode_path.name,
            episode=episode,
        )
        if episode_index % 10 == 0:
            print(
                f"converted episode {episode_index:02d}: onset={episode.onset_frame}, "
                f"end={episode.tail_end_frame_exclusive}, retained={episode.frame_count}, "
                f"valid_actions={episode.valid_action_position_count}",
                flush=True,
            )

    if total_source_frames != EXPECTED_SOURCE_FRAMES:
        raise RuntimeError(f"read {total_source_frames} source frames, expected {EXPECTED_SOURCE_FRAMES}")
    if sorted(onset_frames) != list(range(EXPECTED_EPISODES)):
        raise RuntimeError("onset map does not cover every episode exactly once")
    counts = (train_frames, validation_frames, total_output_frames)
    expected_counts = (EXPECTED_TRAIN_FRAMES, EXPECTED_VALIDATION_FRAMES, EXPECTED_OUTPUT_FRAMES)
    if counts != expected_counts:
        raise RuntimeError(f"v3 frame counts changed: got {counts}, expected {expected_counts}")
    action_counts = (
        valid_train_actions,
        valid_validation_actions,
        valid_train_actions + valid_validation_actions,
    )
    expected_action_counts = (
        EXPECTED_VALID_TRAIN_ACTION_POSITIONS,
        EXPECTED_VALID_VALIDATION_ACTION_POSITIONS,
        EXPECTED_VALID_ACTION_POSITIONS,
    )
    if action_counts != expected_action_counts:
        raise RuntimeError(
            f"v3 valid action-position counts changed: got {action_counts}, expected {expected_action_counts}"
        )

    task_frame = pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK], name="task"))
    task_frame.to_parquet(output_root / "meta/tasks.parquet")

    output_info = copy.deepcopy(source_info)
    features = output_info["features"]
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
    output_info["splits"] = {"train": "0:52", "validation": "52:54"}
    output_info["robot_type"] = "bimanual UMI grippers mounted on YAM"
    output_info["total_frames"] = total_output_frames
    (output_root / "meta/info.json").write_text(json.dumps(output_info, indent=2) + "\n", encoding="utf-8")
    (output_root / "meta/stats.json").write_text(
        json.dumps(statistics.finalize(), indent=2) + "\n", encoding="utf-8"
    )

    split_manifest = {
        "schema_id": UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
        "dataset_repository": OUTPUT_REPO_ID,
        "split_unit": "episode",
        "source_revision": SOURCE_REVISION,
        "train_episodes": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "selection": "locked 52/2 episode split inherited from v1/v2; no frame-level randomization",
        "v2_tail_end_frame_exclusive": {
            str(index): TAIL_RETAINED_LENGTHS.get(index, "source_episode_length")
            for index in range(EXPECTED_EPISODES)
        },
        "v3_source_end_frame_exclusive": {
            str(key): value for key, value in SOURCE_SUFFIX_END_FRAME_EXCLUSIVE.items()
        },
        "final_tail_end_frame_exclusive": {str(key): value for key, value in tail_ends.items()},
        "onset_frame_by_episode": {str(key): value for key, value in onset_frames.items()},
        "retained_length_by_episode": {str(key): value for key, value in retained_lengths.items()},
        "train_frames": train_frames,
        "validation_frames": validation_frames,
        "valid_train_action_positions": valid_train_actions,
        "valid_validation_action_positions": valid_validation_actions,
    }
    (output_root / UMI_CURRENTREL_SPLIT_PATH).write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / UMI_CURRENTREL_ONSET_V3_METADATA_PATH).write_text(
        json.dumps(
            semantic_metadata(onset_frames=onset_frames, retained_lengths=retained_lengths),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(source_root / "meta/filtering.json", output_root / "meta/source_filtering.json")

    readme = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
---

# {OUTPUT_DATASET_NAME}

Onset-v3 query-anchored jaw-centre TCP targets derived from `{SOURCE_REPO_ID}`
at pinned revision `{SOURCE_REVISION}`. It preserves v2's pose-only sustained
motion onset, adds three audited training-only suffix cuts, and applies no
global smoothing or retiming. All 108 videos are regular byte-for-byte copies
covered by the artifact checksum manifest.

Every query uses `inverse(T_tcp(t)) @ T_tcp(t+k)` for `k=1..24`. Targets beyond
the retained episode end remain clamped in storage but are marked by
`action_is_pad`, excluded from action statistics, propagated by the MolmoAct2
processor to `action_horizon_is_pad`, and excluded from training loss. See
`{UMI_CURRENTREL_ONSET_V3_METADATA_PATH}` for the full contract and provenance.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(
        output_root,
        source_root,
        valid_train_actions=valid_train_actions,
        valid_validation_actions=valid_validation_actions,
        gripper_event_frames=gripper_event_frames,
    )
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
