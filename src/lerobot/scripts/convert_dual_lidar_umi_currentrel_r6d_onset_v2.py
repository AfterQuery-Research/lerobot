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
"""Build the scene-free onset-aligned dual-LiDAR UMI current-relative v2 artifact.

This is an isolated successor to ``dual-lidar-umi-currentrel-r6d-v1``. It keeps
the same jaw-TCP, Rotation6D, same-query-anchor, split, tail-cleanup, and
hardware gripper calibration contracts. The sole label-distribution change is
a locked, pose-only leading-prefix trim. No image, object, bowl, or scene cue is
used to choose an onset.
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
from scipy.spatial.transform import Rotation

from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_HELPER_DIM,
    UMI_CURRENTREL_ONSET_METADATA_PATH,
    UMI_CURRENTREL_ONSET_SCHEMA_ID,
    UMI_CURRENTREL_SPLIT_PATH,
    UMI_CURRENTREL_STATE_DIM,
    UMI_TCP_WINDOW_KEY,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d import (
    ACTION_NAMES,
    EXPECTED_EPISODES,
    EXPECTED_SOURCE_FRAMES,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_VALIDATION_FRAMES,
    HELPER_NAMES,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    STATE_NAMES,
    TAIL_RETAINED_LENGTHS,
    TASK,
    TRAIN_EPISODES,
    VALIDATION_EPISODES,
    _fixed_size_float_array,
    _semantic_metadata as _v1_semantic_metadata,
    _stats_block,
    build_episode_arrays,
    stored_pose_track_to_tcp,
)

OUTPUT_DATASET_NAME = "dual-lidar-umi-currentrel-r6d-onset-v2"
OUTPUT_REPO_ID = f"brandonyang/{OUTPUT_DATASET_NAME}"

FPS = 30
ONSET_TRANSLATION_THRESHOLD_M = 0.002
ONSET_ROTATION_THRESHOLD_DEG = 1.0
ONSET_CONSECUTIVE_FRAMES = 3
EXPECTED_ONSET_TRAIN_FRAMES = 47_618
EXPECTED_ONSET_VALIDATION_FRAMES = 1_884
EXPECTED_ONSET_OUTPUT_FRAMES = EXPECTED_ONSET_TRAIN_FRAMES + EXPECTED_ONSET_VALIDATION_FRAMES


@dataclass(frozen=True)
class OnsetAlignedEpisode:
    """Converted numerical payload and immutable trim provenance for one episode."""

    episode_index: int
    source_frame_count: int
    tail_end_frame_exclusive: int
    onset_frame: int
    dataset_from_index: int
    table: pa.Table
    state: np.ndarray
    action: np.ndarray
    helper: np.ndarray
    chunks: np.ndarray
    scalar_values: dict[str, np.ndarray]

    @property
    def frame_count(self) -> int:
        return len(self.table)


class TrainOnlyStatsAccumulator:
    """Accumulate normalization statistics while structurally excluding holdouts."""

    def __init__(self) -> None:
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

    def add(self, episode: OnsetAlignedEpisode) -> None:
        if episode.episode_index not in TRAIN_EPISODES:
            return
        self._values["observation.state"].append(episode.state)
        self._values["action"].append(episode.chunks.reshape(-1, UMI_CURRENTREL_STATE_DIM))
        self._values[UMI_TCP_WINDOW_KEY].append(episode.helper)
        for key, values in episode.scalar_values.items():
            self._values[key].append(values)

    def finalize(self) -> dict[str, dict[str, list[float] | list[int]]]:
        empty = [key for key, values in self._values.items() if not values]
        if empty:
            raise RuntimeError(f"no training values accumulated for {empty}")
        return {key: _stats_block(np.concatenate(values, axis=0)) for key, values in self._values.items()}


def _source_tcp_tracks(table: pa.Table) -> np.ndarray:
    if "action" in table.column_names:
        raise ValueError("the filtered observation-only source must not contain an action column")
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    if state.ndim != 2 or state.shape[1] != 12 or not np.isfinite(state).all():
        raise ValueError(f"source observation.state must have finite shape (N, 12), got {state.shape}")
    return np.stack(
        (stored_pose_track_to_tcp(state[:, :6]), stored_pose_track_to_tcp(state[:, 6:])),
        axis=1,
    )


def find_motion_onset(tcp: np.ndarray) -> int:
    """Return the first frame in the first three-frame cumulative-motion run.

    Motion is measured relative to each arm's frame-0 jaw TCP. A frame is
    active when either arm has translation strictly greater than 2 mm or
    geodesic rotation strictly greater than 1 degree. The returned onset is the
    first frame of the earliest three consecutive active frames.
    """

    transforms = np.asarray(tcp, dtype=np.float64)
    if transforms.ndim != 4 or transforms.shape[1:] != (2, 4, 4):
        raise ValueError(f"tcp must have shape (N, 2, 4, 4), got {transforms.shape}")
    if transforms.shape[0] < ONSET_CONSECUTIVE_FRAMES:
        raise ValueError(
            f"episode has {transforms.shape[0]} frames; onset needs "
            f"{ONSET_CONSECUTIVE_FRAMES} consecutive frames"
        )
    if not np.isfinite(transforms).all():
        raise ValueError("tcp contains non-finite values")

    translation_from_frame0 = np.linalg.norm(transforms[:, :, :3, 3] - transforms[0, :, :3, 3], axis=-1)
    frame0_rotation_t = np.swapaxes(transforms[0, :, :3, :3], -1, -2)
    relative_rotation = np.einsum("aij,tajk->taik", frame0_rotation_t, transforms[:, :, :3, :3])
    rotation_from_frame0_deg = np.rad2deg(
        Rotation.from_matrix(relative_rotation.reshape(-1, 3, 3)).magnitude().reshape(transforms.shape[0], 2)
    )
    active = np.any(
        (translation_from_frame0 > ONSET_TRANSLATION_THRESHOLD_M)
        | (rotation_from_frame0_deg > ONSET_ROTATION_THRESHOLD_DEG),
        axis=1,
    )
    run_starts = np.flatnonzero(
        active[: -(ONSET_CONSECUTIVE_FRAMES - 1)] & active[1 : -(ONSET_CONSECUTIVE_FRAMES - 2)] & active[2:]
    )
    if run_starts.size == 0:
        raise ValueError(
            "no scene-free motion onset: no three consecutive frames exceed either "
            f"{ONSET_TRANSLATION_THRESHOLD_M} m or {ONSET_ROTATION_THRESHOLD_DEG} degree"
        )
    return int(run_starts[0])


def find_motion_onset_in_table(table: pa.Table) -> int:
    """Decode both stored jaw-TCP tracks and apply the locked onset rule."""

    return find_motion_onset(_source_tcp_tracks(table))


def _tail_end_frame_exclusive(episode_index: int, source_frame_count: int) -> int:
    end = TAIL_RETAINED_LENGTHS.get(episode_index, source_frame_count)
    if not 1 <= end <= source_frame_count:
        raise ValueError(
            f"invalid tail retained length {end} for episode {episode_index} "
            f"with {source_frame_count} source frames"
        )
    return end


def prepare_onset_aligned_episode(
    table: pa.Table,
    *,
    episode_index: int,
    dataset_from_index: int,
) -> OnsetAlignedEpisode:
    """Tail-trim, onset-trim, reindex, and encode one source episode."""

    source_frame_count = len(table)
    if source_frame_count == 0:
        raise ValueError(f"episode {episode_index} is empty")
    source_episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    if not np.all(source_episode_indices == episode_index):
        raise ValueError(f"episode_index column does not uniformly equal {episode_index}")
    tail_end = _tail_end_frame_exclusive(episode_index, source_frame_count)
    tail_clean = table.slice(0, tail_end)
    onset_frame = find_motion_onset_in_table(tail_clean)
    trimmed = tail_clean.slice(onset_frame, tail_end - onset_frame)
    state, action, helper, chunks = build_episode_arrays(trimmed)
    frame_count = len(trimmed)

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
    return OnsetAlignedEpisode(
        episode_index=episode_index,
        source_frame_count=source_frame_count,
        tail_end_frame_exclusive=tail_end,
        onset_frame=onset_frame,
        dataset_from_index=dataset_from_index,
        table=converted,
        state=state,
        action=action,
        helper=helper,
        chunks=chunks,
        scalar_values=scalar_values,
    )


def rewrite_onset_episode_metadata(
    source_path: Path,
    output_path: Path,
    *,
    episode: OnsetAlignedEpisode,
) -> None:
    """Rewrite one metadata row and expose only the retained source-video interval."""

    source = pq.read_table(source_path)
    if source.num_rows != 1:
        raise ValueError(f"episode metadata must contain exactly one row, got {source.num_rows}")
    # Every old statistic covers excluded frames. Drop all of them, then write
    # fresh numerical statistics for the retained interval below.
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
        if retained_to > source_to + 1e-6:
            raise ValueError(
                f"retained video interval [{retained_from}, {retained_to}] exceeds "
                f"source interval [{source_from}, {source_to}]"
            )
        columns[from_key] = pa.array([retained_from], type=source.schema.field(from_key).type)
        columns[to_key] = pa.array([retained_to], type=source.schema.field(to_key).type)

    episode_stats = {
        "observation.state": _stats_block(episode.state),
        "action": _stats_block(episode.chunks.reshape(-1, UMI_CURRENTREL_STATE_DIM)),
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
    train_frames: int,
    validation_frames: int,
) -> dict:
    """Return the complete v2 contract while inheriting the reviewed v1 semantics."""

    metadata = copy.deepcopy(_v1_semantic_metadata())
    metadata["schema_id"] = UMI_CURRENTREL_ONSET_SCHEMA_ID
    metadata["schema_version"] = 2
    metadata["dataset_name"] = OUTPUT_DATASET_NAME
    metadata["repository"] = OUTPUT_REPO_ID
    metadata["state_semantics"] = (
        "history_i(t) = inverse(T_tcp_i(t)) @ T_tcp_i(t-1); "
        "the onset-aligned local frame 0 history is exact identity"
    )
    metadata["onset_alignment"] = {
        "variant": "scene-free cumulative jaw-TCP motion onset",
        "reference": "each arm's source episode frame-0 jaw TCP",
        "arm_combination": "either arm",
        "frame_predicate": (
            "cumulative translation strictly exceeds 0.002 m OR geodesic rotation strictly exceeds 1 degree"
        ),
        "translation_threshold_m": ONSET_TRANSLATION_THRESHOLD_M,
        "rotation_threshold_deg": ONSET_ROTATION_THRESHOLD_DEG,
        "threshold_comparison": "strict greater-than",
        "consecutive_frames": ONSET_CONSECUTIVE_FRAMES,
        "selection": "first frame of the earliest qualifying consecutive run",
        "trim": "discard frames before onset; retain onset as local frame_index=0 and timestamp=0",
        "onset_frame_by_episode": {str(key): value for key, value in onset_frames.items()},
        "retained_length_by_episode": {str(key): value for key, value in retained_lengths.items()},
        "removed_prefix_frames": {
            "train": EXPECTED_TRAIN_FRAMES - train_frames,
            "validation": EXPECTED_VALIDATION_FRAMES - validation_frames,
            "total": (EXPECTED_TRAIN_FRAMES + EXPECTED_VALIDATION_FRAMES)
            - (train_frames + validation_frames),
        },
        "scene_or_object_features_used": False,
        "fit_to_task_video": False,
    }
    metadata["tail_cleanup"]["training_frames_after_tail_cleanup_before_onset_alignment"] = metadata[
        "tail_cleanup"
    ].pop("training_frames_after_cleanup")
    metadata["tail_cleanup"].pop("validation_frames")
    metadata["tail_cleanup"].pop("output_frames")
    metadata["tail_cleanup"]["training_frames_after_onset_alignment"] = train_frames
    metadata["tail_cleanup"]["validation_frames_after_onset_alignment"] = validation_frames
    metadata["tail_cleanup"]["output_frames_after_onset_alignment"] = train_frames + validation_frames
    metadata["tail_cleanup"]["video_handling"] = (
        "full source videos are copied byte-for-byte into the immutable artifact and hash-bound; "
        "per-episode from_timestamp is advanced by onset_frame/fps and to_timestamp exposes "
        "exactly the retained duration"
    )
    metadata["dual_arm_common_frame_note"] = (
        "the artifact deliberately carries no invented inter-gripper transform; "
        "runtime controller handles collision"
    )
    metadata["statistics"] = {
        "scope": "training episodes 0..51 only after tail and onset alignment",
        "excluded_episodes": VALIDATION_EPISODES,
        "state_samples": "one 20-D row per retained training frame",
        "action_samples": "all 24 query-anchored 20-D rows per retained training frame",
        "image_statistics": (
            "omitted: full copied-video statistics include excluded prefix/tail/holdout pixels"
        ),
        "fresh": True,
    }
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_manifest(
    output_root: Path,
    source_root: Path,
    *,
    train_frames: int,
    validation_frames: int,
) -> None:
    """Record hashes for every generated file, including copied training videos."""

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
    if len(video_paths) != EXPECTED_EPISODES * 2:
        raise ValueError(f"expected {EXPECTED_EPISODES * 2} copied videos, got {len(video_paths)}")
    if any(path.is_symlink() or not path.is_file() for path in video_paths):
        raise ValueError("artifact videos must be regular copied files, not symlinks")
    filtering_path = source_root / "meta/filtering.json"
    manifest = {
        "schema_id": UMI_CURRENTREL_ONSET_SCHEMA_ID,
        "dataset_repository": OUTPUT_REPO_ID,
        "source": {
            "repository": SOURCE_REPO_ID,
            "revision": SOURCE_REVISION,
            "root": str(source_root.resolve(strict=True)),
            "info_sha256": _sha256(source_root / "meta/info.json"),
            "filtering_sha256": _sha256(filtering_path) if filtering_path.is_file() else None,
        },
        "videos": {
            "handling": "byte-for-byte source copies included in generated_file_sha256",
            "file_count": len(video_paths),
            "total_bytes": sum(path.stat().st_size for path in video_paths),
        },
        "split": {
            "train_episodes": TRAIN_EPISODES,
            "validation_episodes": VALIDATION_EPISODES,
            "train_frames": train_frames,
            "validation_frames": validation_frames,
        },
        "generated_file_sha256": generated_hashes,
        "checksum_file": checksum_path.relative_to(output_root).as_posix(),
        "checksum_scope": ("all generated data/meta/video files and README except the checksum file itself"),
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


def convert(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")

    source_info = json.loads((source_root / "meta/info.json").read_text(encoding="utf-8"))
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
    shutil.copytree(source_root / "videos", output_root / "videos", symlinks=False)

    statistics = TrainOnlyStatsAccumulator()
    onset_frames: dict[int, int] = {}
    retained_lengths: dict[int, int] = {}
    total_source_frames = 0
    total_output_frames = 0
    train_frames = 0
    validation_frames = 0
    for data_path, episode_path in zip(data_files, episode_files, strict=True):
        source_table = pq.read_table(data_path)
        episode_index = int(source_table["episode_index"][0].as_py())
        expected_index = int(data_path.stem.split("-")[-1])
        if episode_index != expected_index:
            raise ValueError(f"episode/file mismatch: {episode_index} vs {data_path}")
        total_source_frames += len(source_table)
        episode = prepare_onset_aligned_episode(
            source_table,
            episode_index=episode_index,
            dataset_from_index=total_output_frames,
        )
        total_output_frames += episode.frame_count
        onset_frames[episode_index] = episode.onset_frame
        retained_lengths[episode_index] = episode.frame_count
        if episode_index in TRAIN_EPISODES:
            train_frames += episode.frame_count
        elif episode_index in VALIDATION_EPISODES:
            validation_frames += episode.frame_count
        else:
            raise ValueError(f"episode {episode_index} is outside the locked split")
        statistics.add(episode)

        pq.write_table(episode.table, output_root / "data/chunk-000" / data_path.name)
        rewrite_onset_episode_metadata(
            episode_path,
            output_root / "meta/episodes/chunk-000" / episode_path.name,
            episode=episode,
        )
        if episode_index % 10 == 0:
            print(
                f"converted episode {episode_index:02d}: onset={episode.onset_frame}, "
                f"retained={episode.frame_count}",
                flush=True,
            )

    if total_source_frames != EXPECTED_SOURCE_FRAMES:
        raise RuntimeError(f"read {total_source_frames} source frames, expected {EXPECTED_SOURCE_FRAMES}")
    if sorted(onset_frames) != list(range(EXPECTED_EPISODES)):
        raise RuntimeError("onset map does not cover every episode exactly once")
    if (train_frames, validation_frames, total_output_frames) != (
        EXPECTED_ONSET_TRAIN_FRAMES,
        EXPECTED_ONSET_VALIDATION_FRAMES,
        EXPECTED_ONSET_OUTPUT_FRAMES,
    ):
        raise RuntimeError(
            "onset-aligned frame counts changed for the pinned source: "
            f"got {train_frames}/{validation_frames}/{total_output_frames}, expected "
            f"{EXPECTED_ONSET_TRAIN_FRAMES}/{EXPECTED_ONSET_VALIDATION_FRAMES}/"
            f"{EXPECTED_ONSET_OUTPUT_FRAMES}"
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
        "schema_id": UMI_CURRENTREL_ONSET_SCHEMA_ID,
        "dataset_repository": OUTPUT_REPO_ID,
        "split_unit": "episode",
        "source_revision": SOURCE_REVISION,
        "train_episodes": TRAIN_EPISODES,
        "validation_episodes": VALIDATION_EPISODES,
        "selection": "locked 52/2 episode split inherited from v1; no frame-level randomization",
        "tail_end_frame_exclusive": {
            str(index): TAIL_RETAINED_LENGTHS.get(index, "source_episode_length")
            for index in range(EXPECTED_EPISODES)
        },
        "onset_frame_by_episode": {str(key): value for key, value in onset_frames.items()},
        "retained_length_by_episode": {str(key): value for key, value in retained_lengths.items()},
        "train_frames": train_frames,
        "validation_frames": validation_frames,
    }
    (output_root / UMI_CURRENTREL_SPLIT_PATH).write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / UMI_CURRENTREL_ONSET_METADATA_PATH).write_text(
        json.dumps(
            semantic_metadata(
                onset_frames=onset_frames,
                retained_lengths=retained_lengths,
                train_frames=train_frames,
                validation_frames=validation_frames,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
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

Scene-free onset-aligned query-anchored jaw-centre TCP targets derived from
`{SOURCE_REPO_ID}` at pinned revision `{SOURCE_REVISION}`. Images are copied
byte-for-byte into this artifact and covered by its checksum manifest. Each
video interval starts at the episode's pose-only onset while tabular timestamp
and frame_index restart at zero.

For every query frame `t`, all 24 target rows use
`inverse(T_tcp(t)) @ T_tcp(t+k)` for `k=1..24`. See
`{UMI_CURRENTREL_ONSET_METADATA_PATH}` for the locked onset rule, transform
direction, units, provenance, feature order, gripper endpoint contract, and
training-only statistics scope.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    write_artifact_manifest(
        output_root,
        source_root,
        train_frames=train_frames,
        validation_frames=validation_frames,
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
