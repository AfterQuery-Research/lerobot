#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Derive trainable H24 joint actions from the pinned published BiYAM joint dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _load_episode_with_stats, _write_parquet
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.umi_yam import ACTION_HORIZON
from lerobot.datasets.umi_yam_ee_dataset import (
    LEFT_VIDEO,
    RIGHT_VIDEO,
    write_payload_manifest,
)
from lerobot.utils.constants import ACTION, OBS_STATE

SOURCE_REPO_ID = "brandonyang/dual-lidar-combined-filtered-joint-positions-long-gripper"
SOURCE_REVISION = "387d696eb36de411a11c8b2612a6f19822ca3a53"
OUTPUT_TASK = "pick up oranges and place them in the bowl"
EXPECTED_EPISODES, EXPECTED_FRAMES, EXPECTED_QUERIES = 182, 179_951, 175_583
EXPECTED_ACTION_VALUES = EXPECTED_QUERIES * ACTION_HORIZON
I2RT_REVISION = "7ed46f4e4e316133a0c39aa6cf34a73d2718e850"
VIDEOS = (LEFT_VIDEO, RIGHT_VIDEO)
SCALARS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
JOINT_NAMES = tuple(
    name
    for arm in ("left", "right")
    for name in (*(f"{arm}_joint_{i}.pos" for i in range(6)), f"{arm}_gripper.pos")
)
SOURCE_KEYS = {OBS_STATE, *VIDEOS, *SCALARS}
KINEMATICS = {
    "implementation": "AfterQuery-Research/i2rt",
    "revision": I2RT_REVISION,
    "arm_model": "i2rt/robot_models/arm/yam/yam.xml",
    "gripper_model": "i2rt/robot_models/gripper/linear_4310/linear_4310.xml",
    "grasp_offset_mm": 220.0,
    "grasp_axes": "UMI-compatible: +X up, +Y right, +Z forward",
}


def _exact_stats(arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Preserve the published artifact's per-column floating-point reductions."""

    if not arrays or any(array.ndim != 2 for array in arrays):
        raise ValueError("stats require non-empty 2-D arrays")
    dimension = arrays[0].shape[1]
    if any(array.shape[1] != dimension for array in arrays):
        raise ValueError("stats dimensions differ")
    columns = [
        np.concatenate([array[:, index].astype(np.float64) for array in arrays]) for index in range(dimension)
    ]
    return {
        "min": np.array([column.min() for column in columns]),
        "max": np.array([column.max() for column in columns]),
        "mean": np.array([column.mean() for column in columns]),
        "std": np.array([column.std() for column in columns]),
        "count": np.array([sum(len(array) for array in arrays)]),
        **{
            f"q{percentile:02d}": np.array([np.quantile(column, percentile / 100) for column in columns])
            for percentile in (1, 10, 50, 90, 99)
        },
    }


def derive_episode(state: Any) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(state)
    if state.dtype != np.float32 or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError("published state must be float32 (frames, 14)")
    if len(state) <= ACTION_HORIZON or not np.isfinite(state).all():
        raise ValueError("published state must contain more than 24 finite frames")
    if np.any((state[:, (6, 13)] < 0) | (state[:, (6, 13)] > 1)):
        raise ValueError("published grippers must use normalized 0=closed, 1=open values")
    action = np.empty_like(state)
    action[:-1], action[-1] = state[1:], state[-1]
    return state, action


def _training_arrays(state: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    queries = len(state) - ACTION_HORIZON
    return state[:queries], [state[offset : offset + queries] for offset in range(1, ACTION_HORIZON + 1)]


def _validate_source(meta: LeRobotDatasetMetadata, root: Path) -> None:
    if (meta.fps, meta.total_episodes, meta.total_frames) != (
        30,
        EXPECTED_EPISODES,
        EXPECTED_FRAMES,
    ):
        raise ValueError("published source counts do not match the pinned snapshot")
    if set(meta.features) != SOURCE_KEYS:
        raise ValueError("published source feature keys do not match the pinned schema")
    state = meta.features[OBS_STATE]
    if (state["dtype"], tuple(state["shape"]), tuple(state["names"])) != (
        "float32",
        (14,),
        JOINT_NAMES,
    ):
        raise ValueError("published source state does not match the pinned 14-D schema")
    if meta.tasks is None or list(meta.tasks.index) != ["bimanual umi demo"]:
        raise ValueError("published source task does not match the pinned schema")
    provenance = json.loads((root / "meta/filtering.json").read_text())
    expected = (
        "combined_zero_origin_v2_i2rt_220mm_umi_grasp_frame",
        KINEMATICS,
        True,
        EXPECTED_EPISODES,
        EXPECTED_FRAMES,
    )
    actual = (
        provenance.get("pipeline_version"),
        provenance.get("kinematics"),
        provenance.get("action_removed"),
        provenance.get("episodes"),
        provenance.get("frames"),
    )
    if actual != expected:
        raise ValueError("published source provenance does not match the long-gripper contract")


def _read_episode(meta: LeRobotDatasetMetadata, root: Path, episode: int) -> pd.DataFrame:
    frame = pd.read_parquet(root / meta.get_data_file_path(episode))
    length = int(meta.episodes[episode]["length"])
    if len(frame) != length or set(frame.columns) != SOURCE_KEYS - set(VIDEOS):
        raise ValueError(f"episode {episode} data schema/count mismatch")
    expected_frame = np.arange(length)
    expected_start = int(meta.episodes[episode]["dataset_from_index"])
    if not (
        np.array_equal(frame["frame_index"].to_numpy(), expected_frame)
        and np.all(frame["episode_index"].to_numpy() == episode)
        and np.array_equal(frame["index"].to_numpy(), expected_start + expected_frame)
        and np.all(frame["task_index"].to_numpy() == 0)
        and np.allclose(frame["timestamp"].to_numpy(), expected_frame / 30, atol=2e-6, rtol=0)
    ):
        raise ValueError(f"episode {episode} scalar alignment mismatch")
    return frame


def build_manifest(lengths: list[int]) -> dict[str, Any]:
    rows, frame_start, query_start = [], 0, 0
    for episode, frames in enumerate(lengths):
        queries = frames - ACTION_HORIZON
        rows.append(
            {
                "source_episode": episode,
                "output_episode": episode,
                "frames": frames,
                "valid_query_dataset_range": [frame_start, frame_start + queries],
                "valid_query_compact_range": [query_start, query_start + queries],
            }
        )
        frame_start += frames
        query_start += queries
    if len(rows) != EXPECTED_EPISODES or frame_start != EXPECTED_FRAMES or query_start != EXPECTED_QUERIES:
        raise ValueError("published all-182 counts changed")
    return {
        "schema": "umi_yam.published_joint.long_gripper182.v1",
        "source": {"repo_id": SOURCE_REPO_ID, "revision": SOURCE_REVISION},
        "kinematics": KINEMATICS,
        "task": OUTPUT_TASK,
        "algorithm": "preserve published 14-D state; action[t]=state[t+1]",
        "action_horizon": ACTION_HORIZON,
        "terminal_query_policy": "exclude final 24 query rows per episode",
        "stats": {
            "observation.state": "exact global values at valid query rows t=0..N-25",
            "action": "exact global values over H24 targets state[t+1:t+25]",
            "state_count": EXPECTED_QUERIES,
            "action_count": EXPECTED_ACTION_VALUES,
        },
        "episodes": rows,
    }


def _link_or_copy(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _video_stats(source_row: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    prefix = f"stats/{key}/"
    result = {}
    for name, value in source_row.items():
        if name.startswith(prefix):
            stat = name.removeprefix(prefix)
            flat = np.array([np.asarray(item).reshape(-1)[0] for item in value], dtype=np.float64)
            result[stat] = flat.astype(np.int64) if stat == "count" else flat.reshape(-1, 1, 1)
    if not result:
        raise ValueError(f"missing source video stats for {key}")
    return result


def materialize(output_root: Path, repo_id: str) -> None:
    if output_root.exists():
        raise FileExistsError(output_root)
    source_root = Path(snapshot_download(SOURCE_REPO_ID, repo_type="dataset", revision=SOURCE_REVISION))
    source_meta = LeRobotDatasetMetadata(SOURCE_REPO_ID, root=source_root, revision=SOURCE_REVISION)
    _validate_source(source_meta, source_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    temporary = temporary_parent / "dataset"
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    lengths: list[int] = []
    try:
        features = copy.deepcopy(source_meta.features)
        features[ACTION] = copy.deepcopy(features[OBS_STATE])
        output_meta = LeRobotDatasetMetadata.create(repo_id, 30, features, root=temporary, use_videos=True)
        output_meta.save_episode_tasks([OUTPUT_TASK])
        source_dataset = argparse.Namespace(meta=source_meta, root=source_root)
        for episode in range(EXPECTED_EPISODES):
            frame = _read_episode(source_meta, source_root, episode)
            state, action = derive_episode(np.stack(frame[OBS_STATE]))
            frame[ACTION] = list(action)
            data_path = temporary / output_meta.data_path.format(chunk_index=0, file_index=episode)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            _write_parquet(frame, data_path, output_meta)

            source_row = _load_episode_with_stats(source_dataset, episode)
            video_metadata: dict[str, int | float] = {"data/chunk_index": 0, "data/file_index": episode}
            for key in VIDEOS:
                _link_or_copy(
                    source_root / source_meta.get_video_file_path(episode, key),
                    temporary
                    / output_meta.video_path.format(video_key=key, chunk_index=0, file_index=episode),
                )
                prefix = f"videos/{key}"
                video_metadata.update(
                    {
                        f"{prefix}/chunk_index": 0,
                        f"{prefix}/file_index": episode,
                        f"{prefix}/from_timestamp": source_row[f"{prefix}/from_timestamp"],
                        f"{prefix}/to_timestamp": source_row[f"{prefix}/to_timestamp"],
                    }
                )

            numeric = {
                key: np.stack(frame[key]) if key in (OBS_STATE, ACTION) else frame[key].to_numpy()
                for key in features
                if key not in VIDEOS
            }
            stats = compute_episode_stats(numeric, features)
            query_state, targets = _training_arrays(state)
            stats[OBS_STATE], stats[ACTION] = _exact_stats([query_state]), _exact_stats(targets)
            stats.update({key: _video_stats(source_row, key) for key in VIDEOS})
            output_meta.save_episode(episode, len(frame), [OUTPUT_TASK], stats, video_metadata)
            state_values.append(query_state)
            action_values.extend(targets)
            lengths.append(len(frame))

        manifest = build_manifest(lengths)
        output_meta.finalize()
        output_meta.stats[OBS_STATE] = _exact_stats(state_values)
        output_meta.stats[ACTION] = _exact_stats(action_values)
        if (
            int(output_meta.stats[OBS_STATE]["count"].item()) != EXPECTED_QUERIES
            or int(output_meta.stats[ACTION]["count"].item()) != EXPECTED_ACTION_VALUES
        ):
            raise ValueError("training-distribution stats counts changed")
        write_stats(output_meta.stats, temporary)
        (temporary / "meta/materialization.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        (temporary / "meta/valid_queries.json").write_text(
            json.dumps({"action_horizon": ACTION_HORIZON, "episodes": manifest["episodes"]}, indent=2) + "\n"
        )
        (temporary / "README.md").write_text(
            "---\nlicense: apache-2.0\ntask_categories: [robotics]\ntags: [LeRobot]\n---\n\n"
            "# Long-gripper dual-UMI BiYAM joints\n\n"
            "Published 14-D states are preserved exactly; see `meta/materialization.json`.\n"
        )
        write_payload_manifest(temporary)
        os.replace(temporary, output_root)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    materialize(args.output_root.resolve(), args.repo_id)


if __name__ == "__main__":
    main()
