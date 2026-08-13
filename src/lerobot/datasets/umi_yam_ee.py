#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Runtime view of standard dual-UMI data as query-relative EE actions."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub.utils import WeakFileLock

from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STATE

from .compute_stats import RunningQuantileStats
from .io_utils import load_stats, write_stats

ACTION_HORIZON = 24
ACTION_DIM = 20
ACTION_CONTRACT = "umi_yam.ee.current_relative.r6d_rows.v1"

_SOURCE_NAMES = tuple(
    f"umi{arm}_{name}" for arm in (1, 2) for name in ("x", "y", "z", "rx", "ry", "rz", "gripper")
)
_ARM_NAMES = ("x", "y", "z", "r0x", "r0y", "r0z", "r1x", "r1y", "r1z", "gripper")
FEATURE_NAMES = tuple(f"{side}_{name}" for side in ("left", "right") for name in _ARM_NAMES)
_IMAGE_KEYS = ("observation.images.umi1", "observation.images.umi2")
_IDENTITY_R6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _validate_source(source: Any) -> None:
    if source.meta.fps != 30:
        raise ValueError(f"UMI EE training requires 30 FPS, got {source.meta.fps}")
    if tuple(source.meta.camera_keys) != _IMAGE_KEYS:
        raise ValueError(f"UMI EE training requires cameras in this exact order: {_IMAGE_KEYS}")
    for key in (OBS_STATE, ACTION):
        feature = source.meta.features.get(key)
        actual = (
            None
            if feature is None
            else (
                feature.get("dtype"),
                tuple(feature.get("shape", ())),
                tuple(feature.get("names") or ()),
            )
        )
        if actual != ("float32", (14,), _SOURCE_NAMES):
            raise ValueError(f"{key} must use the standard dual-UMI 14-D schema")
    for key in _IMAGE_KEYS:
        feature = source.meta.features.get(key)
        if (
            feature is None
            or feature.get("dtype") != "video"
            or tuple(feature.get("shape", ()))
            != (
                600,
                800,
                3,
            )
        ):
            raise ValueError(f"{key} must be an 800x600 RGB video")


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues conversion for arrays shaped ``(..., 3)``."""

    value = np.asarray(rotvec, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise ValueError(f"rotation vectors must be finite (..., 3), got {value.shape}")
    theta2 = np.sum(value * value, axis=-1)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-12
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    a[~small] = np.sin(theta[~small]) / theta[~small]
    b[~small] = (1.0 - np.cos(theta[~small])) / theta2[~small]
    a[small] = 1.0 - theta2[small] / 6.0 + theta2[small] ** 2 / 120.0
    b[small] = 0.5 - theta2[small] / 24.0 + theta2[small] ** 2 / 720.0

    x, y, z = np.moveaxis(value, -1, 0)
    skew = np.zeros(value.shape[:-1] + (3, 3), dtype=np.float64)
    skew[..., 0, 1], skew[..., 0, 2] = -z, y
    skew[..., 1, 0], skew[..., 1, 2] = z, -x
    skew[..., 2, 0], skew[..., 2, 1] = -y, x
    identity = np.broadcast_to(np.eye(3), skew.shape)
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def _validate_rows(value: Any, *, name: str, leading_shape: tuple[int, ...]) -> np.ndarray:
    rows = np.asarray(value, dtype=np.float64)
    if rows.shape != leading_shape + (14,) or not np.isfinite(rows).all():
        raise ValueError(f"{name} must be finite {leading_shape + (14,)}, got {rows.shape}")
    grippers = rows[..., (6, 13)]
    if np.any((grippers < 0.0) | (grippers > 1.0)):
        raise ValueError(f"{name} grippers must be normalized with 0=closed and 1=open")
    return rows


def _derive_state(current: Any) -> np.ndarray:
    rows = np.asarray(current)
    rows = _validate_rows(rows, name="current UMI state", leading_shape=rows.shape[:-1])
    identity = np.broadcast_to(_IDENTITY_R6D, rows.shape[:-1] + (6,))
    zeros = np.zeros(rows.shape[:-1] + (3,))
    left = np.concatenate((zeros, identity, rows[..., 6, None]), axis=-1)
    right = np.concatenate((zeros, identity, rows[..., 13, None]), axis=-1)
    return np.concatenate((left, right), axis=-1).astype(np.float32)


def _derive_arm_actions(current: np.ndarray, future: np.ndarray, offset: int) -> np.ndarray:
    query_rotation = _rotvec_to_matrix(current[..., offset + 3 : offset + 6])
    future_rotation = _rotvec_to_matrix(future[..., offset + 3 : offset + 6])
    query_rotation_t = np.swapaxes(query_rotation, -1, -2)
    delta_position = future[..., offset : offset + 3] - current[..., None, offset : offset + 3]
    relative_position = np.einsum("...ij,...hj->...hi", query_rotation_t, delta_position)
    relative_rotation = np.einsum("...ij,...hjk->...hik", query_rotation_t, future_rotation)
    r6d = relative_rotation[..., :2, :].reshape(relative_rotation.shape[:-2] + (6,))
    return np.concatenate((relative_position, r6d, future[..., offset + 6, None]), axis=-1)


def derive_query_relative_actions(current: Any, future: Any) -> np.ndarray:
    """Convert standard 14-D rows into H24 EE20 actions for each query."""

    query = np.asarray(current)
    query = _validate_rows(query, name="current UMI state", leading_shape=query.shape[:-1])
    targets = _validate_rows(
        future,
        name="future UMI action",
        leading_shape=query.shape[:-1] + (ACTION_HORIZON,),
    )
    left = _derive_arm_actions(query, targets, 0)
    right = _derive_arm_actions(query, targets, 7)
    return np.concatenate((left, right), axis=-1).astype(np.float32)


def _numeric_columns(source: Any, *names: str) -> dict[str, np.ndarray]:
    data = source.hf_dataset.select_columns(list(names)).with_format("numpy")[:]
    return {name: np.asarray(data[name]) for name in names}


def _episode_slices(episode_index: np.ndarray) -> list[slice]:
    if episode_index.ndim != 1 or len(episode_index) == 0:
        raise ValueError("UMI dataset must contain episode indices")
    boundaries = np.flatnonzero(np.diff(episode_index) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(episode_index)]))
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _derived_episode_rows(source: Any) -> tuple[list[dict[str, Any]], dict[int, int], int]:
    rows = []
    starts = {}
    total_frames = 0
    for source_row in source.meta.episodes:
        row = dict(source_row)
        episode_id = int(row["episode_index"])
        length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        if length <= ACTION_HORIZON:
            raise ValueError(f"UMI episode {episode_id} is too short for H24")
        derived_length = length - ACTION_HORIZON
        starts[episode_id] = total_frames
        row["dataset_from_index"] = total_frames
        total_frames += derived_length
        row["dataset_to_index"] = total_frames
        row["length"] = derived_length
        rows.append(row)
    return rows, starts, total_frames


def _valid_source_rows(source: Any) -> tuple[np.ndarray, np.ndarray]:
    columns = _numeric_columns(source, "index", "episode_index", "frame_index")
    absolute = columns["index"].reshape(-1).astype(np.int64)
    episode = columns["episode_index"].reshape(-1).astype(np.int64)
    frame = columns["frame_index"].reshape(-1).astype(np.int64)
    valid = np.zeros(len(absolute), dtype=bool)
    derived_absolute = np.full(len(absolute), -1, dtype=np.int64)
    _, derived_starts, _ = _derived_episode_rows(source)
    seen_episodes: set[int] = set()
    for ep_slice in _episode_slices(episode):
        episode_id = int(episode[ep_slice.start])
        if episode_id in seen_episodes:
            raise ValueError(f"UMI episode {episode_id} occurs in multiple row blocks")
        seen_episodes.add(episode_id)
        if not 0 <= episode_id < len(source.meta.episodes):
            raise ValueError(f"UMI episode {episode_id} is missing from episode metadata")
        metadata = source.meta.episodes[episode_id]
        if int(metadata["episode_index"]) != episode_id:
            raise ValueError(f"UMI episode metadata row {episode_id} has a mismatched episode index")
        length = ep_slice.stop - ep_slice.start
        metadata_start = int(metadata["dataset_from_index"])
        metadata_stop = int(metadata["dataset_to_index"])
        if length != metadata_stop - metadata_start:
            raise ValueError(f"UMI episode {episode_id} row count disagrees with episode metadata")
        if length <= ACTION_HORIZON:
            raise ValueError(f"UMI episode {episode_id} has only {length} frames")
        if not np.array_equal(frame[ep_slice], np.arange(length)):
            raise ValueError(f"UMI episode {episode_id} frame indices are not contiguous")
        if not np.array_equal(absolute[ep_slice], metadata_start + np.arange(length)):
            raise ValueError(f"UMI episode {episode_id} absolute indices disagree with episode metadata")
        valid_stop = ep_slice.stop - ACTION_HORIZON
        valid[ep_slice.start : valid_stop] = True
        derived_absolute[ep_slice.start : valid_stop] = derived_starts[episode_id] + np.arange(
            length - ACTION_HORIZON
        )
    return np.flatnonzero(valid), derived_absolute[valid]


def _snap_constant_stats(stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    constant = stats["min"] == stats["max"]
    if constant.any():
        stats["std"][constant] = 0.0
        for key in ("q01", "q10", "q50", "q90", "q99"):
            stats[key][constant] = stats["min"][constant]
    return stats


def _compute_stats(source: Any) -> dict[str, dict[str, np.ndarray]]:
    columns = _numeric_columns(source, OBS_STATE, ACTION, "episode_index")
    state = columns[OBS_STATE].astype(np.float32, copy=False)
    action = columns[ACTION].astype(np.float32, copy=False)
    episode = columns["episode_index"].reshape(-1)
    if state.shape != action.shape or state.shape[1:] != (14,):
        raise ValueError("UMI state and action must both have shape (frames, 14)")
    if not np.array_equal(state.view(np.uint32), action.view(np.uint32)):
        raise ValueError("standard UMI action[t] must be a bit-exact copy of observation.state[t]")

    state_stats = RunningQuantileStats()
    action_stats = RunningQuantileStats()
    for ep_slice in _episode_slices(episode):
        ep_state = state[ep_slice]
        ep_action = action[ep_slice]
        if np.any(ep_state[0, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]] != 0.0):
            raise ValueError(f"UMI episode {episode[ep_slice.start]} does not start at both arm origins")
        queries = len(ep_state) - ACTION_HORIZON
        target_indices = np.arange(queries)[:, None] + np.arange(1, ACTION_HORIZON + 1)
        state_stats.update(_derive_state(ep_state[:queries]))
        action_stats.update(derive_query_relative_actions(ep_state[:queries], ep_action[target_indices]))
    return {
        OBS_STATE: _snap_constant_stats(state_stats.get_statistics()),
        ACTION: _snap_constant_stats(action_stats.get_statistics()),
    }


def _cached_stats(source: Any) -> dict[str, dict[str, np.ndarray]]:
    identity = json.dumps(
        {
            "contract": ACTION_CONTRACT,
            "repo_id": source.repo_id,
            "revision": source.revision,
            "episodes": source.episodes,
            "frames": source.num_frames,
            "fingerprint": source.hf_dataset._fingerprint,
        },
        sort_keys=True,
    )
    cache_root = HF_LEROBOT_HOME / "derived_stats" / hashlib.sha256(identity.encode()).hexdigest()
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(cache_root.with_suffix(".lock")):
        try:
            stats = load_stats(cache_root)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            stats = None
        if stats is None:
            stats = _compute_stats(source)
            write_stats(stats, cache_root)
    return stats


def _derived_meta(source: Any, stats: dict[str, dict[str, np.ndarray]]) -> Any:
    meta = copy.copy(source.meta)
    meta.info = copy.deepcopy(source.meta.info)
    meta.info.features = copy.deepcopy(source.meta.features)
    vector_feature = {"dtype": "float32", "shape": (ACTION_DIM,), "names": FEATURE_NAMES}
    meta.info.features[OBS_STATE] = copy.deepcopy(vector_feature)
    meta.info.features[ACTION] = vector_feature

    episode_rows, _, total_frames = _derived_episode_rows(source)
    meta.episodes = Dataset.from_list(episode_rows)
    meta.info.total_frames = total_frames
    meta.stats = {
        key: copy.deepcopy(source.meta.stats[key])
        for key in source.meta.camera_keys
        if source.meta.stats is not None and key in source.meta.stats
    }
    meta.stats.update(copy.deepcopy(stats))
    return meta


class UMIYAMEEDataset(torch.utils.data.Dataset):
    """In-memory EE20/H24 view; source parquet and videos remain unchanged."""

    def __init__(self, source: Any, *, stats: dict[str, dict[str, np.ndarray]] | None = None) -> None:
        _validate_source(source)
        self.source = source
        self._source_rows, self._absolute_indices = _valid_source_rows(source)
        self._action_rows = source.hf_dataset.select_columns(ACTION).with_format("numpy")
        self._hf_dataset: Dataset | None = None
        self._absolute_to_relative_idx = {
            int(absolute): relative for relative, absolute in enumerate(self._absolute_indices)
        }
        derived_stats = _cached_stats(source) if stats is None else stats
        self.meta = _derived_meta(source, derived_stats)
        self.repo_id, self.revision = source.repo_id, source.revision
        self.root, self.fps, self.episodes = source.root, source.fps, source.episodes

    def __len__(self) -> int:
        return len(self._source_rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index = int(self._source_rows[int(index)])
        item = dict(self.source[source_index])
        current = item[OBS_STATE]
        future = np.asarray(self._action_rows[source_index + 1 : source_index + 1 + ACTION_HORIZON][ACTION])
        item["index"] = torch.tensor(int(self._absolute_indices[int(index)]), dtype=torch.int64)
        item[OBS_STATE] = torch.from_numpy(_derive_state(current))
        item[ACTION] = torch.from_numpy(derive_query_relative_actions(current, future))
        item["action_is_pad"] = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
        return item

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return self.source.num_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_dataset(self) -> Dataset:
        if self._hf_dataset is None:
            selected = self.source.hf_dataset.select(self._source_rows.tolist())
            self._hf_dataset = selected.remove_columns("index").add_column("index", self._absolute_indices)
        return self._hf_dataset

    @property
    def absolute_to_relative_idx(self) -> dict[int, int]:
        return self._absolute_to_relative_idx
