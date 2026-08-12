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

"""Pinned observation-only UMI dataset adapter for 20-D bimanual EE training."""

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset

from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.umi_yam import ACTION_DIM, ACTION_HORIZON, encode_bimanual_action
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.rotation import Rotation

SOURCE_REPO_ID = "brandonyang/dual-lidar-combined-filtered-long-gripper"
SOURCE_REVISION = "a29ae6a5531584fb950c7bb3bb5895f18421b108"
JOINT_REPO_ID = "brandonyang/dual-lidar-combined-filtered-joint-positions-long-gripper"
JOINT_REVISION = "387d696eb36de411a11c8b2612a6f19822ca3a53"
GRIPPER_SEMANTICS = "normalized_0_closed_1_open"
STATE_SEMANTICS = "identity_r6d_rows_current_gripper"
TASK = "pick up oranges and place them in the bowl"
DATASET_VERSION = "umi_yam.long_gripper.all_182.v1"
APPROVED_QUERY_COUNT = 175_583
_CACHE_MANIFEST = "meta/umi_yam_ee_long_gripper.json"
PAYLOAD_MANIFEST = "meta/payload.sha256"
_CACHE_CONTRACT = {
    "complete": True,
    "source_revision": SOURCE_REVISION,
    "joint_revision": JOINT_REVISION,
    "dataset_version": DATASET_VERSION,
    "count": APPROVED_QUERY_COUNT,
    "action_shape": [ACTION_HORIZON, ACTION_DIM],
    "gripper_semantics": GRIPPER_SEMANTICS,
    "state_semantics": STATE_SEMANTICS,
    "task": TASK,
    "payload_manifest": PAYLOAD_MANIFEST,
}

LEFT_GRIPPER = "observation.gripper_width.umi1"
RIGHT_GRIPPER = "observation.gripper_width.umi2"
LEFT_VIDEO = "observation.images.umi1"
RIGHT_VIDEO = "observation.images.umi2"

_STATE_NAMES = tuple(f"umi{arm}_{axis}" for arm in (1, 2) for axis in ("x", "y", "z", "rx", "ry", "rz"))
_EXPECTED_FEATURES = {
    OBS_STATE: ("float32", (12,), _STATE_NAMES),
    LEFT_VIDEO: ("video", (600, 800, 3), ("height", "width", "channels")),
    RIGHT_VIDEO: ("video", (600, 800, 3), ("height", "width", "channels")),
    LEFT_GRIPPER: ("float32", (1,), ("umi1_width_mm",)),
    RIGHT_GRIPPER: ("float32", (1,), ("umi2_width_mm",)),
    "timestamp": ("float32", (1,), None),
    "frame_index": ("int64", (1,), None),
    "episode_index": ("int64", (1,), None),
    "index": ("int64", (1,), None),
    "task_index": ("int64", (1,), None),
}
_JOINT_NAMES = (
    *(f"left_joint_{i}.pos" for i in range(6)),
    "left_gripper.pos",
    *(f"right_joint_{i}.pos" for i in range(6)),
    "right_gripper.pos",
)
_ARM_NAMES = ("x", "y", "z", "r0x", "r0y", "r0z", "r1x", "r1y", "r1z", "gripper")
FEATURE_NAMES = tuple(f"{arm}_{name}" for arm in ("left", "right") for name in _ARM_NAMES)
_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


@dataclass(frozen=True)
class SourceContract:
    repo_id: str = SOURCE_REPO_ID
    revision: str = SOURCE_REVISION
    fps: int = 30
    episodes: int = 182
    frames: int = 179_951

    def validate_identity(self, source: Any) -> None:
        actual = (
            source.repo_id,
            source.revision,
            source.meta.fps,
            source.meta.total_episodes,
            source.meta.total_frames,
        )
        expected = (self.repo_id, self.revision, self.fps, self.episodes, self.frames)
        if actual != expected:
            raise ValueError(f"UMI source identity mismatch: expected {expected}, got {actual}")

    def validate(self, source: Any) -> None:
        self.validate_identity(source)
        actual = {
            key: (
                feature["dtype"],
                tuple(feature["shape"]),
                None if feature.get("names") is None else tuple(feature["names"]),
            )
            for key, feature in source.meta.features.items()
        }
        if actual != _EXPECTED_FEATURES:
            raise ValueError("UMI source features do not match the pinned observation-only schema")
        if source.meta.tasks is None or list(source.meta.tasks.index) != ["bimanual umi demo"]:
            raise ValueError("UMI source task does not match the pinned schema")


AUTHORITATIVE_SOURCE = SourceContract()
AUTHORITATIVE_JOINT_SOURCE = SourceContract(repo_id=JOINT_REPO_ID, revision=JOINT_REVISION)


@dataclass(frozen=True)
class EEEpisode:
    state: np.ndarray
    action: np.ndarray


def _pose_matrices(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 6 or not np.isfinite(pose).all():
        raise ValueError(f"pose must be finite (frames, 6), got {pose.shape}")
    result = np.repeat(np.eye(4)[None], len(pose), axis=0)
    result[:, :3, 3] = pose[:, :3]
    result[:, :3, :3] = np.stack([Rotation.from_rotvec(v).as_matrix() for v in pose[:, 3:]])
    return result


def materialize_episode(state: Any, left_gripper: Any, right_gripper: Any) -> EEEpisode:
    source = np.asarray(state, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 12 or not np.isfinite(source).all():
        raise ValueError(f"state must be finite (frames, 12), got {source.shape}")
    frames = len(source)
    if frames <= ACTION_HORIZON:
        raise ValueError(f"episode needs more than {ACTION_HORIZON} frames")
    left, right = (np.asarray(value, dtype=np.float64).reshape(-1) for value in (left_gripper, right_gripper))
    if left.shape != (frames,) or right.shape != (frames,) or not np.isfinite([left, right]).all():
        raise ValueError("grippers must contain one finite value per frame")
    if min(left.min(), right.min()) < 0 or max(left.max(), right.max()) > 1:
        raise ValueError(f"grippers must use {GRIPPER_SEMANTICS}")

    left_pose, right_pose = _pose_matrices(source[:, :6]), _pose_matrices(source[:, 6:])
    queries = frames - ACTION_HORIZON
    q = np.arange(queries)
    identity = np.broadcast_to(np.eye(4), (queries, 4, 4))
    left_inverse, right_inverse = np.linalg.inv(left_pose[q]), np.linalg.inv(right_pose[q])
    derived_state = encode_bimanual_action(
        left_delta_transform=identity,
        left_gripper=left[q],
        right_delta_transform=identity,
        right_gripper=right[q],
    )
    targets = q[:, None] + np.arange(1, ACTION_HORIZON + 1)
    action = encode_bimanual_action(
        left_delta_transform=left_inverse[:, None] @ left_pose[targets],
        left_gripper=left[targets],
        right_delta_transform=right_inverse[:, None] @ right_pose[targets],
        right_gripper=right[targets],
    )
    return EEEpisode(derived_state.astype(np.float32), action.astype(np.float32))


def exact_vector_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Compute exact per-dimension training stats for vector or chunk arrays."""

    values = np.asarray(values)
    if values.ndim < 2 or values.shape[-1] == 0 or not values.size:
        raise ValueError("stats require a non-empty array of vectors")
    values = values.reshape(-1, values.shape[-1]).astype(np.float64)
    result = {name: getattr(values, name)(0) for name in ("min", "max", "mean", "std")}
    result["count"] = np.array([len(values)])
    for quantile, value in zip(_QUANTILES, np.quantile(values, _QUANTILES, axis=0), strict=True):
        result[f"q{int(quantile * 100):02d}"] = value
    return result


def write_payload_manifest(root: str | Path) -> Path:
    """Hash every cache payload once so a launcher can verify it before distributed use."""

    root = Path(root)
    output = root / PAYLOAD_MANIFEST
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path != output)
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError("payload must contain regular files only")
    lines = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}\n")
    output.write_text("".join(lines))
    return output


def _stack(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.stack([v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values])


def _batch(source: Any, episode: int) -> dict[str, Any]:
    row = source.meta.episodes[episode]
    return source.hf_dataset[list(range(int(row["dataset_from_index"]), int(row["dataset_to_index"])))]


def validate_joint_source(source: Any, contract: SourceContract = AUTHORITATIVE_JOINT_SOURCE) -> None:
    contract.validate_identity(source)
    expected_keys = set(_EXPECTED_FEATURES) - {LEFT_GRIPPER, RIGHT_GRIPPER}
    if set(source.meta.features) != expected_keys:
        raise ValueError("paired joint source feature keys do not match the pinned schema")
    state = source.meta.features[OBS_STATE]
    if (state["dtype"], tuple(state["shape"]), tuple(state["names"])) != ("float32", (14,), _JOINT_NAMES):
        raise ValueError("paired joint source state does not match the pinned 14-D schema")
    if source.episodes is not None:
        raise ValueError("paired joint source must be unfiltered")


def _paired_grippers(ee: dict[str, Any], joint: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    for key in ("index", "episode_index", "frame_index"):
        if not np.array_equal(_stack(ee[key]), _stack(joint[key])):
            raise ValueError(f"paired sources are not row-aligned at {key}")
    if not np.allclose(_stack(ee["timestamp"]), _stack(joint["timestamp"]), atol=1e-6, rtol=0):
        raise ValueError("paired sources are not timestamp-aligned")
    state = _stack(joint[OBS_STATE])
    return state[:, 6], state[:, 13]


def materialize_cache(
    source: Any,
    joint_source: Any,
    root: str | Path,
    contract: SourceContract = AUTHORITATIVE_SOURCE,
    joint_contract: SourceContract = AUTHORITATIVE_JOINT_SOURCE,
) -> dict[str, dict[str, np.ndarray]]:
    """Write shared memory-mapped numeric features; source videos stay in place."""

    contract.validate(source)
    validate_joint_source(joint_source, joint_contract)
    if source.episodes is not None:
        raise ValueError("materialization requires the unfiltered pinned source")
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    write_json({"complete": False}, root / _CACHE_MANIFEST)
    count = sum(
        int(source.meta.episodes[i]["dataset_to_index"])
        - int(source.meta.episodes[i]["dataset_from_index"])
        - ACTION_HORIZON
        for i in range(contract.episodes)
    )
    mmap = np.lib.format.open_memmap
    state = mmap(root / "state.npy", mode="w+", dtype="float32", shape=(count, ACTION_DIM))
    action = mmap(root / "action.npy", mode="w+", dtype="float32", shape=(count, ACTION_HORIZON, ACTION_DIM))
    source_index = mmap(root / "source_index.npy", mode="w+", dtype="int64", shape=(count,))
    offset = 0
    for episode in range(contract.episodes):
        ee, joint = _batch(source, episode), _batch(joint_source, episode)
        derived = materialize_episode(_stack(ee[OBS_STATE]), *_paired_grippers(ee, joint))
        size, row = len(derived.state), source.meta.episodes[episode]
        state[offset : offset + size], action[offset : offset + size] = derived.state, derived.action
        source_index[offset : offset + size] = np.arange(
            int(row["dataset_from_index"]), int(row["dataset_to_index"]) - ACTION_HORIZON
        )
        offset += size
    for array in (state, action, source_index):
        array.flush()
    stats = {OBS_STATE: exact_vector_stats(state), ACTION: exact_vector_stats(action)}
    write_stats(stats, root)
    manifest = _CACHE_CONTRACT | {
        "source_revision": contract.revision,
        "joint_revision": joint_contract.revision,
        "count": count,
    }
    write_json(manifest, root / _CACHE_MANIFEST)
    write_payload_manifest(root)
    return stats


def _derived_meta(source: Any, stats: dict, contract: SourceContract) -> Any:
    if source.meta.tasks is None or len(source.meta.tasks) != 1:
        raise ValueError("pinned EE source must contain exactly one task")
    meta = copy.copy(source.meta)
    meta.info = copy.deepcopy(source.meta.info)
    meta.info.features = copy.deepcopy(source.meta.features)
    meta.info.features.pop(LEFT_GRIPPER)
    meta.info.features.pop(RIGHT_GRIPPER)
    vector_feature = {"dtype": "float32", "shape": (ACTION_DIM,), "names": FEATURE_NAMES}
    meta.info.features[OBS_STATE] = copy.deepcopy(vector_feature)
    meta.info.features[ACTION] = vector_feature
    rows, start = [], 0
    for episode in range(len(source.meta.episodes)):
        row = dict(source.meta.episodes[episode])
        length = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        queries = length - ACTION_HORIZON
        if length <= ACTION_HORIZON:
            raise ValueError(f"episode {episode} is too short")
        row.update(dataset_from_index=start, dataset_to_index=start + queries, length=queries)
        row["tasks"] = [TASK]
        rows.append(row)
        start += queries
    if contract == AUTHORITATIVE_SOURCE and start != APPROVED_QUERY_COUNT:
        raise ValueError(f"pinned split expected {APPROVED_QUERY_COUNT} queries, got {start}")
    if (
        int(np.asarray(stats[OBS_STATE]["count"]).item()) != start
        or int(np.asarray(stats[ACTION]["count"]).item()) != start * ACTION_HORIZON
    ):
        raise ValueError("stats counts do not match the pinned split")
    meta.info.total_frames = start
    meta.episodes = Dataset.from_list(rows)
    meta.tasks = source.meta.tasks.copy()
    meta.tasks.index = [TASK]
    meta.tasks.index.name = "task"
    meta.info.total_tasks = 1
    meta.stats = copy.deepcopy(source.meta.stats or {})
    meta.stats.pop(LEFT_GRIPPER, None)
    meta.stats.pop(RIGHT_GRIPPER, None)
    meta.stats.update(copy.deepcopy(stats))
    return meta


class UMIYAMEEDataset(torch.utils.data.Dataset):
    """Runtime view that preserves source observations/videos and replaces state/action."""

    def __init__(
        self,
        source: Any,
        cache_root: str | Path,
        *,
        contract: SourceContract = AUTHORITATIVE_SOURCE,
    ) -> None:
        contract.validate(source)
        if source.episodes is not None:
            raise ValueError("pass episode selection to UMIYAMEEDataset, not its source")
        cache_root = Path(cache_root)
        manifest = load_json(cache_root / _CACHE_MANIFEST)
        if contract == AUTHORITATIVE_SOURCE and any(manifest.get(k) != v for k, v in _CACHE_CONTRACT.items()):
            raise ValueError(f"EE cache provenance mismatch: expected {_CACHE_CONTRACT}, got {manifest}")
        self.source = source
        self._cache_root = cache_root
        self._open_cache_arrays()
        if not (cache_root / PAYLOAD_MANIFEST).is_file():
            raise ValueError("EE cache payload manifest is missing")
        stats = load_stats(cache_root)
        expected_shapes = (
            (int(manifest["count"]), ACTION_DIM),
            (int(manifest["count"]), ACTION_HORIZON, ACTION_DIM),
            (int(manifest["count"]),),
        )
        if (
            stats is None
            or (self._state.shape, self._action.shape, self._source_index.shape) != expected_shapes
        ):
            raise ValueError("EE cache is incomplete")
        self.meta = _derived_meta(source, stats, contract)
        self.episodes = list(range(contract.episodes))
        self.repo_id, self.revision = source.repo_id, source.revision
        self.root, self.fps = source.root, source.fps
        self.absolute_to_relative_idx = None
        self.num_frames, self.num_episodes, self.features = len(self), len(self.episodes), self.meta.features

    def _open_cache_arrays(self) -> None:
        self._state = np.load(self._cache_root / "state.npy", mmap_mode="r")
        self._action = np.load(self._cache_root / "action.npy", mmap_mode="r")
        self._source_index = np.load(self._cache_root / "source_index.npy", mmap_mode="r")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for name in ("_state", "_action", "_source_index"):
            state.pop(name, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open_cache_arrays()

    def __len__(self) -> int:
        return len(self._state)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        item = dict(self.source[int(self._source_index[index])])
        item.pop(LEFT_GRIPPER, None)
        item.pop(RIGHT_GRIPPER, None)
        item[OBS_STATE] = torch.from_numpy(self._state[index].copy())
        item[ACTION] = torch.from_numpy(self._action[index].copy())
        item["action_is_pad"] = torch.zeros(ACTION_HORIZON, dtype=torch.bool)
        item["task"] = TASK
        return item
