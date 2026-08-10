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
"""Query-anchored SE(3) sampling for the dual-LiDAR UMI dataset.

The v1 artifact stores a compact per-frame helper containing both jaw-centre
TCP poses and gripper apertures.  A training target cannot be materialized as a
single per-frame column: all 24 future poses must be expressed in the frame of
the *sampled query* pose.  :class:`UmiCurrentRelativeR6dDataset` therefore asks
the regular LeRobot reader for the helper window ``[t, t+1, ..., t+H]`` and
constructs the complete target only after the sampler has selected ``t``.

Rotation6D convention
---------------------
The six numbers are the first two **columns** of an active 3x3 rotation matrix,
concatenated column-by-column::

    [R00, R10, R20, R01, R11, R21]

Decoding applies Gram-Schmidt to those columns and obtains the third column as
``cross(column_0, column_1)``.  In particular, identity is represented by
``[1, 0, 0, 0, 1, 0]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import DEFAULT_DEPTH_UNIT
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_STATE

UMI_CURRENTREL_SCHEMA_ID = "dual-lidar-umi-currentrel-r6d-v1"
UMI_CURRENTREL_METADATA_PATH = Path("meta/umi_current_relative_r6d_v1.json")
UMI_CURRENTREL_ONSET_SCHEMA_ID = "dual-lidar-umi-currentrel-r6d-onset-v2"
UMI_CURRENTREL_ONSET_METADATA_PATH = Path("meta/umi_current_relative_r6d_onset_v2.json")
UMI_CURRENTREL_ONSET_V3_SCHEMA_ID = "dual-lidar-umi-currentrel-r6d-onset-v3"
UMI_CURRENTREL_ONSET_V3_METADATA_PATH = Path("meta/umi_current_relative_r6d_onset_v3.json")
UMI_CURRENTREL_SPLIT_PATH = Path("meta/split_manifest.json")
UMI_TCP_WINDOW_KEY = "umi.tcp_and_gripper"
PADDING_SUPERVISE_CLAMPED_FUTURE_ROWS = "supervise_clamped_future_rows"
PADDING_EXCLUDE_PADDED_FUTURE_ROWS = "exclude_padded_future_rows"

_UMI_CURRENTREL_SCHEMA_BY_METADATA_PATH = {
    UMI_CURRENTREL_METADATA_PATH: UMI_CURRENTREL_SCHEMA_ID,
    UMI_CURRENTREL_ONSET_METADATA_PATH: UMI_CURRENTREL_ONSET_SCHEMA_ID,
    UMI_CURRENTREL_ONSET_V3_METADATA_PATH: UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
}
_PADDING_SEMANTICS_BY_SCHEMA = {
    UMI_CURRENTREL_SCHEMA_ID: PADDING_SUPERVISE_CLAMPED_FUTURE_ROWS,
    UMI_CURRENTREL_ONSET_SCHEMA_ID: PADDING_SUPERVISE_CLAMPED_FUTURE_ROWS,
    UMI_CURRENTREL_ONSET_V3_SCHEMA_ID: PADDING_EXCLUDE_PADDED_FUTURE_ROWS,
}

UMI_CURRENTREL_HORIZON = 24
UMI_CURRENTREL_STATE_DIM = 20
UMI_CURRENTREL_HELPER_DIM = 16

UMI_CURRENTREL_POSE_R6D_NAMES = (
    "relative_x_m",
    "relative_y_m",
    "relative_z_m",
    "relative_r6d_col0_x",
    "relative_r6d_col0_y",
    "relative_r6d_col0_z",
    "relative_r6d_col1_x",
    "relative_r6d_col1_y",
    "relative_r6d_col1_z",
)
UMI_CURRENTREL_STATE_NAMES = (
    *(f"left_previous_{name}" for name in UMI_CURRENTREL_POSE_R6D_NAMES),
    "left_current_gripper",
    *(f"right_previous_{name}" for name in UMI_CURRENTREL_POSE_R6D_NAMES),
    "right_current_gripper",
)
UMI_CURRENTREL_ACTION_NAMES = (
    *(f"left_future_{name}" for name in UMI_CURRENTREL_POSE_R6D_NAMES),
    "left_future_gripper",
    *(f"right_future_{name}" for name in UMI_CURRENTREL_POSE_R6D_NAMES),
    "right_future_gripper",
)

LEFT_STATE_SLICE = slice(0, 10)
RIGHT_STATE_SLICE = slice(10, 20)


def validate_rigid_transform(transform: np.ndarray, *, name: str = "transform", atol: float = 1e-7) -> None:
    """Validate a finite, right-handed homogeneous rigid transform."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{name} rotation must have determinant +1")


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one homogeneous transform without a generic matrix inverse."""

    matrix = np.asarray(transform, dtype=np.float64)
    validate_rigid_transform(matrix)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def matrix_to_rotation_6d(rotation: np.ndarray) -> np.ndarray:
    """Encode active rotation matrices using their first two columns."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in shape (3, 3), got {matrix.shape}")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def rotation_6d_to_matrix(rotation_6d: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Decode first-two-column Rotation6D with right-handed Gram-Schmidt."""

    vector = np.asarray(rotation_6d, dtype=np.float64)
    if vector.shape[-1] != 6:
        raise ValueError(f"Rotation6D must have width 6, got {vector.shape}")
    first = vector[..., :3]
    second = vector[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < eps):
        raise ValueError("Rotation6D first column is degenerate")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < eps):
        raise ValueError("Rotation6D columns are degenerate")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def xyzw_pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert ``xyz + quaternion_xyzw`` pose vectors to homogeneous matrices."""

    from scipy.spatial.transform import Rotation

    value = np.asarray(pose, dtype=np.float64)
    if value.shape[-1] != 7:
        raise ValueError(f"pose must have width 7, got {value.shape}")
    matrices = np.broadcast_to(np.eye(4, dtype=np.float64), (*value.shape[:-1], 4, 4)).copy()
    matrices[..., :3, :3] = Rotation.from_quat(value[..., 3:]).as_matrix()
    matrices[..., :3, 3] = value[..., :3]
    return matrices


def matrix_to_xyzw_pose(transform: np.ndarray) -> np.ndarray:
    """Convert homogeneous matrices to ``xyz + quaternion_xyzw`` vectors."""

    from scipy.spatial.transform import Rotation

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"transform must end in shape (4, 4), got {matrix.shape}")
    return np.concatenate((matrix[..., :3, 3], Rotation.from_matrix(matrix[..., :3, :3]).as_quat()), axis=-1)


def relative_transform(anchor: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``inverse(anchor) @ target`` for active homogeneous transforms."""

    anchor_matrix = np.asarray(anchor, dtype=np.float64)
    target_matrix = np.asarray(target, dtype=np.float64)
    if anchor_matrix.shape != (4, 4) or target_matrix.shape != (4, 4):
        raise ValueError("anchor and target must each have shape (4, 4)")
    return invert_rigid_transform(anchor_matrix) @ target_matrix


def encode_relative_pose(transform: np.ndarray) -> np.ndarray:
    """Encode one relative transform as ``xyz + first-two-column Rotation6D``."""

    matrix = np.asarray(transform, dtype=np.float64)
    validate_rigid_transform(matrix)
    return np.concatenate((matrix[:3, 3], matrix_to_rotation_6d(matrix[:3, :3])))


def decode_relative_pose(value: np.ndarray) -> np.ndarray:
    """Decode ``xyz + Rotation6D`` into a homogeneous transform."""

    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (9,):
        raise ValueError(f"relative pose must have shape (9,), got {vector.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_6d_to_matrix(vector[3:])
    matrix[:3, 3] = vector[:3]
    return matrix


def pack_tcp_and_gripper(tcp: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Pack tracks as ``left xyz+quat+grip, right xyz+quat+grip``."""

    transforms = np.asarray(tcp, dtype=np.float64)
    apertures = np.asarray(gripper, dtype=np.float64)
    if transforms.ndim != 4 or transforms.shape[1:] != (2, 4, 4):
        raise ValueError(f"tcp must have shape (N, 2, 4, 4), got {transforms.shape}")
    if apertures.shape != (transforms.shape[0], 2):
        raise ValueError(f"gripper must have shape ({transforms.shape[0]}, 2), got {apertures.shape}")
    poses = matrix_to_xyzw_pose(transforms)
    return np.concatenate((poses[:, 0], apertures[:, 0, None], poses[:, 1], apertures[:, 1, None]), axis=-1)


def unpack_tcp_and_gripper(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_tcp_and_gripper` for any leading window shape."""

    values = np.asarray(window, dtype=np.float64)
    if values.shape[-1] != UMI_CURRENTREL_HELPER_DIM:
        raise ValueError(f"TCP helper must have width {UMI_CURRENTREL_HELPER_DIM}, got {values.shape}")
    left_pose = values[..., :7]
    left_gripper = values[..., 7]
    right_pose = values[..., 8:15]
    right_gripper = values[..., 15]
    tcp = np.stack((xyzw_pose_to_matrix(left_pose), xyzw_pose_to_matrix(right_pose)), axis=-3)
    gripper = np.stack((left_gripper, right_gripper), axis=-1)
    return tcp, gripper


def build_query_state(tcp: np.ndarray, gripper: np.ndarray, query_index: int) -> np.ndarray:
    """Build the 20-D state containing measured ``inverse(T_t) @ T_(t-1)`` history."""

    transforms = np.asarray(tcp, dtype=np.float64)
    apertures = np.asarray(gripper, dtype=np.float64)
    if transforms.ndim != 4 or transforms.shape[1:] != (2, 4, 4):
        raise ValueError(f"tcp must have shape (N, 2, 4, 4), got {transforms.shape}")
    if apertures.shape != (transforms.shape[0], 2):
        raise ValueError(f"gripper must have shape ({transforms.shape[0]}, 2), got {apertures.shape}")
    if not 0 <= query_index < transforms.shape[0]:
        raise IndexError(query_index)
    previous_index = max(0, query_index - 1)
    state = np.empty(UMI_CURRENTREL_STATE_DIM, dtype=np.float64)
    for arm, arm_slice in ((0, LEFT_STATE_SLICE), (1, RIGHT_STATE_SLICE)):
        history = relative_transform(transforms[query_index, arm], transforms[previous_index, arm])
        state[arm_slice.start : arm_slice.stop - 1] = encode_relative_pose(history)
        state[arm_slice.stop - 1] = apertures[query_index, arm]
    return state


def build_same_anchor_action_chunk(
    tcp: np.ndarray,
    gripper: np.ndarray,
    query_index: int,
    *,
    horizon: int = UMI_CURRENTREL_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``inverse(T_t) @ T_(t+k)`` for ``k=1..horizon``.

    Every row uses the same ``T_t`` anchor.  Future indices clamp to the last
    episode frame; the returned Boolean vector identifies clamped rows.
    """

    transforms = np.asarray(tcp, dtype=np.float64)
    apertures = np.asarray(gripper, dtype=np.float64)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if transforms.ndim != 4 or transforms.shape[1:] != (2, 4, 4):
        raise ValueError(f"tcp must have shape (N, 2, 4, 4), got {transforms.shape}")
    if apertures.shape != (transforms.shape[0], 2):
        raise ValueError(f"gripper must have shape ({transforms.shape[0]}, 2), got {apertures.shape}")
    if not 0 <= query_index < transforms.shape[0]:
        raise IndexError(query_index)

    chunk = np.empty((horizon, UMI_CURRENTREL_STATE_DIM), dtype=np.float64)
    is_pad = np.empty(horizon, dtype=bool)
    final_index = transforms.shape[0] - 1
    for row, offset in enumerate(range(1, horizon + 1)):
        unclamped_index = query_index + offset
        target_index = min(unclamped_index, final_index)
        is_pad[row] = unclamped_index > final_index
        for arm, arm_slice in ((0, LEFT_STATE_SLICE), (1, RIGHT_STATE_SLICE)):
            target = relative_transform(transforms[query_index, arm], transforms[target_index, arm])
            chunk[row, arm_slice.start : arm_slice.stop - 1] = encode_relative_pose(target)
            chunk[row, arm_slice.stop - 1] = apertures[target_index, arm]
    return chunk, is_pad


def build_same_anchor_action_chunk_from_window(window: np.ndarray) -> np.ndarray:
    """Build a chunk from the reader window ``[t, t+1, ..., t+H]``."""

    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != UMI_CURRENTREL_HELPER_DIM:
        raise ValueError(
            f"helper window must have shape (horizon + 1, {UMI_CURRENTREL_HELPER_DIM}), got {values.shape}"
        )
    tcp, gripper = unpack_tcp_and_gripper(values)
    chunk, _ = build_same_anchor_action_chunk(tcp, gripper, 0, horizon=values.shape[0] - 1)
    return chunk


def load_current_relative_metadata(root: str | Path) -> dict[str, Any]:
    """Load and minimally validate the versioned semantic sidecar."""

    root = Path(root)
    present_paths = [
        relative for relative in _UMI_CURRENTREL_SCHEMA_BY_METADATA_PATH if (root / relative).is_file()
    ]
    if len(present_paths) > 1:
        raise ValueError(
            "ambiguous UMI current-relative artifact: multiple semantic sidecars are present: "
            f"{[str(path) for path in present_paths]}"
        )
    relative_path = present_paths[0] if present_paths else UMI_CURRENTREL_METADATA_PATH
    path = root / relative_path
    with path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    expected_schema = _UMI_CURRENTREL_SCHEMA_BY_METADATA_PATH[relative_path]
    if metadata.get("schema_id") != expected_schema:
        raise ValueError(
            f"unsupported UMI current-relative schema {metadata.get('schema_id')!r}; "
            f"sidecar {relative_path} requires {expected_schema!r}"
        )
    if int(metadata.get("action_horizon", -1)) != UMI_CURRENTREL_HORIZON:
        raise ValueError("UMI current-relative metadata must declare action_horizon=24")
    required_padding_semantics = _PADDING_SEMANTICS_BY_SCHEMA[expected_schema]
    padding_semantics = metadata.get("padding_semantics", required_padding_semantics)
    if padding_semantics != required_padding_semantics:
        raise ValueError(
            f"schema {expected_schema!r} requires padding_semantics={required_padding_semantics!r}, "
            f"got {padding_semantics!r}"
        )
    metadata["padding_semantics"] = padding_semantics
    return metadata


def load_current_relative_split_manifest(root: str | Path) -> dict[str, list[int]]:
    """Load the deterministic, episode-level train/validation split."""

    path = Path(root) / UMI_CURRENTREL_SPLIT_PATH
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    train = [int(value) for value in manifest.get("train_episodes", [])]
    validation = [int(value) for value in manifest.get("validation_episodes", [])]
    if len(train) != 52 or len(validation) != 2:
        raise ValueError(
            f"v1 split must contain 52 train and 2 validation episodes, got {len(train)}/{len(validation)}"
        )
    if set(train) & set(validation):
        raise ValueError("train and validation episode manifests overlap")
    if train != list(range(52)) or validation != [52, 53]:
        raise ValueError("v1 split must be train=0..51 and validation=52..53")
    if sorted(train + validation) != list(range(54)):
        raise ValueError("v1 split must cover each source episode exactly once")
    return {"train": train, "validation": validation}


def is_umi_current_relative_dataset(root: str | Path) -> bool:
    """Return whether ``root`` declares a supported query-anchored schema."""

    root = Path(root)
    if not any((root / relative).is_file() for relative in _UMI_CURRENTREL_SCHEMA_BY_METADATA_PATH):
        return False
    # A present sidecar is authoritative. Corruption or an unsupported schema
    # must fail closed: silently falling back to LeRobotDataset would train on
    # per-frame convenience actions instead of the query-anchored 24-row target.
    return (
        load_current_relative_metadata(root).get("schema_id")
        in _UMI_CURRENTREL_SCHEMA_BY_METADATA_PATH.values()
    )


class UmiCurrentRelativeR6dDataset(LeRobotDataset):
    """LeRobot dataset that constructs a query-anchored 24x20 target on read."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: ImageTransforms | None = None,
        revision: str | None = None,
        video_backend: str | None = None,
        return_uint8: bool = False,
        depth_output_unit: str = DEFAULT_DEPTH_UNIT,
        tolerance_s: float = 1e-4,
        *,
        action_horizon: int = UMI_CURRENTREL_HORIZON,
    ) -> None:
        if root is None:
            raise ValueError("the versioned UMI current-relative dataset requires an explicit local root")
        metadata = load_current_relative_metadata(root)
        if action_horizon != int(metadata["action_horizon"]):
            raise ValueError(
                f"policy chunk_size={action_horizon} does not match the dataset action_horizon="
                f"{metadata['action_horizon']}"
            )
        self.current_relative_metadata = metadata
        self.action_horizon = action_horizon
        self.padding_semantics = metadata["padding_semantics"]
        fps = int(metadata["fps"])
        helper_offsets = [offset / fps for offset in range(action_horizon + 1)]
        super().__init__(
            repo_id,
            root=root,
            episodes=episodes,
            delta_timestamps={UMI_TCP_WINDOW_KEY: helper_offsets},
            image_transforms=image_transforms,
            revision=revision,
            video_backend=video_backend,
            return_uint8=return_uint8,
            depth_output_unit=depth_output_unit,
            tolerance_s=tolerance_s,
        )
        if tuple(self.meta.features[OBS_STATE]["shape"]) != (UMI_CURRENTREL_STATE_DIM,):
            raise ValueError("UMI current-relative observation.state must have shape (20,)")
        if tuple(self.meta.features[ACTION]["shape"]) != (UMI_CURRENTREL_STATE_DIM,):
            raise ValueError("UMI current-relative action feature width must be 20")
        if tuple(self.meta.features[UMI_TCP_WINDOW_KEY]["shape"]) != (UMI_CURRENTREL_HELPER_DIM,):
            raise ValueError("UMI current-relative TCP helper width must be 16")

    @property
    def split_manifest(self) -> dict[str, list[int]]:
        return load_current_relative_split_manifest(self.root)

    def __getitem__(self, idx: int | slice) -> dict | list[dict]:
        if isinstance(idx, slice):
            return [self[item_idx] for item_idx in range(*idx.indices(len(self)))]
        item = super().__getitem__(idx)
        helper = item.pop(UMI_TCP_WINDOW_KEY)
        helper_pad = item.pop(f"{UMI_TCP_WINDOW_KEY}_is_pad")
        helper_np = helper.detach().cpu().numpy() if torch.is_tensor(helper) else np.asarray(helper)
        chunk = build_same_anchor_action_chunk_from_window(helper_np)
        if chunk.shape != (self.action_horizon, UMI_CURRENTREL_STATE_DIM):
            raise RuntimeError(f"constructed action has wrong shape {chunk.shape}")
        helper_pad_tensor = torch.as_tensor(helper_pad)
        if helper_pad_tensor.dtype != torch.bool:
            raise ValueError(f"{UMI_TCP_WINDOW_KEY}_is_pad must be Boolean, got {helper_pad_tensor.dtype}")
        if helper_pad_tensor.ndim != 1 or tuple(helper_pad_tensor.shape) != (self.action_horizon + 1,):
            raise ValueError(
                f"{UMI_TCP_WINDOW_KEY}_is_pad must have shape ({self.action_horizon + 1},), "
                f"got {tuple(helper_pad_tensor.shape)}"
            )
        if bool(helper_pad_tensor[0]):
            raise ValueError("the query-time TCP helper row must not be padding")

        # The action rows are k=1..H, so helper row zero (the query anchor) is
        # excluded and every beyond-episode future target is masked.
        item[ACTION] = torch.from_numpy(chunk.astype(np.float32, copy=False))
        if self.padding_semantics == PADDING_EXCLUDE_PADDED_FUTURE_ROWS:
            item[f"{ACTION}_is_pad"] = helper_pad_tensor[1:].clone()
        elif self.padding_semantics != PADDING_SUPERVISE_CLAMPED_FUTURE_ROWS:
            raise RuntimeError(f"unsupported padding semantics {self.padding_semantics!r}")
        return item
