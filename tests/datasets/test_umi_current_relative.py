from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.umi_current_relative import (
    PADDING_EXCLUDE_PADDED_FUTURE_ROWS,
    PADDING_SUPERVISE_CLAMPED_FUTURE_ROWS,
    UMI_CURRENTREL_HELPER_DIM,
    UMI_CURRENTREL_HORIZON,
    UMI_CURRENTREL_METADATA_PATH,
    UMI_CURRENTREL_SCHEMA_ID,
    UMI_CURRENTREL_SPLIT_PATH,
    UMI_CURRENTREL_STATE_DIM,
    UMI_TCP_WINDOW_KEY,
    UmiCurrentRelativeR6dDataset,
    build_query_state,
    build_same_anchor_action_chunk,
    decode_relative_pose,
    encode_relative_pose,
    invert_rigid_transform,
    is_umi_current_relative_dataset,
    load_current_relative_split_manifest,
    matrix_to_rotation_6d,
    matrix_to_xyzw_pose,
    pack_tcp_and_gripper,
    rotation_6d_to_matrix,
    validate_rigid_transform,
    xyzw_pose_to_matrix,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d import (
    STORED_POSE_T_TCP,
    TAIL_RETAINED_LENGTHS,
    TASK,
    _rewrite_episode_metadata,
    _semantic_metadata,
    _stats_block,
    build_episode_arrays,
    stored_pose_track_to_tcp,
)
from lerobot.utils.constants import ACTION, OBS_STATE

IDENTITY_R6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _transform(position=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = position
    if rotation is not None:
        transform[:3, :3] = rotation
    return transform


def _tracks(positions: list[float]) -> tuple[np.ndarray, np.ndarray]:
    tcp = np.empty((len(positions), 2, 4, 4))
    for index, position in enumerate(positions):
        tcp[index, 0] = _transform((position, 0.0, 0.0))
        tcp[index, 1] = _transform((0.0, 2.0 * position, 0.0))
    gripper = np.stack(
        (np.linspace(0.2, 0.8, len(positions)), np.linspace(0.9, 0.3, len(positions))), axis=-1
    )
    return tcp, gripper


def _source_table(frame_count: int = 4, *, include_action: bool = False) -> pa.Table:
    raw_state = np.zeros((frame_count, 12), dtype=np.float32)
    raw_state[:, 0] = np.arange(frame_count, dtype=np.float32) * 0.02
    raw_state[:, 7] = np.arange(frame_count, dtype=np.float32) * 0.03
    columns = {
        OBS_STATE: pa.FixedSizeListArray.from_arrays(pa.array(raw_state.reshape(-1)), 12),
        "observation.gripper_width.umi1": pa.array(np.linspace(25.0, 100.0, frame_count)),
        "observation.gripper_width.umi2": pa.array(np.linspace(110.0, 50.0, frame_count)),
        "timestamp": pa.array(np.arange(frame_count, dtype=np.float32) / 30.0),
        "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
        "episode_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
        "index": pa.array(np.arange(frame_count, dtype=np.int64)),
        "task_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
    }
    if include_action:
        columns[ACTION] = pa.FixedSizeListArray.from_arrays(pa.array(raw_state.reshape(-1)), 12)
    return pa.table(columns)


def test_stored_pose_is_tcp_identity_without_double_lever_arm() -> None:
    validate_rigid_transform(STORED_POSE_T_TCP, name="STORED_POSE_T_TCP")
    np.testing.assert_array_equal(STORED_POSE_T_TCP, np.eye(4))
    stored = np.array([[0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2]])
    tcp = stored_pose_track_to_tcp(stored)[0]
    expected = _transform(stored[0, :3], Rotation.from_rotvec(stored[0, 3:]).as_matrix())
    np.testing.assert_allclose(tcp, expected, atol=1e-12)
    np.testing.assert_array_equal(tcp[:3, 3], stored[0, :3])

    metadata = _semantic_metadata()["stored_pose_to_tcp"]
    assert metadata["symbol"] == "T_stored_pose_tcp"
    assert metadata["application"] == "T_episode_tcp = T_episode_stored_pose @ T_stored_pose_tcp"
    assert metadata["identity"] is True
    assert metadata["scene_fitted"] is False
    np.testing.assert_array_equal(metadata["matrix"], STORED_POSE_T_TCP)


def test_gripper_semantics_require_provenance_bearing_runtime_endpoint_schema() -> None:
    metadata = _semantic_metadata()["gripper"]

    assert metadata["runtime_mapping_requires_verified_endpoint_calibration"] is True
    assert metadata["runtime_endpoint_schema_version"] == 2
    assert metadata["runtime_endpoint_assignments"] == {"umi1": "left", "umi2": "right"}
    assert metadata["runtime_endpoint_fields"] == [
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
    ]


def test_se3_and_pose_round_trips() -> None:
    transform = _transform((0.17, -0.08, 0.31), Rotation.from_euler("xyz", [0.4, -0.2, 0.7]).as_matrix())
    validate_rigid_transform(transform)
    np.testing.assert_allclose(invert_rigid_transform(transform) @ transform, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(xyzw_pose_to_matrix(matrix_to_xyzw_pose(transform)), transform, atol=1e-12)
    np.testing.assert_allclose(decode_relative_pose(encode_relative_pose(transform)), transform, atol=1e-12)


def test_rotation6d_first_two_columns_round_trip_and_identity_exact() -> None:
    rotations = Rotation.random(32, random_state=np.random.default_rng(7)).as_matrix()
    encoded = matrix_to_rotation_6d(rotations)
    decoded = rotation_6d_to_matrix(encoded)
    np.testing.assert_allclose(decoded, rotations, atol=1e-12)
    np.testing.assert_array_equal(matrix_to_rotation_6d(np.eye(3)), IDENTITY_R6D)
    np.testing.assert_array_equal(rotation_6d_to_matrix(IDENTITY_R6D), np.eye(3))


def test_every_future_row_uses_query_anchor_not_previous_target() -> None:
    tcp, gripper = _tracks([0.0, 0.02, 0.04, 0.05])
    chunk, is_pad = build_same_anchor_action_chunk(tcp, gripper, 1, horizon=2)
    # Query is at 2 cm. Futures are 4 and 5 cm, hence +2 and +3 cm.
    np.testing.assert_allclose(chunk[:, 0], [0.02, 0.03], atol=1e-12)
    # A per-step delta implementation would incorrectly emit +1 cm in row 2.
    assert not np.isclose(chunk[1, 0], 0.01)
    np.testing.assert_array_equal(is_pad, [False, False])


def test_rotation_composition_is_not_rotvec_subtraction() -> None:
    tcp, gripper = _tracks([0.0, 0.0])
    query_rotation = Rotation.from_euler("xy", [0.7, -0.4])
    target_rotation = Rotation.from_euler("yz", [0.6, 0.5])
    tcp[0, 0, :3, :3] = query_rotation.as_matrix()
    tcp[1, 0, :3, :3] = target_rotation.as_matrix()
    chunk, _ = build_same_anchor_action_chunk(tcp, gripper, 0, horizon=1)
    composed = rotation_6d_to_matrix(chunk[0, 3:9])
    expected = query_rotation.inv().as_matrix() @ target_rotation.as_matrix()
    np.testing.assert_allclose(composed, expected, atol=1e-12)

    subtracted = Rotation.from_rotvec(target_rotation.as_rotvec() - query_rotation.as_rotvec()).as_matrix()
    assert not np.allclose(composed, subtracted, atol=1e-3)


def test_first_frame_history_is_exact_identity() -> None:
    tcp, gripper = _tracks([0.0, 0.02])
    state = build_query_state(tcp, gripper, 0)
    np.testing.assert_array_equal(state[:3], np.zeros(3))
    np.testing.assert_array_equal(state[3:9], IDENTITY_R6D)
    np.testing.assert_array_equal(state[10:13], np.zeros(3))
    np.testing.assert_array_equal(state[13:19], IDENTITY_R6D)
    assert state[9] == gripper[0, 0]
    assert state[19] == gripper[0, 1]


def test_terminal_padding_holds_final_tcp_and_gripper() -> None:
    tcp, gripper = _tracks([0.0, 0.02, 0.04, 0.05])
    chunk, is_pad = build_same_anchor_action_chunk(tcp, gripper, 2, horizon=4)
    np.testing.assert_allclose(chunk[:, 0], [0.01, 0.01, 0.01, 0.01], atol=1e-12)
    np.testing.assert_allclose(chunk[:, 9], gripper[-1, 0], atol=0.0)
    np.testing.assert_allclose(chunk[:, 19], gripper[-1, 1], atol=0.0)
    np.testing.assert_array_equal(is_pad, [False, True, True, True])


def test_left_right_order_and_converted_shapes() -> None:
    state, action, helper, chunks = build_episode_arrays(_source_table())
    assert state.shape == (4, UMI_CURRENTREL_STATE_DIM)
    assert action.shape == (4, UMI_CURRENTREL_STATE_DIM)
    assert helper.shape == (4, UMI_CURRENTREL_HELPER_DIM)
    assert chunks.shape == (4, UMI_CURRENTREL_HORIZON, UMI_CURRENTREL_STATE_DIM)
    # left is x motion in dims 0:10; right is y motion in dims 10:20.
    assert action[0, 0] > 0
    assert action[0, 11] > 0
    assert action[0, 1] == pytest.approx(0.0, abs=1e-7)
    assert action[0, 10] == pytest.approx(0.0, abs=1e-7)
    assert np.isfinite(state).all() and np.isfinite(chunks).all()


def test_converter_requires_observation_only_source() -> None:
    with pytest.raises(ValueError, match="must not contain an action"):
        build_episode_arrays(_source_table(include_action=True))


def test_converter_rewrites_episode_task_metadata(tmp_path) -> None:
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "output.parquet"
    pq.write_table(
        pa.table(
            {
                "tasks": pa.array([["stale source task"]]),
                "length": [8],
                "dataset_from_index": [10],
                "dataset_to_index": [18],
                "videos/observation.images.umi1/to_timestamp": [8 / 30],
            }
        ),
        source_path,
    )
    scalar_stats = {
        key: _stats_block(np.arange(4, dtype=np.float64).reshape(-1, 1))
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
    }
    _rewrite_episode_metadata(
        source_path,
        output_path,
        state_stats=_stats_block(np.zeros((4, UMI_CURRENTREL_STATE_DIM))),
        action_stats=_stats_block(np.zeros((4, UMI_CURRENTREL_STATE_DIM))),
        helper_stats=_stats_block(np.zeros((4, UMI_CURRENTREL_HELPER_DIM))),
        scalar_stats=scalar_stats,
        frame_count=4,
        dataset_from_index=20,
    )
    output = pq.read_table(output_path)
    assert output["tasks"].to_pylist() == [[TASK]]
    assert output["length"].to_pylist() == [4]
    assert output["dataset_from_index"].to_pylist() == [20]
    assert output["dataset_to_index"].to_pylist() == [24]
    assert output["videos/observation.images.umi1/to_timestamp"].to_pylist() == [4 / 30]
    assert output["stats/index/count"].to_pylist() == [[4]]


def test_tail_cleanup_keeps_all_episodes_and_a_full_policy_horizon() -> None:
    assert TAIL_RETAINED_LENGTHS == {2: 627, 5: 689, 19: 816, 25: 904, 30: 1_144, 34: 939, 47: 707}
    metadata = _semantic_metadata()["tail_cleanup"]
    assert metadata["all_training_episodes_retained"] is True
    assert metadata["removed_training_frames"] == 1_133
    assert metadata["training_frames_after_cleanup"] == 48_997


def test_episode_split_has_no_leakage(tmp_path) -> None:
    (tmp_path / UMI_CURRENTREL_METADATA_PATH.parent).mkdir(parents=True)
    (tmp_path / UMI_CURRENTREL_METADATA_PATH).write_text(
        json.dumps(
            {
                "schema_id": UMI_CURRENTREL_SCHEMA_ID,
                "action_horizon": UMI_CURRENTREL_HORIZON,
                "fps": 30,
            }
        )
    )
    (tmp_path / UMI_CURRENTREL_SPLIT_PATH).write_text(
        json.dumps(
            {
                "train_episodes": list(range(52)),
                "validation_episodes": [52, 53],
            }
        )
    )
    split = load_current_relative_split_manifest(tmp_path)
    assert len(split["train"]) == 52
    assert len(split["validation"]) == 2
    assert not set(split["train"]) & set(split["validation"])


def test_semantic_sidecar_detection_fails_closed_instead_of_using_generic_actions(tmp_path) -> None:
    assert is_umi_current_relative_dataset(tmp_path) is False
    sidecar = tmp_path / UMI_CURRENTREL_METADATA_PATH
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{not-json")
    with pytest.raises(json.JSONDecodeError):
        is_umi_current_relative_dataset(tmp_path)

    sidecar.write_text(json.dumps({"schema_id": "wrong-schema", "action_horizon": 24}))
    with pytest.raises(ValueError, match="unsupported UMI current-relative schema"):
        is_umi_current_relative_dataset(tmp_path)


def test_dataset_constructs_batch_after_query_and_preserves_camera_order(monkeypatch) -> None:
    tcp, gripper = _tracks([0.0, 0.02, 0.04, 0.05])
    future_indices = np.minimum(np.arange(UMI_CURRENTREL_HORIZON + 1), len(tcp) - 1)
    helper = torch.from_numpy(pack_tcp_and_gripper(tcp[future_indices], gripper[future_indices])).float()
    left_image = torch.full((3, 2, 2), 1, dtype=torch.uint8)
    right_image = torch.full((3, 2, 2), 2, dtype=torch.uint8)

    def fake_getitem(_self, _idx):
        return {
            OBS_STATE: torch.zeros(UMI_CURRENTREL_STATE_DIM),
            ACTION: torch.zeros(UMI_CURRENTREL_STATE_DIM),
            UMI_TCP_WINDOW_KEY: helper.clone(),
            f"{UMI_TCP_WINDOW_KEY}_is_pad": torch.tensor(
                [False] * len(tcp) + [True] * (UMI_CURRENTREL_HORIZON + 1 - len(tcp))
            ),
            "observation.images.umi1": left_image,
            "observation.images.umi2": right_image,
        }

    monkeypatch.setattr(LeRobotDataset, "__getitem__", fake_getitem)
    dataset = object.__new__(UmiCurrentRelativeR6dDataset)
    dataset.action_horizon = UMI_CURRENTREL_HORIZON
    dataset.padding_semantics = PADDING_SUPERVISE_CLAMPED_FUTURE_ROWS
    item = dataset[0]
    assert item[OBS_STATE].shape == (UMI_CURRENTREL_STATE_DIM,)
    assert item[ACTION].shape == (UMI_CURRENTREL_HORIZON, UMI_CURRENTREL_STATE_DIM)
    assert f"{ACTION}_is_pad" not in item
    assert torch.equal(item["observation.images.umi1"], left_image)
    assert torch.equal(item["observation.images.umi2"], right_image)
    assert UMI_TCP_WINDOW_KEY not in item

    batch = torch.utils.data.default_collate([item, item])
    assert batch[OBS_STATE].shape == (2, UMI_CURRENTREL_STATE_DIM)
    assert batch[ACTION].shape == (2, UMI_CURRENTREL_HORIZON, UMI_CURRENTREL_STATE_DIM)
    assert f"{ACTION}_is_pad" not in batch

    dataset.padding_semantics = PADDING_EXCLUDE_PADDED_FUTURE_ROWS
    masked_item = dataset[0]
    assert torch.equal(
        masked_item[f"{ACTION}_is_pad"],
        torch.tensor([False] * (len(tcp) - 1) + [True] * (UMI_CURRENTREL_HORIZON - len(tcp) + 1)),
    )


def test_finite_quantile_normalized_values_and_initial_loss() -> None:
    _, _, _, chunks = build_episode_arrays(_source_table(frame_count=8))
    actions = chunks.reshape(-1, UMI_CURRENTREL_STATE_DIM).astype(np.float64)
    q01, q99 = np.percentile(actions, [1, 99], axis=0)
    scale = np.maximum(q99 - q01, 1e-6)
    normalized = np.clip(2.0 * (actions - q01) / scale - 1.0, -1.0, 1.0)
    initial_prediction = np.zeros_like(normalized)
    initial_loss = np.square(initial_prediction - normalized).mean()
    assert np.isfinite(normalized).all()
    assert np.isfinite(initial_loss)
