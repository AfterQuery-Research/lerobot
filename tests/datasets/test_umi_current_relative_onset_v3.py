from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.datasets.factory import _apply_imagenet_stats
from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_HORIZON,
    UMI_CURRENTREL_ONSET_V3_METADATA_PATH,
    UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
    UMI_TCP_WINDOW_KEY,
    is_umi_current_relative_dataset,
    load_current_relative_metadata,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v3 import (
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_V3_RETAINED_LENGTHS,
    EXPECTED_VALID_ACTION_POSITIONS,
    EXPECTED_VALID_TRAIN_ACTION_POSITIONS,
    EXPECTED_VALID_VALIDATION_ACTION_POSITIONS,
    EXPECTED_VALIDATION_FRAMES,
    FPS,
    OUTPUT_DATASET_NAME,
    OUTPUT_REPO_ID,
    PADDED_ACTION_POSITIONS_PER_EPISODE,
    SOURCE_SUFFIX_END_FRAME_EXCLUSIVE,
    TrainOnlyStatsAccumulator,
    action_padding_mask,
    prepare_onset_v3_episode,
    rewrite_episode_metadata,
    semantic_metadata,
)
from lerobot.utils.constants import IMAGENET_STATS


def _source_table(*, frame_count: int, onset: int, episode_index: int) -> pa.Table:
    state = np.zeros((frame_count, 12), dtype=np.float32)
    state[onset:, 0] = 0.003 + np.arange(frame_count - onset, dtype=np.float32) * 0.0001
    return pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), 12),
            "observation.gripper_width.umi1": pa.array(
                np.linspace(30.0, 70.0, frame_count, dtype=np.float32)
            ),
            "observation.gripper_width.umi2": pa.array(
                np.linspace(100.0, 60.0, frame_count, dtype=np.float32)
            ),
            "timestamp": pa.array(np.arange(frame_count, dtype=np.float32) / FPS),
            "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
            "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(frame_count, dtype=np.int64)),
            "task_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
        }
    )


def test_imagenet_stats_populate_missing_camera_entries_without_touching_depth() -> None:
    rgb_key = "observation.images.umi1"
    depth_key = "observation.images.depth"
    dataset = SimpleNamespace(
        meta=SimpleNamespace(camera_keys=[rgb_key, depth_key], depth_keys=[depth_key], stats={})
    )

    _apply_imagenet_stats(dataset)

    assert depth_key not in dataset.meta.stats
    assert set(dataset.meta.stats[rgb_key]) == set(IMAGENET_STATS)
    for stats_type, expected in IMAGENET_STATS.items():
        torch.testing.assert_close(
            dataset.meta.stats[rgb_key][stats_type], torch.tensor(expected, dtype=torch.float32)
        )


@pytest.mark.parametrize("frame_count", [25, 47, 1_088])
def test_action_padding_mask_covers_exactly_beyond_end_k1_through_k24(frame_count: int) -> None:
    mask = action_padding_mask(frame_count)

    assert mask.dtype == np.bool_
    assert mask.shape == (frame_count, UMI_CURRENTREL_HORIZON)
    assert int(mask.sum()) == PADDED_ACTION_POSITIONS_PER_EPISODE == 300
    for query in range(frame_count):
        for row, offset in enumerate(range(1, UMI_CURRENTREL_HORIZON + 1)):
            assert bool(mask[query, row]) is (query + offset >= frame_count)


@pytest.mark.parametrize(
    ("episode_index", "source_frames", "onset"),
    [(0, 1_288, 38), (40, 904, 27), (46, 898, 35)],
)
def test_v3_applies_only_locked_source_exclusive_suffix_cuts(
    episode_index: int,
    source_frames: int,
    onset: int,
) -> None:
    episode = prepare_onset_v3_episode(
        _source_table(frame_count=source_frames, onset=onset, episode_index=episode_index),
        episode_index=episode_index,
        dataset_from_index=17,
    )

    assert episode.onset_frame == onset
    assert episode.tail_end_frame_exclusive == SOURCE_SUFFIX_END_FRAME_EXCLUSIVE[episode_index]
    assert episode.frame_count == EXPECTED_V3_RETAINED_LENGTHS[episode_index]
    assert episode.valid_action_position_count == episode.frame_count * UMI_CURRENTREL_HORIZON - 300
    np.testing.assert_array_equal(episode.table["frame_index"].to_numpy(), np.arange(episode.frame_count))
    np.testing.assert_array_equal(episode.table["index"].to_numpy(), np.arange(17, 17 + episode.frame_count))


def test_v3_stats_exclude_holdout_and_every_padded_action_row() -> None:
    training = prepare_onset_v3_episode(
        _source_table(frame_count=1_288, onset=38, episode_index=0),
        episode_index=0,
        dataset_from_index=0,
    )
    holdout = replace(
        training,
        episode_index=52,
        state=np.full_like(training.state, 999.0),
        action=np.full_like(training.action, 999.0),
        helper=np.full_like(training.helper, 999.0),
        chunks=np.full_like(training.chunks, 999.0),
        scalar_values={key: np.full_like(value, 999.0) for key, value in training.scalar_values.items()},
    )
    accumulator = TrainOnlyStatsAccumulator(expected_action_count=None)
    accumulator.add(training)
    accumulator.add(holdout)
    stats = accumulator.finalize()

    assert stats["observation.state"]["count"] == [training.frame_count]
    assert stats["action"]["count"] == [training.valid_action_position_count]
    assert stats["action"]["count"] != [training.frame_count * UMI_CURRENTREL_HORIZON]
    assert stats[UMI_TCP_WINDOW_KEY]["count"] == [training.frame_count]
    assert max(stats["observation.state"]["max"]) < 999.0


def test_v3_episode_metadata_uses_valid_action_count_and_exact_video_cut(tmp_path: Path) -> None:
    episode = prepare_onset_v3_episode(
        _source_table(frame_count=1_288, onset=38, episode_index=0),
        episode_index=0,
        dataset_from_index=20,
    )
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "output.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "tasks": pa.array([["stale"]], type=pa.list_(pa.string())),
                "length": [1_288],
                "dataset_from_index": [0],
                "dataset_to_index": [1_288],
                "videos/observation.images.umi1/from_timestamp": [10.0],
                "videos/observation.images.umi1/to_timestamp": [10.0 + 1_288 / FPS],
                "stats/action/count": pa.array([[99]], type=pa.list_(pa.int64())),
            }
        ),
        source_path,
    )

    rewrite_episode_metadata(source_path, output_path, episode=episode)
    metadata = pq.read_table(output_path)

    assert metadata["length"].to_pylist() == [1_088]
    assert metadata["stats/action/count"].to_pylist() == [[episode.valid_action_position_count]]
    assert metadata["videos/observation.images.umi1/from_timestamp"].to_pylist() == [10.0 + 38 / FPS]
    assert metadata["videos/observation.images.umi1/to_timestamp"].to_pylist() == [10.0 + 1_126 / FPS]


def test_v3_semantics_record_suffix_padding_and_external_start_pose() -> None:
    metadata = semantic_metadata(
        onset_frames=dict.fromkeys(range(54), 3),
        retained_lengths=dict.fromkeys(range(54), 100),
    )

    assert metadata["schema_id"] == UMI_CURRENTREL_ONSET_V3_SCHEMA_ID
    assert metadata["dataset_name"] == OUTPUT_DATASET_NAME
    assert metadata["repository"] == OUTPUT_REPO_ID
    assert metadata["v3_suffix_cleanup"]["source_end_frame_exclusive_by_episode"] == {
        str(key): value for key, value in SOURCE_SUFFIX_END_FRAME_EXCLUSIVE.items()
    }
    assert metadata["v3_suffix_cleanup"]["global_smoothing_applied"] is False
    assert metadata["v3_suffix_cleanup"]["retiming_applied"] is False
    assert metadata["terminal_padding"]["valid_action_positions"] == {
        "train": EXPECTED_VALID_TRAIN_ACTION_POSITIONS,
        "validation": EXPECTED_VALID_VALIDATION_ACTION_POSITIONS,
        "all": EXPECTED_VALID_ACTION_POSITIONS,
    }
    assert metadata["deployment_start_pose_audit"]["part_of_training_tensors"] is False
    assert metadata["deployment_start_pose_audit"]["hardware_start_verified"] is False
    assert metadata["deployment_start_pose_audit"]["candidate_symmetric_arm_joints_rad"] == [
        0.0,
        0.05,
        0.05,
        0.0,
        0.0,
        0.0,
    ]


def test_v3_schema_is_recognized_without_shadowing_older_sidecars(tmp_path: Path) -> None:
    sidecar = tmp_path / UMI_CURRENTREL_ONSET_V3_METADATA_PATH
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "schema_id": UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
                "action_horizon": UMI_CURRENTREL_HORIZON,
                "fps": FPS,
            }
        )
    )

    assert is_umi_current_relative_dataset(tmp_path) is True
    assert load_current_relative_metadata(tmp_path)["schema_id"] == UMI_CURRENTREL_ONSET_V3_SCHEMA_ID
    assert EXPECTED_TRAIN_FRAMES + EXPECTED_VALIDATION_FRAMES == 49_121
