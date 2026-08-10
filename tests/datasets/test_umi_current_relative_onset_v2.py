from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.umi_yam.validate_currentrel_onset_v2_training_run import (
    _validate_artifact_checksum_manifest,
)
from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_HORIZON,
    UMI_CURRENTREL_METADATA_PATH,
    UMI_CURRENTREL_ONSET_METADATA_PATH,
    UMI_CURRENTREL_ONSET_SCHEMA_ID,
    UMI_TCP_WINDOW_KEY,
    is_umi_current_relative_dataset,
    load_current_relative_metadata,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d import TAIL_RETAINED_LENGTHS
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v2 import (
    FPS,
    OUTPUT_DATASET_NAME,
    OUTPUT_REPO_ID,
    TrainOnlyStatsAccumulator,
    find_motion_onset_in_table,
    prepare_onset_aligned_episode,
    rewrite_onset_episode_metadata,
    semantic_metadata,
    write_artifact_manifest,
)

IDENTITY_R6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _source_table(
    left_x: list[float],
    *,
    right_rotation_deg: list[float] | None = None,
    episode_index: int = 0,
) -> pa.Table:
    frame_count = len(left_x)
    if right_rotation_deg is None:
        right_rotation_deg = [0.0] * frame_count
    state = np.zeros((frame_count, 12), dtype=np.float32)
    state[:, 0] = left_x
    state[:, 10] = np.deg2rad(right_rotation_deg)
    return pa.table(
        {
            "observation.state": pa.FixedSizeListArray.from_arrays(pa.array(state.reshape(-1)), 12),
            "observation.gripper_width.umi1": pa.array(np.linspace(30.0, 70.0, frame_count)),
            "observation.gripper_width.umi2": pa.array(np.linspace(100.0, 60.0, frame_count)),
            "timestamp": pa.array(np.arange(frame_count, dtype=np.float32) / FPS),
            "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
            "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(frame_count, dtype=np.int64)),
            "task_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
        }
    )


def _source_episode_metadata(path: Path, *, source_from: float, source_to: float) -> None:
    pq.write_table(
        pa.table(
            {
                "episode_index": [0],
                "tasks": pa.array([["stale task"]], type=pa.list_(pa.string())),
                "length": [99],
                "dataset_from_index": [0],
                "dataset_to_index": [99],
                "videos/observation.images.umi1/chunk_index": [0],
                "videos/observation.images.umi1/file_index": [0],
                "videos/observation.images.umi1/from_timestamp": [source_from],
                "videos/observation.images.umi1/to_timestamp": [source_to],
                "stats/observation.images.umi1/mean": pa.array([[[[0.5]]]]),
            }
        ),
        path,
    )


def test_scene_free_onset_is_first_of_three_consecutive_pose_motion_frames() -> None:
    # A one-frame 3 mm translation spike must not qualify. Persistent rotation
    # of the other arm starts at frame 5, so frame 5 is the onset.
    table = _source_table(
        [0.0, 0.001, 0.003, 0.001, 0.0, 0.0, 0.0, 0.0],
        right_rotation_deg=[0.0, 0.0, 0.0, 0.0, 1.0, 1.1, 1.2, 1.3],
    )
    assert find_motion_onset_in_table(table) == 5

    no_run = _source_table([0.0, 0.003, 0.0, 0.003, 0.0])
    with pytest.raises(ValueError, match="no scene-free motion onset"):
        find_motion_onset_in_table(no_run)


def test_onset_trim_resets_indices_and_builds_identity_state_and_same_anchor_action() -> None:
    source = _source_table([0.0, 0.001, 0.0015, 0.003, 0.005, 0.008, 0.010])
    episode = prepare_onset_aligned_episode(
        source,
        episode_index=0,
        dataset_from_index=17,
    )
    assert episode.onset_frame == 3
    assert episode.frame_count == 4
    np.testing.assert_array_equal(episode.table["frame_index"].to_numpy(), np.arange(4))
    np.testing.assert_allclose(episode.table["timestamp"].to_numpy(), np.arange(4) / FPS, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(episode.table["index"].to_numpy(), np.arange(17, 21))

    for start in (0, 10):
        np.testing.assert_array_equal(episode.state[0, start : start + 3], np.zeros(3))
        np.testing.assert_array_equal(episode.state[0, start + 3 : start + 9], IDENTITY_R6D)
        np.testing.assert_array_equal(episode.action[-1, start : start + 3], np.zeros(3))
        np.testing.assert_array_equal(episode.action[-1, start + 3 : start + 9], IDENTITY_R6D)

    assert episode.action[0, 0] == pytest.approx(0.002, abs=1e-7)
    # Row two is source frame 5 relative to the one fixed source-frame-3
    # query anchor: 8 mm - 3 mm = 5 mm, not the 3 mm incremental step.
    assert episode.chunks[0, 1, 0] == pytest.approx(0.005, abs=1e-7)


def test_episode_video_metadata_advances_from_timestamp_by_onset(tmp_path: Path) -> None:
    source = _source_table([0.0, 0.001, 0.0015, 0.003, 0.005, 0.008, 0.010])
    episode = prepare_onset_aligned_episode(
        source,
        episode_index=0,
        dataset_from_index=20,
    )
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "output.parquet"
    _source_episode_metadata(source_path, source_from=10.0, source_to=20.0)
    rewrite_onset_episode_metadata(source_path, output_path, episode=episode)
    metadata = pq.read_table(output_path)

    assert metadata["length"].to_pylist() == [4]
    assert metadata["dataset_from_index"].to_pylist() == [20]
    assert metadata["dataset_to_index"].to_pylist() == [24]
    assert metadata["videos/observation.images.umi1/from_timestamp"].to_pylist() == [10.0 + 3 / FPS]
    assert metadata["videos/observation.images.umi1/to_timestamp"].to_pylist() == [10.0 + 7 / FPS]
    assert not any(name.startswith("stats/observation.images") for name in metadata.column_names)
    assert metadata["stats/index/count"].to_pylist() == [[4]]


def test_global_statistics_structurally_exclude_validation_episodes() -> None:
    source = _source_table([0.0, 0.001, 0.0015, 0.003, 0.005, 0.008, 0.010])
    training = prepare_onset_aligned_episode(
        source,
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
        scalar_values={key: np.full_like(values, 999.0) for key, values in training.scalar_values.items()},
    )
    accumulator = TrainOnlyStatsAccumulator()
    accumulator.add(training)
    accumulator.add(holdout)
    stats = accumulator.finalize()

    assert stats["observation.state"]["count"] == [training.frame_count]
    assert stats["action"]["count"] == [training.frame_count * UMI_CURRENTREL_HORIZON]
    assert stats[UMI_TCP_WINDOW_KEY]["count"] == [training.frame_count]
    assert stats["episode_index"]["max"] == [0.0]
    assert max(stats["observation.state"]["max"]) < 999.0


def test_v2_metadata_is_distinct_and_does_not_shadow_v1(tmp_path: Path) -> None:
    v2_root = tmp_path / "v2"
    sidecar = v2_root / UMI_CURRENTREL_ONSET_METADATA_PATH
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        json.dumps(
            {
                "schema_id": UMI_CURRENTREL_ONSET_SCHEMA_ID,
                "action_horizon": UMI_CURRENTREL_HORIZON,
                "fps": FPS,
            }
        )
    )
    assert is_umi_current_relative_dataset(v2_root) is True
    assert load_current_relative_metadata(v2_root)["schema_id"] == UMI_CURRENTREL_ONSET_SCHEMA_ID

    legacy_sidecar = v2_root / UMI_CURRENTREL_METADATA_PATH
    legacy_sidecar.write_text("{}")
    with pytest.raises(ValueError, match="multiple semantic sidecars"):
        load_current_relative_metadata(v2_root)


def test_semantic_contract_preserves_tail_and_hardware_gripper_requirements() -> None:
    metadata = semantic_metadata(
        onset_frames=dict.fromkeys(range(54), 3),
        retained_lengths=dict.fromkeys(range(54), 100),
        train_frames=5_200,
        validation_frames=200,
    )
    assert metadata["dataset_name"] == OUTPUT_DATASET_NAME
    assert metadata["repository"] == OUTPUT_REPO_ID
    assert metadata["schema_id"] == UMI_CURRENTREL_ONSET_SCHEMA_ID
    assert metadata["padding_semantics"] == "supervise_clamped_future_rows"
    assert metadata["onset_alignment"]["translation_threshold_m"] == 0.002
    assert metadata["onset_alignment"]["rotation_threshold_deg"] == 1.0
    assert metadata["onset_alignment"]["consecutive_frames"] == 3
    assert metadata["tail_cleanup"]["retained_lengths_by_output_episode"] == {
        str(index): length for index, length in TAIL_RETAINED_LENGTHS.items()
    }
    assert metadata["gripper"]["runtime_endpoint_schema_version"] == 2
    assert metadata["gripper"]["runtime_mapping_requires_verified_endpoint_calibration"] is True


def test_artifact_manifest_hashes_copied_videos(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    (source / "videos").mkdir()
    (source / "meta/info.json").write_text("{}\n")
    (source / "meta/filtering.json").write_text("{}\n")
    for episode_index in range(54):
        for arm_index, arm in enumerate(("umi1", "umi2")):
            video = source / "videos" / arm / f"episode-{episode_index:03d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"video-{episode_index}-{arm_index}".encode())
    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta").mkdir()
    (output / "data/chunk-000/file-000.parquet").write_bytes(b"data")
    (output / "meta/info.json").write_text("{}\n")
    (output / "README.md").write_text("readme\n")
    shutil.copytree(source / "videos", output / "videos")

    write_artifact_manifest(
        output,
        source,
        train_frames=4,
        validation_frames=2,
    )
    manifest = json.loads((output / "meta/artifact_manifest.json").read_text())
    video_keys = sorted(path for path in manifest["generated_file_sha256"] if path.startswith("videos/"))
    assert len(video_keys) == 108
    assert manifest["videos"] == {
        "handling": "byte-for-byte source copies included in generated_file_sha256",
        "file_count": 108,
        "total_bytes": sum(path.stat().st_size for path in (output / "videos").rglob("*.mp4")),
    }
    assert manifest["generated_file_sha256"]["videos/umi1/episode-000.mp4"]
    checksum = (output / "meta/artifact_manifest.sha256").read_text()
    assert "data/chunk-000/file-000.parquet" in checksum
    assert "videos/umi1/episode-000.mp4" in checksum
    assert "meta/artifact_manifest.json" in checksum


def test_training_validator_hashes_video_contents_not_only_size(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    video = root / "videos/umi1/file-000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"original-video-bytes")
    checksum_path = root / "meta/artifact_manifest.sha256"
    checksum_path.parent.mkdir()
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  videos/umi1/file-000.mp4\n")
    manifest_digest = hashlib.sha256(checksum_path.read_bytes()).hexdigest()

    assert _validate_artifact_checksum_manifest(root, manifest_digest) == {"videos/umi1/file-000.mp4": digest}

    # Same-length corruption must fail; path and total-byte checks alone would
    # not detect this mutation.
    video.write_bytes(b"corrupt!-video-bytes")
    assert video.stat().st_size == len(b"original-video-bytes")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        _validate_artifact_checksum_manifest(root, manifest_digest)
