#!/usr/bin/env python

"""Fail-fast validation for the onset-aligned UMI current-relative v2 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_HORIZON,
    UMI_CURRENTREL_ONSET_METADATA_PATH,
    UMI_CURRENTREL_ONSET_SCHEMA_ID,
    UMI_TCP_WINDOW_KEY,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d import (
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    TAIL_RETAINED_LENGTHS,
    TASK,
    TRAIN_EPISODES,
    VALIDATION_EPISODES,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v2 import (
    EXPECTED_ONSET_OUTPUT_FRAMES,
    EXPECTED_ONSET_TRAIN_FRAMES,
    EXPECTED_ONSET_VALIDATION_FRAMES,
    FPS,
    ONSET_CONSECUTIVE_FRAMES,
    ONSET_ROTATION_THRESHOLD_DEG,
    ONSET_TRANSLATION_THRESHOLD_M,
    OUTPUT_REPO_ID,
)

IDENTITY_R6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_finite(values: object, *, name: str) -> None:
    if isinstance(values, list):
        if not values:
            raise AssertionError(f"empty statistic {name}")
        for index, value in enumerate(values):
            _assert_finite(value, name=f"{name}[{index}]")
        return
    if not math.isfinite(float(values)):
        raise AssertionError(f"non-finite statistic {name}: {values}")


def _validate_artifact_hashes(root: Path, manifest: dict) -> None:
    checksum_path = root / "meta/artifact_manifest.sha256"
    checksum_entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        assert separator and len(digest) == 64, f"malformed checksum line: {line!r}"
        path = Path(relative)
        assert not path.is_absolute() and ".." not in path.parts
        assert relative not in checksum_entries, f"duplicate checksum entry {relative}"
        checksum_entries[relative] = digest
        assert _sha256(root / path) == digest, f"checksum mismatch for {relative}"

    declared = manifest["generated_file_sha256"]
    assert set(checksum_entries) == {*declared, "meta/artifact_manifest.json"}
    for relative, digest in declared.items():
        assert checksum_entries[relative] == digest


def validate(root: Path) -> dict:
    root = root.resolve(strict=True)
    semantics = json.loads((root / UMI_CURRENTREL_ONSET_METADATA_PATH).read_text())
    split = json.loads((root / "meta/split_manifest.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    stats = json.loads((root / "meta/stats.json").read_text())
    manifest = json.loads((root / "meta/artifact_manifest.json").read_text())
    _validate_artifact_hashes(root, manifest)

    assert semantics["schema_id"] == UMI_CURRENTREL_ONSET_SCHEMA_ID
    assert semantics["schema_version"] == 2
    assert semantics["repository"] == OUTPUT_REPO_ID
    assert semantics["source"]["repository"] == SOURCE_REPO_ID
    assert semantics["source"]["revision"] == SOURCE_REVISION
    assert semantics["action_horizon"] == UMI_CURRENTREL_HORIZON
    assert semantics["statistics"]["scope"] == ("training episodes 0..51 only after tail and onset alignment")
    assert semantics["statistics"]["excluded_episodes"] == VALIDATION_EPISODES
    assert semantics["stored_pose_to_tcp"]["identity"] is True
    assert semantics["gripper"]["runtime_mapping_requires_verified_endpoint_calibration"] is True
    assert semantics["gripper"]["runtime_endpoint_schema_version"] == 2
    assert semantics["gripper"]["runtime_endpoint_assignments"] == {
        "umi1": "left",
        "umi2": "right",
    }
    onset = semantics["onset_alignment"]
    assert onset["translation_threshold_m"] == ONSET_TRANSLATION_THRESHOLD_M
    assert onset["rotation_threshold_deg"] == ONSET_ROTATION_THRESHOLD_DEG
    assert onset["consecutive_frames"] == ONSET_CONSECUTIVE_FRAMES
    assert onset["threshold_comparison"] == "strict greater-than"
    assert onset["scene_or_object_features_used"] is False
    assert onset["fit_to_task_video"] is False

    assert split["schema_id"] == UMI_CURRENTREL_ONSET_SCHEMA_ID
    assert split["dataset_repository"] == OUTPUT_REPO_ID
    assert split["source_revision"] == SOURCE_REVISION
    assert split["train_episodes"] == TRAIN_EPISODES
    assert split["validation_episodes"] == VALIDATION_EPISODES
    assert not set(TRAIN_EPISODES) & set(VALIDATION_EPISODES)
    assert sorted(map(int, split["onset_frame_by_episode"])) == list(range(54))
    assert sorted(map(int, split["retained_length_by_episode"])) == list(range(54))

    train_frames = int(split["train_frames"])
    validation_frames = int(split["validation_frames"])
    total_frames = train_frames + validation_frames
    assert train_frames == EXPECTED_ONSET_TRAIN_FRAMES
    assert validation_frames == EXPECTED_ONSET_VALIDATION_FRAMES
    assert total_frames == EXPECTED_ONSET_OUTPUT_FRAMES
    assert manifest["schema_id"] == UMI_CURRENTREL_ONSET_SCHEMA_ID
    assert manifest["dataset_repository"] == OUTPUT_REPO_ID
    assert manifest["source"]["repository"] == SOURCE_REPO_ID
    assert manifest["source"]["revision"] == SOURCE_REVISION
    assert manifest["split"]["train_frames"] == train_frames
    assert manifest["split"]["validation_frames"] == validation_frames
    source_root = Path(manifest["source"]["root"])
    assert source_root.resolve(strict=True).is_dir()
    assert _sha256(source_root / "meta/info.json") == manifest["source"]["info_sha256"]
    if manifest["source"]["filtering_sha256"] is not None:
        assert _sha256(source_root / "meta/filtering.json") == manifest["source"]["filtering_sha256"]
    videos = root / "videos"
    assert videos.is_dir() and not videos.is_symlink(), "videos must be copied into the artifact"
    video_paths = sorted(videos.rglob("*.mp4"))
    source_video_root = source_root / "videos"
    source_video_paths = sorted(source_video_root.rglob("*.mp4"))
    video_relatives = [path.relative_to(videos) for path in video_paths]
    source_video_relatives = [path.relative_to(source_video_root) for path in source_video_paths]
    assert len(video_paths) == 108
    assert video_relatives == source_video_relatives
    assert all(path.is_file() and not path.is_symlink() for path in video_paths)
    assert manifest["videos"] == {
        "handling": "byte-for-byte source copies included in generated_file_sha256",
        "file_count": len(video_paths),
        "total_bytes": sum(path.stat().st_size for path in video_paths),
    }
    for relative, source_video in zip(video_relatives, source_video_paths, strict=True):
        manifest_key = (Path("videos") / relative).as_posix()
        assert manifest["generated_file_sha256"][manifest_key] == _sha256(source_video)

    assert info["total_episodes"] == 54
    assert info["total_frames"] == total_frames
    assert info["splits"] == {"train": "0:52", "validation": "52:54"}
    assert info["features"]["observation.state"]["shape"] == [20]
    assert info["features"]["action"]["shape"] == [20]
    assert info["features"][UMI_TCP_WINDOW_KEY]["shape"] == [16]
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    assert tasks["task"] == [TASK]
    assert tasks["task_index"] == [0]

    cumulative_index = 0
    counted_train_frames = 0
    counted_validation_frames = 0
    for episode_index in TRAIN_EPISODES + VALIDATION_EPISODES:
        data_path = root / "data/chunk-000" / f"file-{episode_index:03d}.parquet"
        episode_path = root / "meta/episodes/chunk-000" / f"file-{episode_index:03d}.parquet"
        source_episode_path = source_root / "meta/episodes/chunk-000" / f"file-{episode_index:03d}.parquet"
        data = pq.read_table(data_path)
        episode = pq.read_table(episode_path)
        source_episode = pq.read_table(source_episode_path)
        frame_count = len(data)
        onset_frame = int(split["onset_frame_by_episode"][str(episode_index)])
        assert frame_count == int(split["retained_length_by_episode"][str(episode_index)])
        source_length = int(source_episode["length"][0].as_py())
        tail_end = TAIL_RETAINED_LENGTHS.get(episode_index, source_length)
        assert onset_frame + frame_count == tail_end
        assert episode["tasks"].to_pylist() == [[TASK]]
        assert episode["length"].to_pylist() == [frame_count]
        assert episode["dataset_from_index"].to_pylist() == [cumulative_index]
        assert episode["dataset_to_index"].to_pylist() == [cumulative_index + frame_count]

        frame_index = np.asarray(data["frame_index"].to_numpy(), dtype=np.int64)
        timestamp = np.asarray(data["timestamp"].to_numpy(), dtype=np.float64)
        index = np.asarray(data["index"].to_numpy(), dtype=np.int64)
        np.testing.assert_array_equal(frame_index, np.arange(frame_count))
        # The LeRobot timestamp feature is float32.  Compare against the exact
        # representable values written by the converter rather than applying a
        # fixed float64 tolerance that eventually rejects valid long episodes.
        expected_timestamp = (np.arange(frame_count, dtype=np.float64) / FPS).astype(np.float32)
        np.testing.assert_array_equal(timestamp, expected_timestamp.astype(np.float64))
        np.testing.assert_array_equal(index, np.arange(cumulative_index, cumulative_index + frame_count))

        first_state = np.asarray(data["observation.state"][0].as_py(), dtype=np.float32)
        final_action = np.asarray(data["action"][-1].as_py(), dtype=np.float32)
        for start in (0, 10):
            np.testing.assert_array_equal(first_state[start : start + 3], np.zeros(3))
            np.testing.assert_array_equal(first_state[start + 3 : start + 9], IDENTITY_R6D)
            np.testing.assert_array_equal(final_action[start : start + 3], np.zeros(3))
            np.testing.assert_array_equal(final_action[start + 3 : start + 9], IDENTITY_R6D)

        for arm in ("umi1", "umi2"):
            key = f"videos/observation.images.{arm}"
            source_from = float(source_episode[f"{key}/from_timestamp"][0].as_py())
            retained_from = float(episode[f"{key}/from_timestamp"][0].as_py())
            retained_to = float(episode[f"{key}/to_timestamp"][0].as_py())
            assert math.isclose(retained_from, source_from + onset_frame / FPS, abs_tol=1e-9)
            assert math.isclose(retained_to - retained_from, frame_count / FPS, abs_tol=1e-9)

        cumulative_index += frame_count
        if episode_index in TRAIN_EPISODES:
            counted_train_frames += frame_count
        else:
            counted_validation_frames += frame_count

    assert cumulative_index == total_frames == info["total_frames"]
    assert counted_train_frames == train_frames
    assert counted_validation_frames == validation_frames
    expected_counts = {
        "observation.state": train_frames,
        "action": train_frames * UMI_CURRENTREL_HORIZON,
        UMI_TCP_WINDOW_KEY: train_frames,
        "timestamp": train_frames,
        "frame_index": train_frames,
        "episode_index": train_frames,
        "index": train_frames,
        "task_index": train_frames,
    }
    assert set(stats) == set(expected_counts), "stale source/image statistics must not be retained"
    for feature, expected_count in expected_counts.items():
        assert stats[feature]["count"] == [expected_count]
        for statistic, values in stats[feature].items():
            if statistic != "count":
                _assert_finite(values, name=f"{feature}/{statistic}")

    report = {
        "schema_id": UMI_CURRENTREL_ONSET_SCHEMA_ID,
        "dataset_repository": OUTPUT_REPO_ID,
        "source_revision": SOURCE_REVISION,
        "train_episodes": len(TRAIN_EPISODES),
        "validation_episodes": len(VALIDATION_EPISODES),
        "train_frames": train_frames,
        "validation_frames": validation_frames,
        "action_stat_rows": train_frames * UMI_CURRENTREL_HORIZON,
        "video_file_count": len(video_paths),
        "video_total_bytes": manifest["videos"]["total_bytes"],
        "artifact_manifest_sha256": _sha256(root / "meta/artifact_manifest.sha256"),
    }
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    validate(args.dataset_root)


if __name__ == "__main__":
    main()
