#!/usr/bin/env python
"""Fail-closed validation for a built onset-v3 artifact.

The validator binds the pinned filtered source, immutable onset-v2 parent, and
onset-v3 successor. It proves that v3 changes only the three declared suffixes,
preserves every available contact/gripper event, masks k=1..24 terminal padding,
and copies all 108 videos byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.umi_current_relative import (
    UMI_CURRENTREL_ONSET_METADATA_PATH,
    UMI_CURRENTREL_ONSET_V3_METADATA_PATH,
    UMI_CURRENTREL_ONSET_V3_SCHEMA_ID,
    UMI_CURRENTREL_SPLIT_PATH,
    UMI_TCP_WINDOW_KEY,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d import (
    TRAIN_EPISODES,
    _stats_block,
    build_episode_arrays,
)
from lerobot.scripts.convert_dual_lidar_umi_currentrel_r6d_onset_v3 import (
    EXPECTED_EPISODES,
    EXPECTED_OUTPUT_FRAMES,
    EXPECTED_SOURCE_FILTERING_SHA256,
    EXPECTED_SOURCE_INFO_SHA256,
    EXPECTED_TRAIN_FRAMES,
    EXPECTED_V3_RETAINED_LENGTHS,
    EXPECTED_VALID_ACTION_POSITIONS,
    EXPECTED_VALID_TRAIN_ACTION_POSITIONS,
    EXPECTED_VALID_VALIDATION_ACTION_POSITIONS,
    EXPECTED_VALIDATION_FRAMES,
    EXPECTED_VIDEO_FILES,
    OUTPUT_DATASET_NAME,
    OUTPUT_REPO_ID,
    SOURCE_SUFFIX_END_FRAME_EXCLUSIVE,
    _sha256,
    _v3_tail_end,
    _validate_preserved_gripper_events,
    _validate_source_root,
    action_padding_mask,
)

EXPECTED_V2_CHECKSUM_LIST_SHA256 = "49d0840ebe17d43ac68250d6329ad356d75756818170a67c6f58225366f440fa"
PREFIX_COLUMNS = (
    "observation.state",
    UMI_TCP_WINDOW_KEY,
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _episode_index(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def _indexed_parquets(root: Path, relative: str) -> dict[int, Path]:
    paths = sorted((root / relative).rglob("*.parquet"))
    indexed = {_episode_index(path): path for path in paths}
    _require(
        len(paths) == len(indexed) == EXPECTED_EPISODES,
        f"{root / relative} must contain one uniquely indexed parquet per episode",
    )
    _require(set(indexed) == set(range(EXPECTED_EPISODES)), "episode parquet indices must be 0..53")
    return indexed


def _validate_checksum_list(root: Path) -> tuple[str, dict[str, str]]:
    checksum_path = root / "meta/artifact_manifest.sha256"
    checksum_digest = _sha256(checksum_path)
    entries: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        _require(bool(separator) and len(digest) == 64 and bool(relative), f"invalid checksum line: {line}")
        _require(relative not in entries, f"duplicate checksum entry: {relative}")
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"checksummed path is not regular: {relative}")
        _require(_sha256(path) == digest, f"artifact checksum mismatch: {relative}")
        entries[relative] = digest

    manifest = json.loads((root / "meta/artifact_manifest.json").read_text(encoding="utf-8"))
    generated = manifest.get("generated_file_sha256")
    _require(isinstance(generated, dict), "artifact manifest has no generated_file_sha256 mapping")
    for relative, digest in generated.items():
        _require(entries.get(relative) == digest, f"manifest/checksum-list mismatch: {relative}")
    _require(
        entries.get("meta/artifact_manifest.json") == _sha256(root / "meta/artifact_manifest.json"),
        "artifact manifest itself is not bound by the checksum list",
    )
    return checksum_digest, entries


def _validate_video_copies(source_root: Path, v2_root: Path, v3_root: Path) -> dict[str, str]:
    relative_sets = []
    for root in (source_root, v2_root, v3_root):
        videos = sorted((root / "videos").rglob("*.mp4"))
        _require(len(videos) == EXPECTED_VIDEO_FILES, f"{root} must contain 108 MP4 files")
        if root == v3_root:
            _require(
                all(path.is_file() and not path.is_symlink() for path in videos),
                "all v3 videos must be regular copied files",
            )
        relative_sets.append({path.relative_to(root / "videos").as_posix() for path in videos})
    _require(relative_sets[0] == relative_sets[1] == relative_sets[2], "video path sets differ")

    hashes: dict[str, str] = {}
    for relative in sorted(relative_sets[0]):
        digests = {_sha256(root / "videos" / relative) for root in (source_root, v2_root, v3_root)}
        _require(len(digests) == 1, f"source/v2/v3 video SHA-256 differs: {relative}")
        hashes[relative] = digests.pop()
    return hashes


def _column_prefix_equal(v2: pa.Table, v3: pa.Table, column: str) -> bool:
    return v3[column].equals(v2[column].slice(0, len(v3)))


def _validate_video_intervals(
    source_metadata: pa.Table,
    v3_metadata: pa.Table,
    *,
    episode_index: int,
    onset_frame: int,
    tail_end_frame_exclusive: int,
) -> None:
    from_keys = sorted(
        key
        for key in source_metadata.column_names
        if key.startswith("videos/") and key.endswith("/from_timestamp")
    )
    _require(len(from_keys) == 2, f"episode {episode_index} must have two source-camera intervals")
    _require(
        from_keys
        == sorted(
            key
            for key in v3_metadata.column_names
            if key.startswith("videos/") and key.endswith("/from_timestamp")
        ),
        f"episode {episode_index} v3 camera interval keys changed",
    )
    for from_key in from_keys:
        to_key = f"{from_key.removesuffix('/from_timestamp')}/to_timestamp"
        _require(to_key in source_metadata.column_names, f"source interval missing {to_key}")
        _require(to_key in v3_metadata.column_names, f"v3 interval missing {to_key}")
        source_from = float(source_metadata[from_key][0].as_py())
        source_to = float(source_metadata[to_key][0].as_py())
        expected_from = source_from + onset_frame / 30.0
        expected_to = source_from + tail_end_frame_exclusive / 30.0
        actual_from = float(v3_metadata[from_key][0].as_py())
        actual_to = float(v3_metadata[to_key][0].as_py())
        _require(
            np.isclose(actual_from, expected_from, atol=1e-6, rtol=0.0),
            f"episode {episode_index} {from_key} does not start at onset",
        )
        _require(
            np.isclose(actual_to, expected_to, atol=1e-6, rtol=0.0),
            f"episode {episode_index} {to_key} does not end at the source-exclusive cut",
        )
        _require(actual_to <= source_to + 1e-6, f"episode {episode_index} interval exceeds source")


def _validate_parent_prefixes(
    source_root: Path,
    v2_root: Path,
    v3_root: Path,
) -> tuple[dict[str, int], int]:
    source_files = _indexed_parquets(source_root, "data")
    source_episode_files = _indexed_parquets(source_root, "meta/episodes")
    v2_files = _indexed_parquets(v2_root, "data")
    v3_files = _indexed_parquets(v3_root, "data")
    v2_semantics = json.loads((v2_root / UMI_CURRENTREL_ONSET_METADATA_PATH).read_text(encoding="utf-8"))
    v3_semantics = json.loads((v3_root / UMI_CURRENTREL_ONSET_V3_METADATA_PATH).read_text(encoding="utf-8"))
    v2_onsets = {
        int(key): int(value)
        for key, value in v2_semantics["onset_alignment"]["onset_frame_by_episode"].items()
    }
    v3_onsets = {
        int(key): int(value)
        for key, value in v3_semantics["onset_alignment"]["onset_frame_by_episode"].items()
    }
    _require(v3_onsets == v2_onsets, "v3 onset indices must exactly equal immutable v2")

    next_global_index = 0
    retained: dict[str, int] = {}
    total_valid = 0
    for episode_index in range(EXPECTED_EPISODES):
        source = pq.read_table(source_files[episode_index])
        v2 = pq.read_table(v2_files[episode_index])
        v3 = pq.read_table(v3_files[episode_index])
        _, expected_source_end = _v3_tail_end(episode_index, len(source))
        expected_length = expected_source_end - v2_onsets[episode_index]
        _require(len(v3) == expected_length, f"episode {episode_index} v3 length changed")
        if episode_index in EXPECTED_V3_RETAINED_LENGTHS:
            _require(
                expected_length == EXPECTED_V3_RETAINED_LENGTHS[episode_index],
                f"episode {episode_index} locked retained length changed",
            )
        else:
            _require(len(v3) == len(v2), f"undeclared suffix change in episode {episode_index}")
        for column in PREFIX_COLUMNS:
            _require(
                _column_prefix_equal(v2, v3, column),
                f"episode {episode_index} is not an exact v2 prefix for {column}",
            )
        action_prefix_length = (
            len(v3) if episode_index not in SOURCE_SUFFIX_END_FRAME_EXCLUSIVE else len(v3) - 1
        )
        _require(
            v3["action"].slice(0, action_prefix_length).equals(v2["action"].slice(0, action_prefix_length)),
            f"episode {episode_index} changed a nonterminal convenience action",
        )
        expected_indices = np.arange(next_global_index, next_global_index + len(v3), dtype=np.int64)
        np.testing.assert_array_equal(v3["index"].to_numpy(), expected_indices)
        next_global_index += len(v3)

        padding = action_padding_mask(len(v3))
        valid_positions = int((~padding).sum())
        total_valid += valid_positions
        retained[str(episode_index)] = len(v3)

        episode_metadata = pq.read_table(v3_root / "meta/episodes/chunk-000" / v3_files[episode_index].name)
        _require(episode_metadata["length"].to_pylist() == [len(v3)], "episode length metadata mismatch")
        _require(
            episode_metadata["stats/action/count"].to_pylist() == [[valid_positions]],
            f"episode {episode_index} padded actions leaked into statistics",
        )
        _validate_video_intervals(
            pq.read_table(source_episode_files[episode_index]),
            episode_metadata,
            episode_index=episode_index,
            onset_frame=v2_onsets[episode_index],
            tail_end_frame_exclusive=expected_source_end,
        )

    _require(next_global_index == EXPECTED_OUTPUT_FRAMES, "v3 global row count changed")
    _require(total_valid == EXPECTED_VALID_ACTION_POSITIONS, "v3 valid action count changed")
    return retained, total_valid


def _validate_full_train_only_stats(source_root: Path, v3_root: Path) -> dict[str, int]:
    source_files = _indexed_parquets(source_root, "data")
    semantics = json.loads((v3_root / UMI_CURRENTREL_ONSET_V3_METADATA_PATH).read_text(encoding="utf-8"))
    onsets = {
        int(key): int(value) for key, value in semantics["onset_alignment"]["onset_frame_by_episode"].items()
    }
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    helpers: list[np.ndarray] = []
    for episode_index in TRAIN_EPISODES:
        source = pq.read_table(source_files[episode_index])
        _, tail_end = _v3_tail_end(episode_index, len(source))
        onset = onsets[episode_index]
        state, _, helper, chunks = build_episode_arrays(source.slice(onset, tail_end - onset))
        padding = action_padding_mask(len(state))
        states.append(state)
        actions.append(chunks[~padding])
        helpers.append(helper)

    recomputed = {
        "observation.state": _stats_block(np.concatenate(states, axis=0)),
        "action": _stats_block(np.concatenate(actions, axis=0)),
        UMI_TCP_WINDOW_KEY: _stats_block(np.concatenate(helpers, axis=0)),
    }
    stored = json.loads((v3_root / "meta/stats.json").read_text(encoding="utf-8"))
    for feature, feature_stats in recomputed.items():
        _require(feature in stored, f"stored stats omit {feature}")
        _require(set(stored[feature]) == set(feature_stats), f"stored {feature} stat keys changed")
        for statistic, expected in feature_stats.items():
            actual = stored[feature][statistic]
            if statistic == "count":
                _require(actual == expected, f"stored {feature} count is not train-only recomputation")
            else:
                np.testing.assert_allclose(
                    np.asarray(actual, dtype=np.float64),
                    np.asarray(expected, dtype=np.float64),
                    atol=0.0,
                    rtol=0.0,
                    err_msg=f"stored {feature}/{statistic} is not train-only recomputation",
                )
    return {feature: int(values["count"][0]) for feature, values in recomputed.items()}


def _validate_source_events(source_root: Path, v3_root: Path) -> dict[str, object]:
    _, source_files, _ = _validate_source_root(source_root)
    semantics = json.loads((v3_root / UMI_CURRENTREL_ONSET_V3_METADATA_PATH).read_text(encoding="utf-8"))
    onsets = {
        int(key): int(value) for key, value in semantics["onset_alignment"]["onset_frame_by_episode"].items()
    }
    filtering = json.loads((source_root / "meta/filtering.json").read_text(encoding="utf-8"))
    optimizer_results = {
        int(result["output_episode_index"]): result for result in filtering["optimizer_results"]
    }
    total_gripper_event_frames = 0
    last_events: dict[str, list[int | None]] = {}
    contact_frames: dict[str, dict[str, list[int]]] = {}
    total_contact_anchors = 0
    for source_path in source_files:
        table = pq.read_table(source_path)
        episode_index = int(table["episode_index"][0].as_py())
        _, tail_end = _v3_tail_end(episode_index, len(table))
        anchors = optimizer_results[episode_index]["retarget"]["contact_events"]["anchors"]
        episode_contacts: dict[str, list[int]] = {}
        for anchor in anchors:
            arm = str(anchor["arm"])
            frame = int(anchor["frame"])
            _require(bool(anchor["feasible"]), f"episode {episode_index} has infeasible contact anchor")
            _require(
                onsets[episode_index] <= frame < tail_end,
                f"episode {episode_index} contact frame {frame} is outside the v3 retained interval",
            )
            episode_contacts.setdefault(arm, []).append(frame)
            total_contact_anchors += 1
        contact_frames[str(episode_index)] = episode_contacts
        counts, last = _validate_preserved_gripper_events(
            table,
            tail_end_frame_exclusive=tail_end,
        )
        total_gripper_event_frames += sum(counts)
        if episode_index in SOURCE_SUFFIX_END_FRAME_EXCLUSIVE:
            last_events[str(episode_index)] = list(last)
    return {
        "optimizer_contact_anchors_preserved": total_contact_anchors,
        "optimizer_contact_frames_by_episode_and_arm": contact_frames,
        "detected_gripper_event_frames_preserved": total_gripper_event_frames,
        "last_gripper_event_frame_in_cut_episodes": last_events,
    }


def validate(source_root: Path, v2_root: Path, v3_root: Path) -> dict[str, object]:
    source_root = source_root.resolve(strict=True)
    v2_root = v2_root.resolve(strict=True)
    v3_root = v3_root.resolve(strict=True)
    _require(
        _sha256(source_root / "meta/info.json") == EXPECTED_SOURCE_INFO_SHA256, "source info hash mismatch"
    )
    _require(
        _sha256(source_root / "meta/filtering.json") == EXPECTED_SOURCE_FILTERING_SHA256,
        "source filtering hash mismatch",
    )
    v2_checksum_digest, v2_checksum_entries = _validate_checksum_list(v2_root)
    _require(v2_checksum_digest == EXPECTED_V2_CHECKSUM_LIST_SHA256, "v2 checksum-list digest mismatch")
    v3_checksum_digest, checksum_entries = _validate_checksum_list(v3_root)
    semantics = json.loads((v3_root / UMI_CURRENTREL_ONSET_V3_METADATA_PATH).read_text(encoding="utf-8"))
    split = json.loads((v3_root / UMI_CURRENTREL_SPLIT_PATH).read_text(encoding="utf-8"))
    info = json.loads((v3_root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((v3_root / "meta/stats.json").read_text(encoding="utf-8"))
    _require(semantics.get("schema_id") == UMI_CURRENTREL_ONSET_V3_SCHEMA_ID, "v3 schema mismatch")
    _require(semantics.get("dataset_name") == OUTPUT_DATASET_NAME, "v3 dataset name mismatch")
    _require(semantics.get("repository") == OUTPUT_REPO_ID, "v3 repository mismatch")
    _require(info.get("total_frames") == EXPECTED_OUTPUT_FRAMES, "v3 info frame count mismatch")
    _require(split.get("train_frames") == EXPECTED_TRAIN_FRAMES, "v3 train frame count mismatch")
    _require(
        split.get("validation_frames") == EXPECTED_VALIDATION_FRAMES,
        "v3 validation frame count mismatch",
    )
    _require(
        stats.get("action", {}).get("count") == [EXPECTED_VALID_TRAIN_ACTION_POSITIONS],
        "v3 global action stats include padding or holdouts",
    )
    _require(
        split.get("valid_train_action_positions") == EXPECTED_VALID_TRAIN_ACTION_POSITIONS,
        "v3 split train action count mismatch",
    )
    _require(
        split.get("valid_validation_action_positions") == EXPECTED_VALID_VALIDATION_ACTION_POSITIONS,
        "v3 split validation action count mismatch",
    )

    video_hashes = _validate_video_copies(source_root, v2_root, v3_root)
    retained, valid_positions = _validate_parent_prefixes(source_root, v2_root, v3_root)
    recomputed_stat_counts = _validate_full_train_only_stats(source_root, v3_root)
    event_report = _validate_source_events(source_root, v3_root)
    _require(event_report["optimizer_contact_anchors_preserved"] == 883, "contact count changed")
    return {
        "schema_id": "dual-lidar-umi-currentrel-r6d-onset-v3-artifact-validation-v1",
        "passed": True,
        "source_root": str(source_root),
        "v2_root": str(v2_root),
        "v3_root": str(v3_root),
        "source_info_sha256": EXPECTED_SOURCE_INFO_SHA256,
        "source_filtering_sha256": EXPECTED_SOURCE_FILTERING_SHA256,
        "v2_checksum_list_sha256": EXPECTED_V2_CHECKSUM_LIST_SHA256,
        "checksummed_v2_files": len(v2_checksum_entries),
        "v3_checksum_list_sha256": v3_checksum_digest,
        "checksummed_v3_files": len(checksum_entries),
        "retained_length_by_episode": retained,
        "valid_action_positions": valid_positions,
        "recomputed_train_only_stat_counts": recomputed_stat_counts,
        "video_files_sha256_equal": len(video_hashes),
        **event_report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = validate(args.source_root, args.v2_root, args.v3_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
