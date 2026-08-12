#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
import pytest

from lerobot.scripts.materialize_umi_yam_joint_dataset import (
    ACTION_HORIZON,
    EXPECTED_ACTION_VALUES,
    EXPECTED_EPISODES,
    EXPECTED_FRAMES,
    EXPECTED_QUERIES,
    I2RT_REVISION,
    KINEMATICS,
    OUTPUT_TASK,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    _link_or_copy,
    _training_arrays,
    build_manifest,
    derive_episode,
)


def _state(frames: int = ACTION_HORIZON + 3) -> np.ndarray:
    state = np.arange(frames * 14, dtype=np.float32).reshape(frames, 14) / 100
    state[:, 6] = np.linspace(0.1, 1, frames, dtype=np.float32)
    state[:, 13] = np.linspace(1, 0.2, frames, dtype=np.float32)
    return state


def test_pins_published_long_gripper_source_and_i2rt_provenance() -> None:
    assert SOURCE_REPO_ID == "brandonyang/dual-lidar-combined-filtered-joint-positions-long-gripper"
    assert SOURCE_REVISION == "387d696eb36de411a11c8b2612a6f19822ca3a53"
    assert I2RT_REVISION == "7ed46f4e4e316133a0c39aa6cf34a73d2718e850"
    assert KINEMATICS["grasp_offset_mm"] == 220
    assert KINEMATICS["grasp_axes"] == "UMI-compatible: +X up, +Y right, +Z forward"
    assert OUTPUT_TASK == "pick up oranges and place them in the bowl"
    assert (EXPECTED_EPISODES, EXPECTED_FRAMES, EXPECTED_QUERIES) == (182, 179_951, 175_583)
    assert EXPECTED_ACTION_VALUES == 4_213_992


def test_derivation_preserves_published_state_and_shifts_action() -> None:
    published = _state()
    state, action = derive_episode(published)

    assert state is published
    np.testing.assert_array_equal(state, published)
    np.testing.assert_array_equal(action[:-1], published[1:])
    np.testing.assert_array_equal(action[-1], published[-1])
    np.testing.assert_array_equal(state[:, (6, 13)], published[:, (6, 13)])


def test_derivation_fails_closed_on_noncanonical_state() -> None:
    with pytest.raises(ValueError, match="float32"):
        derive_episode(_state().astype(np.float64))
    invalid = _state()
    invalid[0, 6] = 1.01
    with pytest.raises(ValueError, match="0=closed, 1=open"):
        derive_episode(invalid)
    with pytest.raises(ValueError, match="more than 24"):
        derive_episode(_state(ACTION_HORIZON))


def test_training_distribution_is_query_rows_and_24_future_targets() -> None:
    state = _state(ACTION_HORIZON + 2)
    query_state, targets = _training_arrays(state)

    np.testing.assert_array_equal(query_state, state[:2])
    assert len(targets) == ACTION_HORIZON
    for offset, target in enumerate(targets, start=1):
        np.testing.assert_array_equal(target, state[offset : offset + 2])
    assert sum(len(target) for target in targets) == 2 * ACTION_HORIZON


def test_manifest_keeps_all_182_and_records_tail_and_stats_contracts() -> None:
    lengths = [EXPECTED_FRAMES - (EXPECTED_EPISODES - 1) * 25, *([25] * (EXPECTED_EPISODES - 1))]
    manifest = build_manifest(lengths)

    assert manifest["schema"] == "umi_yam.published_joint.long_gripper182.v1"
    assert manifest["algorithm"] == "preserve published 14-D state; action[t]=state[t+1]"
    assert manifest["terminal_query_policy"] == "exclude final 24 query rows per episode"
    assert len(manifest["episodes"]) == EXPECTED_EPISODES
    assert manifest["episodes"][-1]["valid_query_compact_range"][1] == EXPECTED_QUERIES
    assert manifest["stats"]["state_count"] == EXPECTED_QUERIES
    assert manifest["stats"]["action_count"] == EXPECTED_ACTION_VALUES
    assert "excluded_source_episodes" not in manifest


def test_video_copy_resolves_hugging_face_snapshot_symlink(tmp_path) -> None:
    blob = tmp_path / "blobs/video.mp4"
    blob.parent.mkdir()
    blob.write_bytes(b"video")
    source = tmp_path / "snapshot/video.mp4"
    source.parent.mkdir()
    source.symlink_to(blob)
    destination = tmp_path / "output/video.mp4"

    _link_or_copy(source, destination)

    assert destination.is_file() and not destination.is_symlink()
    assert destination.read_bytes() == b"video"
