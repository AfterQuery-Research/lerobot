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

import copy
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from datasets import Dataset

from lerobot.datasets import factory as dataset_factory
from lerobot.datasets.umi_yam import ACTION_DIM, ACTION_HORIZON
from lerobot.datasets.umi_yam_ee_dataset import (
    _EXPECTED_FEATURES,
    _JOINT_NAMES,
    APPROVED_QUERY_COUNT,
    AUTHORITATIVE_SOURCE,
    DATASET_VERSION,
    JOINT_REPO_ID,
    JOINT_REVISION,
    LEFT_GRIPPER,
    LEFT_VIDEO,
    RIGHT_GRIPPER,
    RIGHT_VIDEO,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    STATE_SEMANTICS,
    TASK,
    SourceContract,
    UMIYAMEEDataset,
    exact_vector_stats,
    materialize_cache,
    materialize_episode,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


def _trajectory(frames=ACTION_HORIZON + 2):
    state = np.zeros((frames, 12), dtype=np.float32)
    state[:, 0] = np.arange(frames) ** 2
    state[:, 7] = 2 * np.arange(frames)
    left = np.linspace(0.2, 1, frames, dtype=np.float32)[:, None]
    right = np.linspace(0.9, 0.3, frames, dtype=np.float32)[:, None]
    return state, left, right


def test_pinned_long_gripper_sources_and_all_episode_contract():
    assert AUTHORITATIVE_SOURCE.repo_id == SOURCE_REPO_ID
    assert AUTHORITATIVE_SOURCE.revision == SOURCE_REVISION
    assert SOURCE_REVISION == "a29ae6a5531584fb950c7bb3bb5895f18421b108"
    assert JOINT_REVISION == "387d696eb36de411a11c8b2612a6f19822ca3a53"
    assert DATASET_VERSION == "umi_yam.long_gripper.all_182.v1"
    assert STATE_SEMANTICS == "identity_r6d_rows_current_gripper"
    assert TASK == "pick up oranges and place them in the bowl"
    assert (AUTHORITATIVE_SOURCE.episodes, AUTHORITATIVE_SOURCE.frames) == (182, 179_951)
    assert APPROVED_QUERY_COUNT == 175_583


def test_materialization_is_query_anchored_and_drops_the_tail():
    state, left, right = _trajectory()
    episode = materialize_episode(state, left, right)

    assert episode.state.shape == (2, ACTION_DIM)
    assert episode.action.shape == (2, ACTION_HORIZON, ACTION_DIM)
    np.testing.assert_allclose(episode.state[:, :9], [[0, 0, 0, 1, 0, 0, 0, 1, 0]] * 2)
    assert episode.state[1, 9] == pytest.approx(left[1, 0])
    np.testing.assert_allclose(episode.action[0, :2, 0], [1, 4])
    np.testing.assert_allclose(episode.action[0, :2, 11], [2, 4])
    assert episode.state[0, 9] == pytest.approx(0.2)
    assert episode.state[0, 19] == pytest.approx(0.9)
    assert episode.action[-1, -1, 9] == pytest.approx(1)
    with pytest.raises(ValueError, match="normalized_0_closed_1_open"):
        materialize_episode(state, left * 2, right)


def test_stats_are_exact_over_global_samples_not_episode_quantiles():
    values = np.zeros((4, ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    values[:, :, 0] = np.array([0, 100, 200, 300])[:, None]

    stats = exact_vector_stats(values)

    assert stats["count"].item() == 4 * ACTION_HORIZON
    assert stats["q50"][0] == pytest.approx(150)


class _Meta:
    def __init__(self, contract, joint=False):
        self.repo_id, self.revision = contract.repo_id, contract.revision
        self.info = SimpleNamespace(
            features=copy.deepcopy(_EXPECTED_FEATURES),
            fps=contract.fps,
            total_episodes=contract.episodes,
            total_frames=contract.frames,
        )
        self.info.features = {
            key: {"dtype": dtype, "shape": shape, "names": names}
            for key, (dtype, shape, names) in self.info.features.items()
        }
        if joint:
            self.info.features.pop(LEFT_GRIPPER)
            self.info.features.pop(RIGHT_GRIPPER)
            self.info.features[OBS_STATE] = {
                "dtype": "float32",
                "shape": (14,),
                "names": _JOINT_NAMES,
            }
        self.episodes = Dataset.from_list(
            [{"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": contract.frames}]
        )
        self.tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index(["bimanual umi demo"], name="task"))
        self.stats = {LEFT_GRIPPER: {"mean": [1]}, RIGHT_GRIPPER: {"mean": [1]}}

    def __getattr__(self, name):
        if name in {"fps", "total_episodes", "total_frames", "features"}:
            return getattr(self.info, name)
        raise AttributeError(name)


class _HF(list):
    def __getitem__(self, indices):
        if isinstance(indices, list):
            rows = [list.__getitem__(self, i) for i in indices]
            return {key: [row[key] for row in rows] for key in rows[0]}
        return list.__getitem__(self, indices)


class _Source:
    def __init__(self, contract, state, left=None, right=None, joint=False):
        self.repo_id, self.revision = contract.repo_id, contract.revision
        self.root, self.fps, self.episodes = "/tmp/fake", contract.fps, None
        self.meta = _Meta(contract, joint=joint)
        rows = []
        for i in range(len(state)):
            rows.append(
                {
                    OBS_STATE: torch.from_numpy(state[i]),
                    LEFT_VIDEO: torch.tensor([i, 1]),
                    RIGHT_VIDEO: torch.tensor([i, 2]),
                    "index": torch.tensor(i),
                    "episode_index": torch.tensor(0),
                    "frame_index": torch.tensor(i),
                    "timestamp": torch.tensor(i / contract.fps),
                }
            )
            if not joint:
                rows[-1][LEFT_GRIPPER] = torch.from_numpy(left[i])
                rows[-1][RIGHT_GRIPPER] = torch.from_numpy(right[i])
        self.hf_dataset = _HF(rows)

    def __getitem__(self, index):
        return self.hf_dataset[index]


def test_cache_fails_closed_and_runtime_preserves_source_observations(tmp_path):
    state, left, right = _trajectory()
    contract = SourceContract(episodes=1, frames=len(state))
    source = _Source(contract, state, left, right)
    joint_contract = SourceContract(
        repo_id=JOINT_REPO_ID,
        revision=JOINT_REVISION,
        episodes=1,
        frames=len(state),
    )
    joint_state = np.zeros((len(state), 14), dtype=np.float32)
    joint_state[:, 6], joint_state[:, 13] = left[:, 0], right[:, 0]
    joint_source = _Source(joint_contract, joint_state, joint=True)
    source.revision = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        contract.validate(source)
    source.revision = contract.revision
    source.meta.features[OBS_STATE]["shape"] = (11,)
    with pytest.raises(ValueError, match="pinned observation-only schema"):
        contract.validate(source)
    source.meta.features[OBS_STATE]["shape"] = (12,)
    joint_source.hf_dataset[0]["frame_index"] = torch.tensor(99)
    with pytest.raises(ValueError, match="row-aligned"):
        materialize_cache(source, joint_source, tmp_path, contract, joint_contract)
    joint_source.hf_dataset[0]["frame_index"] = torch.tensor(0)
    materialize_cache(source, joint_source, tmp_path, contract, joint_contract)
    dataset = UMIYAMEEDataset(source, tmp_path, contract=contract)
    item = dataset[-1]
    assert len(dataset) == 2
    assert dataset.meta.total_frames == 2
    torch.testing.assert_close(item[LEFT_VIDEO], torch.tensor([1, 1]))
    assert item[ACTION].shape == (ACTION_HORIZON, ACTION_DIM)
    assert not item["action_is_pad"].any()
    assert item[ACTION][-1, 9] == pytest.approx(1)
    assert item["task"] == TASK
    assert dataset.meta.tasks.index.tolist() == [TASK]
    assert dataset.meta.episodes[0]["tasks"] == [TASK]
    assert LEFT_GRIPPER not in item and RIGHT_GRIPPER not in item
    assert LEFT_GRIPPER not in dataset.meta.stats and RIGHT_GRIPPER not in dataset.meta.stats
    assert set(dataset_to_policy_features(dataset.meta.features)) == {
        OBS_STATE,
        ACTION,
        LEFT_VIDEO,
        RIGHT_VIDEO,
    }
    assert (dataset._state.shape, dataset._action.shape) == ((2, ACTION_DIM), (2, ACTION_HORIZON, ACTION_DIM))


def test_factory_opt_in_is_pinned_and_never_applies_delta_windows(monkeypatch, tmp_path):
    ds_cfg = SimpleNamespace(
        repo_id=SOURCE_REPO_ID,
        revision=SOURCE_REVISION,
        streaming=False,
        episodes=None,
        eval_split=0,
        drop_n_last_frames=None,
        root="/fake/ee",
        video_backend="pyav",
        depth_output_unit="m",
        umi_yam_ee_cache_root=str(tmp_path),
    )
    cfg = SimpleNamespace(
        dataset=ds_cfg,
        tolerance_s=1e-4,
        trainable_config=SimpleNamespace(chunk_size=24, n_action_steps=24),
    )
    ee, wrapped = object(), object()
    source_ctor = Mock(return_value=ee)
    wrapper_ctor = Mock(return_value=wrapped)
    monkeypatch.setattr(dataset_factory, "LeRobotDataset", source_ctor)
    monkeypatch.setattr(dataset_factory, "UMIYAMEEDataset", wrapper_ctor)

    assert dataset_factory._make_umi_yam_ee_dataset(cfg, None) is wrapped
    source_ctor.assert_called_once()
    assert source_ctor.call_args.args[0] == SOURCE_REPO_ID
    assert source_ctor.call_args.kwargs["delta_timestamps"] is None
    wrapper_ctor.assert_called_once_with(ee, str(tmp_path))
    assert ds_cfg.drop_n_last_frames == 0

    wrapper_ctor.side_effect = FileNotFoundError("missing immutable cache")
    with pytest.raises(FileNotFoundError, match="missing immutable cache"):
        dataset_factory._make_umi_yam_ee_dataset(cfg, None)

    ds_cfg.revision = "main"
    with pytest.raises(ValueError, match="requires pinned source"):
        dataset_factory._make_umi_yam_ee_dataset(cfg, None)
    ds_cfg.revision, ds_cfg.drop_n_last_frames = SOURCE_REVISION, 24
    with pytest.raises(ValueError, match="already excludes"):
        dataset_factory._make_umi_yam_ee_dataset(cfg, None)
    ds_cfg.drop_n_last_frames, ds_cfg.umi_yam_ee_cache_root = 0, None
    with pytest.raises(ValueError, match="shared across ranks/nodes"):
        dataset_factory._make_umi_yam_ee_dataset(cfg, None)


def test_memmaps_reopen_after_small_spawn_pickle(tmp_path):
    state = np.lib.format.open_memmap(tmp_path / "state.npy", mode="w+", dtype="float32", shape=(1024, 20))
    action = np.lib.format.open_memmap(
        tmp_path / "action.npy", mode="w+", dtype="float32", shape=(1024, 24, 20)
    )
    source_index = np.lib.format.open_memmap(
        tmp_path / "source_index.npy", mode="w+", dtype="int64", shape=(1024,)
    )
    state[17, 3], action[17, 5, 7], source_index[17] = 1.25, -0.75, 99
    for array in (state, action, source_index):
        array.flush()

    dataset = UMIYAMEEDataset.__new__(UMIYAMEEDataset)
    dataset._cache_root = tmp_path
    dataset._state, dataset._action, dataset._source_index = state, action, source_index
    payload = pickle.dumps(dataset)

    assert len(payload) < 2_000
    restored = pickle.loads(payload)
    assert isinstance(restored._action, np.memmap)
    assert restored._state[17, 3] == pytest.approx(1.25)
    assert restored._action[17, 5, 7] == pytest.approx(-0.75)
    assert restored._source_index[17] == 99
