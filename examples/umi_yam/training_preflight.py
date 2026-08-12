#!/usr/bin/env python

"""Fail-closed dataset and checkpoint checks for the four UMI/YAM training jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _env_path(name: str) -> Path:
    return Path(os.environ[name])


def validate_dataset() -> None:
    import torch

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STATE

    kind = os.environ["DATA_KIND"]
    repo = os.environ["DATA_REPO"]
    root = Path(os.environ["DATA_ROOT"]) if os.environ["DATA_ROOT"] else None
    revision = os.environ["DATA_REVISION"] or None
    task = os.environ["TASK"]
    action_dim = int(os.environ["ACTION_DIM"])
    drop_last = int(os.environ["DROP_LAST"])
    manifest_path = _env_path("DATA_MANIFEST")
    assert manifest_path.is_file() and not manifest_path.is_symlink()

    if kind == "ee":
        from lerobot.datasets.umi_yam_ee_dataset import (
            APPROVED_QUERY_COUNT,
            AUTHORITATIVE_SOURCE,
            TASK,
            UMIYAMEEDataset,
        )

        assert (repo, revision, task, action_dim, drop_last) == (
            AUTHORITATIVE_SOURCE.repo_id,
            AUTHORITATIVE_SOURCE.revision,
            TASK,
            20,
            0,
        )
        source = LeRobotDataset(repo, revision=revision, video_backend="pyav", return_uint8=True)
        dataset = UMIYAMEEDataset(source, os.environ["UMI_YAM_EE_CACHE_ROOT"])
        assert (source.meta.total_frames, source.meta.total_episodes) == (179_951, 182)
        assert (len(dataset), dataset.num_episodes) == (APPROVED_QUERY_COUNT, 182)
    else:
        from lerobot.scripts.materialize_umi_yam_joint_dataset import (
            EXPECTED_ACTION_VALUES,
            EXPECTED_EPISODES,
            EXPECTED_FRAMES,
            EXPECTED_QUERIES,
            I2RT_REVISION,
            KINEMATICS,
            OUTPUT_TASK,
            SOURCE_REPO_ID,
            SOURCE_REVISION,
        )

        assert root is not None and (task, action_dim, drop_last) == (OUTPUT_TASK, 14, 24)
        assert repo == "ASethi04/dual-lidar-combined-filtered-joint-positions-long-gripper-trainable"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == "umi_yam.published_joint.long_gripper182.v1"
        assert manifest["source"] == {"repo_id": SOURCE_REPO_ID, "revision": SOURCE_REVISION}
        assert manifest["kinematics"] == KINEMATICS
        assert manifest["kinematics"]["revision"] == I2RT_REVISION
        assert manifest["task"] == task and manifest["action_horizon"] == 24
        assert manifest["terminal_query_policy"] == "exclude final 24 query rows per episode"
        assert manifest["stats"]["state_count"] == EXPECTED_QUERIES
        assert manifest["stats"]["action_count"] == EXPECTED_ACTION_VALUES
        assert len(manifest["episodes"]) == EXPECTED_EPISODES
        assert manifest["episodes"][-1]["valid_query_compact_range"][1] == EXPECTED_QUERIES
        metadata = LeRobotDatasetMetadata(repo, root=root)
        assert (metadata.total_episodes, metadata.total_frames) == (EXPECTED_EPISODES, EXPECTED_FRAMES)
        dataset = LeRobotDataset(
            repo,
            root=root,
            delta_timestamps={ACTION: [index / 30 for index in range(24)]},
            video_backend="pyav",
            return_uint8=True,
        )

    assert list(dataset.meta.tasks.index) == [task]
    sample = dataset[0]
    assert sample["task"] == task
    assert tuple(sample[OBS_STATE].shape) == (action_dim,)
    assert tuple(sample[ACTION].shape) == (24, action_dim)
    if kind == "ee":
        assert tuple(sample["action_is_pad"].shape) == (24,) and not torch.any(sample["action_is_pad"])
    print(f"DATASET_OK kind={kind} task={task!r} state={action_dim} action=24x{action_dim}")


def _regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _checkpoint_config(checkpoint: Path, *, incomplete_ok: bool) -> Path | None:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise RuntimeError(f"numeric checkpoint must be a regular directory: {checkpoint}")
    step = int(checkpoint.name)
    if not (0 < step <= 12_000 and step % 1000 == 0):
        raise RuntimeError(f"invalid checkpoint boundary: {checkpoint}")
    pretrained, state_dir = checkpoint / "pretrained_model", checkpoint / "training_state"
    required = [
        *(
            pretrained / name
            for name in (
                "train_config.json",
                "config.json",
                "model.safetensors",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            )
        ),
        *(
            state_dir / name
            for name in (
                "training_step.json",
                "optimizer_state.safetensors",
                "optimizer_param_groups.json",
                "rng_state.safetensors",
                "scheduler_state.json",
            )
        ),
    ]
    if not all(path.exists() or path.is_symlink() for path in required):
        if incomplete_ok:
            return None
        raise RuntimeError(f"incomplete checkpoint: {checkpoint}")
    if not all(_regular(path) for path in required):
        raise RuntimeError(f"malformed checkpoint files: {checkpoint}")
    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = json.loads((pretrained / processor_name).read_text())
        if not isinstance(processor.get("steps"), list):
            raise RuntimeError(f"malformed processor config in {checkpoint}")
        for item in processor["steps"]:
            if not isinstance(item, dict):
                raise RuntimeError(f"malformed processor step in {checkpoint}")
            if "state_file" not in item:
                continue
            state_file = item["state_file"]
            if not isinstance(state_file, str) or Path(state_file).name != state_file:
                raise RuntimeError(f"unsafe processor state path in {checkpoint}: {state_file!r}")
            required.append(pretrained / state_file)
    if not all(path.exists() or path.is_symlink() for path in required):
        if incomplete_ok:
            return None
        raise RuntimeError(f"incomplete processor state: {checkpoint}")
    if not all(_regular(path) for path in required):
        raise RuntimeError(f"malformed processor state in {checkpoint}")
    state = json.loads((state_dir / "training_step.json").read_text())
    if state != {"step": step, "num_processes": 16, "batch_size": 4}:
        raise RuntimeError(f"invalid training state in {checkpoint}: {state}")
    return pretrained / "train_config.json"


def _validate_train_config(config_path: Path, run_dir: Path) -> int:
    profile = os.environ["UMI_YAM_PROFILE"]
    model = "molmoact2" if profile.startswith("molmoact2-") else "pi05"
    dimension = 20 if profile.endswith("ee20") else 14
    checkpoint = config_path.parent.parent
    assert config_path == _checkpoint_config(checkpoint, incomplete_ok=False)
    assert checkpoint.parent.resolve() == (run_dir / "checkpoints").resolve()
    cfg = json.loads(config_path.read_text())
    assert Path(cfg["output_dir"]).resolve() == run_dir.resolve()
    assert cfg["job_name"] == f"umi-yam-{profile}-{os.environ['UMI_YAM_RUN_ID']}"
    assert (cfg["batch_size"], cfg["seed"], cfg["steps"]) == (4, 1000, 12_000)
    assert (cfg["save_freq"], cfg["eval_steps"]) == (1000, 0)
    assert cfg["dataset"]["repo_id"] == os.environ["DATA_REPO"]
    assert cfg["dataset"]["drop_n_last_frames"] == (0 if dimension == 20 else 24)
    policy = cfg["policy"]
    assert policy["type"] == model and policy["chunk_size"] == policy["n_action_steps"] == 24
    assert policy["output_features"]["action"]["shape"] == [dimension]
    return int(checkpoint.name)


def select_resume() -> None:
    run_dir = _env_path("RUN_DIR")
    restart = int(os.environ["RESTART_COUNT"])
    requested = os.environ.get("REQUESTED_CONFIG", "")
    selected: Path | None = None

    if restart:
        if not run_dir.exists():
            print("FRESH")
            return
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise RuntimeError(f"requeue output is not a regular directory: {run_dir}")
        checkpoints = run_dir / "checkpoints"
        if checkpoints.exists() and (checkpoints.is_symlink() or not checkpoints.is_dir()):
            raise RuntimeError(f"invalid checkpoints directory: {checkpoints}")
        complete: dict[int, tuple[Path, Path]] = {}
        if checkpoints.is_dir():
            for child in checkpoints.iterdir():
                if child.name.isdigit():
                    config = _checkpoint_config(child, incomplete_ok=True)
                    if config is not None:
                        step = int(child.name)
                        if step in complete:
                            raise RuntimeError(f"duplicate numeric checkpoint step {step}")
                        complete[step] = (child.resolve(), config.resolve())
        last, last_step = checkpoints / "last", None
        if last.is_symlink():
            target = last.resolve(strict=True)
            if target.parent != checkpoints.resolve() or int(target.name) not in complete:
                raise RuntimeError(f"checkpoints/last is not a complete numeric checkpoint: {target}")
            last_step = int(target.name)
        elif last.exists():
            raise RuntimeError("checkpoints/last must be a symlink")
        if complete:
            step = max(complete)
            if last_step is not None and last_step < step:
                print(f"STALE_LAST\t{last_step}\t{step}", file=os.sys.stderr)
            selected = complete[step][1]
        elif checkpoints.is_dir() and any(
            child.name.isdigit() and (child / "training_state").exists() for child in checkpoints.iterdir()
        ):
            raise RuntimeError("requeue has training-state fragments but no complete checkpoint")
    elif requested:
        candidate = Path(requested).resolve(strict=True)
        if candidate.name != "train_config.json" or candidate.parent.name != "pretrained_model":
            raise RuntimeError(f"invalid resume config path: {candidate}")
        selected = candidate

    if selected is None:
        print("FRESH")
        return
    step = _validate_train_config(selected, run_dir)
    print(f"{'COMPLETE' if step == 12_000 else 'RESUME'}\t{selected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dataset", "resume"))
    args = parser.parse_args()
    if args.mode == "dataset":
        validate_dataset()
    else:
        select_resume()


if __name__ == "__main__":
    main()
