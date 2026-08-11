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

"""Safely reconstruct and replay a YAM-feasible filtered UMI episode.

The filtered dataset is deliberately observation-only. Its Cartesian pose track
is the achieved FK path from the offline YAM optimizer, mapped back into the UMI
coordinate convention. This module reverses that mapping, solves the path against
the installed YAM URDF, and only touches hardware when ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation

from lerobot.remote_inference.yam_current_relative_r6d import (
    RateLimitedActionChunk,
    rate_limit_action_chunk,
    resolve_yam_urdf,
)
from lerobot.remote_inference.yam_umi_ee_bridge import (
    YAM_JOINT_LIMITS,
    YAM_SCALAR_KEYS,
    GripperMap,
    IkResidualError,
    YamArmKinematics,
    invert_pose,
)
from lerobot.robots.bi_yam.bi_yam import BiYAMFollower
from lerobot.robots.bi_yam.config_bi_yam import BiYAMFollowerConfig, YAMArmConfig
from lerobot.utils.robot_utils import precise_sleep

DEFAULT_REPO_ID = "brandonyang/dual-lidar-umi-relative-filtered"
DEFAULT_REVISION = "e3080246342126e507d92c083e57c5d38fbfdb17"
GRIPPER_WIDTH_NORMALIZER_MM = 125.0

# dataset-filtering/src/dataset_filtering/retarget.py
# UMI world: +X up, +Y right, +Z forward.
# YAM base:  +X forward, +Y left, +Z up.
UMI_TO_YAM_BASIS = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# Exact zero-joint grasp-site pose from the kinematic model used to create the
# filtered artifact. The runtime derives the corresponding installed-URDF flange
# offset from this anchor instead of pretending the two tool frames coincide.
FILTERED_YAM_HOME_TCP = np.asarray(
    [
        [
            7.346456685676744e-06,
            -3.673205103099526e-06,
            0.9999999999662681,
            0.2449969451701382,
        ],
        [
            -6.3267679111271535e-06,
            0.9999999999732392,
            3.6732515825685255e-06,
            2.0077571273949478e-06,
        ],
        [
            -0.9999999999530003,
            -6.326794896334455e-06,
            7.346433446001038e-06,
            0.16400171938345195,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilteredEpisode:
    state: np.ndarray
    gripper_width_mm: np.ndarray
    fps: float
    filtering: dict


@dataclass(frozen=True)
class PreparedReplay:
    raw_actions: np.ndarray
    limited: RateLimitedActionChunk
    source_fps: float
    max_joint_step_rad: float
    max_joint_velocity_rad_s: float
    max_joint_acceleration_rad_s2: float
    max_position_residual_m: float
    max_orientation_residual_rad: float


def filtered_tcp_targets(state: np.ndarray) -> np.ndarray:
    """Reverse the artifact's YAM-to-UMI export into per-arm YAM-base TCP targets."""

    values = np.asarray(state, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12 or not np.isfinite(values).all():
        raise ValueError(f"expected a finite (N, 12) filtered state, got {values.shape}")

    targets = np.broadcast_to(np.eye(4), (len(values), 2, 4, 4)).copy()
    for arm, offset in ((0, 0), (1, 6)):
        positions = values[:, offset : offset + 3]
        rotations = Rotation.from_rotvec(values[:, offset + 3 : offset + 6])
        relative_positions_umi = positions - positions[0]
        targets[:, arm, :3, 3] = (
            FILTERED_YAM_HOME_TCP[:3, 3] + (UMI_TO_YAM_BASIS @ relative_positions_umi.T).T
        )

        relative_rotations_umi = rotations[0].inv() * rotations
        relative_rotations_yam = np.einsum(
            "ij,njk,kl->nil",
            UMI_TO_YAM_BASIS,
            relative_rotations_umi.as_matrix(),
            UMI_TO_YAM_BASIS.T,
        )
        targets[:, arm, :3, :3] = np.einsum(
            "ij,njk->nik",
            FILTERED_YAM_HOME_TCP[:3, :3],
            relative_rotations_yam,
        )
    return targets


def clamp_joints_to_operational_limits(joints: np.ndarray) -> np.ndarray:
    """Clamp solver roundoff at a limit while rejecting real limit violations."""

    values = np.asarray(joints, dtype=np.float64)
    if values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError(f"expected finite joint vectors ending in dimension 6, got {values.shape}")
    lower = np.asarray([limit[0] for limit in YAM_JOINT_LIMITS])
    upper = np.asarray([limit[1] for limit in YAM_JOINT_LIMITS])
    if np.any(values < lower - 1e-6) or np.any(values > upper + 1e-6):
        raise ValueError("reconstructed episode exceeds YAM operational joint limits")
    return np.clip(values, lower, upper)


def load_filtered_episode(
    repo_id: str,
    revision: str,
    episode: int,
    *,
    cache_dir: str | Path | None = None,
) -> FilteredEpisode:
    if episode < 0:
        raise ValueError("episode must be non-negative")

    download_kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "cache_dir": cache_dir,
    }
    info_path = hf_hub_download(filename="meta/info.json", **download_kwargs)
    filtering_path = hf_hub_download(filename="meta/filtering.json", **download_kwargs)
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    filtering = json.loads(Path(filtering_path).read_text(encoding="utf-8"))

    total_episodes = int(info["total_episodes"])
    if episode >= total_episodes:
        raise ValueError(f"episode {episode} is outside dataset range 0..{total_episodes - 1}")
    if "action" in info["features"] or not filtering.get("action_removed", False):
        raise ValueError("expected the observation-only relative-filtered artifact")

    episode_metadata = next(
        (row for row in filtering["optimizer_results"] if int(row["output_episode_index"]) == episode),
        None,
    )
    if episode_metadata is None:
        raise ValueError(f"filtering metadata has no output episode {episode}")
    validation = episode_metadata["retarget"]["validation"]
    if not episode_metadata["retarget"].get("continuously_replayable", False):
        raise ValueError(f"episode {episode} is not marked continuously replayable")
    required_checks = (
        "passes_joint_limits",
        "passes_velocity_limits",
        "passes_acceleration_limits",
        "passes_interarm_clearance",
        "passes_self_collision_clearance",
        "passes_tracking_accuracy",
        "passes_contact_accuracy",
    )
    failed_checks = [name for name in required_checks if not validation.get(name, False)]
    if failed_checks:
        raise ValueError(f"episode {episode} failed validation: {failed_checks}")

    data_path = hf_hub_download(
        filename=f"data/chunk-000/file-{episode:03d}.parquet",
        **download_kwargs,
    )
    table = pq.read_table(data_path)
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    gripper_width_mm = np.column_stack(
        (
            table["observation.gripper_width.umi1"].to_numpy(zero_copy_only=False),
            table["observation.gripper_width.umi2"].to_numpy(zero_copy_only=False),
        )
    ).astype(np.float64)
    if len(state) != int(episode_metadata["frames"]):
        raise ValueError(
            f"episode row count {len(state)} does not match metadata {episode_metadata['frames']}"
        )
    if gripper_width_mm.shape != (len(state), 2) or not np.isfinite(gripper_width_mm).all():
        raise ValueError(f"invalid gripper widths: {gripper_width_mm.shape}")
    return FilteredEpisode(
        state=state,
        gripper_width_mm=gripper_width_mm,
        fps=float(info["fps"]),
        filtering=episode_metadata,
    )


def reconstruct_joint_actions(
    episode: FilteredEpisode,
    *,
    urdf_path: str | Path | None = None,
    max_joint_delta: float = 0.08,
    max_gripper_delta: float = 0.05,
    max_dispatches_per_waypoint: int = 16,
) -> PreparedReplay:
    targets = filtered_tcp_targets(episode.state)
    solvers = (
        YamArmKinematics(str(resolve_yam_urdf(urdf_path)), iterations=10),
        YamArmKinematics(str(resolve_yam_urdf(urdf_path)), iterations=10),
    )
    home = np.zeros(6, dtype=np.float64)
    flange_to_filtered_tcp = invert_pose(solvers[0].fk(home)) @ FILTERED_YAM_HOME_TCP
    filtered_tcp_to_flange = invert_pose(flange_to_filtered_tcp)

    joints = np.empty((len(episode.state), 2, 6), dtype=np.float64)
    max_position_residual_m = 0.0
    max_orientation_residual_rad = 0.0
    for arm, solver in enumerate(solvers):
        seed = home.copy()
        for frame_index, target_tcp in enumerate(targets[:, arm]):
            target_flange = target_tcp @ filtered_tcp_to_flange
            try:
                solution = solver.ik(target_flange, seed, check=True)
            except IkResidualError as exc:
                raise IkResidualError(
                    f"episode frame {frame_index}/{len(targets) - 1}, arm {arm}: {exc}"
                ) from exc
            position_residual_m, orientation_residual_rad = solver.residual(solution, target_flange)
            max_position_residual_m = max(max_position_residual_m, position_residual_m)
            max_orientation_residual_rad = max(max_orientation_residual_rad, orientation_residual_rad)
            joints[frame_index, arm] = solution
            seed = solution

    if not np.allclose(joints[0], 0.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"episode does not reconstruct to the down/home start: {joints[0].tolist()}")
    joints = clamp_joints_to_operational_limits(joints)

    gripper_map = GripperMap()
    normalized_width = np.clip(
        episode.gripper_width_mm / GRIPPER_WIDTH_NORMALIZER_MM,
        0.0,
        1.0,
    )
    actions = np.empty((len(joints), len(YAM_SCALAR_KEYS)), dtype=np.float64)
    actions[:, :6] = joints[:, 0]
    actions[:, 6] = [gripper_map.umi_to_yam(value, left=True) for value in normalized_width[:, 0]]
    actions[:, 7:13] = joints[:, 1]
    actions[:, 13] = [gripper_map.umi_to_yam(value, left=False) for value in normalized_width[:, 1]]

    joint_steps = np.diff(joints, axis=0)
    joint_velocity = joint_steps * episode.fps
    joint_acceleration = np.diff(joint_velocity, axis=0) * episode.fps
    max_joint_step_rad = float(np.max(np.abs(joint_steps), initial=0.0))
    max_joint_velocity_rad_s = float(np.max(np.abs(joint_velocity), initial=0.0))
    max_joint_acceleration_rad_s2 = float(np.max(np.abs(joint_acceleration), initial=0.0))
    acceleration_limit = float(episode.filtering["retarget"]["config"]["planning_acceleration_limit_rad_s2"])
    if max_joint_velocity_rad_s > 10.0 + 1e-6:
        raise ValueError(f"reconstructed velocity {max_joint_velocity_rad_s:.3f} rad/s exceeds 10")
    if max_joint_acceleration_rad_s2 > 1.05 * acceleration_limit:
        raise ValueError(
            f"reconstructed acceleration {max_joint_acceleration_rad_s2:.3f} rad/s^2 exceeds "
            f"the {acceleration_limit:.3f} planning assumption"
        )

    limited = rate_limit_action_chunk(
        actions,
        actions[0],
        max_joint_delta=max_joint_delta,
        max_gripper_delta=max_gripper_delta,
        max_dispatches_per_waypoint=max_dispatches_per_waypoint,
    )
    return PreparedReplay(
        raw_actions=np.ascontiguousarray(actions, dtype=np.float32),
        limited=limited,
        source_fps=episode.fps,
        max_joint_step_rad=max_joint_step_rad,
        max_joint_velocity_rad_s=max_joint_velocity_rad_s,
        max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
        max_position_residual_m=max_position_residual_m,
        max_orientation_residual_rad=max_orientation_residual_rad,
    )


def replay_on_hardware(args: argparse.Namespace, replay: PreparedReplay) -> None:
    if not args.robot_id or not args.left_adapter_serial or not args.right_adapter_serial:
        raise ValueError("--execute requires --robot-id, --left-adapter-serial, and --right-adapter-serial")
    if not 0.0 < args.speed <= 1.0:
        raise ValueError("--speed must be in (0, 1] for hardware replay")

    first_action = tuple(float(value) for value in replay.raw_actions[0])
    telemetry_path = (
        Path(args.telemetry_path).expanduser()
        if args.telemetry_path
        else Path(f"replay-relative-filtered-episode-{args.episode:03d}.jsonl")
    )
    config = BiYAMFollowerConfig(
        id=args.robot_id,
        left_arm_config=YAMArmConfig(adapter_serial=args.left_adapter_serial),
        right_arm_config=YAMArmConfig(adapter_serial=args.right_adapter_serial),
        cameras={},
        max_joint_delta=args.max_joint_delta,
        max_gripper_delta=args.max_gripper_delta,
        policy_start_position=first_action,
        policy_reset_step_size=0.01,
        policy_reset_max_steps=400,
        policy_reset_timeout_s=45.0,
        control_telemetry_path=telemetry_path,
    )
    robot = BiYAMFollower(config)

    robot.connect()
    try:
        robot.arm()
        robot.reset_for_policy()

        interval_s = 1.0 / (replay.source_fps * args.speed)
        logger.info(
            "Replaying %d dispatches at %.2fx speed (estimated %.1f seconds)",
            len(replay.limited.actions),
            args.speed,
            len(replay.limited.actions) * interval_s,
        )
        for row in replay.limited.actions:
            started_at = time.perf_counter()
            action = {key: float(value) for key, value in zip(YAM_SCALAR_KEYS, row, strict=True)}
            robot.send_action(action)
            precise_sleep(max(0.0, interval_s - (time.perf_counter() - started_at)))
    finally:
        if robot.is_connected:
            robot.disarm()
            robot.disconnect()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--cache-dir")
    parser.add_argument("--urdf-path")
    parser.add_argument("--max-joint-delta", type=float, default=0.08)
    parser.add_argument("--max-gripper-delta", type=float, default=0.05)
    parser.add_argument("--max-dispatches-per-waypoint", type=int, default=16)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--robot-id")
    parser.add_argument("--left-adapter-serial")
    parser.add_argument("--right-adapter-serial")
    parser.add_argument("--telemetry-path")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    episode = load_filtered_episode(
        args.repo_id,
        args.revision,
        args.episode,
        cache_dir=args.cache_dir,
    )
    replay = reconstruct_joint_actions(
        episode,
        urdf_path=args.urdf_path,
        max_joint_delta=args.max_joint_delta,
        max_gripper_delta=args.max_gripper_delta,
        max_dispatches_per_waypoint=args.max_dispatches_per_waypoint,
    )
    estimated_duration_s = len(replay.limited.actions) / (replay.source_fps * args.speed)
    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "episode": args.episode,
        "source_frames": len(replay.raw_actions),
        "dispatches_after_rate_limit": len(replay.limited.actions),
        "source_duration_s": len(replay.raw_actions) / replay.source_fps,
        "estimated_replay_duration_s": estimated_duration_s,
        "speed": args.speed,
        "start_left_joints": replay.raw_actions[0, :6].tolist(),
        "start_right_joints": replay.raw_actions[0, 7:13].tolist(),
        "max_joint_step_rad": replay.max_joint_step_rad,
        "max_joint_velocity_rad_s": replay.max_joint_velocity_rad_s,
        "max_joint_acceleration_rad_s2": replay.max_joint_acceleration_rad_s2,
        "max_ik_position_residual_mm": replay.max_position_residual_m * 1000.0,
        "max_ik_orientation_residual_deg": np.degrees(replay.max_orientation_residual_rad),
        "metadata_minimum_interarm_clearance_mm": episode.filtering["retarget"]["validation"][
            "minimum_interarm_surface_clearance_mm"
        ],
        "metadata_minimum_self_clearance_mm": episode.filtering["retarget"]["validation"][
            "minimum_self_surface_clearance_mm"
        ],
        "table_clearance_validated": episode.filtering["retarget"]["validation"][
            "minimum_table_surface_clearance_mm"
        ]
        is not None,
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to connect to hardware.")
        return
    replay_on_hardware(args, replay)


if __name__ == "__main__":
    main()
