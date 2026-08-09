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
"""Reference remote-inference client for bimanual UMI end-effector-pose policies.

The stock ``lerobot-rollout`` embodiment builder only advertises robot features named
``*.pos``/``*.vel``, so it can never open a session against a policy whose manifest
uses bare end-effector feature names (``umi1_x`` ... ``umi2_gripper``). This module is
the supported path for such policies: a thin client that owns the embodiment manifest
and documents the state/action conventions the policy was trained with.

Conventions (must match the fine-tune dataset exactly):

- ``state``: float32 (14,) = ``[umi1 xyz, umi1 rotvec, umi1_gripper, umi2 xyz,
  umi2 rotvec, umi2_gripper]``. Each arm's pose is expressed in that arm's
  EPISODE-START frame (frame 0 = identity): ``p_t = R0^T (p_world_t - p_world_0)``,
  ``r_t = rotvec(R0^T R_world_t)``. Grippers are ``width_mm / gripper_scale_mm``
  clipped to [0, 1] (open is ~1.0).
- returned actions: float32 (horizon, 14) ABSOLUTE next-frame targets in the same
  frames — NOT deltas. Execute at ``control_hz``; re-query every ``execution_horizon``
  steps. The caller must rebase the episode-start frames whenever ``reset()`` is
  called (new episode = new frame anchors).
- images: HWC uint8 RGB at the trained resolution, in manifest camera order
  (umi1 = LEFT gripper first).

Run this module directly to validate a served checkpoint by replaying dataset
observations through the wire and scoring returned chunks against ground truth:

    python -m lerobot.remote_inference.umi_ee_client \
        --server 127.0.0.1:8081 \
        --data_root /path/to/converted-14d-dataset \
        --repo_id user/dual-lidar-umi \
        --episodes 3 17 41 --stride 60 \
        --task "Put all oranges in the bowl"

fps, camera order/resolution, and feature names all come from the dataset metadata,
so the harness works for any dataset whose layout matches the served policy.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from .client import RemotePolicyClient, RemotePolicyClientConfig
from .schema import CameraSpec, EmbodimentManifest, ImageEncoding, ImageFrame, PolicyObservation

UMI_EE_FEATURE_NAMES: tuple[str, ...] = (
    "umi1_x",
    "umi1_y",
    "umi1_z",
    "umi1_rx",
    "umi1_ry",
    "umi1_rz",
    "umi1_gripper",
    "umi2_x",
    "umi2_y",
    "umi2_z",
    "umi2_rx",
    "umi2_ry",
    "umi2_rz",
    "umi2_gripper",
)


@dataclass
class UmiEeClientConfig:
    server_address: str
    task: str
    camera_keys: tuple[str, ...] = ("umi1", "umi2")
    image_width: int = 800
    image_height: int = 600
    control_hz: float = 30.0
    feature_names: tuple[str, ...] = UMI_EE_FEATURE_NAMES
    robot_id: str = "umi-rig"
    robot_type: str = "bimanual_umi_ee"
    # a 5B-class VLA needs far more than the transport default of 2 s
    inference_timeout_s: float = 20.0
    connect_timeout_s: float = 30.0


class UmiEeRemoteClient:
    """Session wrapper for bimanual-UMI EE-pose policies served by lerobot-policy-server."""

    def __init__(self, config: UmiEeClientConfig):
        self._config = config
        self._manifest = EmbodimentManifest(
            schema_id="lerobot-remote-v1",
            robot_id=config.robot_id,
            robot_type=config.robot_type,
            control_hz=config.control_hz,
            state_features=tuple(config.feature_names),
            action_features=tuple(config.feature_names),
            cameras=tuple(
                CameraSpec(
                    key=key,
                    width=config.image_width,
                    height=config.image_height,
                    encoding=ImageEncoding.JPEG,
                )
                for key in config.camera_keys
            ),
        ).signed()
        self._client = RemotePolicyClient(
            RemotePolicyClientConfig(
                server_address=config.server_address,
                connect_timeout_s=config.connect_timeout_s,
                inference_timeout_s=config.inference_timeout_s,
            )
        )
        self._episode_id = uuid.uuid4().hex
        self._sequence = 0

    @property
    def model_manifest(self):
        return self._session.model

    def connect(self):
        self._session = self._client.connect(self._manifest, task=self._config.task)
        return self._session

    def predict(self, state: np.ndarray, images: dict[str, np.ndarray], tick: int) -> np.ndarray:
        """One inference round trip. Returns the (horizon, 14) absolute-target chunk."""
        frames = tuple(
            ImageFrame(key=key, array=images[key], capture_monotonic_ns=time.monotonic_ns())
            for key in self._config.camera_keys
        )
        observation = PolicyObservation(
            episode_id=self._episode_id,
            sequence=self._sequence,
            capture_tick=tick,
            capture_monotonic_ns=time.monotonic_ns(),
            state=np.ascontiguousarray(state, dtype=np.float32),
            images=frames,
            task=self._config.task,
            last_executed_tick=max(tick - 1, 0),
            action_queue_depth=0,
        )
        chunk = self._client.infer(observation)
        self._sequence += 1
        return chunk.actions

    def reset(self) -> None:
        """New episode: the caller must rebase episode-start pose frames alongside this."""
        self._client.reset()
        self._episode_id = uuid.uuid4().hex
        self._sequence = 0

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------------------
# Replay validation harness (python -m lerobot.remote_inference.umi_ee_client)
# --------------------------------------------------------------------------------------


@dataclass
class _ReplayArgs:
    server: str
    data_root: str
    repo_id: str
    task: str
    episodes: list[int] = field(default_factory=lambda: [3, 17, 41])
    stride: int = 60
    max_obs_per_episode: int = 8
    gripper_dims: tuple[int, ...] = (6, 13)
    out: str | None = None


def _to_hwc_uint8(image) -> np.ndarray:
    """LeRobotDataset yields CHW float32 in [0, 1]; the wire protocol wants HWC uint8 RGB."""
    array = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    return np.ascontiguousarray(array)


def _replay(args: _ReplayArgs) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    # dataset metadata is the source of truth for fps, camera order, and resolution
    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=args.data_root)
    image_keys = [key for key in meta.features if key.startswith("observation.images.")]
    if not image_keys:
        raise RuntimeError(f"{args.repo_id} declares no camera features")
    camera_keys = tuple(key.removeprefix("observation.images.") for key in image_keys)
    height, width = (int(v) for v in meta.features[image_keys[0]]["shape"][:2])
    state_names = tuple(meta.features["observation.state"]["names"])

    client = UmiEeRemoteClient(
        UmiEeClientConfig(
            server_address=args.server,
            task=args.task,
            camera_keys=camera_keys,
            image_width=width,
            image_height=height,
            control_hz=float(meta.fps),
            feature_names=state_names,
        )
    )
    session = client.connect()
    model = session.model
    horizon = model.action_horizon
    action_dim = model.action_dim
    print(
        f"session open: model={model.model_id} horizon={horizon} dim={action_dim} "
        f"cameras={model.camera_keys} fps={meta.fps} image={width}x{height}"
    )

    pose_dims = [d for d in range(action_dim) if d not in args.gripper_dims]
    per_obs = []
    try:
        for episode in args.episodes:
            # go through the dataset rather than guessing file paths: data files are
            # size-based (an episode is not file-<episode>), may hold several episodes,
            # and concatenated videos need per-episode timestamp offsets
            dataset = LeRobotDataset(
                repo_id=args.repo_id,
                root=args.data_root,
                episodes=[episode],
                delta_timestamps={"action": [i / meta.fps for i in range(horizon)]},
                image_transforms=None,  # validation must not augment
                video_backend="pyav",
            )
            indices = list(range(0, len(dataset), args.stride))[: args.max_obs_per_episode]
            client.reset()
            for index in indices:
                item = dataset[index]
                state = item["observation.state"].numpy().astype(np.float32)
                target = item["action"].numpy().astype(np.float32)
                images = {
                    key: _to_hwc_uint8(item[image_key])
                    for key, image_key in zip(camera_keys, image_keys, strict=True)
                }
                started = time.perf_counter()
                chunk = client.predict(state, images, index)
                latency_s = time.perf_counter() - started
                hold = np.repeat(state[None, :], horizon, axis=0)
                pad = item.get("action_is_pad")
                record = {
                    "episode": episode,
                    "index": index,
                    "latency_s": round(latency_s, 3),
                    "padded_rows": int(pad.sum()) if pad is not None else 0,
                    "l1_pose": float(np.abs(chunk[:, pose_dims] - target[:, pose_dims]).mean()),
                    "l1_gripper": float(
                        np.abs(chunk[:, list(args.gripper_dims)] - target[:, list(args.gripper_dims)]).mean()
                    ),
                    "hold_l1_pose": float(np.abs(hold[:, pose_dims] - target[:, pose_dims]).mean()),
                    "gripper_min": float(chunk[:, list(args.gripper_dims)].min()),
                    "gripper_max": float(chunk[:, list(args.gripper_dims)].max()),
                    "finite": bool(np.isfinite(chunk).all()),
                }
                per_obs.append(record)
                print(record)
    finally:
        # the server allows one active session; leaking it blocks retries until the
        # idle timeout expires
        client.close()

    if not per_obs:
        raise RuntimeError("replay produced no observations")
    l1_pose = float(np.mean([r["l1_pose"] for r in per_obs]))
    l1_grip = float(np.mean([r["l1_gripper"] for r in per_obs]))
    hold_pose = float(np.mean([r["hold_l1_pose"] for r in per_obs]))
    gr_min = min(r["gripper_min"] for r in per_obs)
    gr_max = max(r["gripper_max"] for r in per_obs)
    structural_ok = all(r["finite"] for r in per_obs) and gr_min >= -0.05 and gr_max <= 1.05
    summary = {
        "n_obs": len(per_obs),
        "l1_pose": l1_pose,
        "l1_gripper": l1_grip,
        "hold_baseline_l1_pose": hold_pose,
        "beats_hold_baseline": l1_pose < hold_pose,
        "gripper_range": [gr_min, gr_max],
        "structural_ok": structural_ok,
        "per_obs": per_obs,
    }
    print(
        f"\nSUMMARY n={len(per_obs)} l1_pose={l1_pose:.4f} (hold {hold_pose:.4f}) "
        f"l1_gripper={l1_grip:.4f} gripper_range=[{gr_min:.3f},{gr_max:.3f}]"
    )
    print("WIRE_TEST_OK" if structural_ok else "WIRE_TEST_FAILED")
    return summary


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--repo_id", required=True, help="repo id of the dataset at --data_root")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[3, 17, 41])
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max_obs_per_episode", type=int, default=8)
    parser.add_argument("--out", default=None)
    ns = parser.parse_args()
    summary = _replay(
        _ReplayArgs(
            server=ns.server,
            data_root=ns.data_root,
            repo_id=ns.repo_id,
            task=ns.task,
            episodes=ns.episodes,
            stride=ns.stride,
            max_obs_per_episode=ns.max_obs_per_episode,
            out=ns.out,
        )
    )
    if ns.out:
        with open(ns.out, "w") as f:
            json.dump(summary, f, indent=2)
    raise SystemExit(0 if summary["structural_ok"] else 1)


if __name__ == "__main__":
    main()
