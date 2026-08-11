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

"""Capture cameras, run one remote inference, and permanently discard its actions.

This diagnostic intentionally has no robot construction or control dependency. State
must be supplied explicitly so a probe can never enable motors merely to obtain it.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import draccus
import numpy as np
from PIL import Image

from lerobot.cameras import Camera, CameraConfig, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.remote_inference import (
    CameraSpec,
    EmbodimentManifest,
    ImageEncoding,
    ImageFrame,
    PolicyObservation,
    RemotePolicyClient,
    RemotePolicyClientConfig,
)

logger = logging.getLogger(__name__)

YAM_FEATURES = (
    *(f"left_joint_{index}.pos" for index in range(6)),
    "left_gripper.pos",
    *(f"right_joint_{index}.pos" for index in range(6)),
    "right_gripper.pos",
)

YAM_CAMERA_KEYS = ("top", "left", "right")


@dataclass
class PolicyProbeConfig:
    server_address: str = "127.0.0.1:8081"
    robot_id: str = ""
    robot_type: str = "bi_yam_follower"
    task: str = ""
    state: list[float] = field(default_factory=list)
    state_source: str = ""
    output_dir: Path = Path("outputs/policy_probe")
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    schema_id: str = "molmoact2-bimanual-yam-observation-probe-v1"
    connect_timeout_s: float = 180.0
    inference_timeout_s: float = 180.0
    max_message_bytes: int = 16 * 1024 * 1024
    jpeg_quality: int = 95

    def validate(self) -> None:
        if not self.server_address.strip():
            raise ValueError("server_address must not be empty")
        if not self.robot_id.strip():
            raise ValueError("robot_id must not be empty")
        if not self.robot_type.strip():
            raise ValueError("robot_type must not be empty")
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if len(self.state) != len(YAM_FEATURES):
            raise ValueError(f"state must contain exactly {len(YAM_FEATURES)} values")
        if not np.isfinite(np.asarray(self.state, dtype=np.float64)).all():
            raise ValueError("state values must be finite")
        if not self.state_source.strip():
            raise ValueError("state_source must describe where the supplied state came from")
        camera_keys = tuple(self.cameras)
        if not camera_keys:
            raise ValueError("at least one camera must be configured")
        unknown_camera_keys = tuple(key for key in camera_keys if key not in YAM_CAMERA_KEYS)
        if unknown_camera_keys:
            raise ValueError(f"unsupported YAM camera keys: {unknown_camera_keys}")
        canonical_camera_keys = tuple(key for key in YAM_CAMERA_KEYS if key in self.cameras)
        if camera_keys != canonical_camera_keys:
            raise ValueError("cameras must follow the canonical order: top, left, right")
        if self.connect_timeout_s <= 0 or self.inference_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_message_bytes <= 0 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("message and JPEG limits are invalid")


@dataclass(frozen=True)
class PolicyProbeResult:
    evidence_dir: Path
    model_id: str
    policy_type: str
    action_shape: tuple[int, int]
    round_trip_ms: float
    server_compute_ms: float
    action_min: float
    action_max: float
    actions_discarded: bool = True


CameraFactory = Callable[[dict[str, CameraConfig]], dict[str, Camera]]
ClientFactory = Callable[[RemotePolicyClientConfig], RemotePolicyClient]


def _capture_frames(
    configs: dict[str, CameraConfig],
    *,
    camera_factory: CameraFactory,
) -> dict[str, np.ndarray]:
    cameras = camera_factory(configs)
    connected: list[Camera] = []
    try:
        for camera in cameras.values():
            camera.connect()
            connected.append(camera)
        frames = {
            name: np.asarray(camera.async_read(), dtype=np.uint8).copy() for name, camera in cameras.items()
        }
    finally:
        for camera in reversed(connected):
            try:
                camera.disconnect()
            except Exception:
                logger.exception("Failed to disconnect diagnostic camera")
    return frames


def _manifest(cfg: PolicyProbeConfig) -> EmbodimentManifest:
    cameras = tuple(
        CameraSpec(
            key=name,
            width=int(camera.width),
            height=int(camera.height),
            channels=3,
            encoding=ImageEncoding.JPEG,
        )
        for name, camera in cfg.cameras.items()
    )
    return EmbodimentManifest(
        schema_id=cfg.schema_id,
        robot_id=cfg.robot_id,
        robot_type=cfg.robot_type,
        control_hz=30.0,
        state_features=YAM_FEATURES,
        action_features=YAM_FEATURES,
        cameras=cameras,
    ).signed()


def _new_evidence_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = root / stamp
    suffix = 1
    while path.exists():
        path = root / f"{stamp}-{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def _save_frames(frames: dict[str, np.ndarray], evidence_dir: Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    for name, frame in frames.items():
        path = evidence_dir / f"{name}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        metadata[name] = {
            "path": path.name,
            "shape": list(frame.shape),
            "sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "mean": float(frame.mean()),
            "min": int(frame.min()),
            "max": int(frame.max()),
        }
    return metadata


def run_policy_probe(
    cfg: PolicyProbeConfig,
    *,
    camera_factory: CameraFactory = make_cameras_from_configs,
    client_factory: ClientFactory = RemotePolicyClient,
) -> PolicyProbeResult:
    """Run one observation-only request; the returned action array never leaves this function."""

    cfg.validate()
    evidence_dir = _new_evidence_dir(cfg.output_dir)
    frames = _capture_frames(cfg.cameras, camera_factory=camera_factory)
    frame_metadata = _save_frames(frames, evidence_dir)
    now_ns = time.monotonic_ns()
    embodiment = _manifest(cfg)
    observation = PolicyObservation(
        episode_id="observation-only-probe",
        sequence=0,
        capture_tick=0,
        capture_monotonic_ns=now_ns,
        state=np.asarray(cfg.state, dtype=np.float32),
        images=tuple(
            ImageFrame(key=name, array=frames[name], capture_monotonic_ns=now_ns) for name in cfg.cameras
        ),
        task=cfg.task,
        last_executed_tick=0,
        action_queue_depth=0,
    )
    observation.validate(embodiment)

    client_config = RemotePolicyClientConfig(
        server_address=cfg.server_address,
        connect_timeout_s=cfg.connect_timeout_s,
        inference_timeout_s=cfg.inference_timeout_s,
        max_message_bytes=cfg.max_message_bytes,
        jpeg_quality=cfg.jpeg_quality,
    )
    with client_factory(client_config) as client:
        session = client.connect(embodiment, task=cfg.task, client_instance_id="observation-only-probe")
        started_ns = time.perf_counter_ns()
        chunk = client.infer(observation)
        round_trip_ns = time.perf_counter_ns() - started_ns

    action_shape = tuple(int(value) for value in chunk.actions.shape)
    action_min = float(chunk.actions.min())
    action_max = float(chunk.actions.max())
    server_compute_ns = int(chunk.server_compute_ns)
    result = PolicyProbeResult(
        evidence_dir=evidence_dir,
        model_id=session.model.model_id,
        policy_type=session.model.policy_type,
        action_shape=action_shape,
        round_trip_ms=round_trip_ns / 1e6,
        server_compute_ms=server_compute_ns / 1e6,
        action_min=action_min,
        action_max=action_max,
    )

    report = {
        "probe": {
            "server_address": cfg.server_address,
            "robot_id": cfg.robot_id,
            "robot_type": cfg.robot_type,
            "task": cfg.task,
            "state_source": cfg.state_source,
            "state_features": list(YAM_FEATURES),
            "state": [float(value) for value in cfg.state],
        },
        "frames": frame_metadata,
        "result": {**asdict(result), "evidence_dir": str(result.evidence_dir)},
    }
    (evidence_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # The process retains only scalar diagnostics; no action vector is returned or persisted.
    del chunk
    logger.info(
        "Observation-only inference succeeded: model=%s shape=%s round_trip=%.1fms; actions discarded",
        result.model_id,
        result.action_shape,
        result.round_trip_ms,
    )
    return result


@draccus.wrap()
def policy_probe(cfg: PolicyProbeConfig) -> None:
    result = run_policy_probe(cfg)
    print(json.dumps({**asdict(result), "evidence_dir": str(result.evidence_dir)}, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    policy_probe()


if __name__ == "__main__":
    main()
