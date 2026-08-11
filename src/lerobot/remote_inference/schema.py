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

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

PROTOCOL_VERSION = 1


class ProtocolValidationError(ValueError):
    """Raised when a remote inference message violates the protocol contract."""


class ImageEncoding(IntEnum):
    RAW_RGB = 1
    PNG = 2
    JPEG = 3


def _require_unique(values: tuple[str, ...], field_name: str) -> None:
    if not values or any(not value for value in values):
        raise ProtocolValidationError(f"{field_name} must contain non-empty values")
    if len(set(values)) != len(values):
        raise ProtocolValidationError(f"{field_name} contains duplicate values")


@dataclass(frozen=True)
class CameraSpec:
    key: str
    width: int
    height: int
    channels: int = 3
    encoding: ImageEncoding = ImageEncoding.JPEG
    calibration_sha256: str = ""

    def validate(self) -> None:
        if not self.key:
            raise ProtocolValidationError("camera key must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ProtocolValidationError(f"camera {self.key!r} has invalid dimensions")
        if self.channels != 3:
            raise ProtocolValidationError(f"camera {self.key!r} must be RGB")
        if not isinstance(self.encoding, ImageEncoding):
            raise ProtocolValidationError(f"camera {self.key!r} has an invalid encoding")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "encoding": int(self.encoding),
            "calibration_sha256": self.calibration_sha256,
        }


@dataclass(frozen=True)
class EmbodimentManifest:
    schema_id: str
    robot_id: str
    robot_type: str
    control_hz: float
    state_features: tuple[str, ...]
    action_features: tuple[str, ...]
    cameras: tuple[CameraSpec, ...]
    protocol_version: int = PROTOCOL_VERSION
    manifest_sha256: str = ""

    def validate(self, *, verify_digest: bool = True) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ProtocolValidationError(
                f"protocol version {self.protocol_version} is unsupported; expected {PROTOCOL_VERSION}"
            )
        if not self.schema_id or not self.robot_id or not self.robot_type:
            raise ProtocolValidationError("schema_id, robot_id, and robot_type are required")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ProtocolValidationError("control_hz must be finite and positive")
        _require_unique(self.state_features, "state_features")
        _require_unique(self.action_features, "action_features")
        for camera in self.cameras:
            camera.validate()
        _require_unique(tuple(camera.key for camera in self.cameras), "camera keys")
        if verify_digest and self.manifest_sha256 != self.digest():
            raise ProtocolValidationError("embodiment manifest digest does not match its contents")

    def canonical_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "schema_id": self.schema_id,
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "control_hz": self.control_hz,
            "state_features": list(self.state_features),
            "action_features": list(self.action_features),
            "cameras": [camera.canonical_dict() for camera in self.cameras],
        }

    def digest(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def signed(self) -> EmbodimentManifest:
        return replace(self, manifest_sha256=self.digest())


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    revision: str
    policy_type: str
    norm_tag: str
    action_horizon: int
    action_dim: int
    state_features: tuple[str, ...]
    action_features: tuple[str, ...]
    camera_keys: tuple[str, ...]
    fingerprint: str

    def validate(self) -> None:
        if not self.model_id or not self.policy_type or not self.fingerprint:
            raise ProtocolValidationError("model id, policy type, and fingerprint are required")
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ProtocolValidationError("model action dimensions must be positive")
        _require_unique(self.state_features, "model state_features")
        _require_unique(self.action_features, "model action_features")
        _require_unique(self.camera_keys, "model camera_keys")
        if len(self.action_features) != self.action_dim:
            raise ProtocolValidationError("model action feature count does not match action_dim")

    def assert_compatible(self, embodiment: EmbodimentManifest) -> None:
        embodiment.validate()
        self.validate()
        if embodiment.state_features != self.state_features:
            raise ProtocolValidationError("robot state feature order does not match the model")
        if embodiment.action_features != self.action_features:
            raise ProtocolValidationError("robot action feature order does not match the model")
        if tuple(camera.key for camera in embodiment.cameras) != self.camera_keys:
            raise ProtocolValidationError("robot camera order does not match the model")


@dataclass(frozen=True)
class ImageFrame:
    key: str
    array: np.ndarray
    capture_monotonic_ns: int

    def validate(self, expected: CameraSpec | None = None) -> None:
        if not self.key:
            raise ProtocolValidationError("image key must not be empty")
        if self.capture_monotonic_ns < 0:
            raise ProtocolValidationError("image capture timestamp must be non-negative")
        if self.array.dtype != np.uint8 or self.array.ndim != 3 or self.array.shape[2] != 3:
            raise ProtocolValidationError(f"image {self.key!r} must be an HWC uint8 RGB array")
        if expected is not None:
            shape = (expected.height, expected.width, expected.channels)
            if self.key != expected.key or self.array.shape != shape:
                raise ProtocolValidationError(
                    f"image {self.key!r} shape {self.array.shape} does not match expected {shape}"
                )


@dataclass(frozen=True)
class PolicyObservation:
    episode_id: str
    sequence: int
    capture_tick: int
    capture_monotonic_ns: int
    state: np.ndarray
    images: tuple[ImageFrame, ...]
    task: str
    last_executed_tick: int
    action_queue_depth: int

    def validate(self, embodiment: EmbodimentManifest) -> None:
        if self.sequence < 0 or self.capture_tick < 0 or self.capture_monotonic_ns < 0:
            raise ProtocolValidationError("observation sequence and timestamps must be non-negative")
        if self.last_executed_tick < 0 or self.action_queue_depth < 0:
            raise ProtocolValidationError("action execution metadata must be non-negative")
        if self.state.shape != (len(embodiment.state_features),):
            raise ProtocolValidationError("observation state has the wrong shape")
        if self.state.dtype != np.float32 or not np.isfinite(self.state).all():
            raise ProtocolValidationError("observation state must be finite float32")
        if len(self.images) != len(embodiment.cameras):
            raise ProtocolValidationError("observation has the wrong number of camera frames")
        for image, camera in zip(self.images, embodiment.cameras, strict=True):
            image.validate(camera)


@dataclass(frozen=True)
class PolicyActionChunk:
    observation_sequence: int
    first_action_tick: int
    actions: np.ndarray
    model_fingerprint: str
    server_compute_ns: int = 0

    def validate(self, model: ModelManifest) -> None:
        expected = (model.action_horizon, model.action_dim)
        if self.actions.shape != expected:
            raise ProtocolValidationError(
                f"action chunk shape {self.actions.shape} does not match {expected}"
            )
        if self.actions.dtype != np.float32 or not np.isfinite(self.actions).all():
            raise ProtocolValidationError("action chunk must contain finite float32 values")
        if self.observation_sequence < 0 or self.first_action_tick < 0 or self.server_compute_ns < 0:
            raise ProtocolValidationError("action chunk metadata must be non-negative")
        if self.model_fingerprint != model.fingerprint:
            raise ProtocolValidationError("action chunk model fingerprint does not match the session")
