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

import io

import numpy as np
from PIL import Image

from . import remote_policy_pb2
from .schema import (
    CameraSpec,
    EmbodimentManifest,
    ImageEncoding,
    ImageFrame,
    ModelManifest,
    PolicyActionChunk,
    PolicyObservation,
    ProtocolValidationError,
)


def embodiment_to_proto(manifest: EmbodimentManifest) -> remote_policy_pb2.EmbodimentManifest:
    manifest.validate()
    return remote_policy_pb2.EmbodimentManifest(
        protocol_version=manifest.protocol_version,
        schema_id=manifest.schema_id,
        robot_id=manifest.robot_id,
        robot_type=manifest.robot_type,
        control_hz=manifest.control_hz,
        state_features=manifest.state_features,
        action_features=manifest.action_features,
        cameras=[
            remote_policy_pb2.CameraSpec(
                key=camera.key,
                width=camera.width,
                height=camera.height,
                channels=camera.channels,
                encoding=int(camera.encoding),
                calibration_sha256=camera.calibration_sha256,
            )
            for camera in manifest.cameras
        ],
        manifest_sha256=manifest.manifest_sha256,
    )


def embodiment_from_proto(message: remote_policy_pb2.EmbodimentManifest) -> EmbodimentManifest:
    try:
        cameras = tuple(
            CameraSpec(
                key=camera.key,
                width=camera.width,
                height=camera.height,
                channels=camera.channels,
                encoding=ImageEncoding(camera.encoding),
                calibration_sha256=camera.calibration_sha256,
            )
            for camera in message.cameras
        )
    except ValueError as exc:
        raise ProtocolValidationError("manifest contains an unknown image encoding") from exc
    manifest = EmbodimentManifest(
        protocol_version=message.protocol_version,
        schema_id=message.schema_id,
        robot_id=message.robot_id,
        robot_type=message.robot_type,
        control_hz=message.control_hz,
        state_features=tuple(message.state_features),
        action_features=tuple(message.action_features),
        cameras=cameras,
        manifest_sha256=message.manifest_sha256,
    )
    manifest.validate()
    return manifest


def model_to_proto(model: ModelManifest) -> remote_policy_pb2.ModelManifest:
    model.validate()
    return remote_policy_pb2.ModelManifest(
        model_id=model.model_id,
        revision=model.revision,
        policy_type=model.policy_type,
        norm_tag=model.norm_tag,
        action_horizon=model.action_horizon,
        action_dim=model.action_dim,
        state_features=model.state_features,
        action_features=model.action_features,
        camera_keys=model.camera_keys,
        fingerprint=model.fingerprint,
    )


def model_from_proto(message: remote_policy_pb2.ModelManifest) -> ModelManifest:
    model = ModelManifest(
        model_id=message.model_id,
        revision=message.revision,
        policy_type=message.policy_type,
        norm_tag=message.norm_tag,
        action_horizon=message.action_horizon,
        action_dim=message.action_dim,
        state_features=tuple(message.state_features),
        action_features=tuple(message.action_features),
        camera_keys=tuple(message.camera_keys),
        fingerprint=message.fingerprint,
    )
    model.validate()
    return model


def _encode_image(array: np.ndarray, encoding: ImageEncoding, jpeg_quality: int) -> bytes:
    if encoding is ImageEncoding.RAW_RGB:
        return array.tobytes(order="C")
    output = io.BytesIO()
    image = Image.fromarray(array, mode="RGB")
    if encoding is ImageEncoding.PNG:
        image.save(output, format="PNG")
    elif encoding is ImageEncoding.JPEG:
        image.save(output, format="JPEG", quality=jpeg_quality, subsampling=0)
    else:
        raise ProtocolValidationError(f"unsupported image encoding: {encoding}")
    return output.getvalue()


def _decode_image(
    message: remote_policy_pb2.ImageFrame,
    max_image_bytes: int,
    max_decoded_image_bytes: int,
    expected: CameraSpec,
) -> np.ndarray:
    if len(message.data) > max_image_bytes:
        raise ProtocolValidationError(f"image {message.key!r} exceeds the configured byte limit")
    try:
        encoding = ImageEncoding(message.encoding)
    except ValueError as exc:
        raise ProtocolValidationError("image uses an unknown encoding") from exc
    shape = (message.height, message.width, message.channels)
    if message.channels != 3 or message.width <= 0 or message.height <= 0:
        raise ProtocolValidationError(f"image {message.key!r} has invalid dimensions")
    if (
        message.key != expected.key
        or message.width != expected.width
        or message.height != expected.height
        or message.channels != expected.channels
        or encoding is not expected.encoding
    ):
        raise ProtocolValidationError(f"image {message.key!r} metadata does not match the session manifest")
    decoded_bytes = message.width * message.height * message.channels
    if decoded_bytes > max_decoded_image_bytes:
        raise ProtocolValidationError(f"image {message.key!r} exceeds the decoded byte limit")
    if encoding is ImageEncoding.RAW_RGB:
        expected_bytes = int(np.prod(shape))
        if len(message.data) != expected_bytes:
            raise ProtocolValidationError(f"raw image {message.key!r} has the wrong byte length")
        return np.frombuffer(message.data, dtype=np.uint8).reshape(shape).copy()
    try:
        with Image.open(io.BytesIO(message.data)) as image:
            if image.size != (message.width, message.height):
                raise ProtocolValidationError(
                    f"encoded image {message.key!r} dimensions do not match its metadata"
                )
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ProtocolValidationError:
        raise
    except Exception as exc:
        raise ProtocolValidationError(f"image {message.key!r} could not be decoded") from exc
    if array.shape != shape:
        raise ProtocolValidationError(
            f"decoded image {message.key!r} shape {array.shape} does not match declared {shape}"
        )
    return array


def observation_to_proto(
    observation: PolicyObservation,
    *,
    session_id: str,
    embodiment: EmbodimentManifest,
    jpeg_quality: int = 95,
) -> remote_policy_pb2.InferRequest:
    observation.validate(embodiment)
    if not 1 <= jpeg_quality <= 100:
        raise ProtocolValidationError("jpeg_quality must be between 1 and 100")
    images = []
    for frame, camera in zip(observation.images, embodiment.cameras, strict=True):
        images.append(
            remote_policy_pb2.ImageFrame(
                key=frame.key,
                width=camera.width,
                height=camera.height,
                channels=camera.channels,
                encoding=int(camera.encoding),
                data=_encode_image(frame.array, camera.encoding, jpeg_quality),
                capture_monotonic_ns=frame.capture_monotonic_ns,
            )
        )
    return remote_policy_pb2.InferRequest(
        session_id=session_id,
        episode_id=observation.episode_id,
        observation_sequence=observation.sequence,
        capture_tick=observation.capture_tick,
        capture_monotonic_ns=observation.capture_monotonic_ns,
        state=observation.state,
        images=images,
        task=observation.task,
        last_executed_tick=observation.last_executed_tick,
        action_queue_depth=observation.action_queue_depth,
    )


def observation_from_proto(
    message: remote_policy_pb2.InferRequest,
    *,
    embodiment: EmbodimentManifest,
    max_image_bytes: int,
    max_decoded_image_bytes: int = 64 * 1024 * 1024,
) -> PolicyObservation:
    if len(message.images) != len(embodiment.cameras):
        raise ProtocolValidationError("observation has the wrong number of camera frames")
    observation = PolicyObservation(
        episode_id=message.episode_id,
        sequence=message.observation_sequence,
        capture_tick=message.capture_tick,
        capture_monotonic_ns=message.capture_monotonic_ns,
        state=np.asarray(message.state, dtype=np.float32),
        images=tuple(
            ImageFrame(
                key=image.key,
                array=_decode_image(image, max_image_bytes, max_decoded_image_bytes, camera),
                capture_monotonic_ns=image.capture_monotonic_ns,
            )
            for image, camera in zip(message.images, embodiment.cameras, strict=True)
        ),
        task=message.task,
        last_executed_tick=message.last_executed_tick,
        action_queue_depth=message.action_queue_depth,
    )
    observation.validate(embodiment)
    return observation


def action_to_proto(
    action: PolicyActionChunk,
    *,
    session_id: str,
    model: ModelManifest,
) -> remote_policy_pb2.ActionChunk:
    action.validate(model)
    return remote_policy_pb2.ActionChunk(
        session_id=session_id,
        observation_sequence=action.observation_sequence,
        first_action_tick=action.first_action_tick,
        action_horizon=model.action_horizon,
        action_dim=model.action_dim,
        actions=action.actions.reshape(-1),
        model_fingerprint=action.model_fingerprint,
        server_compute_ns=action.server_compute_ns,
    )


def action_from_proto(
    message: remote_policy_pb2.ActionChunk,
    *,
    expected_session_id: str,
    model: ModelManifest,
) -> PolicyActionChunk:
    if message.session_id != expected_session_id:
        raise ProtocolValidationError("action chunk belongs to a different session")
    if message.action_horizon != model.action_horizon or message.action_dim != model.action_dim:
        raise ProtocolValidationError("action chunk dimensions do not match the model manifest")
    action = PolicyActionChunk(
        observation_sequence=message.observation_sequence,
        first_action_tick=message.first_action_tick,
        actions=np.asarray(message.actions, dtype=np.float32).reshape(model.action_horizon, model.action_dim),
        model_fingerprint=message.model_fingerprint,
        server_compute_ns=message.server_compute_ns,
    )
    action.validate(model)
    return action
