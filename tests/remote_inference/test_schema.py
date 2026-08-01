# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("grpc")

from lerobot.remote_inference.codec import (
    action_from_proto,
    action_to_proto,
    embodiment_from_proto,
    embodiment_to_proto,
    observation_from_proto,
    observation_to_proto,
)
from lerobot.remote_inference.schema import (
    CameraSpec,
    EmbodimentManifest,
    ImageEncoding,
    ImageFrame,
    ModelManifest,
    PolicyActionChunk,
    PolicyObservation,
    ProtocolValidationError,
)

STATE_FEATURES = tuple(f"joint_{index}.pos" for index in range(3))
CAMERA_KEYS = ("top", "left", "right")


def make_manifest(encoding: ImageEncoding = ImageEncoding.PNG) -> EmbodimentManifest:
    return EmbodimentManifest(
        schema_id="test-v1",
        robot_id="test-robot",
        robot_type="test",
        control_hz=30.0,
        state_features=STATE_FEATURES,
        action_features=STATE_FEATURES,
        cameras=tuple(CameraSpec(key, 4, 3, encoding=encoding) for key in CAMERA_KEYS),
    ).signed()


def make_model() -> ModelManifest:
    return ModelManifest(
        model_id="test/model",
        revision="abc123",
        policy_type="deterministic",
        norm_tag="",
        action_horizon=4,
        action_dim=3,
        state_features=STATE_FEATURES,
        action_features=STATE_FEATURES,
        camera_keys=CAMERA_KEYS,
        fingerprint="fingerprint",
    )


def make_observation() -> PolicyObservation:
    return PolicyObservation(
        episode_id="episode",
        sequence=7,
        capture_tick=11,
        capture_monotonic_ns=1234,
        state=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        images=tuple(
            ImageFrame(key, np.full((3, 4, 3), index, dtype=np.uint8), 1200 + index)
            for index, key in enumerate(CAMERA_KEYS)
        ),
        task="test",
        last_executed_tick=10,
        action_queue_depth=2,
    )


def test_embodiment_digest_roundtrip():
    manifest = make_manifest()
    decoded = embodiment_from_proto(embodiment_to_proto(manifest))
    assert decoded == manifest
    assert decoded.manifest_sha256 == decoded.digest()


def test_embodiment_rejects_tampered_contents():
    manifest = replace(make_manifest(), robot_id="another-robot")
    with pytest.raises(ProtocolValidationError, match="digest"):
        manifest.validate()


@pytest.mark.parametrize("encoding", list(ImageEncoding))
def test_observation_image_codec_roundtrip(encoding: ImageEncoding):
    manifest = make_manifest(encoding)
    observation = make_observation()
    message = observation_to_proto(observation, session_id="session", embodiment=manifest)
    decoded = observation_from_proto(message, embodiment=manifest, max_image_bytes=1024 * 1024)

    assert decoded.sequence == observation.sequence
    assert np.array_equal(decoded.state, observation.state)
    for original, reconstructed in zip(observation.images, decoded.images, strict=True):
        assert reconstructed.array.shape == original.array.shape
        if encoding is not ImageEncoding.JPEG:
            assert np.array_equal(reconstructed.array, original.array)


def test_observation_rejects_non_finite_state():
    manifest = make_manifest()
    observation = replace(make_observation(), state=np.asarray([0.0, np.nan, 1.0], dtype=np.float32))
    with pytest.raises(ProtocolValidationError, match="finite"):
        observation_to_proto(observation, session_id="session", embodiment=manifest)


def test_observation_rejects_oversized_encoded_image():
    manifest = make_manifest(ImageEncoding.RAW_RGB)
    message = observation_to_proto(make_observation(), session_id="session", embodiment=manifest)
    with pytest.raises(ProtocolValidationError, match="byte limit"):
        observation_from_proto(message, embodiment=manifest, max_image_bytes=4)


def test_observation_rejects_image_metadata_before_decoding():
    manifest = make_manifest(ImageEncoding.PNG)
    message = observation_to_proto(make_observation(), session_id="session", embodiment=manifest)
    message.images[0].width = 4000

    with pytest.raises(ProtocolValidationError, match="session manifest"):
        observation_from_proto(message, embodiment=manifest, max_image_bytes=1024 * 1024)


def test_observation_rejects_encoded_dimensions_that_disagree_with_metadata():
    manifest = make_manifest(ImageEncoding.PNG)
    message = observation_to_proto(make_observation(), session_id="session", embodiment=manifest)
    payload = io.BytesIO()
    Image.fromarray(np.zeros((30, 40, 3), dtype=np.uint8), mode="RGB").save(payload, format="PNG")
    message.images[0].data = payload.getvalue()

    with pytest.raises(ProtocolValidationError, match="dimensions"):
        observation_from_proto(message, embodiment=manifest, max_image_bytes=1024 * 1024)


def test_observation_rejects_oversized_decoded_image():
    manifest = make_manifest(ImageEncoding.RAW_RGB)
    message = observation_to_proto(make_observation(), session_id="session", embodiment=manifest)

    with pytest.raises(ProtocolValidationError, match="decoded byte limit"):
        observation_from_proto(
            message,
            embodiment=manifest,
            max_image_bytes=1024 * 1024,
            max_decoded_image_bytes=8,
        )


def test_action_codec_roundtrip_and_fingerprint_validation():
    model = make_model()
    action = PolicyActionChunk(
        observation_sequence=7,
        first_action_tick=11,
        actions=np.arange(12, dtype=np.float32).reshape(4, 3),
        model_fingerprint=model.fingerprint,
        server_compute_ns=42,
    )
    message = action_to_proto(action, session_id="session", model=model)
    decoded = action_from_proto(message, expected_session_id="session", model=model)
    assert np.array_equal(decoded.actions, action.actions)

    message.model_fingerprint = "wrong"
    with pytest.raises(ProtocolValidationError, match="fingerprint"):
        action_from_proto(message, expected_session_id="session", model=model)
