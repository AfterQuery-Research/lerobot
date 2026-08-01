# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

from __future__ import annotations

import shutil
import socket
import subprocess
import time

import numpy as np
import pytest

from lerobot.remote_inference.backend import (
    DeterministicPolicyBackend,
    DeterministicPolicyBackendConfig,
)
from lerobot.remote_inference.client import (
    RemotePolicyClient,
    RemotePolicyClientConfig,
    RemotePolicyError,
)
from lerobot.remote_inference.schema import (
    CameraSpec,
    EmbodimentManifest,
    ImageEncoding,
    ImageFrame,
    PolicyObservation,
)
from lerobot.remote_inference.server import RemotePolicyServerConfig, create_grpc_server

STATE_FEATURES = tuple(f"joint_{index}.pos" for index in range(3))
CAMERA_KEYS = ("top", "left", "right")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_manifest(camera_keys: tuple[str, ...] = CAMERA_KEYS) -> EmbodimentManifest:
    return EmbodimentManifest(
        schema_id="test-v1",
        robot_id="test-robot",
        robot_type="test",
        control_hz=30.0,
        state_features=STATE_FEATURES,
        action_features=STATE_FEATURES,
        cameras=tuple(CameraSpec(key, width=4, height=3, encoding=ImageEncoding.PNG) for key in camera_keys),
    ).signed()


def make_observation(sequence: int = 1, task: str = "test task") -> PolicyObservation:
    return PolicyObservation(
        episode_id="episode",
        sequence=sequence,
        capture_tick=sequence,
        capture_monotonic_ns=sequence * 100,
        state=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        images=tuple(
            ImageFrame(key, np.full((3, 4, 3), index, dtype=np.uint8), sequence * 100)
            for index, key in enumerate(CAMERA_KEYS)
        ),
        task=task,
        last_executed_tick=max(0, sequence - 1),
        action_queue_depth=0,
    )


@pytest.fixture
def running_server():
    port = free_port()
    backend = DeterministicPolicyBackend(
        DeterministicPolicyBackendConfig(
            action_horizon=4,
            state_features=STATE_FEATURES,
            action_features=STATE_FEATURES,
            camera_keys=CAMERA_KEYS,
        )
    )
    config = RemotePolicyServerConfig(port=port, auth_token="secret", session_idle_timeout_s=10)
    server, _ = create_grpc_server(config, backend)
    server.start()
    try:
        yield f"127.0.0.1:{port}"
    finally:
        server.stop(grace=0).wait()


def make_client(address: str, token: str = "secret") -> RemotePolicyClient:
    return RemotePolicyClient(
        RemotePolicyClientConfig(
            server_address=address,
            auth_token=token,
            connect_timeout_s=2,
            inference_timeout_s=2,
        )
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tls_client_cert_path": "client.pem", "tls_client_key_path": "client.key"},
        {"tls_server_name_override": "policy.internal"},
    ],
)
def test_client_rejects_tls_options_without_root_certificate(kwargs):
    with pytest.raises(ValueError, match="TLS root certificate"):
        RemotePolicyClient(RemotePolicyClientConfig(server_address="localhost:1", **kwargs))


def test_server_rejects_client_ca_without_tls_identity():
    with pytest.raises(ValueError, match="server certificate"):
        RemotePolicyServerConfig(tls_client_ca_path="ca.pem").validate()


def test_end_to_end_session_and_inference(running_server):
    client = make_client(running_server)
    session = client.connect(make_manifest(), task="test task", client_instance_id="client")
    action = client.infer(make_observation())

    assert session.model.action_horizon == 4
    assert action.actions.shape == (4, 3)
    assert np.allclose(action.actions, [[0.1, 0.2, 0.3]] * 4)
    assert action.observation_sequence == 1
    client.reset()
    client.close()


def test_authentication_is_required(running_server):
    client = make_client(running_server, token="wrong")
    with pytest.raises(RemotePolicyError, match="UNAUTHENTICATED"):
        client.connect(make_manifest())
    client.close()


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl is required for TLS integration")
def test_tls_session_and_inference(tmp_path):
    certificate = tmp_path / "server.pem"
    private_key = tmp_path / "server.key"
    subprocess.run(
        [
            shutil.which("openssl"),
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    backend = DeterministicPolicyBackend(
        DeterministicPolicyBackendConfig(
            action_horizon=4,
            state_features=STATE_FEATURES,
            action_features=STATE_FEATURES,
            camera_keys=CAMERA_KEYS,
        )
    )
    port = free_port()
    server, _ = create_grpc_server(
        RemotePolicyServerConfig(
            port=port,
            auth_token="secret",
            tls_cert_path=str(certificate),
            tls_key_path=str(private_key),
        ),
        backend,
    )
    server.start()
    client = RemotePolicyClient(
        RemotePolicyClientConfig(
            server_address=f"127.0.0.1:{port}",
            auth_token="secret",
            tls_root_cert_path=str(certificate),
            tls_server_name_override="localhost",
            connect_timeout_s=2,
            inference_timeout_s=2,
        )
    )
    try:
        client.connect(make_manifest(), task="test task", client_instance_id="tls-client")
        action = client.infer(make_observation())
        assert action.actions.shape == (4, 3)
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_manifest_mismatch_fails_before_inference(running_server):
    client = make_client(running_server)
    with pytest.raises(RemotePolicyError, match="camera order"):
        client.connect(make_manifest(("left", "top", "right")))
    client.close()


def test_client_rejects_duplicate_sequence(running_server):
    client = make_client(running_server)
    client.connect(make_manifest(), task="test task", client_instance_id="client")
    client.infer(make_observation(sequence=2))
    with pytest.raises(RemotePolicyError, match="increase monotonically"):
        client.infer(make_observation(sequence=2))
    client.close()


def test_second_robot_cannot_replace_active_session(running_server):
    first = make_client(running_server)
    second = make_client(running_server)
    first.connect(make_manifest(), task="test task", client_instance_id="first")
    with pytest.raises(RemotePolicyError, match="RESOURCE_EXHAUSTED"):
        second.connect(make_manifest(), task="test task", client_instance_id="second")
    first.close()
    second.close()


def test_another_robot_can_connect_after_idle_session_expires():
    port = free_port()
    backend = DeterministicPolicyBackend(
        DeterministicPolicyBackendConfig(
            action_horizon=4,
            state_features=STATE_FEATURES,
            action_features=STATE_FEATURES,
            camera_keys=CAMERA_KEYS,
        )
    )
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port, session_idle_timeout_s=0.05), backend)
    server.start()
    first = make_client(f"127.0.0.1:{port}", token="")
    second = make_client(f"127.0.0.1:{port}", token="")
    try:
        first.connect(make_manifest(), task="test task", client_instance_id="first")
        time.sleep(0.08)
        second.connect(make_manifest(), task="test task", client_instance_id="second")
        assert second.connected
    finally:
        first.close()
        second.close()
        server.stop(grace=0).wait()


def test_server_rejects_duplicate_sequence_even_from_raw_rpc(running_server):
    client = make_client(running_server)
    client.connect(make_manifest(), task="test task", client_instance_id="client")
    client.infer(make_observation(sequence=3))
    with pytest.raises(RemotePolicyError, match="increase monotonically"):
        client.infer(make_observation(sequence=3))
    client.close()


def test_task_is_fixed_for_session(running_server):
    client = make_client(running_server)
    client.connect(make_manifest(), task="session task", client_instance_id="client")

    with pytest.raises(RemotePolicyError, match="INVALID_ARGUMENT"):
        client.infer(make_observation(task="different task"))

    client.close()


def test_session_prepares_exact_task_before_connect_returns():
    class RecordingBackend(DeterministicPolicyBackend):
        def __init__(self):
            super().__init__(
                DeterministicPolicyBackendConfig(
                    action_horizon=4,
                    state_features=STATE_FEATURES,
                    action_features=STATE_FEATURES,
                    camera_keys=CAMERA_KEYS,
                )
            )
            self.prepared: list[tuple[EmbodimentManifest, str]] = []

        def prepare(self, embodiment: EmbodimentManifest, task: str) -> None:
            self.prepared.append((embodiment, task))

    port = free_port()
    backend = RecordingBackend()
    server, _ = create_grpc_server(RemotePolicyServerConfig(port=port), backend)
    server.start()
    client = make_client(f"127.0.0.1:{port}", token="")
    try:
        client.connect(make_manifest(), task="pick up the cube")
        assert [(manifest.robot_id, task) for manifest, task in backend.prepared] == [
            ("test-robot", "pick up the cube")
        ]
    finally:
        client.close()
        server.stop(grace=0).wait()


def test_unreachable_server_has_bounded_connect_timeout():
    client = make_client(f"127.0.0.1:{free_port()}")
    with pytest.raises(RemotePolicyError, match="could not open"):
        client.connect(make_manifest())
    client.close()
