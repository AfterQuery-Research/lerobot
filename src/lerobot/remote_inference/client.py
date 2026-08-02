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

import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import grpc

from . import remote_policy_pb2, remote_policy_pb2_grpc
from .codec import action_from_proto, embodiment_to_proto, model_from_proto, observation_to_proto
from .schema import (
    PROTOCOL_VERSION,
    EmbodimentManifest,
    ModelManifest,
    PolicyActionChunk,
    PolicyObservation,
    ProtocolValidationError,
)


class RemotePolicyError(RuntimeError):
    """A transport, session, or remote policy failure."""


@dataclass(frozen=True)
class RemotePolicyClientConfig:
    server_address: str
    connect_timeout_s: float = 10.0
    inference_timeout_s: float = 2.0
    max_message_bytes: int = 16 * 1024 * 1024
    jpeg_quality: int = 95
    tls_root_cert_path: str | None = None
    tls_client_cert_path: str | None = None
    tls_client_key_path: str | None = None
    tls_server_name_override: str | None = None

    def validate(self) -> None:
        if not self.server_address:
            raise ValueError("server_address is required")
        if self.connect_timeout_s <= 0 or self.inference_timeout_s <= 0:
            raise ValueError("remote policy timeouts must be positive")
        if self.max_message_bytes <= 0 or not 1 <= self.jpeg_quality <= 100:
            raise ValueError("remote policy message and JPEG limits are invalid")
        client_tls = (self.tls_client_cert_path, self.tls_client_key_path)
        if any(client_tls) and not all(client_tls):
            raise ValueError("both client TLS certificate and key are required")
        if any(client_tls) and self.tls_root_cert_path is None:
            raise ValueError("a TLS root certificate is required when client certificates are configured")
        if self.tls_server_name_override and self.tls_root_cert_path is None:
            raise ValueError("TLS server name override requires a TLS root certificate")


@dataclass(frozen=True)
class RemotePolicySession:
    session_id: str
    model: ModelManifest
    command_ttl_ms: int


class RemotePolicyClient:
    """Synchronous request client with explicit session and reconnect semantics."""

    def __init__(self, config: RemotePolicyClientConfig):
        config.validate()
        self._config = config
        self._channel = self._make_channel()
        self._stub = remote_policy_pb2_grpc.RemotePolicyServiceStub(self._channel)
        self._embodiment: EmbodimentManifest | None = None
        self._session: RemotePolicySession | None = None
        self._infer_lock = threading.Lock()
        self._last_sequence = -1

    def _make_channel(self) -> grpc.Channel:
        options: list[tuple[str, int | str]] = [
            ("grpc.max_receive_message_length", self._config.max_message_bytes),
            ("grpc.max_send_message_length", self._config.max_message_bytes),
            ("grpc.enable_retries", 0),
        ]
        if self._config.tls_server_name_override:
            options.append(("grpc.ssl_target_name_override", self._config.tls_server_name_override))
        if self._config.tls_root_cert_path is None:
            return grpc.insecure_channel(self._config.server_address, options=options)
        root = Path(self._config.tls_root_cert_path).read_bytes()
        cert = (
            Path(self._config.tls_client_cert_path).read_bytes()
            if self._config.tls_client_cert_path
            else None
        )
        key = (
            Path(self._config.tls_client_key_path).read_bytes() if self._config.tls_client_key_path else None
        )
        credentials = grpc.ssl_channel_credentials(
            root_certificates=root, private_key=key, certificate_chain=cert
        )
        return grpc.secure_channel(self._config.server_address, credentials, options=options)

    @property
    def session(self) -> RemotePolicySession | None:
        return self._session

    @property
    def connected(self) -> bool:
        return self._session is not None

    def connect(
        self,
        embodiment: EmbodimentManifest,
        *,
        task: str = "",
        requested_model_id: str = "",
        client_instance_id: str | None = None,
    ) -> RemotePolicySession:
        embodiment.validate()
        try:
            grpc.channel_ready_future(self._channel).result(timeout=self._config.connect_timeout_s)
            info = self._stub.GetServerInfo(
                remote_policy_pb2.Empty(),
                timeout=self._config.connect_timeout_s,
            )
            if info.protocol_version != PROTOCOL_VERSION or not info.ready:
                raise RemotePolicyError("remote policy server is not compatible or ready")
            advertised_model = model_from_proto(info.model)
            advertised_model.assert_compatible(embodiment)
            response = self._stub.OpenSession(
                remote_policy_pb2.OpenSessionRequest(
                    client_instance_id=client_instance_id or uuid.uuid4().hex,
                    requested_model_id=requested_model_id,
                    embodiment=embodiment_to_proto(embodiment),
                    task=task,
                ),
                timeout=self._config.connect_timeout_s,
            )
            model = model_from_proto(response.model)
            model.assert_compatible(embodiment)
            if model.fingerprint != advertised_model.fingerprint:
                raise RemotePolicyError("server model changed during session setup")
        except grpc.FutureTimeoutError as exc:
            raise RemotePolicyError("could not open remote policy session: connection timed out") from exc
        except grpc.RpcError as exc:
            raise RemotePolicyError(f"could not open remote policy session: {exc.code().name}") from exc
        except ProtocolValidationError as exc:
            raise RemotePolicyError(f"remote policy manifest is incompatible: {exc}") from exc
        self._embodiment = embodiment
        self._session = RemotePolicySession(
            session_id=response.session_id,
            model=model,
            command_ttl_ms=response.command_ttl_ms,
        )
        self._last_sequence = -1
        return self._session

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        if self._session is None or self._embodiment is None:
            raise RemotePolicyError("remote policy client is not connected")
        if observation.sequence <= self._last_sequence:
            raise RemotePolicyError("observation sequence must increase monotonically")
        request = observation_to_proto(
            observation,
            session_id=self._session.session_id,
            embodiment=self._embodiment,
            jpeg_quality=self._config.jpeg_quality,
        )
        if not self._infer_lock.acquire(blocking=False):
            raise RemotePolicyError("only one inference request may be active")
        try:
            response = self._stub.Infer(
                request,
                timeout=self._config.inference_timeout_s,
                wait_for_ready=False,
            )
            action = action_from_proto(
                response,
                expected_session_id=self._session.session_id,
                model=self._session.model,
            )
        except grpc.RpcError as exc:
            raise RemotePolicyError(f"remote inference failed: {exc.code().name}") from exc
        except (ProtocolValidationError, ValueError) as exc:
            raise RemotePolicyError(f"remote inference returned an invalid action chunk: {exc}") from exc
        finally:
            self._infer_lock.release()
        if action.observation_sequence != observation.sequence:
            raise RemotePolicyError("remote action chunk does not match the submitted observation")
        self._last_sequence = observation.sequence
        return action

    def reset(self) -> None:
        if self._session is None:
            return
        try:
            self._stub.ResetSession(
                remote_policy_pb2.SessionRequest(session_id=self._session.session_id),
                timeout=self._config.connect_timeout_s,
            )
        except grpc.RpcError as exc:
            raise RemotePolicyError(f"could not reset remote policy session: {exc.code().name}") from exc
        self._last_sequence = -1

    def close(self) -> None:
        if self._session is not None:
            with suppress(grpc.RpcError):
                self._stub.CloseSession(
                    remote_policy_pb2.SessionRequest(session_id=self._session.session_id),
                    timeout=self._config.connect_timeout_s,
                )
        self._session = None
        self._embodiment = None
        self._channel.close()

    def __enter__(self) -> RemotePolicyClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
