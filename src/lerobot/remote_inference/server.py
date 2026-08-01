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

import hmac
import logging
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, replace
from pathlib import Path

import grpc

from . import remote_policy_pb2, remote_policy_pb2_grpc
from .backend import PolicyBackend
from .codec import action_to_proto, embodiment_from_proto, model_to_proto, observation_from_proto
from .schema import PROTOCOL_VERSION, EmbodimentManifest, ProtocolValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemotePolicyServerConfig:
    host: str = "127.0.0.1"
    port: int = 8081
    service_name: str = "lerobot-remote-policy"
    max_message_bytes: int = 16 * 1024 * 1024
    max_image_bytes: int = 8 * 1024 * 1024
    max_decoded_image_bytes: int = 64 * 1024 * 1024
    max_task_chars: int = 4096
    command_ttl_ms: int = 200
    session_idle_timeout_s: float = 30.0
    max_workers: int = 2
    auth_token: str | None = None
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    tls_client_ca_path: str | None = None

    def validate(self) -> None:
        if not self.host or not 1 <= self.port <= 65535:
            raise ValueError("server host and port are invalid")
        if self.max_message_bytes <= 0 or self.max_image_bytes <= 0 or self.max_decoded_image_bytes <= 0:
            raise ValueError("message and image byte limits must be positive")
        if self.max_image_bytes > self.max_message_bytes:
            raise ValueError("max_image_bytes cannot exceed max_message_bytes")
        if self.max_task_chars <= 0 or self.command_ttl_ms <= 0 or self.session_idle_timeout_s <= 0:
            raise ValueError("task, command TTL, and session timeout limits must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        tls_values = (self.tls_cert_path, self.tls_key_path)
        if any(tls_values) and not all(tls_values):
            raise ValueError("both TLS certificate and key paths are required")
        if self.tls_client_ca_path is not None and self.tls_cert_path is None:
            raise ValueError("a TLS client CA requires a server certificate and key")


@dataclass
class _Session:
    session_id: str
    client_instance_id: str
    embodiment: EmbodimentManifest
    task: str
    last_sequence: int = -1
    last_activity_monotonic: float = 0.0
    inference_active: bool = False


class RemotePolicyService(remote_policy_pb2_grpc.RemotePolicyServiceServicer):
    """Typed, single-robot remote policy service."""

    def __init__(self, config: RemotePolicyServerConfig, backend: PolicyBackend):
        config.validate()
        backend.manifest.validate()
        self._config = config
        self._backend = backend
        self._session: _Session | None = None
        self._lock = threading.Lock()

    def _abort(self, context: grpc.ServicerContext, code: grpc.StatusCode, detail: str):
        context.abort(code, detail)

    def _authenticate(self, context: grpc.ServicerContext) -> None:
        if self._config.auth_token is None:
            return
        metadata = dict(context.invocation_metadata())
        supplied = metadata.get("authorization", "")
        expected = f"Bearer {self._config.auth_token}"
        if not hmac.compare_digest(supplied, expected):
            self._abort(context, grpc.StatusCode.UNAUTHENTICATED, "invalid authorization token")

    def _active_session_locked(self, session_id: str, context: grpc.ServicerContext) -> _Session:
        session = self._session
        if session is None or session.session_id != session_id:
            self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "session is not active")
        assert session is not None
        now = time.monotonic()
        if now - session.last_activity_monotonic > self._config.session_idle_timeout_s:
            self._session = None
            self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "session expired")
        session.last_activity_monotonic = now
        return session

    def GetServerInfo(self, request, context):  # noqa: N802
        del request
        self._authenticate(context)
        return remote_policy_pb2.ServerInfo(
            protocol_version=PROTOCOL_VERSION,
            service_name=self._config.service_name,
            ready=True,
            model=model_to_proto(self._backend.manifest),
        )

    def OpenSession(self, request, context):  # noqa: N802
        self._authenticate(context)
        try:
            embodiment = embodiment_from_proto(request.embodiment)
            self._backend.manifest.assert_compatible(embodiment)
        except ProtocolValidationError as exc:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        if request.requested_model_id and request.requested_model_id != self._backend.manifest.model_id:
            self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "requested model is not served here")
        if not request.client_instance_id:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "client_instance_id is required")
        if len(request.task) > self._config.max_task_chars:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "task exceeds the configured length limit")
        with self._lock:
            if (
                self._session is not None
                and not self._session.inference_active
                and time.monotonic() - self._session.last_activity_monotonic
                > self._config.session_idle_timeout_s
            ):
                logger.info("Expiring idle remote policy session %s", self._session.session_id)
                self._backend.reset()
                self._session = None
            if self._session is not None:
                if self._session.client_instance_id != request.client_instance_id:
                    self._abort(
                        context, grpc.StatusCode.RESOURCE_EXHAUSTED, "another robot session is active"
                    )
                if self._session.inference_active:
                    self._abort(
                        context,
                        grpc.StatusCode.FAILED_PRECONDITION,
                        "cannot replace a session while inference is active",
                    )
                logger.info("Replacing session for client %s", request.client_instance_id)
            try:
                self._backend.prepare(embodiment, request.task)
                self._backend.reset()
            except Exception:
                logger.exception("Policy preparation failed for client %s", request.client_instance_id)
                self._abort(context, grpc.StatusCode.INTERNAL, "policy preparation failed")
            session_id = uuid.uuid4().hex
            self._session = _Session(
                session_id=session_id,
                client_instance_id=request.client_instance_id,
                embodiment=embodiment,
                task=request.task,
                last_activity_monotonic=time.monotonic(),
            )
        logger.info("Opened remote policy session %s for %s", session_id, request.client_instance_id)
        return remote_policy_pb2.OpenSessionResponse(
            session_id=session_id,
            model=model_to_proto(self._backend.manifest),
            command_ttl_ms=self._config.command_ttl_ms,
        )

    def Infer(self, request, context):  # noqa: N802
        self._authenticate(context)
        if len(request.task) > self._config.max_task_chars:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, "task exceeds the configured length limit")
        with self._lock:
            session = self._active_session_locked(request.session_id, context)
            if request.task != session.task:
                self._abort(
                    context, grpc.StatusCode.INVALID_ARGUMENT, "task does not match the active session"
                )
            if session.inference_active:
                self._abort(
                    context, grpc.StatusCode.RESOURCE_EXHAUSTED, "an inference request is already active"
                )
            if request.observation_sequence <= session.last_sequence:
                self._abort(
                    context, grpc.StatusCode.OUT_OF_RANGE, "observation sequence is stale or duplicated"
                )
            session.inference_active = True
        try:
            observation = observation_from_proto(
                request,
                embodiment=session.embodiment,
                max_image_bytes=self._config.max_image_bytes,
                max_decoded_image_bytes=self._config.max_decoded_image_bytes,
            )
            started_ns = time.perf_counter_ns()
            action = self._backend.infer(observation)
            compute_ns = time.perf_counter_ns() - started_ns
            action = replace(action, server_compute_ns=compute_ns)
            response = action_to_proto(
                action,
                session_id=session.session_id,
                model=self._backend.manifest,
            )
        except ProtocolValidationError as exc:
            self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception:
            logger.exception("Policy inference failed for observation %d", request.observation_sequence)
            self._abort(context, grpc.StatusCode.INTERNAL, "policy inference failed")
        finally:
            with self._lock:
                session.inference_active = False
        with self._lock:
            session.last_sequence = request.observation_sequence
            session.last_activity_monotonic = time.monotonic()
        return response

    def ResetSession(self, request, context):  # noqa: N802
        self._authenticate(context)
        with self._lock:
            session = self._active_session_locked(request.session_id, context)
            if session.inference_active:
                self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "cannot reset during inference")
            self._backend.reset()
            session.last_sequence = -1
            session.last_activity_monotonic = time.monotonic()
        return remote_policy_pb2.Empty()

    def CloseSession(self, request, context):  # noqa: N802
        self._authenticate(context)
        with self._lock:
            session = self._active_session_locked(request.session_id, context)
            if session.inference_active:
                self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, "cannot close during inference")
            self._backend.reset()
            self._session = None
        logger.info("Closed remote policy session %s", request.session_id)
        return remote_policy_pb2.Empty()


def create_grpc_server(
    config: RemotePolicyServerConfig,
    backend: PolicyBackend,
) -> tuple[grpc.Server, RemotePolicyService]:
    config.validate()
    options = (
        ("grpc.max_receive_message_length", config.max_message_bytes),
        ("grpc.max_send_message_length", config.max_message_bytes),
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers), options=options)
    service = RemotePolicyService(config, backend)
    remote_policy_pb2_grpc.add_RemotePolicyServiceServicer_to_server(service, server)
    address = f"{config.host}:{config.port}"
    if config.tls_cert_path is not None:
        cert_chain = Path(config.tls_cert_path).read_bytes()
        private_key = Path(config.tls_key_path or "").read_bytes()
        root_certificates = (
            Path(config.tls_client_ca_path).read_bytes() if config.tls_client_ca_path else None
        )
        credentials = grpc.ssl_server_credentials(
            ((private_key, cert_chain),),
            root_certificates=root_certificates,
            require_client_auth=root_certificates is not None,
        )
        bound_port = server.add_secure_port(address, credentials)
    else:
        bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"could not bind remote policy server to {address}")
    return server, service


def serve(config: RemotePolicyServerConfig, backend: PolicyBackend) -> None:
    config.validate()
    backend.warmup()
    server, _ = create_grpc_server(config, backend)
    server.start()
    logger.info("Remote policy server listening on %s:%d", config.host, config.port)
    try:
        server.wait_for_termination()
    finally:
        server.stop(grace=2.0).wait()
        backend.close()
