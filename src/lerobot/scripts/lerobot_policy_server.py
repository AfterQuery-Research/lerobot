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

import logging
from dataclasses import dataclass, field

import draccus

from lerobot.remote_inference import (
    LeRobotPolicyBackend,
    LeRobotPolicyBackendConfig,
    RemotePolicyServerConfig,
    serve,
)


@dataclass
class PolicyServerAppConfig:
    policy: LeRobotPolicyBackendConfig = field(
        default_factory=lambda: LeRobotPolicyBackendConfig(pretrained_name_or_path="")
    )
    host: str = "127.0.0.1"
    port: int = 8081
    max_message_bytes: int = 16 * 1024 * 1024
    max_image_bytes: int = 8 * 1024 * 1024
    max_decoded_image_bytes: int = 64 * 1024 * 1024
    max_task_chars: int = 4096
    command_ttl_ms: int = 200
    session_idle_timeout_s: float = 30.0
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    tls_client_ca_path: str | None = None


@draccus.wrap()
def run_server(cfg: PolicyServerAppConfig) -> None:
    if not cfg.policy.pretrained_name_or_path:
        raise ValueError("--policy.pretrained_name_or_path is required")
    server_config = RemotePolicyServerConfig(
        host=cfg.host,
        port=cfg.port,
        max_message_bytes=cfg.max_message_bytes,
        max_image_bytes=cfg.max_image_bytes,
        max_decoded_image_bytes=cfg.max_decoded_image_bytes,
        max_task_chars=cfg.max_task_chars,
        command_ttl_ms=cfg.command_ttl_ms,
        session_idle_timeout_s=cfg.session_idle_timeout_s,
        tls_cert_path=cfg.tls_cert_path,
        tls_key_path=cfg.tls_key_path,
        tls_client_ca_path=cfg.tls_client_ca_path,
    )
    server_config.validate()
    backend = LeRobotPolicyBackend(cfg.policy)
    serve(server_config, backend)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_server()


if __name__ == "__main__":
    main()
