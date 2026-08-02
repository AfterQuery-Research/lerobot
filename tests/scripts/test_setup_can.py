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

from types import SimpleNamespace

import pytest

from lerobot.scripts import lerobot_setup_can as can_setup_module
from lerobot.utils.can import CANInterfaceInfo


def _interface(name: str, serial: str, *, ready: bool = True) -> CANInterfaceInfo:
    return CANInterfaceInfo(
        name=name,
        adapter_serial=serial,
        driver="gs_usb",
        is_up=ready,
        bitrate=1_000_000 if ready else None,
        is_fd=False,
    )


def test_serial_selection_defaults_to_safe_classic_can_setup():
    cfg = can_setup_module.CANSetupConfig(
        left_adapter_serial="LEFT",
        right_adapter_serial="RIGHT",
    )
    discovered = [_interface("can0", "RIGHT"), _interface("can1", "LEFT")]

    assert cfg.effective_mode == "setup"
    assert cfg.effective_use_fd is False
    assert cfg.get_interfaces(discovered) == ["can1", "can0"]


def test_no_arguments_defaults_to_read_only_listing():
    cfg = can_setup_module.CANSetupConfig()

    assert cfg.effective_mode == "list"
    assert cfg.adapter_serials == ()


def test_setup_config_rejects_ambiguous_selectors():
    with pytest.raises(ValueError, match="either interfaces or adapter serials"):
        can_setup_module.CANSetupConfig(interfaces="can0", left_adapter_serial="LEFT")
    with pytest.raises(ValueError, match="must be different"):
        can_setup_module.CANSetupConfig(left_adapter_serial="SAME", right_adapter_serial="SAME")


def test_run_setup_resolves_serials_and_only_configures_links(monkeypatch, capsys):
    cfg = can_setup_module.CANSetupConfig(left_adapter_serial="LEFT", right_adapter_serial="RIGHT")
    discovered = [_interface("can0", "RIGHT"), _interface("can1", "LEFT")]
    configured = []
    monkeypatch.setattr(can_setup_module, "discover_can_interfaces", lambda: discovered)
    monkeypatch.setattr(
        can_setup_module,
        "setup_interface",
        lambda name, bitrate, data_bitrate, use_fd: (
            configured.append((name, bitrate, data_bitrate, use_fd)) or True
        ),
    )

    can_setup_module.run_setup(cfg)

    assert configured == [
        ("can1", 1_000_000, 5_000_000, False),
        ("can0", 1_000_000, 5_000_000, False),
    ]
    output = capsys.readouterr().out
    assert "No CAN frames or motor commands were sent" in output
    assert "--mode=test" not in output


def test_classic_setup_explicitly_disables_fd(monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(can_setup_module.subprocess, "run", fake_run)

    assert can_setup_module.setup_interface("can7", 1_000_000, 5_000_000, False)
    assert commands == [
        ["sudo", "ip", "link", "set", "can7", "down"],
        ["sudo", "ip", "link", "set", "can7", "type", "can", "bitrate", "1000000", "fd", "off"],
        ["sudo", "ip", "link", "set", "can7", "up"],
    ]
