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

from pathlib import Path

import pytest

from lerobot.utils.can import (
    CANInterfaceInfo,
    can_interface_readiness_error,
    discover_can_interfaces,
    format_can_interfaces,
    resolve_can_interface,
)


def _add_interface(root: Path, name: str, serial: str, *, link_type: int = 280) -> None:
    interface = root / "net" / name
    interface.mkdir(parents=True)
    (interface / "type").write_text(str(link_type))

    adapter = root / "devices" / f"usb-{name}"
    usb_interface = adapter / f"{name}:1.0"
    usb_interface.mkdir(parents=True)
    (adapter / "serial").write_text(serial)
    driver = root / "drivers" / "gs_usb"
    driver.mkdir(parents=True, exist_ok=True)
    (usb_interface / "driver").symlink_to(driver, target_is_directory=True)
    (interface / "device").symlink_to(usb_interface, target_is_directory=True)


def _details(*, up: bool, bitrate: int | None, fd: bool = False):
    info_data = {"ctrlmode": ["FD"] if fd else []}
    if bitrate is not None:
        info_data["bittiming"] = {"bitrate": bitrate}
    return {
        "flags": ["UP", "NOARP"] if up else ["NOARP"],
        "linkinfo": {"info_data": info_data},
    }


def test_discovery_uses_usb_serial_instead_of_interface_order(tmp_path):
    _add_interface(tmp_path, "can1", "LEFT-SERIAL")
    _add_interface(tmp_path, "can0", "RIGHT-SERIAL")
    _add_interface(tmp_path, "eth0", "NOT-CAN", link_type=1)

    interfaces = discover_can_interfaces(
        tmp_path / "net",
        ip_details_loader=lambda name: _details(
            up=name == "can1",
            bitrate=1_000_000 if name == "can1" else None,
        ),
    )

    assert [interface.name for interface in interfaces] == ["can0", "can1"]
    assert resolve_can_interface("left-serial", interfaces=interfaces).name == "can1"
    assert resolve_can_interface("RIGHT-SERIAL", interfaces=interfaces).name == "can0"
    assert all(interface.driver == "gs_usb" for interface in interfaces)
    assert "LEFT-SERIAL" in format_can_interfaces(interfaces)


def test_discovery_ignores_malformed_interface_type(tmp_path):
    interface = tmp_path / "net" / "broken0"
    interface.mkdir(parents=True)
    (interface / "type").write_text("not-a-number")

    assert discover_can_interfaces(tmp_path / "net", ip_details_loader=lambda _: {}) == []


def test_resolution_errors_include_discovered_inventory():
    interfaces = [CANInterfaceInfo("can0", "KNOWN", "gs_usb", False, None, False)]

    with pytest.raises(RuntimeError, match="KNOWN"):
        resolve_can_interface("MISSING", interfaces=interfaces)

    duplicate = [*interfaces, CANInterfaceInfo("can1", "KNOWN", "gs_usb", False, None, False)]
    with pytest.raises(RuntimeError, match="multiple SocketCAN interfaces"):
        resolve_can_interface("KNOWN", interfaces=duplicate)


def test_readiness_requires_up_classic_can_at_expected_bitrate():
    ready = CANInterfaceInfo("can0", "SERIAL", "gs_usb", True, 1_000_000, False)
    assert can_interface_readiness_error(ready, bitrate=1_000_000, use_fd=False) is None

    wrong = CANInterfaceInfo("can0", "SERIAL", "gs_usb", False, 500_000, True)
    message = can_interface_readiness_error(wrong, bitrate=1_000_000, use_fd=False)
    assert message is not None
    assert "DOWN" in message
    assert "500000" in message
    assert "expected Classic CAN" in message
