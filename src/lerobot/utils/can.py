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

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARPHRD_CAN = 280


@dataclass(frozen=True)
class CANInterfaceInfo:
    name: str
    adapter_serial: str | None
    driver: str | None
    is_up: bool
    bitrate: int | None
    is_fd: bool

    @property
    def state(self) -> str:
        return "UP" if self.is_up else "DOWN"


IPDetailsLoader = Callable[[str], dict[str, Any]]


def _read_text(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return None
    return value or None


def _device_attribute(device_path: Path, attribute: str) -> str | None:
    try:
        current = device_path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None

    for parent in (current, *current.parents):
        value = _read_text(parent / attribute)
        if value is not None:
            return value
    return None


def _device_driver(device_path: Path) -> str | None:
    try:
        return (device_path / "driver").resolve(strict=True).name
    except (FileNotFoundError, OSError):
        return None


def load_ip_link_details(interface: str) -> dict[str, Any]:
    try:
        result = subprocess.run(  # nosec B607
            ["ip", "-details", "-json", "link", "show", "dev", interface],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("The `ip` command is required to inspect SocketCAN interfaces") from exc
    if result.returncode != 0:
        message = result.stderr.strip() or f"exit status {result.returncode}"
        raise RuntimeError(f"Could not inspect SocketCAN interface {interface}: {message}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The `ip` command returned invalid JSON for {interface}") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError(f"The `ip` command returned an unexpected result for {interface}")
    return payload[0]


def _parse_link_details(details: dict[str, Any]) -> tuple[bool, int | None, bool]:
    flags = details.get("flags", [])
    is_up = isinstance(flags, list) and "UP" in flags
    linkinfo = details.get("linkinfo", {})
    info_data = linkinfo.get("info_data", {}) if isinstance(linkinfo, dict) else {}
    bittiming = info_data.get("bittiming", {}) if isinstance(info_data, dict) else {}
    bitrate_value = bittiming.get("bitrate") if isinstance(bittiming, dict) else None
    bitrate = int(bitrate_value) if isinstance(bitrate_value, int | float) else None
    ctrlmode = info_data.get("ctrlmode", []) if isinstance(info_data, dict) else []
    is_fd = isinstance(ctrlmode, list) and "FD" in ctrlmode
    return is_up, bitrate, is_fd


def discover_can_interfaces(
    sysfs_net_root: Path = Path("/sys/class/net"),
    *,
    ip_details_loader: IPDetailsLoader = load_ip_link_details,
) -> list[CANInterfaceInfo]:
    if not sysfs_net_root.is_dir():
        return []

    interfaces: list[CANInterfaceInfo] = []
    for interface_path in sorted(sysfs_net_root.iterdir(), key=lambda path: path.name):
        link_type = _read_text(interface_path / "type")
        try:
            is_can = link_type is not None and int(link_type, 0) == ARPHRD_CAN
        except ValueError:
            is_can = False
        if not is_can:
            continue

        details = ip_details_loader(interface_path.name)
        is_up, bitrate, is_fd = _parse_link_details(details)
        device_path = interface_path / "device"
        interfaces.append(
            CANInterfaceInfo(
                name=interface_path.name,
                adapter_serial=_device_attribute(device_path, "serial"),
                driver=_device_driver(device_path),
                is_up=is_up,
                bitrate=bitrate,
                is_fd=is_fd,
            )
        )
    return interfaces


def format_can_interfaces(interfaces: list[CANInterfaceInfo]) -> str:
    if not interfaces:
        return "No SocketCAN interfaces discovered."

    headings = ("Interface", "USB serial", "Driver", "State", "Bitrate", "Mode")
    rows = [
        (
            interface.name,
            interface.adapter_serial or "-",
            interface.driver or "-",
            interface.state,
            str(interface.bitrate) if interface.bitrate is not None else "unset",
            "CAN FD" if interface.is_fd else "Classic CAN",
        )
        for interface in interfaces
    ]
    widths = [max(len(headings[index]), *(len(row[index]) for row in rows)) for index in range(len(headings))]
    rendered = ["  ".join(value.ljust(widths[index]) for index, value in enumerate(headings))]
    rendered.extend("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows)
    return "\n".join(rendered)


def resolve_can_interface(
    adapter_serial: str,
    *,
    interfaces: list[CANInterfaceInfo] | None = None,
) -> CANInterfaceInfo:
    serial = adapter_serial.strip()
    if not serial:
        raise ValueError("CAN adapter serial must not be empty")
    discovered = discover_can_interfaces() if interfaces is None else interfaces
    matches = [
        interface
        for interface in discovered
        if interface.adapter_serial is not None and interface.adapter_serial.casefold() == serial.casefold()
    ]
    if len(matches) == 1:
        return matches[0]

    inventory = format_can_interfaces(discovered)
    if not matches:
        raise RuntimeError(f"No SocketCAN adapter with USB serial {serial!r} was found.\n{inventory}")
    names = ", ".join(interface.name for interface in matches)
    raise RuntimeError(f"USB serial {serial!r} matched multiple SocketCAN interfaces: {names}.\n{inventory}")


def can_interface_readiness_error(
    interface: CANInterfaceInfo,
    *,
    bitrate: int,
    use_fd: bool,
) -> str | None:
    problems = []
    if not interface.is_up:
        problems.append("interface is DOWN")
    if interface.bitrate != bitrate:
        actual = "unset" if interface.bitrate is None else str(interface.bitrate)
        problems.append(f"bitrate is {actual}, expected {bitrate}")
    if interface.is_fd != use_fd:
        expected = "CAN FD" if use_fd else "Classic CAN"
        actual = "CAN FD" if interface.is_fd else "Classic CAN"
        problems.append(f"mode is {actual}, expected {expected}")
    return "; ".join(problems) or None
