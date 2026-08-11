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

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

from lerobot.simulators.bi_yam.config import STATE_SIZE, YAM_JOINT_LIMITS, BiYAMSimulatorConfig
from lerobot.simulators.bi_yam.simulator import _StepDrivenBiYAMBackend

_GRIPPER_TRAVEL_METERS = 0.0475


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(str(value) for value in values)


def _prefix_references(element: ET.Element, prefix: str) -> ET.Element:
    """Prefix names and internal body/joint references in a copied MJCF subtree."""

    copied = deepcopy(element)
    for node in copied.iter():
        if "name" in node.attrib:
            node.set("name", f"{prefix}{node.get('name')}")
        for attribute in ("body1", "body2", "joint", "joint1", "joint2", "site", "site1", "site2"):
            if attribute in node.attrib:
                node.set(attribute, f"{prefix}{node.get(attribute)}")
    return copied


class MujocoBiYAMBackend(_StepDrivenBiYAMBackend):
    """One shared MuJoCo world built from i2rt YAM and linear_4310 assets."""

    backend_name = "mujoco"

    def __init__(self, config: BiYAMSimulatorConfig) -> None:
        super().__init__(config)
        self._mujoco = None
        self._model = None
        self._data = None
        self._renderer = None
        self._joint_qpos_addresses = np.empty(STATE_SIZE, dtype=np.int32)
        self._joint_dof_addresses = np.empty(STATE_SIZE, dtype=np.int32)
        self._gripper_secondary_qpos_addresses = np.empty(2, dtype=np.int32)
        self._gripper_secondary_dof_addresses = np.empty(2, dtype=np.int32)
        self._actuator_ids = np.empty(STATE_SIZE, dtype=np.int32)

    def _reset_backend(self, seed: int, initial_state: np.ndarray) -> None:
        import mujoco

        self._close_backend()
        self._mujoco = mujoco
        xml = self._build_world_xml(seed)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        self._resolve_model_addresses()
        mujoco.mj_resetData(self._model, self._data)
        self._write_command_state(initial_state)
        self._set_control_target(initial_state)
        mujoco.mj_forward(self._model, self._data)

    def _read_state(self) -> np.ndarray:
        state = np.asarray(self._data.qpos[self._joint_qpos_addresses], dtype=np.float32).copy()
        state[[6, 13]] /= _GRIPPER_TRAVEL_METERS
        return state

    def _set_control_target(self, target: np.ndarray) -> None:
        if self._data is None:
            return
        controls = target.astype(np.float64, copy=True)
        controls[[6, 13]] *= _GRIPPER_TRAVEL_METERS
        self._data.ctrl[self._actuator_ids] = controls

    def _step_physics(self, stuck_positions: dict[int, float]) -> None:
        self._mujoco.mj_step(self._model, self._data)
        if not stuck_positions:
            return
        for index, position in stuck_positions.items():
            physical_position = position * _GRIPPER_TRAVEL_METERS if index in (6, 13) else position
            self._data.qpos[self._joint_qpos_addresses[index]] = physical_position
            self._data.qvel[self._joint_dof_addresses[index]] = 0.0
            if index in (6, 13):
                side_index = 0 if index == 6 else 1
                self._data.qpos[self._gripper_secondary_qpos_addresses[side_index]] = physical_position
                self._data.qvel[self._gripper_secondary_dof_addresses[side_index]] = 0.0
        self._mujoco.mj_forward(self._model, self._data)

    def _render_camera(self, camera_name: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(
                self._model,
                height=self.config.camera_height,
                width=self.config.camera_width,
            )
        self._renderer.update_scene(self._data, camera=camera_name)
        return self._renderer.render().copy()

    def _close_backend(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = None
        self._data = None
        self._model = None

    def _write_command_state(self, state: np.ndarray) -> None:
        physical = state.astype(np.float64, copy=True)
        physical[[6, 13]] *= _GRIPPER_TRAVEL_METERS
        self._data.qpos[self._joint_qpos_addresses] = physical
        for side_index, state_index in enumerate((6, 13)):
            self._data.qpos[self._gripper_secondary_qpos_addresses[side_index]] = physical[state_index]
        self._data.qvel[:] = 0.0

    def _resolve_model_addresses(self) -> None:
        for offset, side in enumerate(("left", "right")):
            for local_index in range(7):
                state_index = offset * 7 + local_index
                joint_name = f"{side}_joint{local_index + 1}"
                joint_id = self._name_to_id(self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                actuator_id = self._name_to_id(self._mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_position")
                self._joint_qpos_addresses[state_index] = self._model.jnt_qposadr[joint_id]
                self._joint_dof_addresses[state_index] = self._model.jnt_dofadr[joint_id]
                self._actuator_ids[state_index] = actuator_id

            secondary_id = self._name_to_id(self._mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint8")
            self._gripper_secondary_qpos_addresses[offset] = self._model.jnt_qposadr[secondary_id]
            self._gripper_secondary_dof_addresses[offset] = self._model.jnt_dofadr[secondary_id]

    def _name_to_id(self, object_type, name: str) -> int:
        object_id = self._mujoco.mj_name2id(self._model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"Generated MuJoCo world is missing {name!r}")
        return object_id

    def _build_world_xml(self, seed: int) -> str:
        source_root = self._load_i2rt_robot_xml()
        root = ET.Element("mujoco", {"model": "bi_yam"})
        ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
        ET.SubElement(
            root,
            "option",
            {
                "timestep": str(1.0 / self.config.physics_hz),
                "integrator": "implicitfast",
                "gravity": "0 0 -9.81",
            },
        )
        ET.SubElement(root, "size", {"nconmax": "200", "njmax": "1000"})
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            visual,
            "global",
            {"offwidth": str(self.config.camera_width), "offheight": str(self.config.camera_height)},
        )

        source_assets = source_root.find("asset")
        if source_assets is None:
            raise RuntimeError("i2rt's combined YAM model has no asset section")
        root.append(deepcopy(source_assets))

        worldbody = ET.SubElement(root, "worldbody")
        self._add_workspace(worldbody, seed)

        source_worldbody = source_root.find("worldbody")
        if source_worldbody is None:
            raise RuntimeError("i2rt's combined YAM model has no worldbody")
        source_bodies = source_worldbody.findall("body")
        if len(source_bodies) != 1:
            raise RuntimeError("Expected exactly one root body in i2rt's combined YAM model")

        arm_poses = (
            ("left", self.config.left_base_position, self.config.left_base_quaternion),
            ("right", self.config.right_base_position, self.config.right_base_quaternion),
        )
        for side, position, quaternion in arm_poses:
            arm_body = _prefix_references(source_bodies[0], f"{side}_")
            arm_body.set("pos", _numbers(position))
            arm_body.set("quat", _numbers(quaternion))
            self._configure_robot_joints(arm_body)
            worldbody.append(arm_body)

        self._copy_per_arm_section(source_root, root, "equality")
        self._copy_per_arm_section(source_root, root, "contact")
        self._add_actuators(root)
        return ET.tostring(root, encoding="unicode")

    @staticmethod
    def _load_i2rt_robot_xml() -> ET.Element:
        from i2rt.robot_models import ARM_YAM_XML_PATH, GRIPPER_LINEAR_4310_PATH

        arm_root = ET.parse(ARM_YAM_XML_PATH).getroot()
        gripper_root = ET.parse(GRIPPER_LINEAR_4310_PATH).getroot()
        MujocoBiYAMBackend._resolve_mesh_paths(arm_root, Path(ARM_YAM_XML_PATH).parent)
        MujocoBiYAMBackend._resolve_mesh_paths(gripper_root, Path(GRIPPER_LINEAR_4310_PATH).parent)

        arm_assets = arm_root.find("asset")
        gripper_assets = gripper_root.find("asset")
        if arm_assets is None or gripper_assets is None:
            raise RuntimeError("i2rt YAM and linear_4310 models must contain mesh assets")
        existing_assets = {(element.tag, element.get("name")) for element in arm_assets}
        for asset in gripper_assets:
            key = (asset.tag, asset.get("name"))
            if key not in existing_assets:
                arm_assets.append(deepcopy(asset))
                existing_assets.add(key)

        arm_worldbody = arm_root.find("worldbody")
        gripper_body = gripper_root.find(".//body[@name='gripper']")
        if arm_worldbody is None or gripper_body is None:
            raise RuntimeError("i2rt YAM or linear_4310 model is missing its root body")
        terminal_body = arm_worldbody.find("body")
        if terminal_body is None:
            raise RuntimeError("i2rt YAM model has no arm body")
        while (child_body := terminal_body.find("body")) is not None:
            terminal_body = child_body
        terminal_body.append(deepcopy(gripper_body))

        for tag in ("equality", "contact"):
            source_section = gripper_root.find(tag)
            if source_section is not None:
                arm_root.append(deepcopy(source_section))
        return arm_root

    @staticmethod
    def _resolve_mesh_paths(root: ET.Element, model_directory: Path) -> None:
        compiler = root.find("compiler")
        mesh_directory = compiler.get("meshdir", "") if compiler is not None else ""
        assets = root.find("asset")
        if assets is not None:
            for asset in assets:
                filename = asset.get("file")
                if filename is not None and not Path(filename).is_absolute():
                    asset.set("file", str((model_directory / mesh_directory / filename).resolve()))
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)

    @staticmethod
    def _configure_robot_joints(arm_body: ET.Element) -> None:
        for joint in arm_body.iter("joint"):
            name = joint.get("name", "")
            joint.set("damping", "0.4" if not name.endswith(("joint7", "joint8")) else "2.0")
            joint.set("armature", "0.01" if not name.endswith(("joint7", "joint8")) else "0.001")

    @staticmethod
    def _copy_per_arm_section(source_root: ET.Element, target_root: ET.Element, tag: str) -> None:
        source = source_root.find(tag)
        if source is None:
            return
        target = ET.SubElement(target_root, tag)
        for side in ("left", "right"):
            for child in source:
                target.append(_prefix_references(child, f"{side}_"))

    def _add_actuators(self, root: ET.Element) -> None:
        actuator = ET.SubElement(root, "actuator")
        for side in ("left", "right"):
            for index, (lower, upper) in enumerate(YAM_JOINT_LIMITS, start=1):
                joint_name = f"{side}_joint{index}"
                ET.SubElement(
                    actuator,
                    "position",
                    {
                        "name": f"{joint_name}_position",
                        "joint": joint_name,
                        "kp": str(self.config.mujoco_arm_kp),
                        "ctrlrange": f"{lower} {upper}",
                        "forcerange": "-10 10",
                    },
                )
            joint_name = f"{side}_joint7"
            ET.SubElement(
                actuator,
                "position",
                {
                    "name": f"{joint_name}_position",
                    "joint": joint_name,
                    "kp": str(self.config.mujoco_gripper_kp),
                    "ctrlrange": f"0 {_GRIPPER_TRAVEL_METERS}",
                    "forcerange": "-30 30",
                },
            )

    def _add_workspace(self, worldbody: ET.Element, seed: int) -> None:
        ET.SubElement(
            worldbody,
            "light",
            {"name": "key_light", "pos": "0 -0.5 2.0", "dir": "0 0 -1", "diffuse": "0.8 0.8 0.8"},
        )
        ET.SubElement(
            worldbody,
            "geom",
            {"name": "floor", "type": "plane", "size": "2 2 0.1", "rgba": "0.18 0.19 0.21 1"},
        )
        table = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0 0 0.2"})
        ET.SubElement(
            table,
            "geom",
            {"type": "box", "size": "0.75 0.55 0.2", "rgba": "0.48 0.43 0.35 1"},
        )
        ET.SubElement(
            table,
            "site",
            {
                "name": "workspace",
                "type": "box",
                "pos": "0 0 0.205",
                "size": "0.35 0.25 0.002",
                "rgba": "0.2 0.7 0.45 0.12",
            },
        )

        cameras = (
            ("top", "0 0 1.8", "1 0 0 0 1 0"),
            ("left", "0 1.35 1.15", "-1 0 0 0 -0.55 0.835"),
            ("right", "1.35 0 1.15", "0 1 0 -0.55 0 0.835"),
        )
        for name, position, axes in cameras:
            ET.SubElement(
                worldbody,
                "camera",
                {"name": name, "mode": "fixed", "pos": position, "xyaxes": axes, "fovy": "58"},
            )

        rng = np.random.default_rng(seed)
        object_specs = (
            ("red_cube", "box", "0.035 0.035 0.035", "0.8 0.12 0.1 1"),
            ("blue_cube", "box", "0.03 0.045 0.025", "0.1 0.3 0.8 1"),
            ("green_cylinder", "cylinder", "0.025 0.055", "0.12 0.65 0.25 1"),
        )
        for index, (name, geom_type, size, color) in enumerate(object_specs):
            x, y = rng.uniform((-0.13, -0.16), (0.13, 0.16))
            z = 0.435 if geom_type == "box" else 0.455
            body = ET.SubElement(worldbody, "body", {"name": name, "pos": f"{x} {y} {z}"})
            ET.SubElement(body, "freejoint", {"name": f"{name}_joint"})
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{name}_geom",
                    "type": geom_type,
                    "size": size,
                    "mass": str(0.08 + index * 0.02),
                    "rgba": color,
                    "friction": "0.8 0.01 0.001",
                },
            )
