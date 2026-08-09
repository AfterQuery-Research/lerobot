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
"""Drive a bimanual YAM from a UMI end-effector-pose policy.

`UmiEeRemoteClient` (see ``umi_ee_client.py``) speaks the policy's language: per-arm
end-effector poses in an episode-start frame. `BiYAM` speaks joint space. This module is
the adapter between them, built on the kinematics primitives already in the repo
(``lerobot.model.RobotKinematics``, placo-backed FK/IK from a URDF).

Per control tick::

    joints (rad) --FK--> base-frame EE pose --rebase--> policy state (14,)
    policy chunk row (14,) --unrebase--> base-frame EE target --IK--> joints (rad)

Three conversions in that path are easy to get wrong and silent when wrong; each is
handled here and asserted in ``--self-test``:

1. **Units.** ``RobotKinematics`` takes and returns DEGREES. YAM joints, and every YAM
   dataset in this project, are RADIANS. The conversion lives in `YamArmKinematics` and
   nowhere else.
2. **Gripper convention.** The policy emits UMI jaw width / 125 mm; YAM wants a
   normalized ``gripper.pos``. Both are "larger = more open", but the scale differs
   enough that commanding the policy value directly leaves the jaws ~13 mm too open at a
   grasp — every pick fails while the pose trajectory looks perfect. See `GripperMap`.
3. **Frames.** Policy poses are relative to each arm's episode-start pose, so the anchor
   must be re-captured on every episode (`EpisodeFrames.capture`). If the UMI gripper's
   tool frame is rotated relative to the YAM flange, that constant offset conjugates the
   relative motion and must be supplied as ``tool_offset``; identity is only correct when
   the two tool frames agree.

Offline self-test (no robot, no policy server)::

    python -m lerobot.remote_inference.yam_umi_ee_bridge --self-test \\
        --urdf /path/to/i2rt/robot_models/arm/yam/yam.urdf \\
        --yam_dataset /path/to/yam-vive-teleop --repo_id user/yam-vive-teleop

Closed-loop replay against a live server, using a recorded YAM episode as a stand-in for
the robot (still no hardware)::

    python -m lerobot.remote_inference.yam_umi_ee_bridge --replay \\
        --server host:8081 --urdf ... --yam_dataset ... --repo_id ... \\
        --umi_dataset /path/to/dual-lidar-umi-14d --umi_repo_id user/dual-lidar-umi \\
        --task "Put all oranges in the bowl"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.model import RobotKinematics

# BiYAM's scalar interface, in the order the policy's 14-D vector uses.
LEFT_JOINT_KEYS = tuple(f"left_joint_{i}.pos" for i in range(6))
RIGHT_JOINT_KEYS = tuple(f"right_joint_{i}.pos" for i in range(6))
LEFT_GRIPPER_KEY = "left_gripper.pos"
RIGHT_GRIPPER_KEY = "right_gripper.pos"
YAM_SCALAR_KEYS = (*LEFT_JOINT_KEYS, LEFT_GRIPPER_KEY, *RIGHT_JOINT_KEYS, RIGHT_GRIPPER_KEY)

# Policy vector layout: [left pose(6), left gripper, right pose(6), right gripper]
LEFT_POSE_SLICE = slice(0, 6)
LEFT_GRIPPER_INDEX = 6
RIGHT_POSE_SLICE = slice(7, 13)
RIGHT_GRIPPER_INDEX = 13


def pose_to_vec(transform: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> [x, y, z, rx, ry, rz] (rotation as a rotation vector)."""
    return np.concatenate([transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]).astype(
        np.float64
    )


def vec_to_pose(vec: np.ndarray) -> np.ndarray:
    """[x, y, z, rx, ry, rz] -> 4x4 homogeneous transform."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(np.asarray(vec[3:6], dtype=np.float64)).as_matrix()
    transform[:3, 3] = np.asarray(vec[0:3], dtype=np.float64)
    return transform


def invert_pose(transform: np.ndarray) -> np.ndarray:
    """Inverse of a homogeneous transform, without a general 4x4 inversion."""
    rotation = transform[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


@dataclass(frozen=True)
class GripperMap:
    """UMI jaw width (policy output) <-> YAM ``gripper.pos`` command.

    Derived from data by matching two shared physical anchors between the UMI corpus and
    the YAM teleop corpus: jaws fully open, and jaws holding an orange (the same fruit in
    both, so the same physical jaw width). Per-arm because the two UMI devices have
    different maximum apertures (123.5 mm vs 118.1 mm) while agreeing on the orange to
    0.08 mm. See ``notes_gripper_mapping.md`` in the project notes for the derivation.

    Commanding the policy value directly (a = 1, b = 0) is NOT a safe default: at a grasp
    it is ~0.25 yam-units too open, the jaws travel only ~47% of the way, and they stop
    11-15 mm wider than the orange -- no contact at all, silently.

    The mm figures behind these constants are in the UMI device's frame, which carries a
    device offset. Confirm on hardware with a caliper sweep at commanded pos
    0.00/0.25/0.50/0.75/1.00 before trusting the absolute scale.
    """

    a_left: float = 2.2559
    b_left: float = -1.2290
    a_right: float = 2.7300
    b_right: float = -1.5781
    lo: float = 0.0
    hi: float = 1.0

    def umi_to_yam(self, value: float, *, left: bool) -> float:
        a, b = (self.a_left, self.b_left) if left else (self.a_right, self.b_right)
        return float(np.clip(a * float(value) + b, self.lo, self.hi))

    def yam_to_umi(self, value: float, *, left: bool) -> float:
        a, b = (self.a_left, self.b_left) if left else (self.a_right, self.b_right)
        return float((float(value) - b) / a)


# BiYAMConfig._default_joint_limits(): operational bounds, radians, per arm.
YAM_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.61799, 3.14159),
    (0.0, 3.66519),
    (0.0, 3.14159),
    (-1.69297, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)


@dataclass(frozen=True)
class SafetyLimits:
    """Rate and range caps applied to every commanded joint vector.

    The defaults are the values the YAM teleop corpus was itself generated under: its
    commands are hard-clipped at exactly 0.01500 rad/tick per joint and 0.05000/tick on
    the gripper. ``max_joint_delta`` keeps 33% headroom over the demonstrated clamp;
    ``max_gripper_delta`` matches it, because the stock 0.03 default throttles ~3.4% of
    ticks, nearly all of them at grasp and release edges, stretching a close from 0.33 s
    to 0.55 s.
    """

    max_joint_delta: float = 0.02  # rad per tick
    max_gripper_delta: float = 0.05  # normalized units per tick
    joint_lower: tuple[float, ...] = tuple(lo for lo, _ in YAM_JOINT_LIMITS)
    joint_upper: tuple[float, ...] = tuple(hi for _, hi in YAM_JOINT_LIMITS)

    def clamp_joints(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        limited = current + np.clip(target - current, -self.max_joint_delta, self.max_joint_delta)
        return np.clip(limited, np.asarray(self.joint_lower), np.asarray(self.joint_upper))

    def clamp_gripper(self, target: float, current: float) -> float:
        delta = np.clip(target - current, -self.max_gripper_delta, self.max_gripper_delta)
        return float(np.clip(current + delta, 0.0, 1.0))


YAM_URDF_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
YAM_FLANGE_FRAME = "gripper"
# The fingertip midpoint sits a constant 56.683 mm along the flange frame's local -Z
# (measured across 82 configurations, invariant to <1e-4 um). joint7/joint8 are the
# unactuated finger slides and are deliberately excluded from the IK chain, so targeting
# tip_left/tip_right directly is far less accurate than composing this offset.
YAM_FLANGE_TO_FINGERTIP: np.ndarray = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.056683], [0.0, 0.0, 0.0, 1.0]]
)


class IkResidualError(RuntimeError):
    """IK returned a configuration that does not reach the requested pose.

    placo returns the nearest feasible configuration for an unreachable target, with no
    exception and no flag, so the residual must be checked explicitly. On hardware this
    should stop the episode rather than command the nearest-but-wrong pose.
    """


class YamArmKinematics:
    """FK/IK for one YAM arm, in RADIANS, on top of `RobotKinematics` (which uses degrees).

    One instance per arm, used from one thread. The wrapped solver keeps mutable state (the
    placo model's configuration), and sharing an instance across arms or threads corrupts
    results — measured at 51.6% of calls wrong, deviating by up to 150 degrees, with 8
    threads on one instance.

    Two correctness details this class exists to encapsulate:

    * ``RobotKinematics.inverse_kinematics`` sets the seed joints but never calls
      ``update_kinematics()`` before solving, so the QP linearizes about whatever
      configuration was last evaluated — not the seed. Left to chance that produces a
      median position error of 163 mm (max 1040 mm), and it diverges, because each bad
      solution becomes the next linearization point. `ik` therefore always syncs the model
      to the seed first.
    * ``solve()`` performs a single Gauss-Newton step, which is not enough for a
      command-sized pose delta (max joint error 6.85 degrees at one iteration, 0.0009 at
      three). `iterations` defaults to 3, and chaining is safe because each solve ends
      with an ``update_kinematics()``.
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = YAM_FLANGE_FRAME,
        joint_names: list[str] | None = None,
        *,
        orientation_weight: float = 1.0,  # NOT the 0.01 default: joint6 is unobservable
        position_weight: float = 1.0,
        iterations: int = 3,
        max_position_residual: float = 2e-3,  # metres
        max_orientation_residual: float = np.deg2rad(1.0),
    ):
        joint_names = list(joint_names or YAM_URDF_JOINT_NAMES)
        self._kinematics = RobotKinematics(
            urdf_path=urdf_path, target_frame_name=target_frame_name, joint_names=joint_names
        )
        self.n_joints = len(joint_names)
        self.orientation_weight = orientation_weight
        self.position_weight = position_weight
        self.iterations = max(1, int(iterations))
        self.max_position_residual = max_position_residual
        self.max_orientation_residual = max_orientation_residual

    def _validate_joints(self, joints_rad: np.ndarray, what: str) -> np.ndarray:
        joints_rad = np.asarray(joints_rad, dtype=np.float64)
        if joints_rad.shape != (self.n_joints,):
            raise ValueError(f"expected {self.n_joints} {what}, got {joints_rad.shape}")
        if not np.isfinite(joints_rad).all():
            raise ValueError(f"non-finite {what}: {joints_rad}")
        # Degrees passed as radians is the dangerous unit error: the arm still sits a
        # plausible distance from the base at every frame, so only the magnitude reveals it.
        if np.abs(joints_rad).max() > 2 * np.pi:
            raise ValueError(
                f"{what} exceed 2*pi ({np.abs(joints_rad).max():.2f}) — these look like degrees; "
                "this API takes RADIANS"
            )
        return joints_rad

    def fk(self, joints_rad: np.ndarray) -> np.ndarray:
        """Joint vector (rad) -> 4x4 base->EE transform."""
        joints_rad = self._validate_joints(joints_rad, "joints")
        return np.asarray(self._kinematics.forward_kinematics(np.rad2deg(joints_rad)), dtype=np.float64)

    def ik(self, target_pose: np.ndarray, seed_rad: np.ndarray, *, check: bool = True) -> np.ndarray:
        """4x4 base->EE target + seed configuration (rad) -> joint vector (rad).

        Raises `IkResidualError` when the returned configuration does not reach the target
        within the configured residuals (i.e. the target was unreachable), unless
        ``check=False``.
        """
        seed_rad = self._validate_joints(seed_rad, "seed joints")
        target_pose = np.asarray(target_pose, dtype=np.float64)
        if target_pose.shape != (4, 4) or not np.isfinite(target_pose).all():
            raise ValueError("target pose must be a finite 4x4 transform")

        # Sync the model to the seed so the solver linearizes about it (see class docstring).
        self._kinematics.forward_kinematics(np.rad2deg(seed_rad))
        solution = seed_rad
        for _ in range(self.iterations):
            solution_deg = self._kinematics.inverse_kinematics(
                np.rad2deg(solution),
                target_pose,
                position_weight=self.position_weight,
                orientation_weight=self.orientation_weight,
            )
            solution = np.deg2rad(np.asarray(solution_deg, dtype=np.float64)[: self.n_joints])

        if check:
            position_error, orientation_error = self.residual(solution, target_pose)
            if (
                position_error > self.max_position_residual
                or orientation_error > self.max_orientation_residual
            ):
                raise IkResidualError(
                    f"IK did not reach the target: position error {position_error * 1e3:.2f} mm "
                    f"(limit {self.max_position_residual * 1e3:.2f} mm), orientation error "
                    f"{np.rad2deg(orientation_error):.2f} deg "
                    f"(limit {np.rad2deg(self.max_orientation_residual):.2f} deg). "
                    "The target is most likely outside the arm's reachable workspace."
                )
        return solution

    def residual(self, joints_rad: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
        """(position error in metres, orientation error in radians) of a solution."""
        reached = self.fk(joints_rad)
        position_error = float(np.linalg.norm(reached[:3, 3] - target_pose[:3, 3]))
        relative = reached[:3, :3].T @ target_pose[:3, :3]
        orientation_error = float(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()))
        return position_error, orientation_error


@dataclass
class EpisodeFrames:
    """Per-arm episode-start anchors, plus the constant tool-frame offset.

    The policy's poses are relative to where each arm started the episode. `capture`
    stores those anchors; `to_policy` and `from_policy` move between base-frame poses and
    the policy's frame.

    ``tool_offset`` is the constant transform from the YAM flange frame to the frame the
    policy was trained in (the UMI gripper's tool frame). Because the policy works in
    relative poses, this offset does not cancel -- it conjugates the relative motion:
    a pose that is ``R`` in the UMI frame is ``C^-1 R C`` in the flange frame. Identity is
    correct only if the two tool frames coincide.
    """

    tool_offset: np.ndarray = field(default_factory=lambda: np.eye(4))
    _anchors: dict[str, np.ndarray] = field(default_factory=dict)

    def capture(self, arm: str, base_to_flange: np.ndarray) -> None:
        self._anchors[arm] = np.asarray(base_to_flange, dtype=np.float64) @ self.tool_offset

    @property
    def captured(self) -> bool:
        return {"left", "right"} <= self._anchors.keys()

    def to_policy(self, arm: str, base_to_flange: np.ndarray) -> np.ndarray:
        """base->flange transform -> policy pose vector (6,) in the episode-start frame."""
        anchor = self._require(arm)
        tool = np.asarray(base_to_flange, dtype=np.float64) @ self.tool_offset
        return pose_to_vec(invert_pose(anchor) @ tool)

    def from_policy(self, arm: str, pose_vec: np.ndarray) -> np.ndarray:
        """Policy pose vector (6,) -> base->flange transform."""
        anchor = self._require(arm)
        tool = anchor @ vec_to_pose(np.asarray(pose_vec, dtype=np.float64))
        return tool @ invert_pose(self.tool_offset)

    def _require(self, arm: str) -> np.ndarray:
        if arm not in self._anchors:
            raise RuntimeError(
                f"no episode anchor for {arm!r}; call capture() at the episode start "
                "(and again after every client.reset())"
            )
        return self._anchors[arm]


@dataclass(frozen=True)
class TrackingPolicy:
    """What to do when the policy asks for a pose this arm cannot reach.

    ``strict`` refuses (raises `IkResidualError`) — right for bring-up, because it makes an
    embodiment mismatch impossible to miss.

    ``best_effort`` tracks the target as closely as the arm allows, prioritizing position
    over orientation, and records the shortfall in ``YamUmiEeAdapter.last_residual``. This
    mirrors what a Cartesian impedance controller does in the UMI deployments, where
    orientation is held ~50x more softly than position (translational stiffness 750 vs
    rotational 15 in their Franka controller), so an unreachable wrist angle bleeds off as
    tracking error instead of stopping the episode.

    Neither mode invents workspace. If a policy trained in another embodiment's frame asks
    for motions outside this arm's reachable set, best_effort converts hard failures into
    lag and silent inaccuracy — useful for getting a rollout to run and to see, dangerous
    to mistake for success. Watch ``last_residual``.
    """

    mode: str = "strict"  # "strict" | "best_effort"
    best_effort_orientation_weight: float = 0.02
    # Roll about the gripper's approach axis is a RANGED goal, not a fixed one: a parallel
    # jaw is symmetric under a 180 deg roll, and for a rotationally symmetric object it does
    # not matter at all. Both YAM arms are 6-DoF driving a 6-DoF task, so (I - J+J) = 0 and
    # every classical joint-limit-avoidance method is inert -- there is no null space to
    # exploit. Declaring roll a tolerance band manufactures the missing degree of freedom
    # for free (RangedIK, Wang et al., ICRA 2023). Unlike clamping an infeasible command,
    # this does not distort the task: it picks a different but physically equivalent grasp.
    # Set to np.pi for a symmetric jaw; 0.0 disables it.
    roll_tolerance: float = 0.0
    roll_samples: int = 13

    def __post_init__(self):
        if self.mode not in {"strict", "best_effort"}:
            raise ValueError(f"unknown tracking mode {self.mode!r}")
        if self.roll_tolerance < 0 or self.roll_samples < 1:
            raise ValueError("roll_tolerance must be >= 0 and roll_samples >= 1")


@dataclass
class YamUmiEeAdapter:
    """Translate between `BiYAM`'s 14 joint scalars and a UMI EE-pose policy's 14-D vectors.

    Holds the per-arm solvers, the episode anchors, the gripper map, and the safety caps.
    The caller owns the robot and the policy client; this object owns only the conversion.
    """

    left: YamArmKinematics
    right: YamArmKinematics
    frames: EpisodeFrames = field(default_factory=EpisodeFrames)
    gripper: GripperMap = field(default_factory=GripperMap)
    limits: SafetyLimits = field(default_factory=SafetyLimits)
    tracking: TrackingPolicy = field(default_factory=TrackingPolicy)
    last_residual: dict[str, tuple[float, float]] = field(default_factory=dict)

    def _roll_variants(self, target: np.ndarray) -> list[np.ndarray]:
        """Targets equivalent to ``target`` up to a roll about the gripper's approach axis."""
        tolerance = self.tracking.roll_tolerance
        if tolerance <= 0:
            return [target]
        angles = np.linspace(-tolerance, tolerance, self.tracking.roll_samples)
        variants = []
        for angle in angles:
            roll = np.eye(4)
            roll[:3, :3] = Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()
            variants.append(target @ roll)  # roll in the TOOL frame, so the axis moves with it
        return variants

    def _solve(self, kinematics: YamArmKinematics, target: np.ndarray, seed: np.ndarray, arm: str):
        """IK under the configured tracking policy, recording the residual either way.

        With a roll tolerance, every physically equivalent roll of the target is tried and
        the one the arm reaches best is kept, scored on position error first (roll is what
        we agreed not to care about) and joint motion second.
        """
        strict = self.tracking.mode == "strict"
        original = kinematics.orientation_weight
        if not strict:
            kinematics.orientation_weight = self.tracking.best_effort_orientation_weight
        try:
            best = None
            for candidate in self._roll_variants(target):
                solution = kinematics.ik(candidate, seed, check=False)
                position_error, orientation_error = kinematics.residual(solution, candidate)
                score = (position_error, float(np.abs(solution - seed).max()))
                if best is None or score < best[0]:
                    best = (score, solution, position_error, orientation_error)
        finally:
            kinematics.orientation_weight = original

        _, solution, position_error, orientation_error = best
        self.last_residual[arm] = (position_error, orientation_error)
        if strict and (
            position_error > kinematics.max_position_residual
            or orientation_error > kinematics.max_orientation_residual
        ):
            raise IkResidualError(
                f"[{arm}] IK did not reach the target: position error {position_error * 1e3:.2f} mm, "
                f"orientation error {np.rad2deg(orientation_error):.2f} deg. The target is most "
                "likely outside this arm's reachable workspace."
            )
        return solution

    @staticmethod
    def joints_from_observation(observation: dict) -> tuple[np.ndarray, np.ndarray, float, float]:
        """BiYAM observation dict -> (left joints rad, right joints rad, left grip, right grip)."""
        missing = [key for key in YAM_SCALAR_KEYS if key not in observation]
        if missing:
            raise KeyError(f"observation is missing YAM scalars: {missing}")
        left = np.array([float(observation[k]) for k in LEFT_JOINT_KEYS], dtype=np.float64)
        right = np.array([float(observation[k]) for k in RIGHT_JOINT_KEYS], dtype=np.float64)
        return left, right, float(observation[LEFT_GRIPPER_KEY]), float(observation[RIGHT_GRIPPER_KEY])

    def capture_episode_start(self, observation: dict) -> None:
        """Anchor both arms at the current pose. Call once per episode, with client.reset()."""
        left_q, right_q, _, _ = self.joints_from_observation(observation)
        self.frames.capture("left", self.left.fk(left_q))
        self.frames.capture("right", self.right.fk(right_q))

    def observation_to_policy_state(self, observation: dict) -> np.ndarray:
        """BiYAM observation -> the policy's 14-D state vector (float32)."""
        left_q, right_q, left_grip, right_grip = self.joints_from_observation(observation)
        state = np.empty(14, dtype=np.float64)
        state[LEFT_POSE_SLICE] = self.frames.to_policy("left", self.left.fk(left_q))
        state[LEFT_GRIPPER_INDEX] = self.gripper.yam_to_umi(left_grip, left=True)
        state[RIGHT_POSE_SLICE] = self.frames.to_policy("right", self.right.fk(right_q))
        state[RIGHT_GRIPPER_INDEX] = self.gripper.yam_to_umi(right_grip, left=False)
        return state.astype(np.float32)

    def action_row_to_joint_command(self, row: np.ndarray, observation: dict) -> dict[str, float]:
        """One policy action row (14,) + the current observation -> a BiYAM action dict.

        ``observation`` supplies both the IK seed and the reference for rate limiting, so
        pass the most recent one — rate limits are meaningful only against the present pose.
        """
        row = np.asarray(row, dtype=np.float64)
        if row.shape != (14,):
            raise ValueError(f"expected a (14,) action row, got {row.shape}")
        if not np.isfinite(row).all():
            raise ValueError("policy returned a non-finite action row")

        left_q, right_q, left_grip, right_grip = self.joints_from_observation(observation)
        left_target = self._solve(
            self.left, self.frames.from_policy("left", row[LEFT_POSE_SLICE]), left_q, "left"
        )
        right_target = self._solve(
            self.right, self.frames.from_policy("right", row[RIGHT_POSE_SLICE]), right_q, "right"
        )

        left_safe = self.limits.clamp_joints(left_target, left_q)
        right_safe = self.limits.clamp_joints(right_target, right_q)
        left_grip_cmd = self.limits.clamp_gripper(
            self.gripper.umi_to_yam(row[LEFT_GRIPPER_INDEX], left=True), left_grip
        )
        right_grip_cmd = self.limits.clamp_gripper(
            self.gripper.umi_to_yam(row[RIGHT_GRIPPER_INDEX], left=False), right_grip
        )

        command = {key: float(value) for key, value in zip(LEFT_JOINT_KEYS, left_safe, strict=True)}
        command[LEFT_GRIPPER_KEY] = left_grip_cmd
        command.update({key: float(value) for key, value in zip(RIGHT_JOINT_KEYS, right_safe, strict=True)})
        command[RIGHT_GRIPPER_KEY] = right_grip_cmd
        return command


# --------------------------------------------------------------------------------------
# Offline self-test: the frame algebra and gripper map, with no robot and no policy server
# --------------------------------------------------------------------------------------


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def _test_pose_algebra(rng: np.random.Generator) -> bool:
    """vec<->pose and invert_pose are exact inverses over random poses."""
    worst_vec, worst_inv = 0.0, 0.0
    for _ in range(2000):
        vec = np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-np.pi, np.pi, 3)])
        if np.linalg.norm(vec[3:]) > np.pi:  # rotvec beyond pi aliases; skip
            continue
        transform = vec_to_pose(vec)
        worst_vec = max(worst_vec, float(np.abs(pose_to_vec(transform) - vec).max()))
        identity = invert_pose(transform) @ transform
        worst_inv = max(worst_inv, float(np.abs(identity - np.eye(4)).max()))
    return _check(
        "pose algebra round-trips",
        worst_vec < 1e-9 and worst_inv < 1e-9,
        f"vec {worst_vec:.2e}, inverse {worst_inv:.2e}",
    )


def _test_frames(rng: np.random.Generator) -> bool:
    """to_policy and from_policy invert each other, with and without a tool offset."""
    results = []
    for label, offset in (
        ("identity offset", np.eye(4)),
        ("rotated+translated offset", vec_to_pose(np.array([0.03, -0.01, 0.12, 0.4, -0.2, 1.1]))),
    ):
        frames = EpisodeFrames(tool_offset=offset)
        anchor = vec_to_pose(np.array([0.3, 0.1, 0.4, 0.2, -0.3, 0.5]))
        frames.capture("left", anchor)
        worst = 0.0
        for _ in range(500):
            pose = vec_to_pose(np.concatenate([rng.uniform(-0.5, 0.5, 3), rng.uniform(-1, 1, 3)]))
            recovered = frames.from_policy("left", frames.to_policy("left", pose))
            worst = max(worst, float(np.abs(recovered - pose).max()))
        results.append(_check(f"episode frames round-trip ({label})", worst < 1e-9, f"max {worst:.2e}"))

    # at the anchor itself the policy pose must be exactly the origin
    frames = EpisodeFrames(tool_offset=vec_to_pose(np.array([0.0, 0.0, 0.1, 0.3, 0.0, 0.0])))
    anchor = vec_to_pose(np.array([0.2, 0.2, 0.3, 0.1, 0.1, 0.1]))
    frames.capture("right", anchor)
    at_start = frames.to_policy("right", anchor)
    results.append(
        _check(
            "anchor maps to the origin",
            float(np.abs(at_start).max()) < 1e-9,
            f"max |pose| {float(np.abs(at_start).max()):.2e}",
        )
    )

    # a tool offset must NOT cancel out: relative motion is conjugated by it
    rotated = EpisodeFrames(tool_offset=vec_to_pose(np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])))
    plain = EpisodeFrames()
    for frames_obj in (rotated, plain):
        frames_obj.capture("left", np.eye(4))
    moved = vec_to_pose(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))  # pure +x translation
    difference = float(np.abs(rotated.to_policy("left", moved) - plain.to_policy("left", moved)).max())
    results.append(
        _check(
            "tool offset changes the policy frame (does not cancel)",
            difference > 0.05,
            f"difference {difference:.3f} (a 90deg tool yaw turns +x into +/-y)",
        )
    )
    return all(results)


def _test_gripper_map() -> bool:
    """The map reproduces its anchors, inverts cleanly, and beats the identity map."""
    gripper = GripperMap()
    results = [
        _check(
            "open anchor -> YAM open (left)",
            abs(gripper.umi_to_yam(0.9880, left=True) - 1.0) < 0.02,
            f"{gripper.umi_to_yam(0.9880, left=True):.4f}",
        ),
        _check(
            "open anchor -> YAM open (right)",
            abs(gripper.umi_to_yam(0.9444, left=False) - 1.0) < 0.02,
            f"{gripper.umi_to_yam(0.9444, left=False):.4f}",
        ),
        _check(
            "grasp anchor -> YAM grasp (left)",
            abs(gripper.umi_to_yam(0.7690, left=True) - 0.5056) < 0.02,
            f"{gripper.umi_to_yam(0.7690, left=True):.4f} vs 0.5056",
        ),
        _check(
            "grasp anchor -> YAM grasp (right)",
            abs(gripper.umi_to_yam(0.7690, left=False) - 0.5217) < 0.02,
            f"{gripper.umi_to_yam(0.7690, left=False):.4f} vs 0.5217",
        ),
    ]
    # Invertible only inside each arm's linear region: the map saturates once a*u+b leaves
    # [0, 1], which is intended (a UMI reading wider than the YAM jaws can open must clamp).
    worst = 0.0
    saturation = {}
    for left in (True, False):
        a, b = (gripper.a_left, gripper.b_left) if left else (gripper.a_right, gripper.b_right)
        lo, hi = -b / a, (1.0 - b) / a
        saturation["left" if left else "right"] = (lo, hi)
        for value in np.linspace(lo + 1e-6, hi - 1e-6, 40):
            worst = max(
                worst, abs(gripper.yam_to_umi(gripper.umi_to_yam(value, left=left), left=left) - value)
            )
    results.append(
        _check(
            "gripper map inverts inside its linear region",
            worst < 1e-9,
            f"max {worst:.2e}; saturates outside "
            f"L[{saturation['left'][0]:.4f},{saturation['left'][1]:.4f}] "
            f"R[{saturation['right'][0]:.4f},{saturation['right'][1]:.4f}]",
        )
    )
    results.append(
        _check(
            "map saturates rather than exceeding the YAM range",
            gripper.umi_to_yam(1.0, left=True) == 1.0
            and gripper.umi_to_yam(1.0, left=False) == 1.0
            and gripper.umi_to_yam(0.0, left=True) == 0.0,
        )
    )
    results.append(
        _check(
            "output stays in [0,1]",
            all(
                0.0 <= gripper.umi_to_yam(value, left=is_left) <= 1.0
                for value in np.linspace(0, 1, 101)
                for is_left in (True, False)
            ),
        )
    )
    identity_error = abs(0.7690 - 0.5056)
    mapped_error = abs(gripper.umi_to_yam(0.7690, left=True) - 0.5056)
    results.append(
        _check(
            "beats the identity map at a grasp",
            mapped_error < identity_error / 5,
            f"identity off by {identity_error:.3f}, mapped off by {mapped_error:.3f}",
        )
    )
    return all(results)


def _test_safety_limits(rng: np.random.Generator) -> bool:
    """Rate caps bind, joint limits bind, and a legal small step is untouched."""
    limits = SafetyLimits()
    current = np.zeros(6)
    far = np.full(6, 3.0)
    stepped = limits.clamp_joints(far, current)
    results = [
        _check(
            "joint rate cap binds",
            np.allclose(stepped, np.full(6, limits.max_joint_delta)),
            f"max step {float(np.abs(stepped - current).max()):.4f} rad",
        ),
        _check(
            "small legal step passes through",
            np.allclose(limits.clamp_joints(current + 0.005, current), current + 0.005),
        ),
    ]
    # a target outside the operational bounds is pulled inside
    low = np.array([lo for lo, _ in YAM_JOINT_LIMITS])
    out = limits.clamp_joints(low - 1.0, low)
    results.append(
        _check(
            "joint bounds bind",
            bool((out >= low - 1e-12).all()),
            f"min margin {float((out - low).min()):.4f} rad",
        )
    )
    results.append(
        _check(
            "gripper rate cap binds", abs(limits.clamp_gripper(1.0, 0.0) - limits.max_gripper_delta) < 1e-12
        )
    )
    results.append(
        _check(
            "gripper stays in [0,1]",
            all(
                0.0 <= limits.clamp_gripper(t, c) <= 1.0 for t in (-5, 0, 0.5, 1, 5) for c in (0.0, 0.5, 1.0)
            ),
        )
    )
    del rng
    return all(results)


def _load_yam_episode(data_root: str, repo_id: str, episode: int) -> np.ndarray:
    """Absolute 14-D joint states of one YAM episode, radians, shape (T, 14)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=repo_id, root=data_root, episodes=[episode], image_transforms=None, video_backend="pyav"
    )
    return np.stack([dataset[i]["observation.state"].numpy() for i in range(len(dataset))]).astype(np.float64)


def _observation_from_state(state: np.ndarray) -> dict[str, float]:
    """A 14-D YAM state vector -> the observation dict BiYAM would return (scalars only)."""
    return {key: float(value) for key, value in zip(YAM_SCALAR_KEYS, state, strict=True)}


def _test_kinematics_on_real_data(adapter: YamUmiEeAdapter, states: np.ndarray) -> bool:
    """FK/IK round-trip and full observation->policy->command loop on recorded joint data.

    This is the test that matters: it runs the exact conversion the deployment will run,
    on trajectories a real YAM actually executed, and asks whether the joints that come
    out the far end match the joints that went in.
    """
    results = []
    left_states = states[:, :6]

    # 1. FK -> IK returns the original configuration (seeded at the true answer).
    pose_errors, joint_errors = [], []
    for q in left_states[::10]:
        pose = adapter.left.fk(q)
        solved = adapter.left.ik(pose, q)
        pose_errors.append(float(np.linalg.norm(adapter.left.fk(solved)[:3, 3] - pose[:3, 3])))
        joint_errors.append(float(np.abs(solved - q).max()))
    results.append(
        _check(
            "FK->IK recovers the configuration (seed = truth)",
            float(np.percentile(pose_errors, 99)) < 2e-3,
            f"pos err p50 {np.percentile(pose_errors, 50) * 1e3:.3f} mm, "
            f"p99 {np.percentile(pose_errors, 99) * 1e3:.3f} mm; joint p99 "
            f"{np.rad2deg(np.percentile(joint_errors, 99)):.3f} deg",
        )
    )

    # 2. Control-loop round trip: seed at q_t, target the pose of q_{t+1}.
    step_errors = []
    for index in range(0, len(left_states) - 1, 5):
        target = adapter.left.fk(left_states[index + 1])
        solved = adapter.left.ik(target, left_states[index])
        step_errors.append(float(np.abs(solved - left_states[index + 1]).max()))
    results.append(
        _check(
            "control-loop round trip (seed = previous tick)",
            float(np.percentile(step_errors, 99)) < np.deg2rad(2.0),
            f"joint err p50 {np.rad2deg(np.percentile(step_errors, 50)):.4f} deg, "
            f"p99 {np.rad2deg(np.percentile(step_errors, 99)):.4f} deg, "
            f"max {np.rad2deg(max(step_errors)):.4f} deg",
        )
    )

    # 3. Whole adapter: observation -> policy state -> (identity policy) -> joint command.
    #    Feeding the state straight back as the action must reproduce the same joints.
    adapter.capture_episode_start(_observation_from_state(states[0]))
    identity_errors, gripper_errors = [], []
    for index in range(0, min(len(states), 400), 7):
        observation = _observation_from_state(states[index])
        policy_state = adapter.observation_to_policy_state(observation)
        command = adapter.action_row_to_joint_command(policy_state, observation)
        commanded = np.array([command[key] for key in YAM_SCALAR_KEYS])
        identity_errors.append(float(np.abs(commanded[:6] - states[index][:6]).max()))
        gripper_errors.append(abs(commanded[6] - states[index][6]))
    results.append(
        _check(
            "adapter is identity when the policy echoes the state",
            float(np.max(identity_errors)) < np.deg2rad(1.0) and float(np.max(gripper_errors)) < 0.02,
            f"joint max {np.rad2deg(np.max(identity_errors)):.4f} deg, "
            f"gripper max {np.max(gripper_errors):.4f}",
        )
    )

    # 4. Commands stay inside the operational envelope.
    adapter.capture_episode_start(_observation_from_state(states[0]))
    in_bounds = True
    for index in range(0, min(len(states), 200), 3):
        observation = _observation_from_state(states[index])
        command = adapter.action_row_to_joint_command(
            adapter.observation_to_policy_state(observation), observation
        )
        joints = np.array([command[key] for key in LEFT_JOINT_KEYS])
        lower = np.array([lo for lo, _ in YAM_JOINT_LIMITS])
        upper = np.array([hi for _, hi in YAM_JOINT_LIMITS])
        in_bounds &= bool((joints >= lower - 1e-9).all() and (joints <= upper + 1e-9).all())
    results.append(_check("commands respect operational joint bounds", in_bounds))
    return all(results)


def _test_ik_is_seed_synced(adapter: YamUmiEeAdapter, states: np.ndarray) -> bool:
    """IK must be correct even when the solver's internal state points somewhere else.

    `RobotKinematics.inverse_kinematics` linearizes about the last configuration the model
    evaluated, not about the seed it is handed. Code that happens to call FK on the seed
    beforehand works by luck; this test deliberately poisons the internal state first, so
    only a self-syncing implementation passes.
    """
    left_states = states[:, :6]
    poisoned_errors, clean_errors = [], []
    for index in range(0, min(len(left_states) - 1, 600), 5):
        seed = left_states[index]
        target = adapter.left.fk(left_states[index + 1])

        # poison: evaluate an unrelated configuration so the cached linearization is wrong
        adapter.left.fk(left_states[(index + 137) % len(left_states)])
        poisoned = adapter.left.ik(target, seed, check=False)
        poisoned_errors.append(adapter.left.residual(poisoned, target)[0])

        adapter.left.fk(seed)
        clean_errors.append(adapter.left.residual(adapter.left.ik(target, seed, check=False), target)[0])

    worst_poisoned = float(np.max(poisoned_errors)) if poisoned_errors else 0.0
    worst_clean = float(np.max(clean_errors)) if clean_errors else 0.0
    return _check(
        "IK is immune to a poisoned solver state",
        worst_poisoned < 1e-3,
        f"max position error {worst_poisoned * 1e3:.4f} mm poisoned vs {worst_clean * 1e3:.4f} mm pre-synced",
    )


def _test_ik_rejects_unreachable(adapter: YamUmiEeAdapter, states: np.ndarray) -> bool:
    """An out-of-workspace target must raise rather than return the nearest pose."""
    seed = states[0, :6]
    far = adapter.left.fk(seed)
    far[:3, 3] = far[:3, 3] + np.array([0.0, 0.0, 1.5])  # 1.5 m above; unreachable
    try:
        adapter.left.ik(far, seed)
    except IkResidualError:
        return _check("unreachable target raises IkResidualError", True)
    return _check("unreachable target raises IkResidualError", False, "no exception raised")


def _tool_offset_from_args(args) -> np.ndarray | None:
    """--tool_offset RX RY RZ (radians) -> a 4x4 rotation, or None for identity."""
    if getattr(args, "tool_offset", None) is None:
        return None
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_rotvec(np.asarray(args.tool_offset, dtype=np.float64)).as_matrix()
    return offset


def _build_adapter(
    urdf: str,
    *,
    tool_offset: np.ndarray | None = None,
    tracking: str = "strict",
    roll_tolerance: float = 0.0,
) -> YamUmiEeAdapter:
    frames = EpisodeFrames(tool_offset=np.eye(4) if tool_offset is None else tool_offset)
    return YamUmiEeAdapter(
        left=YamArmKinematics(urdf),
        right=YamArmKinematics(urdf),
        frames=frames,
        tracking=TrackingPolicy(mode=tracking, roll_tolerance=roll_tolerance),
    )


def _self_test(args) -> int:
    rng = np.random.default_rng(0)
    print("=== offline math checks (no robot, no policy server) ===")
    results = [_test_pose_algebra(rng), _test_frames(rng), _test_gripper_map(), _test_safety_limits(rng)]

    if args.urdf and args.yam_dataset and args.repo_id:
        print("\n=== kinematics on recorded YAM trajectories ===")
        adapter = _build_adapter(args.urdf)
        states = _load_yam_episode(args.yam_dataset, args.repo_id, args.episode)
        print(f"episode {args.episode}: {len(states)} frames")
        results.append(_test_kinematics_on_real_data(adapter, states))
        results.append(_test_ik_is_seed_synced(adapter, states))
        results.append(_test_ik_rejects_unreachable(adapter, states))
    else:
        print("\n(skipping kinematics tests: pass --urdf, --yam_dataset and --repo_id to run them)")

    ok = all(results)
    print("\nSELF_TEST_OK" if ok else "\nSELF_TEST_FAILED")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# Closed-loop replay: live policy server + a recorded YAM episode standing in for the robot
# --------------------------------------------------------------------------------------


def _replay(args) -> int:
    """Run the real control loop with a recorded episode in place of the hardware.

    The policy server is real, the client is real, the kinematics are real; only the robot
    is simulated, by replaying recorded joint states and applying the commands to a
    first-order model of a position-controlled arm. This exercises every conversion the
    hardware path uses and is the last check before a rig.
    """
    from .umi_ee_client import UmiEeClientConfig, UmiEeRemoteClient

    adapter = _build_adapter(args.urdf, tool_offset=_tool_offset_from_args(args), tracking=args.tracking)
    states = _load_yam_episode(args.yam_dataset, args.repo_id, args.episode)

    # camera frames come from the UMI dataset: the policy needs images from its own domain
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=args.umi_repo_id, root=args.umi_dataset)
    image_keys = [key for key in meta.features if key.startswith("observation.images.")]
    camera_keys = tuple(key.removeprefix("observation.images.") for key in image_keys)
    height, width = (int(v) for v in meta.features[image_keys[0]]["shape"][:2])
    umi = LeRobotDataset(
        repo_id=args.umi_repo_id,
        root=args.umi_dataset,
        episodes=[args.episode],
        image_transforms=None,
        video_backend="pyav",
    )

    client = UmiEeRemoteClient(
        UmiEeClientConfig(
            server_address=args.server,
            task=args.task,
            camera_keys=camera_keys,
            image_width=width,
            image_height=height,
            control_hz=float(meta.fps),
            feature_names=tuple(meta.features["observation.state"]["names"]),
        )
    )
    session = client.connect()
    print(f"session open: horizon={session.model.action_horizon} dim={session.model.action_dim}")

    from .umi_ee_client import _to_hwc_uint8

    joints = states[0].copy()  # the "robot": current measured position
    adapter.capture_episode_start(_observation_from_state(joints))
    executed, ik_failures = [], 0
    try:
        for tick in range(0, min(args.ticks, len(umi))):
            observation = _observation_from_state(joints)
            item = umi[tick]
            images = {
                key: _to_hwc_uint8(item[image_key])
                for key, image_key in zip(camera_keys, image_keys, strict=True)
            }
            chunk = client.predict(adapter.observation_to_policy_state(observation), images, tick)

            for row in chunk[: args.execution_horizon]:
                try:
                    command = adapter.action_row_to_joint_command(row, _observation_from_state(joints))
                except IkResidualError:
                    ik_failures += 1
                    break
                commanded = np.array([command[key] for key in YAM_SCALAR_KEYS])
                # first-order position-controlled arm: it reaches what it was told
                joints = commanded
                executed.append(commanded.copy())
    finally:
        client.close()

    executed_array = np.array(executed)
    steps = np.abs(np.diff(executed_array[:, :6], axis=0)).max() if len(executed_array) > 1 else 0.0
    lower = np.array([lo for lo, _ in YAM_JOINT_LIMITS])
    upper = np.array([hi for _, hi in YAM_JOINT_LIMITS])
    in_bounds = bool(
        (executed_array[:, :6] >= lower - 1e-9).all() and (executed_array[:, :6] <= upper + 1e-9).all()
    )
    grip = executed_array[:, LEFT_GRIPPER_INDEX]
    ok = (
        len(executed_array) > 0
        and np.isfinite(executed_array).all()
        and in_bounds
        and steps <= adapter.limits.max_joint_delta + 1e-9
        and ik_failures == 0
    )
    print(
        f"\nexecuted {len(executed_array)} commands | max joint step {np.rad2deg(steps):.3f} deg "
        f"(cap {np.rad2deg(adapter.limits.max_joint_delta):.3f}) | in bounds {in_bounds} | "
        f"IK failures {ik_failures} | left gripper range [{grip.min():.3f}, {grip.max():.3f}]"
    )
    if ik_failures:
        print(
            f"\n{ik_failures} chunk(s) stopped early because the policy asked for a pose this arm "
            "cannot reach. That is the safety check doing its job, not a transport fault — but it "
            "means the policy is not executable on this arm as configured. Run --calibrate to find "
            "the tool-frame rotation, and if the best rotation still leaves targets unreachable, "
            "the embodiment gap needs data or fine-tuning in the robot's own frame."
        )
    print("REPLAY_OK" if ok else "REPLAY_FAILED")
    return 0 if ok else 1


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="offline math + kinematics checks")
    parser.add_argument("--replay", action="store_true", help="closed loop against a live server")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="search tool-frame rotations and report how much of the policy this arm can reach",
    )
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--yam_dataset", default=None)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--umi_dataset", default=None)
    parser.add_argument("--umi_repo_id", default=None)
    parser.add_argument("--server", default=None)
    parser.add_argument("--task", default="Put all oranges in the bowl")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--execution_horizon", type=int, default=15)
    parser.add_argument(
        "--tracking",
        choices=("strict", "best_effort"),
        default="strict",
        help="strict refuses unreachable poses; best_effort tracks them position-first and "
        "reports the shortfall",
    )
    parser.add_argument("--stride", type=int, default=20, help="--calibrate: pose sampling stride")
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=None, help="--calibrate: episodes to fit on"
    )
    parser.add_argument(
        "--holdout_episodes",
        type=int,
        nargs="+",
        default=None,
        help="--calibrate: episodes to score the fitted rotation on (believe THIS number)",
    )
    parser.add_argument(
        "--roll_tolerance",
        type=float,
        default=0.0,
        help="radians of free roll about the gripper axis (use pi for a symmetric jaw)",
    )
    parser.add_argument("--n_random", type=int, default=120, help="--calibrate: random rotations tried")
    parser.add_argument(
        "--tool_offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="flange->tool rotation vector in radians (from --calibrate, or measured)",
    )
    args = parser.parse_args()

    if args.calibrate:
        missing = [n for n in ("urdf", "yam_dataset", "repo_id", "umi_dataset") if getattr(args, n) is None]
        if missing:
            parser.error(f"--calibrate needs {', '.join('--' + m for m in missing)}")
        raise SystemExit(_calibrate(args))
    if args.replay:
        missing = [
            n
            for n in ("server", "urdf", "yam_dataset", "repo_id", "umi_dataset", "umi_repo_id")
            if getattr(args, n) is None
        ]
        if missing:
            parser.error(f"--replay needs {', '.join('--' + m for m in missing)}")
        raise SystemExit(_replay(args))
    raise SystemExit(_self_test(args))


# --------------------------------------------------------------------------------------
# Tool-frame calibration: which constant flange->tool rotation makes the policy executable?
# --------------------------------------------------------------------------------------


def estimate_tool_offset(
    kinematics: YamArmKinematics,
    start_joints: np.ndarray,
    policy_poses: np.ndarray,
    *,
    n_random: int = 120,
    seed: int = 0,
    tolerance: float = 2e-3,
) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """Search constant tool rotations for the one that makes the most targets reachable.

    The policy's poses arrive in the frame of the gripper it was trained on. If that frame
    is rotated relative to the YAM flange, every relative pose arrives conjugated by the
    unknown constant, and the arm is asked to move in the wrong directions. This searches
    the 24 axis-aligned mount orientations plus random rotations and scores each by the
    fraction of ``policy_poses`` the arm can actually reach from ``start_joints``.

    Scoring uses position-only IK deliberately: it is the most permissive test, so a pose
    that fails here is unreachable under any orientation policy.

    Returns ``(best_rotation_vector, ranked_results)``. This is an estimate from data, not
    a substitute for measuring the physical mount — validate it before trusting it.
    """
    frames = EpisodeFrames()
    original_weight = kinematics.orientation_weight
    kinematics.orientation_weight = 0.0
    rng = np.random.default_rng(seed)

    candidates: list[tuple[str, np.ndarray]] = [("identity", np.eye(3))]
    for index, rotation in enumerate(Rotation.create_group("O")):
        candidates.append((f"axis-aligned[{index}]", rotation.as_matrix()))
    for index in range(n_random):
        candidates.append(
            (f"random[{index}]", Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix())
        )

    start_pose = kinematics.fk(start_joints)
    results: list[tuple[float, float, str]] = []
    best_matrix = np.eye(3)
    best_score = -1.0
    try:
        for name, matrix in candidates:
            offset = np.eye(4)
            offset[:3, :3] = matrix
            frames.tool_offset = offset
            frames._anchors = {}
            frames.capture("left", start_pose)
            residuals = []
            for row in policy_poses:
                target = frames.from_policy("left", row)
                residuals.append(
                    kinematics.residual(kinematics.ik(target, start_joints, check=False), target)[0]
                )
            residuals = np.array(residuals)
            reachable = float((residuals <= tolerance).mean())
            results.append((reachable, float(np.median(residuals)), name))
            if reachable > best_score:
                best_score, best_matrix = reachable, matrix
    finally:
        kinematics.orientation_weight = original_weight

    results.sort(reverse=True)
    return Rotation.from_matrix(best_matrix).as_rotvec(), results


def _calibrate(args) -> int:
    """Report how executable a policy's pose distribution is on this arm, and the best frame.

    Fits on one set of episodes and reports the score on a HELD-OUT set. A rotation fitted
    to a single episode overfits badly -- measured here, the single-episode winner scored
    worse than no calibration at all on other episodes -- so the held-out column, not the
    fit column, is the number to believe.
    """
    import pyarrow.parquet as pq

    def poses_for(episodes: list[int]) -> np.ndarray:
        chunks = []
        for episode in episodes:
            table = pq.read_table(f"{args.umi_dataset}/data/chunk-000/file-{episode:03d}.parquet")
            states = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
            chunks.append(states[:: args.stride][:80, LEFT_POSE_SLICE])
        return np.concatenate(chunks)

    kinematics = YamArmKinematics(args.urdf)
    yam = _load_yam_episode(args.yam_dataset, args.repo_id, args.episode)
    start_joints = yam[0, :6]
    fit_episodes = args.episodes or [args.episode]
    holdout_episodes = args.holdout_episodes or []

    fit_poses = poses_for(fit_episodes)
    rotvec, results = estimate_tool_offset(kinematics, start_joints, fit_poses, n_random=args.n_random)
    identity = next(r for r in results if r[2] == "identity")
    print(f"fit on episodes {fit_episodes}: {len(fit_poses)} poses (position-only IK, 2 mm tolerance)\n")
    print("best 5 tool rotations on the fit set:")
    for reachable, median_residual, name in results[:5]:
        print(
            f"  {name:<18} reachable {100 * reachable:5.1f}%  median residual {median_residual * 1e3:7.2f} mm"
        )
    print(f"\nidentity (no calibration): {100 * identity[0]:.1f}% reachable")
    print(f"best on fit set:           {100 * results[0][0]:.1f}% reachable")

    if holdout_episodes:
        holdout = poses_for(holdout_episodes)
        best_offset = np.eye(4)
        best_offset[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
        scores = {}
        for label, offset in (("identity", np.eye(4)), ("fitted", best_offset)):
            frames = EpisodeFrames(tool_offset=offset)
            frames.capture("left", kinematics.fk(start_joints))
            weight = kinematics.orientation_weight
            kinematics.orientation_weight = 0.0
            try:
                residuals = [
                    kinematics.residual(
                        kinematics.ik(frames.from_policy("left", row), start_joints, check=False),
                        frames.from_policy("left", row),
                    )[0]
                    for row in holdout
                ]
            finally:
                kinematics.orientation_weight = weight
            scores[label] = float((np.array(residuals) <= 2e-3).mean())
        print(f"\nHELD-OUT episodes {holdout_episodes} ({len(holdout)} poses):")
        print(f"  identity {100 * scores['identity']:.1f}% reachable | fitted {100 * scores['fitted']:.1f}%")
        if scores["fitted"] <= scores["identity"]:
            print(
                "  The fitted rotation does NOT generalize — it is fitting episode-specific\n"
                "  geometry, not a mounting convention. Do not deploy it; measure the physical\n"
                "  mount instead."
            )

    print(f"\nsuggested tool_offset rotation vector (rad): {np.round(rotvec, 4).tolist()}")
    print(
        "\nThis is a DATA-DERIVED ESTIMATE from one start pose. Measure the physical gripper\n"
        "mount before trusting it. If the best rotation still leaves many poses unreachable,\n"
        "the gap is not a frame convention: the policy is asking for motions outside this\n"
        "arm's workspace, and it needs feasibility-aware data curation or fine-tuning in the\n"
        "robot's own frame."
    )
    return 0


if __name__ == "__main__":
    main()
