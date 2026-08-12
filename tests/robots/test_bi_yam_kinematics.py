#!/usr/bin/env python

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

import importlib.util

import numpy as np
import pytest

from lerobot.robots.bi_yam.kinematics import (
    I2RT_GRASP_SITE,
    I2RT_GRIPPER_XML_SHA256,
    I2RT_REVISION,
    I2RT_YAM_JOINT_LIMITS,
    I2RT_YAM_LINEAR_4310_Q0_FK,
    I2RTYAMKinematics,
)


def _translated_q0(x: float) -> np.ndarray:
    pose = I2RT_YAM_LINEAR_4310_Q0_FK.copy()
    pose[0, 3] += x
    return pose


class FakeI2RTKinematics:
    def __init__(self) -> None:
        self.fk_calls: list[tuple[np.ndarray, str | None]] = []
        self.ik_calls: list[tuple[np.ndarray, str, np.ndarray, dict[str, object]]] = []
        self.ik_success = True
        self.ik_solution = np.zeros(8, dtype=np.float64)
        self.ik_verification_pose = I2RT_YAM_LINEAR_4310_Q0_FK.copy()

    def fk(self, q: np.ndarray, site_name: str | None = None) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).copy()
        self.fk_calls.append((q, site_name))
        if len(self.fk_calls) == 1:
            return I2RT_YAM_LINEAR_4310_Q0_FK.copy()
        return self.ik_verification_pose.copy()

    def ik(
        self,
        target_pose: np.ndarray,
        site_name: str,
        init_q: np.ndarray | None = None,
        **kwargs: object,
    ) -> tuple[bool, np.ndarray]:
        assert init_q is not None
        self.ik_calls.append((target_pose.copy(), site_name, init_q.copy(), kwargs.copy()))
        return self.ik_success, self.ik_solution.copy()


def _solver(
    fake: FakeI2RTKinematics | None = None, **kwargs: object
) -> tuple[I2RTYAMKinematics, FakeI2RTKinematics]:
    backend = fake or FakeI2RTKinematics()
    return I2RTYAMKinematics(backend_factory=lambda: backend, **kwargs), backend


def test_contract_is_pinned_to_authoritative_i2rt_model() -> None:
    assert I2RT_REVISION == "7ed46f4e4e316133a0c39aa6cf34a73d2718e850"
    assert I2RT_GRIPPER_XML_SHA256 == "42d7ab67071d33c2fecf73a74f529ccb80337b4c81f4208d5a8ecd56cc89d7a2"
    assert I2RT_GRASP_SITE == "grasp_site"
    np.testing.assert_allclose(
        I2RT_YAM_LINEAR_4310_Q0_FK[:3, 3], [0.330597263, 0.000001793, 0.173502620], atol=0
    )


def test_constructor_checks_i2rt_zero_pose_contract() -> None:
    fake = FakeI2RTKinematics()
    fake_pose = I2RT_YAM_LINEAR_4310_Q0_FK.copy()
    fake_pose[0, 3] += 1e-3

    def wrong_fk(q: np.ndarray, site_name: str | None = None) -> np.ndarray:
        del q, site_name
        return fake_pose

    fake.fk = wrong_fk  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="pinned model contract"):
        I2RTYAMKinematics(backend_factory=lambda: fake)


def test_fk_pads_six_arm_joints_with_zero_jaw_slides_and_uses_grasp_site() -> None:
    solver, fake = _solver()
    q = np.array([0.1, 0.7, 0.8, -0.2, 0.3, -0.4])

    actual = solver.fk(q)

    np.testing.assert_allclose(actual, I2RT_YAM_LINEAR_4310_Q0_FK)
    padded, site = fake.fk_calls[-1]
    np.testing.assert_allclose(padded, [*q, 0.0, 0.0])
    assert site == I2RT_GRASP_SITE


def test_ik_delegates_to_i2rt_with_padded_seed_and_verifies_by_i2rt_fk() -> None:
    target = _translated_q0(5e-5)
    fake = FakeI2RTKinematics()
    fake.ik_solution = np.array([0.1, 0.6, 0.7, -0.2, 0.1, -0.3, 0.0, 0.0])
    fake.ik_verification_pose = target.copy()
    solver, fake = _solver(fake)
    seed = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])

    solution = solver.ik(target, seed)

    np.testing.assert_allclose(solution, fake.ik_solution[:6])
    called_target, site, init_q, kwargs = fake.ik_calls[-1]
    np.testing.assert_allclose(called_target, target)
    np.testing.assert_allclose(init_q, [*seed, 0.0, 0.0])
    assert site == I2RT_GRASP_SITE
    assert kwargs == {"pos_threshold": 1e-4, "ori_threshold": 1e-4}
    verified_q, verified_site = fake.fk_calls[-1]
    np.testing.assert_allclose(verified_q, fake.ik_solution)
    assert verified_site == I2RT_GRASP_SITE


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [
        ("fk", np.zeros(5), "shape"),
        ("fk", np.array([0.0, 0.5, 0.5, 0.0, 0.0, np.nan]), "finite"),
        ("fk", np.array([4.0, 0.5, 0.5, 0.0, 0.0, 0.0]), "joint limits"),
        ("ik_target", np.eye(3), "shape"),
        ("ik_target", np.full((4, 4), np.nan), "finite"),
        ("ik_seed", np.zeros(5), "shape"),
        ("ik_seed", np.array([0.0, -0.1, 0.5, 0.0, 0.0, 0.0]), "joint limits"),
    ],
)
def test_inputs_fail_closed(method: str, value: np.ndarray, message: str) -> None:
    solver, _ = _solver()
    target = I2RT_YAM_LINEAR_4310_Q0_FK
    seed = np.zeros(6)

    with pytest.raises(ValueError, match=message):
        if method == "fk":
            solver.fk(value)
        elif method == "ik_target":
            solver.ik(value, seed)
        else:
            solver.ik(target, value)


def test_invalid_target_rotation_fails_before_calling_i2rt() -> None:
    solver, fake = _solver()
    target = I2RT_YAM_LINEAR_4310_Q0_FK.copy()
    target[:3, :3] = np.diag([1.0, 1.0, -1.0])

    with pytest.raises(ValueError, match="determinant"):
        solver.ik(target, np.zeros(6))
    assert not fake.ik_calls


def test_ik_solver_failure_fails_closed() -> None:
    fake = FakeI2RTKinematics()
    fake.ik_success = False
    solver, _ = _solver(fake)

    with pytest.raises(RuntimeError, match="failed to converge"):
        solver.ik(I2RT_YAM_LINEAR_4310_Q0_FK, np.zeros(6))


@pytest.mark.parametrize(
    ("solution", "message"),
    [
        (np.zeros(7), "shape"),
        (np.array([0.0, 0.5, 0.5, 0.0, 0.0, np.nan, 0.0, 0.0]), "finite"),
        (np.array([4.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), "joint limits"),
        (np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.01, 0.01]), "jaw slides"),
    ],
)
def test_invalid_ik_results_fail_closed(solution: np.ndarray, message: str) -> None:
    fake = FakeI2RTKinematics()
    fake.ik_solution = solution
    solver, _ = _solver(fake)

    with pytest.raises((ValueError, RuntimeError), match=message):
        solver.ik(I2RT_YAM_LINEAR_4310_Q0_FK, np.zeros(6))


def test_ik_result_must_pass_independent_fk_residual_check() -> None:
    fake = FakeI2RTKinematics()
    fake.ik_verification_pose = _translated_q0(2e-4)
    solver, _ = _solver(fake)

    with pytest.raises(RuntimeError, match="residual exceeds tolerance"):
        solver.ik(I2RT_YAM_LINEAR_4310_Q0_FK, np.zeros(6))


def test_joint_limit_endpoints_are_accepted() -> None:
    solver, fake = _solver()

    solver.fk(I2RT_YAM_JOINT_LIMITS[:, 0])
    solver.fk(I2RT_YAM_JOINT_LIMITS[:, 1])

    np.testing.assert_allclose(fake.fk_calls[-2][0][:6], I2RT_YAM_JOINT_LIMITS[:, 0])
    np.testing.assert_allclose(fake.fk_calls[-1][0][:6], I2RT_YAM_JOINT_LIMITS[:, 1])


def test_joint_limit_roundoff_is_clipped_but_real_violation_is_rejected() -> None:
    solver, fake = _solver()
    q = I2RT_YAM_JOINT_LIMITS[:, 0].copy()
    q[1] -= 5e-9
    solver.fk(q)
    assert fake.fk_calls[-1][0][1] == I2RT_YAM_JOINT_LIMITS[1, 0]

    q[1] -= 1e-6
    with pytest.raises(ValueError, match="joint limits"):
        solver.fk(q)


@pytest.mark.skipif(importlib.util.find_spec("i2rt") is None, reason="i2rt is optional")
def test_pinned_i2rt_integration_fk_and_ik_roundtrip() -> None:
    solver = I2RTYAMKinematics()
    np.testing.assert_allclose(solver.fk(np.zeros(6)), I2RT_YAM_LINEAR_4310_Q0_FK, atol=1e-8)

    target_q = np.array([0.15, 0.7, 1.0, -0.3, 0.2, -0.25])
    target_pose = solver.fk(target_q)
    seed = target_q + np.array([0.02, -0.02, 0.02, -0.02, 0.02, -0.02])
    solution = solver.ik(target_pose, seed)
    np.testing.assert_allclose(solver.fk(solution), target_pose, atol=1e-4)
