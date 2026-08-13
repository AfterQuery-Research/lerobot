from __future__ import annotations

import logging

import numpy as np

from lerobot.rollout.inference.bi_yam import BiYAMActionAdapter, BiYAMQueryAnchor


class _IdentityKinematics:
    def fk(self, q: object) -> np.ndarray:
        return np.eye(4, dtype=np.float64)

    def ik(self, target_pose: object, seed_q: object) -> np.ndarray:
        return np.asarray(seed_q, dtype=np.float64)


def test_ee_adapter_logs_raw_xyz_and_r6d_before_ik(caplog) -> None:
    adapter = BiYAMActionAdapter("ee", kinematics_factory=_IdentityKinematics)
    adapter.start()
    anchor = BiYAMQueryAnchor(
        left_pose=np.eye(4),
        right_pose=np.eye(4),
        left_q=np.zeros(6),
        right_q=np.zeros(6),
    )
    action = np.asarray(
        [
            0.1,
            -0.2,
            0.3,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.25,
            -0.4,
            0.5,
            -0.6,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.75,
        ]
    )

    with caplog.at_level(logging.INFO, logger="lerobot.rollout.inference.bi_yam"):
        adapter.to_joint_action(action, anchor)

    message = caplog.messages[-1]
    assert "left_xyz_m(tool)=[ 0.1,-0.2, 0.3]" in message
    assert "left_r6d_rows=[1.,0.,0.,0.,1.,0.]" in message
    assert "right_xyz_m(tool)=[-0.4, 0.5,-0.6]" in message
    assert "right_r6d_rows=[1.,0.,0.,0.,1.,0.]" in message
