"""
Global IK of the keyframe posture: every joint configuration reaching a target pose, from the panda_global_ik package
(GeoFIK, joint 7 sampled over its range), configured with the robot profile of FrankaKinematics (joint offsets, TCP,
base) so its solutions are those of the model the MPPI costs use. Unlike the local DLS solver it does not depend on the
current posture: no solution means the pose is out of reach. Install: pip install packages/panda_global_ik.
"""
from typing import List, Optional, Tuple

import numpy as np


class GlobalIK:
    def __init__(self, kinematics, q7_samples: int = 33):
        from panda_global_ik import PandaIK
        self.kin = kinematics
        tcp = np.eye(4)
        tcp[:3, 3] = np.asarray(kinematics.EE_OFFSET, dtype=np.float64)
        self._ik = PandaIK(tcp_offset=tcp, joint_offsets=np.asarray(kinematics.joint_angle_offsets, dtype=np.float64),
                           q7_samples=q7_samples)

    def solutions(self, target_pos: np.ndarray, target_rot: np.ndarray) -> List[np.ndarray]:
        """Joint configurations (measured space, inside the limits) whose TCP pose matches the target."""
        base = np.eye(4)
        if self.kin.base_position is not None:
            base[:3, 3] = np.asarray(self.kin.base_position, dtype=np.float64)
        self._ik.base, self._ik.base_inv = base, np.linalg.inv(base)
        target = np.eye(4)
        target[:3, :3] = np.asarray(target_rot, dtype=np.float64)
        target[:3, 3] = np.asarray(target_pos, dtype=np.float64)
        return list(self._ik.solutions(target))

    @staticmethod
    def select(solutions: List[np.ndarray], q_now: np.ndarray, kinematics,
               min_margin: float) -> Optional[Tuple[np.ndarray, float, float]]:
        """The solution closest to the current posture among those at least min_margin from every joint limit
        (all of them when none is): (q, joint-limit margin, joint distance)."""
        if not solutions:
            return None
        margins = [float(np.min(np.minimum(q - kinematics.Q_MIN, kinematics.Q_MAX - q))) for q in solutions]
        pool = [i for i, m in enumerate(margins) if m >= min_margin] or list(range(len(solutions)))
        best = min(pool, key=lambda i: float(np.linalg.norm(solutions[i] - q_now)))
        return solutions[best], margins[best], float(np.linalg.norm(solutions[best] - q_now))
