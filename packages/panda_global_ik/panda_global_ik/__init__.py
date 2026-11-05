"""
Global inverse kinematics of the Franka Emika Panda: every joint configuration reaching a TCP pose, from the analytical
solver GeoFIK (Lopez-Custodio, Gong, Figueredo, arXiv:2503.03992) sampled over the joint-7 range. Each solution is
checked with the forward kinematics of this module, so a wrong or out-of-limit solution is never returned.
"""
import ctypes
import glob
import os
from typing import Optional

import numpy as np

__all__ = ['PandaIK', 'Q_MIN', 'Q_MAX', 'FRANKA_HAND_TCP']

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# Official Franka modified DH parameters (a_{i-1}, alpha_{i-1}, d_i); the last row ends at the flange.
_MDH = [(0.0, 0.0, 0.333), (0.0, -np.pi / 2, 0.0), (0.0, np.pi / 2, 0.316), (0.0825, np.pi / 2, 0.0),
        (-0.0825, -np.pi / 2, 0.384), (0.0, np.pi / 2, 0.0), (0.088, np.pi / 2, 0.107)]


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Flange -> TCP of the Franka Hand. GeoFIK solves for this frame whatever TCP the caller uses.
FRANKA_HAND_TCP = np.eye(4)
FRANKA_HAND_TCP[:3, :3] = _rot_z(-np.pi / 4)
FRANKA_HAND_TCP[2, 3] = 0.1034
_HAND_INV = np.linalg.inv(FRANKA_HAND_TCP)


def _load_geofik() -> ctypes.CDLL:
    found = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_geofik*.so'))
    if not found:
        raise ImportError('panda_global_ik: compiled GeoFIK library not found, reinstall with `pip install .`')
    lib = ctypes.CDLL(found[0])
    lib.geofik_ik_q7.restype = ctypes.c_int
    return lib


_DOUBLE_P = ctypes.POINTER(ctypes.c_double)


class PandaIK:
    def __init__(self, tcp_offset: Optional[np.ndarray] = None, joint_offsets: Optional[np.ndarray] = None,
                 base: Optional[np.ndarray] = None, q7_samples: int = 33, q7_refine_samples: int = 133,
                 position_tolerance: float = 1e-3, rotation_tolerance_deg: float = 0.5):
        """
        :param tcp_offset: 4x4 flange -> TCP transform; default the Franka Hand (0.1034 m along z, -45 deg about z).
        :param joint_offsets: added to the joint values before the kinematic chain; zero on a real Panda, non-zero
            only for simulator models whose joint zeros differ from Franka's (e.g. joint 4 of the RLBench Panda).
        :param base: 4x4 world -> robot base transform; default identity, i.e. poses in the base frame.
        :param q7_samples: joint-7 values sampled over its range (33 = 10 deg step).
        :param q7_refine_samples: denser second pass, only when the first finds nothing: about 0.2 % of the reachable
            poses are reachable only in a joint-7 window narrower than 10 deg (0 left out of 4000 with 133).
        """
        self.tcp = np.eye(4) @ (FRANKA_HAND_TCP if tcp_offset is None else np.asarray(tcp_offset, dtype=np.float64))
        self.tcp_inv = np.linalg.inv(self.tcp)
        self.offsets = np.zeros(7) if joint_offsets is None else np.asarray(joint_offsets, dtype=np.float64).reshape(7)
        self.base = np.eye(4) if base is None else np.asarray(base, dtype=np.float64)
        self.base_inv = np.linalg.inv(self.base)
        self.q7_grid = np.linspace(Q_MIN[6] + 1e-3, Q_MAX[6] - 1e-3, int(q7_samples))
        self.q7_refine_grid = np.linspace(Q_MIN[6] + 1e-3, Q_MAX[6] - 1e-3, int(q7_refine_samples)) if q7_refine_samples else None
        self.position_tolerance = position_tolerance
        self.rotation_tolerance = np.radians(rotation_tolerance_deg)
        self._lib = _load_geofik()
        self._out = np.zeros(56)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """World pose of the TCP (4x4) for the joint values q."""
        T = np.eye(4)
        for (a, alpha, d), theta in zip(_MDH, np.asarray(q, dtype=np.float64) + self.offsets):
            ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
            T = T @ np.array([[ct, -st, 0.0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0.0, 0.0, 0.0, 1.0]])
        return self.base @ T @ self.tcp

    def solutions(self, target: np.ndarray, q7: Optional[float] = None) -> np.ndarray:
        """All joint configurations (n x 7, inside the joint limits) whose TCP reaches the 4x4 world pose target,
        over the sampled joint-7 range, or for the given joint-7 value only."""
        target = np.asarray(target, dtype=np.float64)
        if q7 is not None:
            return self._solve(target, [q7 + self.offsets[6]])
        found = self._solve(target, self.q7_grid)
        if len(found) == 0 and self.q7_refine_grid is not None:
            found = self._solve(target, self.q7_refine_grid)
        return found

    def _solve(self, target: np.ndarray, q7_values) -> np.ndarray:
        hand = self.base_inv @ target @ self.tcp_inv @ FRANKA_HAND_TCP
        r = np.ascontiguousarray(hand[:3, 3])
        rot = np.ascontiguousarray(hand[:3, :3]).ravel()
        found = []
        for q7 in q7_values:
            self._lib.geofik_ik_q7(r.ctypes.data_as(_DOUBLE_P), rot.ctypes.data_as(_DOUBLE_P), ctypes.c_double(q7),
                                   ctypes.c_double(np.pi / 2), self._out.ctypes.data_as(_DOUBLE_P))
            for theta in self._out.reshape(8, 7):
                if not np.all(np.isfinite(theta)):
                    continue
                q = theta - self.offsets
                if np.any(q < Q_MIN) or np.any(q > Q_MAX):
                    continue
                if self._reaches(q, target):
                    found.append(q)
        return np.array(found).reshape(-1, 7)

    def best(self, solutions: np.ndarray, q_current: np.ndarray, min_margin: float = 0.2) -> Optional[np.ndarray]:
        """The solution closest to q_current among those at least min_margin [rad] from every joint limit (among all
        of them when none is); None if there are no solutions."""
        if len(solutions) == 0:
            return None
        margins = np.array([self.margin(q) for q in solutions])
        pool = np.flatnonzero(margins >= min_margin)
        if len(pool) == 0:
            pool = np.arange(len(solutions))
        return solutions[pool[np.argmin(np.linalg.norm(solutions[pool] - np.asarray(q_current), axis=1))]]

    def ik(self, target: np.ndarray, q_current: np.ndarray, min_margin: float = 0.2) -> Optional[np.ndarray]:
        """Shortcut: best(solutions(target), q_current). None means the pose is out of reach."""
        return self.best(self.solutions(target), q_current, min_margin)

    @staticmethod
    def margin(q: np.ndarray) -> float:
        """Smallest distance [rad] of q from a joint limit."""
        return float(np.min(np.minimum(np.asarray(q) - Q_MIN, Q_MAX - np.asarray(q))))

    def _reaches(self, q: np.ndarray, target: np.ndarray) -> bool:
        T = self.fk(q)
        cos_angle = np.clip((np.trace(T[:3, :3].T @ target[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        return (np.linalg.norm(T[:3, 3] - target[:3, 3]) < self.position_tolerance and
                np.arccos(cos_angle) < self.rotation_tolerance)
