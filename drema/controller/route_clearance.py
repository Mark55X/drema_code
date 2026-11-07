"""
Route validity with the digital twin's own collision model and the current object roles (their margins included).

A route may run closer than the required clearance to an object only in a stretch joined to one of its ends: while
it leaves where the robot is (resting on a support, still at the object it released) or while it approaches where
the keyframe puts it (a grasp, an insertion). A close stretch with free space on both sides is a passage through or
along the object. Past the stretch joined to the start, overlapping an object (distance <= 0) is a collision, the
arrival included: a keyframe posture in collision cannot be reached.
"""
from typing import Any, Optional, Sequence, Tuple

import numpy as np


def clearance_matrix(twin: Any, kin: Any, Q: np.ndarray, cap: float, sigma_1: float = 0.015, kappa: float = 120.0,
                     rho: float = 0.8, dt: float = 0.05,
                     ids: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distance [n, m] of the arm and what it holds from each object, role margins applied and capped at `cap`, for the
    configurations Q [n, 7]; columns are the object ids (-1 = table), the given ones or every object within the cap.
    """
    Q = np.asarray(Q, dtype=np.float32).reshape(-1, 7)
    saved = (twin.record_collision_points, twin.last_collision_points)
    twin.record_collision_points, twin.last_collision_points = True, None
    try:
        twin.compute_trajectory_collision_costs(Q=Q[:, None, :], QD=np.zeros((len(Q), 1, 7), dtype=np.float32),
                                                sigma_1=min(sigma_1, cap), sigma_2=cap, kappa=kappa, rho=rho,
                                                kin_helper=kin, dt=dt)
        pts = twin.last_collision_points or {}
    finally:
        twin.record_collision_points, twin.last_collision_points = saved
    rows = np.asarray(pts.get('rollout', []), dtype=np.int64)
    objs = np.asarray(pts.get('obstacle', []), dtype=np.int64)
    dist = np.asarray(pts.get('distance', []), dtype=np.float64)
    cols = np.asarray(sorted(set(objs.tolist())) if ids is None else list(ids), dtype=np.int64)
    D = np.full((len(Q), len(cols)), float(cap))
    index = {int(o): j for j, o in enumerate(cols)}
    for k, o, d in zip(rows, objs, dist):
        j = index.get(int(o))
        if j is not None and d < D[k, j]:
            D[k, j] = d
    return D, cols


def route_violations(D: np.ndarray, clearance: float) -> np.ndarray:
    """[n, m] samples of a route (rows, start to end) that break the rule above for each object (columns)."""
    tight = D < clearance
    from_start = np.logical_and.accumulate(tight, axis=0)
    to_end = np.logical_and.accumulate(tight[::-1], axis=0)[::-1]
    return (tight & ~from_start & ~to_end) | ((D <= 0.0) & ~from_start)


def densify(nodes: Sequence[np.ndarray], resolution: float) -> np.ndarray:
    """Straight joint-space segments through the nodes, sampled at most `resolution` rad apart, ends included."""
    nodes = [np.asarray(n, dtype=np.float64) for n in nodes]
    pieces = [a[None] + np.linspace(0.0, 1.0, max(2, int(np.ceil(np.linalg.norm(b - a) / resolution)) + 1))[:-1, None]
              * (b - a)[None] for a, b in zip(nodes[:-1], nodes[1:])]
    return np.concatenate(pieces + [nodes[-1][None]])
