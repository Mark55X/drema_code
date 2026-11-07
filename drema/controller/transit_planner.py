"""
Collision-free joint-space transit paths (RRT-Connect, Kuffner and LaValle 2000) checked with the digital twin's own
collision model under the route rule of route_clearance: the tree rooted at the start may stay close to an object only
along a branch that has been close to it since the root (leaving it), the tree rooted at the goal postures likewise
(approaching it) and never overlapping it. Several attempts are shortcut and the cheapest is kept.
"""
import time
from typing import Callable, Optional, Tuple

import numpy as np

from drema.controller.route_clearance import route_violations

DistFn = Callable[[np.ndarray], np.ndarray]


def _samples(a: np.ndarray, b: np.ndarray, res: float) -> np.ndarray:
    """Points of the segment a->b after a, at most res apart, b included."""
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / res)))
    return a[None] + np.linspace(0.0, 1.0, n + 1)[1:, None] * (b - a)[None]


def _grow(D: np.ndarray, joined: np.ndarray, clearance: float, from_start: bool) -> Optional[np.ndarray]:
    """Per-object flags 'close since the root' after the samples D [n, m] of a new branch, None if the rule breaks."""
    tight = D < clearance
    still = joined[None] & np.logical_and.accumulate(tight, axis=0)
    bad = (tight & ~still) | ((D <= 0.0) & (~still if from_start else True))
    return None if bad.any() else still[-1]


def rrt_connect(dist: DistFn, start: np.ndarray, goals: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                rng: np.random.Generator, budget_s: float, step: float, res: float,
                clearance: float) -> Optional[np.ndarray]:
    D0 = dist(np.vstack([start[None], goals]))
    reachable = ~np.any(D0[1:] <= 0.0, axis=1)
    if not reachable.any():
        return None
    trees = [{'q': [start], 'parent': [-1], 'joined': [D0[0] < clearance], 'start': True},
             {'q': list(goals[reachable]), 'parent': [-1] * int(reachable.sum()),
              'joined': list(D0[1:][reachable] < clearance), 'start': False}]
    t0 = time.time()

    def extend(tree, x):
        i = int(np.argmin(np.linalg.norm(np.asarray(tree['q']) - x, axis=1)))
        qn = tree['q'][i]
        d = float(np.linalg.norm(x - qn))
        qs = x if d <= step else qn + (x - qn) * step / d
        joined = _grow(dist(_samples(qn, qs, res)), tree['joined'][i], clearance, tree['start'])
        if joined is None:
            return None, 'trapped'
        tree['q'].append(qs)
        tree['parent'].append(i)
        tree['joined'].append(joined)
        return len(tree['q']) - 1, 'reached' if d <= step else 'advanced'

    def root_path(tree, i):
        out = []
        while i >= 0:
            out.append(tree['q'][i])
            i = tree['parent'][i]
        return out

    while time.time() - t0 < budget_s:
        a, b = trees
        ia, _ = extend(a, lo + (hi - lo) * rng.random(len(lo)))
        if ia is not None:
            while True:
                ib, status = extend(b, a['q'][ia])
                if status != 'advanced':
                    break
            if status == 'reached':
                pa, pb = root_path(a, ia), root_path(b, ib)
                return np.asarray(pa[::-1] + pb[1:] if a['start'] else pb[::-1] + pa[1:])
        trees.reverse()
    return None


def shortcut(dist: DistFn, path: np.ndarray, rng: np.random.Generator, tries: int, res: float,
             clearance: float) -> np.ndarray:
    """Replaces stretches by straight segments while the whole path still follows the route rule."""
    path = list(path)
    first = dist(path[0][None])
    edges = [dist(_samples(a, b, res)) for a, b in zip(path[:-1], path[1:])]
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        new = dist(_samples(path[i], path[j], res))
        if not route_violations(np.vstack([first] + edges[:i] + [new] + edges[j:]), clearance).any():
            path = path[:i + 1] + path[j:]
            edges = edges[:i] + [new] + edges[j:]
    return np.asarray(path)


def path_cost(path: np.ndarray, wrist_weight: float) -> float:
    """Joint length plus extra weight on joint 7, whose rotation turns a held object about the gripper axis."""
    d = np.abs(np.diff(path, axis=0))
    return float(np.sum(np.linalg.norm(d, axis=1)) + wrist_weight * np.sum(d[:, 6]))


def plan_transit(dist: DistFn, start: np.ndarray, goals: np.ndarray, lo: np.ndarray, hi: np.ndarray, seed: int,
                 clearance: float, attempts: int = 5, budget_s: float = 4.0, step: float = 0.15, res: float = 0.04,
                 shortcut_tries: int = 150, wrist_weight: float = 1.0) -> Tuple[Optional[np.ndarray], dict]:
    """Best of `attempts` shortcut RRT-Connect paths within budget_s overall."""
    t0, best, info = time.time(), None, {'found': 0, 'attempts': 0}
    for k in range(attempts):
        left = budget_s - (time.time() - t0)
        if left <= 0.05:
            break
        rng = np.random.default_rng(seed + k)
        info['attempts'] += 1
        raw = rrt_connect(dist, start, goals, lo, hi, rng, left, step, res, clearance)
        if raw is None:
            continue
        p = shortcut(dist, raw, rng, shortcut_tries, res, clearance)
        info['found'] += 1
        if best is None or path_cost(p, wrist_weight) < path_cost(best, wrist_weight):
            best = p
    info['time_s'] = time.time() - t0
    if best is not None:
        info['cost'] = path_cost(best, wrist_weight)
        info['joint_length'] = float(np.sum(np.linalg.norm(np.diff(best, axis=0), axis=1)))
    return best, info
