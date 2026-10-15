"""
Captures what the MPC "imagines" at each solve (sampled rollouts, planned trajectory, IK guidance, forecast
obstacle poses) as a self-contained frame of numpy arrays, and forwards it to the configured sinks.
"""

import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .sinks import ImaginationSink, warn_limited

# Rollout groups, in the order the engine concatenates its samples
GROUP_NAMES = ("greedy", "sensitive", "mixed", "primitive")


def _link_skeletons(kin: Any, q_batch: np.ndarray) -> np.ndarray:
    """World positions of the link 0..7 frame origins and of the TCP, shape [N, 9, 3]."""
    q_batch = np.asarray(q_batch, dtype=np.float64).reshape(-1, 7)
    T_all = kin._batch_link_transforms(q_batch)
    base = kin.base_position if kin.base_position is not None else np.zeros(3)
    origins = T_all[:, :, :3, 3] + base
    tcp = kin.batch_forward_kinematics_ee(q_batch[:, None, :])[:, 0].astype(np.float64)
    return np.concatenate([origins, tcp[:, None, :]], axis=1)


def _integrate_plan(q0: np.ndarray, qd0: np.ndarray, acc: np.ndarray, dt: float, qd_limit: float,
                    max_acc: float, q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
    """Joint trajectory of the blended plan, with the same velocity projection as the engine rollouts."""
    q, qd = np.asarray(q0, dtype=np.float64).copy(), np.asarray(qd0, dtype=np.float64).copy()
    out = np.zeros((acc.shape[0], 7), dtype=np.float64)
    for h in range(acc.shape[0]):
        u = np.clip(acc[h], (-qd_limit - qd) / dt, (qd_limit - qd) / dt)
        u = np.clip(u, -max_acc, max_acc)
        qd = qd + u * dt
        q = np.clip(q + qd * dt, q_min, q_max)
        out[h] = q
    return out


class ImaginationRecorder:
    """
    Registers itself as the engine rollout observer and turns each solve into an imagination frame.
    On the MPC thread it only copies the solve arrays and the obstacle forecast; selecting the rollouts, their
    FK, the skeletons and the sinks run on a dedicated thread. A full queue drops the frame, never the solve.
    """

    def __init__(
        self,
        engine: Any,
        digital_twin: Any,
        sinks: List[ImaginationSink],
        max_rollouts: int = 40,
        ghost_count: int = 3,
        scene_info: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        self.engine = engine
        self.twin = digital_twin
        self.sinks = list(sinks)
        self.max_rollouts = max(1, int(max_rollouts))
        self.ghost_count = max(0, int(ghost_count))
        self.scene_info = scene_info
        self._context: Dict[str, Any] = {}
        self._scene_ids: Optional[set] = None
        self.dropped = 0
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=8)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def attach(self) -> None:
        self.engine.rollout_observer = self._on_rollouts
        if self.twin is not None:
            self.twin.record_collision_points = True

    def detach(self) -> None:
        if self.engine.rollout_observer == self._on_rollouts:
            self.engine.rollout_observer = None
        if self.twin is not None:
            self.twin.record_collision_points = False

    def begin_action(self, robot_state: Any) -> None:
        """Context of the next solve that the engine does not know (timestep, keyframe flag)."""
        if self.twin is not None:
            self.twin.last_collision_points = None  # a solve that skips the collision query must not show stale points
        self._context = {
            'timestep': int(robot_state.timestep),
            'target_pose': np.asarray(robot_state.target_pose, dtype=np.float64),
            'approach_funnel': int(robot_state.approach_funnel) if robot_state.HasField('approach_funnel') else -1,
        }

    def close(self) -> None:
        self.detach()
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        for sink in self.sinks:
            sink.close()
        if self.dropped:
            print(f"[VISUALIZATION WARNING] {self.dropped} imagination frames dropped (processing queue full)")

    def _on_rollouts(self, snap: Dict[str, Any]) -> None:
        copied = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in snap.items()
                  if k not in ('planner_costs', 'diagnostics')}
        copied['diagnostics'] = {'top_costs': dict(snap['diagnostics'].get('top_costs', {}) or {}),
                                 'calc_time_ms': snap['diagnostics'].get('calc_time_ms', 0.0)}
        obstacles = self._obstacles(snap['Q'].shape[1], float(snap['dt']))
        points = getattr(self.twin, 'last_collision_points', None) if self.twin is not None else None
        copied['collision_points'] = {k: v.copy() for k, v in points.items()} if points else None
        try:
            self._queue.put_nowait((copied, dict(self._context), obstacles))
        except queue.Full:
            self.dropped += 1

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._build_frame(*item)
            except Exception as e:
                warn_limited(self, f"imagination frame failed: {e}")

    def _select_rollouts(self, snap: Dict[str, Any]) -> np.ndarray:
        """Best rollouts by judge cost, spread over the groups, always including the winning candidate."""
        K = snap['Q'].shape[0]
        M = int(snap['samples_per_planner'])
        groups = np.minimum(np.arange(K) // M, 3)
        judge = np.asarray(snap['judge_costs'], dtype=np.float64)
        per_group = max(1, self.max_rollouts // 4)
        keep = [int(snap['best_index'])]
        for g in range(4):
            idx = np.flatnonzero(groups == g)
            keep.extend(idx[np.argsort(judge[idx])[:per_group]].tolist())
        return np.array(sorted(set(keep)), dtype=np.int64)

    def _obstacles(self, horizon: int, dt: float) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        twin = self.twin
        if twin is None:
            return out
        with twin.state_lock:
            predictor = getattr(twin, 'predictor', None)
            preds = predictor.predict_all(horizon=horizon, dt=dt) if predictor is not None else {}
            ids = []
            for oid, obj in twin.tracked_objects.items():
                if obj.get('is_target', False) or 'target_pos' not in obj:
                    continue
                ids.append(int(oid))
                out[f"obs{oid}_pos"] = np.asarray(obj['target_pos'], dtype=np.float64)
                out[f"obs{oid}_quat"] = np.asarray(obj.get('target_quat', (0, 0, 0, 1)), dtype=np.float64)
                if oid in preds:
                    out[f"obs{oid}_pred_pos"] = np.asarray(preds[oid].positions, dtype=np.float64)
                    out[f"obs{oid}_pred_quat"] = np.asarray(preds[oid].orientations, dtype=np.float64)
                    out[f"obs{oid}_pred_vel"] = np.asarray(preds[oid].velocities, dtype=np.float64)
            out['obstacle_ids'] = np.asarray(ids, dtype=np.int64)
            out['min_obstacle_distance'] = np.float64(twin.get_min_obstacle_distance())
        return out

    def _scene(self) -> Optional[Dict[str, Any]]:
        """Static scene sent once: obstacle meshes, table and robot base."""
        twin = self.twin
        scene: Dict[str, Any] = {'robot_base': np.asarray(self.engine.kin.base_position, dtype=np.float64)}
        info = self.scene_info() if self.scene_info is not None else {}
        scene.update({k: np.asarray(v, dtype=np.float64) for k, v in info.items() if v is not None})
        if twin is not None:
            with twin.state_lock:
                objects = list(twin.tracked_objects.items())
            for oid, obj in objects:
                if obj.get('is_target', False):
                    continue
                path = obj.get('mesh_path')
                if path:
                    scene[f"obs{oid}_mesh_path"] = str(path)
                    scene[f"obs{oid}_name"] = str(obj.get('name', oid))
        return scene

    def _build_frame(self, snap: Dict[str, Any], context: Dict[str, Any], obstacles: Dict[str, np.ndarray]) -> None:
        kin = self.engine.kin
        Q = snap['Q']
        H = Q.shape[1]
        sel = self._select_rollouts(snap)
        M = int(snap['samples_per_planner'])

        tcp = kin.batch_forward_kinematics_ee(Q[sel]).astype(np.float32)
        judge = np.asarray(snap['judge_costs'], dtype=np.float64)
        w = np.exp(-(judge[sel] - judge[sel].min()) / max(float(snap['lambda']), 1e-4))
        coll = np.asarray(snap['coll_p_costs'], dtype=np.float64)[sel]

        frame: Dict[str, Any] = dict(context)
        frame.update({
            'q': np.asarray(snap['q_current'], dtype=np.float64),
            'qd': np.asarray(snap['qd_current'], dtype=np.float64),
            'skeleton': _link_skeletons(kin, snap['q_current'][None, :])[0],
            'rollout_index': sel,
            'rollout_group': np.minimum(sel // M, 3).astype(np.int64),
            'rollout_tcp': tcp,
            'rollout_weight': (w / max(w.sum(), 1e-12)).astype(np.float32),
            'rollout_judge_cost': judge[sel].astype(np.float32),
            'rollout_collision': coll.astype(np.float32),
            'best_index': np.int64(snap['best_index']),
            'mixing_weights': np.array([snap['mixing_weights'].get(n, 0.0) for n in ('greedy', 'sensitive')]),
        })
        prim = snap['primitive_names']
        best = int(snap['best_index'])
        frame['best_label'] = (GROUP_NAMES[min(best // M, 3)] if best < 3 * M
                               else f"prim:{prim[best - 3 * M] if best - 3 * M < len(prim) else '?'}")

        if snap['planned_acc'] is not None:
            plan_q = _integrate_plan(snap['q_current'], snap['qd_current'], snap['planned_acc'], snap['dt'],
                                     snap['qd_limit'], snap['max_acc'], kin.Q_MIN, kin.Q_MAX)
            frame['plan_q'] = plan_q
            frame['plan_tcp'] = kin.batch_forward_kinematics_ee(plan_q[None]).astype(np.float64)[0]
            if self.ghost_count > 0:
                ghost_steps = np.linspace(H // self.ghost_count, H - 1, self.ghost_count).astype(int)
                frame['ghost_steps'] = ghost_steps
                frame['ghost_skeletons'] = _link_skeletons(kin, plan_q[ghost_steps])
        if snap['q_des'] is not None:
            frame['q_des'] = np.asarray(snap['q_des'], dtype=np.float64)
            frame['q_des_skeleton'] = _link_skeletons(kin, snap['q_des'][None, :])[0]
        if snap['target_pos'] is not None:
            frame['target_pos'] = np.asarray(snap['target_pos'], dtype=np.float64)
        if snap['target_rot'] is not None:
            frame['target_rot'] = np.asarray(snap['target_rot'], dtype=np.float64)

        diag = snap['diagnostics']
        for name, value in (diag.get('top_costs', {}) or {}).items():
            frame[f"top_cost_{name}"] = np.float64(value)
        frame['calc_time_ms'] = np.float64(diag.get('calc_time_ms', 0.0))
        frame.update(obstacles)
        points = snap.get('collision_points')
        if points is not None and len(points['rollout']):
            index_in_sel = {int(r): i for i, r in enumerate(sel)}
            keep = np.array([int(r) in index_in_sel for r in points['rollout']], dtype=bool)
            frame['contact_rollout'] = np.array([index_in_sel[int(r)] for r in points['rollout'][keep]], dtype=np.int64)
            frame['contact_step'] = points['step'][keep]
            frame['contact_position'] = points['position'][keep].astype(np.float32)
            frame['contact_distance'] = points['distance'][keep].astype(np.float32)
            frame['contact_obstacle'] = points['obstacle'][keep]

        # The scene is (re)sent whenever a new obstacle appears, e.g. objects spawned after the first solve.
        ids = set(int(i) for i in np.asarray(obstacles.get('obstacle_ids', []), dtype=np.int64))
        if self._scene_ids is None or not ids <= self._scene_ids:
            scene = self._scene()
            for sink in self.sinks:
                sink.publish_scene(scene)
            self._scene_ids = ids | (self._scene_ids or set())
        for sink in self.sinks:
            sink.publish(frame)
