"""
Per-action dump of the MPC solve inputs, enough to replay a control cycle offline with the same result.
"""

import os
import time
from typing import Any, Dict, Optional

import numpy as np


class MpcInputDumper:
    """Writes action_<n>_t<timestep>.npz in a timestamped subfolder of root_dir, every `interval` actions."""

    def __init__(self, root_dir: str, interval: int = 1, horizon: int = 15, dt: float = 0.05):
        self.out_dir = os.path.join(root_dir, time.strftime("%Y%m%d_%H%M%S"))
        self.interval = max(1, int(interval))
        self.horizon = int(horizon)
        self.dt = float(dt)
        self._actions = 0
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[VISUALIZATION] MPC input dump active: '{self.out_dir}' (every {self.interval} actions)")

    def snapshot(self, robot_state: Any, controller: Any, twin: Any) -> Optional[Dict[str, Any]]:
        """
        Everything the MPC solve reads besides the static scene: robot state, goal, per obstacle the measured
        pose, the twin body pose, the filter state and the forecast over the horizon, plus the engine state.
        Returns None on the actions that are not dumped.
        """
        self._actions += 1
        if (self._actions - 1) % self.interval:
            return None
        dump: Dict[str, Any] = {
            'action_index': self._actions,
            'timestep': int(robot_state.timestep),
            'q': np.asarray(robot_state.joint_positions, dtype=np.float64),
            'qd': np.asarray(robot_state.joint_velocities, dtype=np.float64),
            'target_pose': np.asarray(robot_state.target_pose, dtype=np.float64),
            'approach_funnel': int(robot_state.approach_funnel) if robot_state.HasField('approach_funnel') else -1,
            'robot_base_pos': np.asarray(robot_state.robot_base_pos, dtype=np.float64),
        }
        with twin.state_lock:
            predictor = getattr(twin, 'predictor', None)
            preds = predictor.predict_all(horizon=self.horizon, dt=self.dt) if predictor is not None else {}
            obstacle_ids = []
            for oid, obj in twin.tracked_objects.items():
                if obj.get('is_target', False):
                    continue
                obstacle_ids.append(int(oid))
                key = f"obs{oid}"
                dump[f"{key}_measured_pos"] = np.asarray(obj.get('target_pos', (np.nan,) * 3), dtype=np.float64)
                dump[f"{key}_measured_quat"] = np.asarray(obj.get('target_quat', (np.nan,) * 4), dtype=np.float64)
                body_pose = twin.get_object_pose(oid)
                if body_pose is not None:
                    dump[f"{key}_body_pos"] = np.asarray(body_pose[0], dtype=np.float64)
                    dump[f"{key}_body_quat"] = np.asarray(body_pose[1], dtype=np.float64)
                st = predictor.get_estimated_state(oid) if predictor is not None else None
                if st is not None:
                    for name, value in st.items():
                        dump[f"{key}_filter_{name}"] = np.asarray(value, dtype=np.float64)
                if oid in preds:
                    dump[f"{key}_pred_positions"] = np.asarray(preds[oid].positions, dtype=np.float64)
                    dump[f"{key}_pred_orientations"] = np.asarray(preds[oid].orientations, dtype=np.float64)
                    dump[f"{key}_pred_velocities"] = np.asarray(preds[oid].velocities, dtype=np.float64)
                dump[f"{key}_mesh_path"] = str(obj.get('mesh_path', obj.get('name', '')))
            dump['obstacle_ids'] = np.asarray(obstacle_ids, dtype=np.int64)
            dump['min_obstacle_distance'] = float(twin.get_min_obstacle_distance())

        # The solve also depends on the planner distributions carried over from previous cycles and on the
        # global sampling RNG: both are needed to replay a cycle exactly.
        engine = controller.engine
        for name, dist in engine.planner_distributions.items():
            dump[f"engine_{name}_mu"] = np.array(dist['mu'], dtype=np.float64)
            dump[f"engine_{name}_sigma"] = np.array(dist['sigma'], dtype=np.float64)
        dump['engine_mu_mixed'] = np.array(engine.mu_mixed, dtype=np.float64)
        dump['engine_sigma_mixed'] = np.array(engine.sigma_mixed, dtype=np.float64)
        dump['engine_mixing_weights'] = np.array([engine.mixing_weights[n] for n in engine.planner_names], dtype=np.float64)
        dump['engine_planner_names'] = np.array(engine.planner_names)
        if engine.last_ik_solution is not None:
            dump['engine_last_ik_solution'] = np.array(engine.last_ik_solution, dtype=np.float64)
        _, rng_keys, rng_pos, rng_has_gauss, rng_cached = np.random.get_state()
        dump['rng_keys'] = rng_keys
        dump['rng_pos_gauss'] = np.array([rng_pos, rng_has_gauss, rng_cached], dtype=np.float64)
        return dump

    def write(self, dump: Dict[str, Any], action: Any, controller: Any) -> None:
        dump['qd_cmd'] = np.asarray(action.joint_velocities, dtype=np.float64)
        dump['safety_stop'] = bool(action.safety_stop)
        dump['status_message'] = str(action.status_message)
        diag = getattr(controller, 'last_diagnostics', {}) or {}
        for name, value in (diag.get('top_costs', {}) or {}).items():
            dump[f"top_cost_{name}"] = float(value)
        dump['top_candidate'] = str(diag.get('top_candidate', ''))
        try:
            path = os.path.join(self.out_dir, f"action_{dump['action_index']:05d}_t{dump['timestep']:04d}.npz")
            np.savez_compressed(path, **dump)
        except Exception as e:
            print(f"[VISUALIZATION WARNING] MPC dump failed: {e}")
