"""
Live overlay of the MPC imagination in the existing viser scene: sampled TCP rollouts colored by planner,
the executed plan with ghost arms, the IK guidance posture, collision markers on the rollouts, the forecast
obstacle poses along the horizon and the active goal with its approach cone.
"""

import threading
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .sinks import _WorkerSink, load_obj_mesh

GROUP_COLORS = np.array([[40, 110, 230], [240, 140, 30], [150, 80, 200], [30, 170, 90]], dtype=np.float64)
ROOT = "/imagination"


def _xyzw_to_wxyz(q) -> tuple:
    q = np.asarray(q, dtype=np.float64)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _skeleton_segments(skeleton: np.ndarray) -> np.ndarray:
    return np.stack([skeleton[:-1], skeleton[1:]], axis=1)


class ViserImaginationOverlay(_WorkerSink):
    """Redraws at most every `every_n` solves, always from the latest frame (older pending frames are dropped)."""

    def __init__(self, server: Any, every_n: int = 2, forecast_steps=(5, 10, 14), mesh_cluster_size: float = 0.01):
        self.server = server
        self.forecast_steps = tuple(int(s) for s in forecast_steps)
        self.mesh_cluster_size = float(mesh_cluster_size)
        self._ghost_handles: Dict[str, list] = {}
        self._handles: Dict[str, Any] = {}
        self._scene: Optional[Dict[str, Any]] = None
        self._scene_dirty = False
        # The GUI callback runs on a viser thread while frames are drawn on the worker thread.
        self._lock = threading.Lock()
        self.supported = hasattr(server.scene, 'add_line_segments')
        self.enabled = self.supported
        if not self.supported:
            print("[VISUALIZATION WARNING] this viser version has no add_line_segments: imagination overlay disabled "
                  "(the offline recording is unaffected)")
        try:
            folder = server.gui.add_folder("MPC imagination")
            with folder:
                checkbox = server.gui.add_checkbox("Show imagination", initial_value=True)
            checkbox.on_update(lambda _: self._set_visible(checkbox.value))
        except Exception:
            pass
        super().__init__(queue_size=1, every_n=every_n)
        if self.supported:
            print(f"[VISUALIZATION] Live imagination overlay active in viser (every {self.every_n} actions)")

    def _set_visible(self, visible: bool) -> None:
        with self._lock:
            self.enabled = bool(visible) and self.supported
            if self.enabled:
                self._scene_dirty = self._scene is not None
                return
            for handle in self._handles.values():
                try:
                    handle.remove()
                except Exception:
                    pass
            self._handles.clear()
            self._ghost_handles.clear()

    def _keep(self, name: str, handle: Any) -> Any:
        self._handles[name] = handle
        return handle

    def publish_scene(self, scene: Dict[str, Any]) -> None:
        # Not queued: the single-slot queue keeps only the latest frame and would discard a pending scene.
        with self._lock:
            self._scene = scene
            self._scene_dirty = True

    def publish(self, frame: Dict[str, Any]) -> None:
        # Keep only the most recent frame: replace a pending one instead of dropping the new one.
        self._count += 1
        if (self._count - 1) % self.every_n:
            return
        try:
            self._queue.get_nowait()
        except Exception:
            pass
        try:
            self._queue.put_nowait(('frame', frame))
        except Exception:
            self.dropped += 1

    def _lines(self, name: str, segments: np.ndarray, colors: np.ndarray, width: float) -> None:
        scene = self.server.scene
        try:
            h = scene.add_line_segments(name, segments.astype(np.float32), colors.astype(np.uint8),
                                        thickness=width, thickness_units='screen')
        except TypeError:
            h = scene.add_line_segments(name, segments.astype(np.float32), colors.astype(np.uint8), line_width=width)
        self._keep(name, h)

    def _draw_scene(self, scene: Dict[str, Any]) -> None:
        for key in [k for k in scene if k.endswith('_mesh_path')]:
            prefix = key[:-len('_mesh_path')]
            mesh = load_obj_mesh(str(scene[key]), self.mesh_cluster_size)
            if mesh is None:
                continue
            handles = []
            for i, _ in enumerate(self.forecast_steps):
                kwargs = dict(color=(220, 60, 60), wireframe=False)
                try:
                    h = self.server.scene.add_mesh_simple(f"{ROOT}/forecast/{prefix}_{i}", mesh[0], mesh[1],
                                                          opacity=0.12 + 0.06 * i, **kwargs)
                except TypeError:
                    h = self.server.scene.add_mesh_simple(f"{ROOT}/forecast/{prefix}_{i}", mesh[0], mesh[1],
                                                          wireframe=True, color=(220, 60, 60))
                handles.append(self._keep(f"{ROOT}/forecast/{prefix}_{i}", h))
            self._ghost_handles[prefix] = handles

    def _handle_frame(self, frame: Dict[str, Any]) -> None:
        with self._lock:
            if not self.enabled:
                return
            if self._scene_dirty:
                self._draw_scene(self._scene)
                self._scene_dirty = False
            self._draw_frame(frame)

    def _draw_frame(self, frame: Dict[str, Any]) -> None:
        tcp = np.asarray(frame['rollout_tcp'], dtype=np.float64)
        groups = np.asarray(frame['rollout_group'])
        weights = np.asarray(frame['rollout_weight'], dtype=np.float64)
        intensity = np.clip(weights / max(weights.max(), 1e-12), 0.0, 1.0) ** 0.5
        base = GROUP_COLORS[groups]
        colors = base * (0.25 + 0.75 * intensity[:, None]) + 255.0 * 0.75 * (1.0 - intensity[:, None])
        segments = np.stack([tcp[:, :-1], tcp[:, 1:]], axis=2).reshape(-1, 2, 3)
        seg_colors = np.repeat(colors, tcp.shape[1] - 1, axis=0)[:, None, :].repeat(2, axis=1)
        self._lines(f"{ROOT}/rollouts", segments, seg_colors, 1.5)

        hit = np.asarray(frame['rollout_collision'], dtype=np.float64) > 0.0
        if np.any(hit):
            pts = tcp[hit].reshape(-1, 3)
            self._keep(f"{ROOT}/collisions", self.server.scene.add_point_cloud(
                f"{ROOT}/collisions", pts.astype(np.float32),
                np.tile([[230, 30, 30]], (len(pts), 1)).astype(np.uint8), point_size=0.008))
        elif f"{ROOT}/collisions" in self._handles:
            self._handles.pop(f"{ROOT}/collisions").remove()

        if 'plan_tcp' in frame:
            plan = np.asarray(frame['plan_tcp'], dtype=np.float64)
            self._lines(f"{ROOT}/plan", np.stack([plan[:-1], plan[1:]], axis=1),
                        np.full((len(plan) - 1, 2, 3), 20.0), 5.0)
        if 'ghost_skeletons' in frame:
            for i, sk in enumerate(np.asarray(frame['ghost_skeletons'])):
                shade = 120.0 + 100.0 * i / max(len(frame['ghost_skeletons']) - 1, 1)
                self._lines(f"{ROOT}/ghost_{i}", _skeleton_segments(sk), np.full((len(sk) - 1, 2, 3), shade), 6.0)
        if 'q_des_skeleton' in frame:
            sk = np.asarray(frame['q_des_skeleton'])
            self._lines(f"{ROOT}/q_des", _skeleton_segments(sk), np.tile([[[40, 200, 90]]], (len(sk) - 1, 2, 1)), 3.0)

        if 'target_pos' in frame:
            pos = np.asarray(frame['target_pos'], dtype=np.float64)
            wxyz = (1.0, 0.0, 0.0, 0.0)
            if 'target_rot' in frame:
                wxyz = _xyzw_to_wxyz(Rotation.from_matrix(np.asarray(frame['target_rot'])).as_quat())
            self._keep(f"{ROOT}/goal", self.server.scene.add_frame(
                f"{ROOT}/goal", axes_length=0.06, axes_radius=0.003, position=tuple(pos), wxyz=wxyz))

        for oid in np.asarray(frame.get('obstacle_ids', []), dtype=np.int64):
            handles = self._ghost_handles.get(f"obs{oid}")
            pred_pos, pred_quat = frame.get(f"obs{oid}_pred_pos"), frame.get(f"obs{oid}_pred_quat")
            if not handles or pred_pos is None:
                continue
            # A static obstacle would show its forecast copies on top of the obstacle itself.
            moving = float(np.linalg.norm(pred_pos[-1] - pred_pos[0])) > 1e-3
            for h, step in zip(handles, self.forecast_steps):
                step = min(step, len(pred_pos) - 1)
                h.visible = moving
                h.position = tuple(np.asarray(pred_pos[step], dtype=np.float64))
                h.wxyz = _xyzw_to_wxyz(pred_quat[step])
