"""
Offline video of the MPC imagination, composed from the server recording (visualization.imagination.record_dir)
and, optionally, the client simulation frames (run_coppelia_client.py --record_dir):

    title       : DREMA Dynamic System, time, chosen candidate, Judge weights
    left top    : simulated scene (client camera)
    right       : 3D imagination - current arm and gripper, sampled TCP rollouts colored by planner and weighted
                  by the softmax, executed plan with ghost arms (and the held object along it), IK guidance
                  posture, collision hits, perceived objects in their colours outlined by role, forecast
                  obstacles, goal gripper pose and approach cone
    left bottom : telemetry up to the current time (distances, cost terms, Judge weights, closest-obstacle speed)

Usage:
    python -m drema.visualization.render_imagination_video --frames <record_dir/run> [--sim <sim_dir/run>]
           [--out imagination.mp4] [--fps 10] [--stride 1] [--zoom 0.32 | --overview] [--still 120 --still 145]
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from drema.roles.registry import PANDA_HAND_FOOTPRINT  # noqa: E402

GROUP_COLORS = np.array([[40, 110, 230], [240, 140, 30], [150, 80, 200], [30, 170, 90]], dtype=np.float64) / 255.0
GROUP_LABELS = ("greedy", "sensitive", "mixed", "primitives")
FORECAST_STEPS = (5, 10, 14)

# Object roles (drema.roles) drawn as outline and label; plain obstacles keep only their perceived mesh
ROLE_COLORS = {'TARGET': (0.1, 0.65, 0.25), 'NEAR': (0.95, 0.55, 0.1), 'HELD': (0.55, 0.25, 0.75),
               'RELEASED': (0.45, 0.8, 0.45), 'FRAGILE': (0.85, 0.1, 0.1)}
DEFAULT_MESH_COLOR = (0.62, 0.62, 0.62)
GOAL_GRIPPER_COLOR = (0.0, 0.5, 0.55)

# Panda gripper in the hand frame (z along the approach, y along the finger opening, origin at the TCP)
HAND_R = Rotation.from_euler('z', PANDA_HAND_FOOTPRINT['finger_axis_yaw_deg'], degrees=True).as_matrix()
PALM_BOX = (np.array([0.0, -0.004, -0.088]), np.array([0.032, 0.104, 0.049]))
FINGER_HALF = np.array([0.011, 0.0125, 0.028])
FINGER_Z = -0.025
FINGER_STROKE = 0.04
BOX_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
BOX_PAIRS = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def load_frames(frames_dir: str) -> List[Dict[str, np.ndarray]]:
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.npz")))
    if not files:
        raise FileNotFoundError(f"no frame_*.npz in '{frames_dir}'")
    return [dict(np.load(f, allow_pickle=False)) for f in files]


def segment_starts(steps: List[int], gap: int) -> List[int]:
    """
    Indices where an episode starts: the step counter goes back (client reset) or jumps by more than `gap` steps
    (the client keeps counting while idle). Applied identically to the server and the client streams.
    """
    return [0] + [i for i in range(1, len(steps)) if steps[i] < steps[i - 1] or steps[i] - steps[i - 1] > gap]


def split_by_steps(items: list, steps: List[int], gap: int) -> List[list]:
    starts = segment_starts(steps, gap) + [len(items)]
    return [items[a:b] for a, b in zip(starts[:-1], starts[1:]) if b > a]


def load_sim_episodes(sim_dir: Optional[str], gap: int) -> List[Dict[int, str]]:
    """Simulation frames per episode, keyed by step, in recording order."""
    if not sim_dir:
        return []
    rows = []
    with open(os.path.join(sim_dir, "index.csv")) as f:
        next(f)
        for line in f:
            _, step, _, name = line.strip().split(",")
            rows.append((int(step), os.path.join(sim_dir, name)))
    rows.sort(key=lambda r: r[1])
    return [dict(seg) for seg in split_by_steps(rows, [r[0] for r in rows], gap)]


def _pose(points: np.ndarray, pos, quat) -> np.ndarray:
    return Rotation.from_quat(np.asarray(quat, dtype=np.float64)).apply(points) + np.asarray(pos, dtype=np.float64)


def _box_edges(vertices: np.ndarray) -> Tuple[np.ndarray, list]:
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return corners, pairs


def _clip_segment(p0: np.ndarray, p1: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Optional[np.ndarray]:
    """Liang-Barsky clipping of a 3D segment to an axis-aligned box (mplot3d does not clip to the axes)."""
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for k in range(3):
        for p, q in ((-d[k], p0[k] - lo[k]), (d[k], hi[k] - p0[k])):
            if abs(p) < 1e-12:
                if q < 0:
                    return None
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return None
    return np.stack([p0 + t0 * d, p0 + t1 * d])


def gripper_segments(tcp: np.ndarray, R_ee: np.ndarray, opening: float) -> List[np.ndarray]:
    """Edges of the palm and of the two fingers, with the fingers at the given opening (1 open, 0 closed)."""
    R = np.asarray(R_ee, dtype=np.float64) @ HAND_R
    inner = FINGER_STROKE * float(np.clip(opening, 0.0, 1.0))
    boxes = [PALM_BOX] + [(np.array([0.0, side * (inner + FINGER_HALF[1]), FINGER_Z]), FINGER_HALF) for side in (-1, 1)]
    segments = []
    for center, half in boxes:
        corners = (center + BOX_SIGNS * half) @ R.T + np.asarray(tcp, dtype=np.float64)
        segments.extend(np.stack([corners[a], corners[b]]) for a, b in BOX_PAIRS)
    return segments


class ImaginationRenderer:
    def __init__(self, frames: List[Dict[str, np.ndarray]], scene: Dict[str, np.ndarray],
                 sim_index: Dict[int, str], dt: float, size=(1920, 1080), dpi: int = 100,
                 view=(22.0, -60.0), zoom: Optional[float] = 0.32, cost_ylim: Optional[float] = None):
        self.frames = frames
        self.scene = scene
        self.sim_index = sim_index
        self.sim_steps = np.array(sorted(sim_index)) if sim_index else np.zeros(0, dtype=int)
        self.dt = dt
        self.view = view
        self.cost_ylim = cost_ylim
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self._telemetry = self._collect_telemetry()
        self.center, self.half = self._scene_limits(zoom)
        self.lo, self.hi = self.center - self.half, self.center + self.half
        self._kin = None

    def _closest_obstacle(self, fr) -> Tuple[Optional[int], float]:
        """Obstacle nearest to the arm (mesh vertices vs link origins) and its forecast speed; objects the robot may
        hold or touch as targets are not obstacles."""
        best, best_d = None, np.inf
        roles = self._roles(fr)
        held = set(int(i) for i in fr.get('held_ids', []))
        for oid in fr.get('obstacle_ids', np.zeros(0, dtype=int)):
            if f"obs{oid}_pos" not in fr or int(oid) in held or roles.get(int(oid)) in ('TARGET', 'HELD', 'RELEASED'):
                continue
            verts = self.scene.get(f"obs{oid}_vertices")
            pts = (_pose(verts, fr[f"obs{oid}_pos"], fr[f"obs{oid}_quat"]) if verts is not None
                   else np.asarray(fr[f"obs{oid}_pos"])[None])
            d = float(np.min(np.linalg.norm(pts[:, None, :] - fr['skeleton'][None, :, :], axis=-1)))
            if d < best_d:
                best, best_d = int(oid), d
        if best is None or f"obs{best}_pred_vel" not in fr:
            return best, 0.0
        return best, float(np.linalg.norm(fr[f"obs{best}_pred_vel"][0]))

    def _collect_telemetry(self) -> Dict[str, np.ndarray]:
        t, d_goal, min_d, goal, coll, funnel, w_grd, w_sns, v_obs, obs_id = ([] for _ in range(10))
        for fr in self.frames:
            t.append(float(fr['timestep']) * self.dt)
            tcp = fr['skeleton'][-1]
            d_goal.append(float(np.linalg.norm(tcp - fr['target_pos'])) if 'target_pos' in fr else np.nan)
            min_d.append(float(fr.get('min_obstacle_distance', np.nan)))
            goal.append(float(fr.get('top_cost_goal', np.nan)))
            coll.append(float(fr.get('top_cost_coll', np.nan)))
            funnel.append(float(fr.get('top_cost_funnel', np.nan)))
            w_grd.append(float(fr['mixing_weights'][0]))
            w_sns.append(float(fr['mixing_weights'][1]))
            oid, speed = self._closest_obstacle(fr)
            v_obs.append(speed)
            obs_id.append(-1 if oid is None else oid)
        return {k: np.asarray(v) for k, v in dict(t=t, d_goal=d_goal, min_d=min_d, goal=goal, coll=coll, funnel=funnel,
                                                   w_grd=w_grd, w_sns=w_sns, v_obs=v_obs, obs_id=obs_id).items()}

    def _obstacle_name(self, oid: int) -> str:
        name = self.scene.get(f"obs{oid}_name")
        return str(name) if name is not None else f"obstacle {oid}"

    def _scene_limits(self, zoom: Optional[float]) -> Tuple[np.ndarray, float]:
        """Close-up cube around the lowest goal (the grasp) or, without zoom, a cube enclosing the whole arm."""
        goals = np.array([fr['target_pos'] for fr in self.frames if 'target_pos' in fr])
        if zoom and len(goals):
            center = goals[np.argmin(goals[:, 2])] + np.array([-0.05, 0.0, 0.12])
            return center, float(zoom)
        pts = np.concatenate([fr['skeleton'] for fr in self.frames[::max(1, len(self.frames) // 20)]] +
                             ([goals[:1]] if len(goals) else []), axis=0)
        return 0.5 * (pts.min(axis=0) + pts.max(axis=0)), 0.55 * float(np.max(pts.max(axis=0) - pts.min(axis=0)))

    def _inside(self, pts: np.ndarray) -> np.ndarray:
        return np.all((pts >= self.lo) & (pts <= self.hi), axis=-1)

    def _polylines(self, ax, polylines, colors, lw: float, ls: str = 'solid', zorder: int = 2) -> None:
        """Draws polylines clipped to the view cube; colors is one RGBA per polyline."""
        segs, cols = [], []
        for line, color in zip(polylines, colors):
            line = np.asarray(line, dtype=np.float64)
            for a, b in zip(line[:-1], line[1:]):
                clipped = _clip_segment(a, b, self.lo, self.hi)
                if clipped is not None:
                    segs.append(clipped)
                    cols.append(color)
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=lw, linestyles=ls, zorder=zorder))

    @staticmethod
    def _roles(fr) -> Dict[int, str]:
        ids, names = fr.get('role_ids'), fr.get('role_names')
        if ids is None or names is None:
            return {}
        return {int(i): str(n) for i, n in zip(ids, names)}

    def _face_colors(self, oid: int, faces: np.ndarray, alpha: float) -> np.ndarray:
        colors = self.scene.get(f"obs{oid}_colors")
        rgb = (np.asarray(colors, dtype=np.float64)[faces].mean(axis=1) if colors is not None
               else np.tile(DEFAULT_MESH_COLOR, (len(faces), 1)))
        return np.concatenate([rgb, np.full((len(faces), 1), alpha)], axis=1)

    def _outline(self, ax, verts: np.ndarray, pos, quat, color, alpha: float = 1.0, lw: float = 1.6,
                 ls: str = 'solid') -> np.ndarray:
        corners, pairs = _box_edges(verts)
        c = _pose(corners, pos, quat)
        self._polylines(ax, [[c[a], c[b]] for a, b in pairs], [(*color, alpha)] * len(pairs), lw=lw, ls=ls, zorder=4)
        return c

    def _draw_obstacles(self, ax, fr) -> None:
        roles = self._roles(fr)
        held = set(int(i) for i in fr.get('held_ids', []))
        for oid in fr.get('obstacle_ids', np.zeros(0, dtype=int)):
            oid = int(oid)
            verts, faces = self.scene.get(f"obs{oid}_vertices"), self.scene.get(f"obs{oid}_faces")
            if verts is None or f"obs{oid}_pos" not in fr:
                continue
            world = _pose(verts, fr[f"obs{oid}_pos"], fr[f"obs{oid}_quat"])
            tris = world[faces]
            inside = np.all(self._inside(tris), axis=1)
            if np.any(inside):
                ax.add_collection3d(Poly3DCollection(tris[inside], facecolor=self._face_colors(oid, faces[inside], 0.6),
                                                     edgecolor=(0.2, 0.2, 0.2, 0.06), linewidths=0.2))
            role = roles.get(oid, 'HELD' if oid in held else 'OBSTACLE')
            color = ROLE_COLORS.get(role)
            if color is not None:
                c = self._outline(ax, verts, fr[f"obs{oid}_pos"], fr[f"obs{oid}_quat"], color)
                top = c[np.argmax(c[:, 2])] + np.array([0.0, 0.0, 0.03])
                if self._inside(top[None])[0]:
                    ax.text(*top, role, color=color, fontsize=9, fontweight='bold', zorder=12)
            for i in range(len(fr.get(f"held{oid}_ghost_pos", []))):
                self._outline(ax, verts, fr[f"held{oid}_ghost_pos"][i], fr[f"held{oid}_ghost_quat"][i],
                              ROLE_COLORS['HELD'], alpha=0.2 + 0.15 * i, lw=1.2, ls='dashed')
            if oid in held:
                continue
            pred_pos, pred_quat = fr.get(f"obs{oid}_pred_pos"), fr.get(f"obs{oid}_pred_quat")
            if pred_pos is None or float(np.linalg.norm(pred_pos[-1] - pred_pos[0])) < 1e-3:
                continue
            corners, pairs = _box_edges(verts)
            for i, step in enumerate(FORECAST_STEPS):
                c = _pose(corners, pred_pos[min(step, len(pred_pos) - 1)], pred_quat[min(step, len(pred_pos) - 1)])
                self._polylines(ax, [[c[a], c[b]] for a, b in pairs], [(0.85, 0.2, 0.2, 0.3 + 0.2 * i)] * 12,
                                lw=1.0, ls='dashed')

    def _draw_table(self, ax) -> None:
        z, b = self.scene.get('table_z'), self.scene.get('table_bounds')
        if z is None or b is None:
            return
        z, (x0, x1, y0, y1) = float(z), np.asarray(b, dtype=np.float64)
        if not (self.lo[2] <= z <= self.hi[2]):
            return
        x0, x1 = max(x0, self.lo[0]), min(x1, self.hi[0])
        y0, y1 = max(y0, self.lo[1]), min(y1, self.hi[1])
        quad = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]])
        ax.add_collection3d(Poly3DCollection([quad], facecolor=(0.8, 0.78, 0.72, 0.45), edgecolor=(0.6, 0.6, 0.6, 0.5)))

    def _draw_goal(self, ax, fr) -> None:
        if 'target_pos' not in fr:
            return
        g = fr['target_pos']
        ax.scatter(*g, marker='*', s=260, color=(0.9, 0.1, 0.1), edgecolor='k', linewidths=0.5, depthshade=False, zorder=10)
        if 'target_rot' in fr and int(fr.get('approach_funnel', -1)) != 0:
            u = -np.asarray(fr['target_rot'])[:, 2]
            a = np.cross(u, [1.0, 0.0, 0.0])
            a = a / max(np.linalg.norm(a), 1e-9)
            b = np.cross(u, a)
            s, r = 0.12, 0.01 + np.tan(np.radians(30.0)) * 0.12
            ring = np.array([g + s * u + r * (np.cos(th) * a + np.sin(th) * b) for th in np.linspace(0, 2 * np.pi, 33)])
            spokes = [np.stack([g, ring[k]]) for k in range(0, 32, 4)]
            self._polylines(ax, [ring] + spokes, [(0.9, 0.3, 0.3, 0.55)] * (1 + len(spokes)), lw=0.9)

    def _tcp_rot(self, fr) -> np.ndarray:
        """Recordings made before the gripper was recorded only have the joints: their TCP rotation comes from FK."""
        if 'tcp_rot' in fr:
            return np.asarray(fr['tcp_rot'], dtype=np.float64)
        if self._kin is None:
            from drema.controller.franka_kinematics import FrankaKinematics
            self._kin = FrankaKinematics(base_position=np.asarray(self.scene.get('robot_base', np.zeros(3))))
        return np.asarray(self._kin.forward_kinematics_ee(np.asarray(fr['q'], dtype=np.float64))[1], dtype=np.float64)

    def _draw_gripper(self, ax, tcp, R_ee, opening: float, color, lw: float, ls: str = 'solid', zorder: int = 5) -> None:
        segments = gripper_segments(tcp, R_ee, opening)
        self._polylines(ax, segments, [color] * len(segments), lw=lw, ls=ls, zorder=zorder)

    def _draw_robot(self, ax, fr) -> None:
        opening = float(fr.get('gripper_open', 1.0))
        ghosts = list(fr.get('ghost_skeletons', []))
        self._polylines(ax, ghosts, [(0.25, 0.25, 0.25, 0.15 + 0.12 * i) for i in range(len(ghosts))], lw=5.0)
        for i, quat in enumerate(fr.get('ghost_tcp_quat', [])):
            self._draw_gripper(ax, ghosts[i][-1], Rotation.from_quat(quat).as_matrix(), opening,
                               (0.25, 0.25, 0.25, 0.15 + 0.12 * i), lw=1.2)
        if 'target_pos' in fr and 'target_rot' in fr:
            self._draw_gripper(ax, fr['target_pos'], fr['target_rot'], 1.0, (*GOAL_GRIPPER_COLOR, 0.8), lw=1.0, ls='dashed')
        if 'q_des_skeleton' in fr:
            self._polylines(ax, [fr['q_des_skeleton']], [(0.15, 0.7, 0.3, 0.85)], lw=2.0, ls='dashed')
        sk = fr['skeleton']
        self._polylines(ax, [sk], [(0.12, 0.12, 0.12, 1.0)], lw=6.0, zorder=5)
        joints = sk[:-1][self._inside(sk[:-1])]
        if len(joints):
            ax.scatter(*joints.T, color=(0.12, 0.12, 0.12), s=30, depthshade=False)
        self._draw_gripper(ax, sk[-1], self._tcp_rot(fr), opening, (0.12, 0.12, 0.12, 1.0), lw=2.0, zorder=6)

    def _draw_rollouts(self, ax, fr) -> None:
        tcp, groups = fr['rollout_tcp'].astype(np.float64), fr['rollout_group']
        w = fr['rollout_weight'].astype(np.float64)
        alpha = 0.15 + 0.8 * np.clip(w / max(w.max(), 1e-12), 0.0, 1.0) ** 0.5
        order = np.argsort(alpha)
        self._polylines(ax, tcp[order], [(*GROUP_COLORS[int(groups[i])], float(alpha[i])) for i in order], lw=1.3)
        if 'contact_position' in fr:
            pts, dist = fr['contact_position'].astype(np.float64), fr['contact_distance'].astype(np.float64)
            inside = self._inside(pts)
            if np.any(inside):
                # darker and larger the deeper the point is in the collision margin
                closeness = np.clip(1.0 - dist[inside] / max(float(dist.max()), 1e-6), 0.0, 1.0)
                ax.scatter(*pts[inside].T, c=[(0.85, 0.05, 0.05, 0.25 + 0.7 * c) for c in closeness],
                           s=4 + 14 * closeness, depthshade=False)
        else:
            hit = fr['rollout_collision'] > 0.0
            if np.any(hit):
                pts = tcp[hit].reshape(-1, 3)
                pts = pts[self._inside(pts)]
                if len(pts):
                    ax.scatter(*pts.T, color=(0.9, 0.1, 0.1), s=4, alpha=0.5, depthshade=False)
        if 'plan_tcp' in fr:
            self._polylines(ax, [fr['plan_tcp']], [(0.0, 0.0, 0.0, 1.0)], lw=3.2, zorder=6)

    def _draw_3d(self, ax, fr) -> None:
        ax.set_xlim(self.lo[0], self.hi[0])
        ax.set_ylim(self.lo[1], self.hi[1])
        ax.set_zlim(self.lo[2], self.hi[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=self.view[0], azim=self.view[1])
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
        ax.tick_params(labelsize=8)
        self._draw_table(ax)
        self._draw_obstacles(ax, fr)
        self._draw_goal(ax, fr)
        self._draw_rollouts(ax, fr)
        self._draw_robot(ax, fr)
        handles = [plt.Line2D([], [], color=(0.12, 0.12, 0.12), lw=5, marker='o', markersize=4),
                   plt.Line2D([], [], color=(0.12, 0.12, 0.12), lw=2, marker='s', markersize=7, fillstyle='none'),
                   Patch(facecolor=(*DEFAULT_MESH_COLOR, 0.6), edgecolor=(0.2, 0.2, 0.2)),
                   plt.Line2D([], [], color=(0.9, 0.1, 0.1), marker='*', ls='', markersize=12, markeredgecolor='k'),
                   plt.Line2D([], [], color=GOAL_GRIPPER_COLOR, lw=1, ls='--', marker='s', markersize=7, fillstyle='none'),
                   plt.Line2D([], [], color=(0.9, 0.3, 0.3), lw=1)]
        handles += [plt.Line2D([], [], color=c, lw=2) for c in GROUP_COLORS]
        handles += [plt.Line2D([], [], color='k', lw=3), plt.Line2D([], [], color=(0.4, 0.4, 0.4), lw=5, alpha=0.4),
                    plt.Line2D([], [], color=(0.15, 0.7, 0.3), lw=2, ls='--'),
                    plt.Line2D([], [], color=(0.85, 0.2, 0.2), lw=1, ls='--'),
                    plt.Line2D([], [], color=(0.85, 0.05, 0.05), marker='o', ls='', markersize=4)]
        labels = ['arm (current configuration)', 'gripper (measured opening)', 'object (perceived mesh and colour)',
                  'active goal', 'gripper pose at the goal', 'approach cone']
        labels += [f"rollouts: {g}" for g in GROUP_LABELS]
        labels += ['executed plan (TCP path)', 'imagined arm along the plan', 'IK guidance posture q_des',
                   'forecast obstacle (+0.25 / 0.5 / 0.7 s)', 'arm points inside the collision margin']
        main_legend = ax.legend(handles, labels, loc='upper left', fontsize=8.5, framealpha=0.85)
        role_names = list(ROLE_COLORS)
        role_handles = [Patch(facecolor='none', edgecolor=ROLE_COLORS[r], lw=1.6) for r in role_names]
        role_handles.append(Patch(facecolor='none', edgecolor=ROLE_COLORS['HELD'], lw=1.2, ls='--'))
        ax.legend(role_handles, [f"role {r}" for r in role_names] + ['held object along the plan'],
                  loc='lower left', fontsize=8.5, framealpha=0.85, title='object outline', title_fontsize=8.5)
        ax.add_artist(main_legend)

    def _draw_sim(self, ax, fr) -> None:
        ax.axis('off')
        if not len(self.sim_steps):
            ax.text(0.5, 0.5, 'no simulation recording', ha='center', va='center', transform=ax.transAxes)
            return
        idx = np.searchsorted(self.sim_steps, int(fr['timestep']), side='right') - 1
        ax.imshow(plt.imread(self.sim_index[int(self.sim_steps[max(idx, 0)])]))
        ax.set_title('Simulated scene (CoppeliaSim)', fontsize=11)

    def _draw_telemetry(self, axes, k: int) -> None:
        T = self._telemetry
        t, now = T['t'][:k + 1], T['t'][k]
        span = (T['t'][0], max(T['t'][-1], T['t'][0] + 1e-3))
        a0, a1, a2 = axes
        a0.plot(t, T['d_goal'][:k + 1] * 100, color='k', label='TCP - active goal [cm]')
        a0.plot(t, T['min_d'][:k + 1] * 100, color=(0.85, 0.2, 0.2), label='min robot - obstacle [cm]')
        a1.plot(t, T['goal'][:k + 1], color=(0.2, 0.2, 0.2), label='goal')
        a1.plot(t, T['coll'][:k + 1], color=(0.85, 0.2, 0.2), label='collision')
        a1.plot(t, T['funnel'][:k + 1], color=(0.6, 0.3, 0.7), label='approach cone')
        a2.plot(t, T['w_grd'][:k + 1], color=GROUP_COLORS[0], label='w greedy')
        a2.plot(t, T['w_sns'][:k + 1], color=GROUP_COLORS[1], label='w sensitive')
        a2b = a2.twinx()
        current = int(T['obs_id'][k])
        label = f"|v| closest obstacle ({self._obstacle_name(current)})" if current >= 0 else '|v| closest obstacle'
        a2b.plot(t, T['v_obs'][:k + 1], color=(0.4, 0.4, 0.4), lw=1, ls=':', label=label)
        a2b.set_ylim(0, max(0.05, float(np.nanmax(T['v_obs'])) * 1.2))
        a2b.set_ylabel('m/s', fontsize=8)
        a2b.tick_params(labelsize=7)
        a2b.legend(fontsize=7, loc='lower right')
        dist = np.r_[T['d_goal'], T['min_d']]
        a0.set_ylim(0, max(10.0, float(np.nanmax(dist[np.isfinite(dist)])) * 100 * 1.05))
        costs = np.r_[T['goal'], T['coll'], T['funnel']]
        costs = costs[np.isfinite(costs)]
        top = self.cost_ylim if self.cost_ylim else (max(10.0, float(costs.max()) * 1.1) if costs.size else 10.0)
        a1.set_ylim(0, top)
        a2.set_ylim(-0.05, 1.05)
        for a, title in zip(axes, ('distances', 'cost terms of the chosen candidate',
                                   'Judge weights and closest-obstacle speed')):
            a.set_xlim(*span)
            a.axvline(now, color=(0.5, 0.5, 0.5), lw=0.8)
            a.set_title(title, fontsize=9, loc='left')
            a.legend(fontsize=7, loc='upper right')
            a.tick_params(labelsize=7)
        a2.set_xlabel('time [s]', fontsize=8)

    def render(self, k: int) -> np.ndarray:
        fig = self.fig
        fig.clf()
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.15],
                      left=0.03, right=0.99, top=0.93, bottom=0.05, wspace=0.12, hspace=0.18)
        fr = self.frames[k]
        self._draw_sim(fig.add_subplot(gs[0, 0]), fr)
        self._draw_3d(fig.add_subplot(gs[:, 1:], projection='3d'), fr)
        fig.text(0.665, 0.012,
                 "Rollouts: TCP paths (gripper centre point) sampled by the MPC over the next 0.75 s; opacity = softmax "
                 "weight. Each cycle only the first step of the executed plan is applied, then the MPC replans.",
                 ha='center', va='bottom', fontsize=9, color=(0.3, 0.3, 0.3))
        sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0], hspace=0.55)
        self._draw_telemetry([fig.add_subplot(sub[i]) for i in range(3)], k)
        funnel = {1: 'on', 0: 'off'}.get(int(fr.get('approach_funnel', -1)), 'default')
        fig.suptitle(f"DREMA Dynamic System  |  t = {float(fr['timestep']) * self.dt:5.2f} s (step {int(fr['timestep'])})"
                     f"  |  chosen: {fr.get('best_label', '')}  |  w_greedy {fr['mixing_weights'][0]:.2f}"
                     f" / w_sensitive {fr['mixing_weights'][1]:.2f}  |  approach cone {funnel}", fontsize=13)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the MPC imagination recording into a video")
    parser.add_argument("--frames", required=True, help="Server imagination recording folder (frame_*.npz, scene.npz)")
    parser.add_argument("--sim", default=None, help="Client simulation recording folder (index.csv, sim_*.jpg)")
    parser.add_argument("--out", default="imagination.mp4", help="Output video (.mp4)")
    parser.add_argument("--fps", type=float, default=10.0, help="Video frame rate (default 10: half speed at 20 Hz control)")
    parser.add_argument("--stride", type=int, default=1, help="Use every N-th recorded frame")
    parser.add_argument("--dt", type=float, default=0.05, help="Control period [s] used for the time axis")
    parser.add_argument("--view", type=float, nargs=2, default=[22.0, -60.0], metavar=("ELEV", "AZIM"))
    parser.add_argument("--zoom", type=float, default=0.32, help="Half size [m] of the close-up cube around the grasp goal")
    parser.add_argument("--overview", action="store_true", help="Frame the whole arm instead of the close-up")
    parser.add_argument("--cost_ylim", type=float, default=None,
                        help="Upper limit of the cost plot (default: 10%% above the largest cost of the run)")
    parser.add_argument("--episode", type=int, default=-1,
                        help="Episode to render when the recording contains client resets (default: the last one)")
    parser.add_argument("--gap", type=int, default=60,
                        help="Steps without frames that separate two episodes (default: 60)")
    parser.add_argument("--still", type=int, action="append", default=[],
                        help="Instead of a video, save a PNG of the frame at this timestep (repeatable)")
    args = parser.parse_args()

    all_frames = load_frames(args.frames)
    episodes = split_by_steps(all_frames, [int(fr['timestep']) for fr in all_frames], args.gap)
    sim_episodes = load_sim_episodes(args.sim, args.gap)
    if len(episodes) > 1 or len(sim_episodes) > 1:
        print(f"episodes found: {len(episodes)} in the server recording, {len(sim_episodes)} in the simulation "
              f"recording; rendering episode {args.episode % len(episodes)}")
    if sim_episodes and len(sim_episodes) != len(episodes):
        print("[warning] the two recordings have a different number of episodes: check the camera panel, or use --gap")
    frames = episodes[args.episode]
    sim_index = sim_episodes[args.episode] if len(sim_episodes) == len(episodes) else (sim_episodes[-1] if sim_episodes else {})
    scene_path = os.path.join(args.frames, "scene.npz")
    scene = dict(np.load(scene_path, allow_pickle=False)) if os.path.exists(scene_path) else {}
    renderer = ImaginationRenderer(frames, scene, sim_index, dt=args.dt, view=tuple(args.view),
                                   zoom=None if args.overview else args.zoom, cost_ylim=args.cost_ylim)
    steps = np.array([int(fr['timestep']) for fr in frames])

    if args.still:
        base = os.path.splitext(args.out)[0]
        for ts in args.still:
            k = int(np.argmin(np.abs(steps - ts)))
            path = f"{base}_t{int(steps[k]):05d}.png"
            plt.imsave(path, renderer.render(k))
            print(f"saved {path}")
        return

    import imageio
    indices = list(range(0, len(frames), max(1, args.stride)))
    with imageio.get_writer(args.out, fps=args.fps, codec='libx264', quality=8, macro_block_size=8) as writer:
        for n, k in enumerate(indices):
            writer.append_data(renderer.render(k))
            if n % 20 == 0:
                print(f"rendered {n + 1}/{len(indices)} frames")
    print(f"saved {args.out} ({len(indices)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
