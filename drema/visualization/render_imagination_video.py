"""
Offline video of the MPC imagination, composed from the server recording (visualization.imagination.record_dir)
and, optionally, the client simulation frames (run_coppelia_client.py --record_dir):

    left top    : simulated scene (client camera)
    right       : 3D imagination - current arm, sampled TCP rollouts colored by planner and weighted by the
                  softmax, executed plan with ghost arms, IK guidance posture, collision hits, current and
                  forecast obstacle, goal and approach cone
    left bottom : telemetry up to the current time (distances, cost terms, Judge weights, obstacle speed)

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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

GROUP_COLORS = np.array([[40, 110, 230], [240, 140, 30], [150, 80, 200], [30, 170, 90]], dtype=np.float64) / 255.0
GROUP_LABELS = ("greedy", "sensitive", "mixed", "primitives")
FORECAST_STEPS = (5, 10, 14)


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


class ImaginationRenderer:
    def __init__(self, frames: List[Dict[str, np.ndarray]], scene: Dict[str, np.ndarray],
                 sim_index: Dict[int, str], dt: float, size=(1920, 1080), dpi: int = 100,
                 view=(22.0, -60.0), zoom: Optional[float] = 0.32):
        self.frames = frames
        self.scene = scene
        self.sim_index = sim_index
        self.sim_steps = np.array(sorted(sim_index)) if sim_index else np.zeros(0, dtype=int)
        self.dt = dt
        self.view = view
        self.fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
        self._telemetry = self._collect_telemetry()
        self.center, self.half = self._scene_limits(zoom)
        self.lo, self.hi = self.center - self.half, self.center + self.half

    def _collect_telemetry(self) -> Dict[str, np.ndarray]:
        t, d_goal, min_d, goal, coll, funnel, w_grd, w_sns, v_obs = ([] for _ in range(9))
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
            ids = fr.get('obstacle_ids', np.zeros(0, dtype=int))
            v_obs.append(max((float(np.linalg.norm(fr[f"obs{o}_pred_vel"][0])) for o in ids if f"obs{o}_pred_vel" in fr),
                             default=0.0))
        return {k: np.asarray(v) for k, v in dict(t=t, d_goal=d_goal, min_d=min_d, goal=goal, coll=coll, funnel=funnel,
                                                   w_grd=w_grd, w_sns=w_sns, v_obs=v_obs).items()}

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

    def _draw_obstacles(self, ax, fr) -> None:
        for oid in fr.get('obstacle_ids', np.zeros(0, dtype=int)):
            verts, faces = self.scene.get(f"obs{oid}_vertices"), self.scene.get(f"obs{oid}_faces")
            if verts is None or f"obs{oid}_pos" not in fr:
                continue
            world = _pose(verts, fr[f"obs{oid}_pos"], fr[f"obs{oid}_quat"])
            tris = world[faces]
            tris = tris[np.all(self._inside(tris), axis=1)]
            if len(tris):
                ax.add_collection3d(Poly3DCollection(tris, facecolor=(0.25, 0.45, 0.85, 0.35),
                                                     edgecolor=(0.2, 0.35, 0.7, 0.08), linewidths=0.2))
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

    def _draw_robot(self, ax, fr) -> None:
        ghosts = list(fr.get('ghost_skeletons', []))
        self._polylines(ax, ghosts, [(0.25, 0.25, 0.25, 0.15 + 0.12 * i) for i in range(len(ghosts))], lw=5.0)
        if 'q_des_skeleton' in fr:
            self._polylines(ax, [fr['q_des_skeleton']], [(0.15, 0.7, 0.3, 0.85)], lw=2.0, ls='dashed')
        sk = fr['skeleton']
        self._polylines(ax, [sk], [(0.12, 0.12, 0.12, 1.0)], lw=6.0, zorder=5)
        joints = sk[:-1][self._inside(sk[:-1])]
        if len(joints):
            ax.scatter(*joints.T, color=(0.12, 0.12, 0.12), s=30, depthshade=False)

    def _draw_rollouts(self, ax, fr) -> None:
        tcp, groups = fr['rollout_tcp'].astype(np.float64), fr['rollout_group']
        w = fr['rollout_weight'].astype(np.float64)
        alpha = 0.15 + 0.8 * np.clip(w / max(w.max(), 1e-12), 0.0, 1.0) ** 0.5
        order = np.argsort(alpha)
        self._polylines(ax, tcp[order], [(*GROUP_COLORS[int(groups[i])], float(alpha[i])) for i in order], lw=1.3)
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
        handles = [plt.Line2D([], [], color=c, lw=2) for c in GROUP_COLORS]
        handles += [plt.Line2D([], [], color='k', lw=3), plt.Line2D([], [], color=(0.4, 0.4, 0.4), lw=5, alpha=0.4),
                    plt.Line2D([], [], color=(0.15, 0.7, 0.3), lw=2, ls='--'),
                    plt.Line2D([], [], color=(0.85, 0.2, 0.2), lw=1, ls='--'),
                    plt.Line2D([], [], color=(0.9, 0.1, 0.1), marker='o', ls='', markersize=3)]
        ax.legend(handles, [f"rollouts: {g}" for g in GROUP_LABELS] +
                  ['executed plan (TCP)', 'imagined arm along the plan', 'IK guidance q_des', 'forecast obstacle',
                   'rollout steps in collision margin'], loc='upper left', fontsize=9, framealpha=0.85)

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
        a2b.plot(t, T['v_obs'][:k + 1], color=(0.4, 0.4, 0.4), lw=1, ls=':', label='|v| obstacle (forecast)')
        a2b.set_ylim(0, max(0.05, float(np.nanmax(T['v_obs'])) * 1.2))
        a2b.set_ylabel('m/s', fontsize=8)
        a2b.tick_params(labelsize=7)
        a2b.legend(fontsize=7, loc='lower right')
        dist = np.r_[T['d_goal'], T['min_d']]
        a0.set_ylim(0, max(10.0, float(np.nanmax(dist[np.isfinite(dist)])) * 100 * 1.05))
        a1.set_ylim(0, max(1.0, float(np.nanpercentile(np.r_[T['goal'], T['coll'], T['funnel']], 98)) * 1.1))
        a2.set_ylim(-0.05, 1.05)
        for a, title in zip(axes, ('distances', 'cost terms of the chosen candidate', 'Judge weights and obstacle speed')):
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
        sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 0], hspace=0.55)
        self._draw_telemetry([fig.add_subplot(sub[i]) for i in range(3)], k)
        funnel = {1: 'on', 0: 'off'}.get(int(fr.get('approach_funnel', -1)), 'default')
        fig.suptitle(f"MP-PMPPI imagination  |  t = {float(fr['timestep']) * self.dt:5.2f} s (step {int(fr['timestep'])})"
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
                                   zoom=None if args.overview else args.zoom)
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
