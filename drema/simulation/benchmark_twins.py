"""
Benchmark of the digital twin backends on a recorded run (visualization.imagination.record_dir):

  concavity      share of probe points in the free space of the obstacle (>= 1.5 cm from every surface of its mesh)
                 that the backend reports as inside the obstacle (0% = concave shape preserved)
  distance error |backend distance - mesh distance| for probe points within 6 cm of the obstacle surface
  min distance   robot-obstacle clearance at the recorded configurations, compared with the reference variant
  costs          C_P and GVM-SDF on the same rollout batch: correlation with the reference variant
  timing         collision-cost evaluation time per batch

Each variant runs in its own process (PyBullet keeps global state). Usage:
    python -m drema.simulation.benchmark_twins --recording logs/imagination/<run> [--mesh <obstacle.obj>]
           [--variants pybullet_vhacd mujoco_vhacd ...] [--out benchmark_twins]

This checks the twins component by component; the task-level comparison (same task, same seeds, statistics on
success and clearance) is drema/simulation/benchmark_task.py.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List

import numpy as np

VARIANTS = ("pybullet_hull", "pybullet_vhacd", "mujoco_hull", "mujoco_vhacd")


def _build_twin(variant: str, scene: Dict[str, Any], mesh: str, pose, q0):
    from drema.controller.franka_kinematics import FrankaKinematics as FK
    offsets = FK.COPPELIA_JOINT_ANGLE_OFFSETS
    base = tuple(scene['robot_base'])
    if variant.startswith('pybullet'):
        from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
        twin = PyBulletDigitalTwin(visualize=False, joint_angle_offsets=offsets, tracking_mode='teleport',
                                   obstacle_convex_decomposition=(variant == 'pybullet_vhacd'))
    else:
        from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
        twin = MuJoCoDigitalTwin(visualize=False, joint_angle_offsets=offsets, tracking_mode='teleport',
                                 obstacle_convex_decomposition=(variant == 'mujoco_vhacd'))
    # Same order as the server: the scene restored from cache first, the robot at the first action.
    twin.spawn_scanned_table(float(scene['table_z']), bounds=tuple(scene['table_bounds']))
    twin.spawn_scanned_mesh_obstacle(mesh, tuple(pose[0]), initial_quat=tuple(pose[1]), obj_id=84)
    twin.load_robot(base, joint_positions=list(q0))
    twin.sync_object_pose(84, tuple(pose[0]), tuple(pose[1]))
    return twin


def _probe_distances(variant: str, twin: Any, probe_world: np.ndarray, pose) -> np.ndarray:
    """Backend distance from each probe point to the obstacle (0 when the point is inside)."""
    if variant.startswith('pybullet'):
        import pybullet as p
        sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001)
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere, basePosition=[0, 0, -5])
        obstacle = twin.tracked_objects[84]['body_id']
        out = np.zeros(len(probe_world))
        for i, pt in enumerate(probe_world):
            p.resetBasePositionAndOrientation(body, pt.tolist(), [0, 0, 0, 1])
            closest = p.getClosestPoints(body, obstacle, 0.3)
            out[i] = max(0.0, min(c[8] for c in closest)) if closest else 0.3
        return out
    # MuJoCo collides every mesh geom through its convex hull: the backend distance is the distance to the union of
    # the part hulls (the plane distance is exact inside a hull and a lower bound outside it).
    import mujoco
    from scipy.spatial import ConvexHull
    m, d = twin.model, twin.data
    mujoco.mj_forward(m, d)
    out = np.full(len(probe_world), np.inf)
    for gid, oid in twin._obstacle_geoms:
        if oid != 84:
            continue
        mid = m.geom_dataid[gid]
        verts = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        local = (probe_world - d.geom_xpos[gid]) @ d.geom_xmat[gid].reshape(3, 3)
        eq = ConvexHull(verts).equations                  # n.x + c <= 0 inside
        signed = np.max(local @ eq[:, :3].T + eq[:, 3], axis=1)
        out = np.minimum(out, np.maximum(signed, 0.0))
    return out


def _worker(variant: str, recording: str, mesh: str, queue: Any) -> None:
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    import trimesh
    from drema.controller.franka_kinematics import FrankaKinematics as FK

    frames = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(recording, 'frame_*.npz')))]
    scene = dict(np.load(os.path.join(recording, 'scene.npz')))
    ref = frames[len(frames) // 3]
    pose = (ref['obs84_pos'], ref['obs84_quat'])
    twin = _build_twin(variant, scene, mesh, pose, frames[0]['q'])
    result: Dict[str, Any] = {'variant': variant}

    # concavity and distance accuracy against the mesh itself
    obj = trimesh.load(mesh, force='mesh')
    rng = np.random.default_rng(0)
    lo, hi = obj.bounds
    probe_local = rng.uniform(lo - 0.03, hi + 0.03, size=(3000, 3))
    true_d, _ = cKDTree(obj.sample(80000)).query(probe_local)
    inside_box = np.all((probe_local > lo) & (probe_local < hi), axis=1)
    free = inside_box & (true_d > 0.015)
    probe_world = Rotation.from_quat(pose[1]).apply(probe_local) + np.asarray(pose[0])
    t0 = time.perf_counter()
    backend_d = _probe_distances(variant, twin, probe_world, pose)
    result['probe_us'] = (time.perf_counter() - t0) / len(probe_world) * 1e6
    near = true_d < 0.06
    err = np.abs(backend_d - true_d)[near & ~(inside_box & (true_d < 0.003))]
    result['free_inside_pct'] = float(np.mean(backend_d[free] < 1e-3) * 100) if free.any() else float('nan')
    result['dist_err_median_mm'] = float(np.median(err) * 1000)
    result['dist_err_p95_mm'] = float(np.percentile(err, 95) * 1000)

    # clearance at the recorded configurations
    clearances = []
    for fr in frames[::3]:
        twin.sync_object_pose(84, tuple(fr['obs84_pos']), tuple(fr['obs84_quat']))
        twin.sync_robot_state(list(fr['q']))
        clearances.append(twin.get_min_obstacle_distance())
    result['clearance'] = [float(c) for c in clearances]

    # collision costs and timing on a fixed rollout batch
    kin = FK(base_position=np.asarray(scene['robot_base']), joint_angle_offsets=FK.COPPELIA_JOINT_ANGLE_OFFSETS)
    rng = np.random.default_rng(1)
    q0 = ref['q']
    K, H = 72, 15
    Q = (q0 + rng.normal(0, 1, (K, 1, 7)) * np.linspace(0, 0.35, H)[None, :, None]).astype(np.float32)
    QD = rng.uniform(-0.6, 0.6, (K, H, 7)).astype(np.float32)
    # Static obstacle for the cost comparison: the poses synced above (without timestamps) would make the
    # twin predictor extrapolate arbitrary velocities, different in every process.
    twin.predictor = None
    twin.sync_object_pose(84, tuple(pose[0]), tuple(pose[1]))
    args = dict(Q=Q, QD=QD, sigma_1=0.015, sigma_2=0.035, kappa=120.0, rho=0.8, kin_helper=kin, dt=0.05)
    cp, cg = twin.compute_trajectory_collision_costs(**args)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        twin.compute_trajectory_collision_costs(**args)
        times.append(time.perf_counter() - t0)
    result['cost_ms'] = float(np.median(times) * 1000)
    result['coll_p'] = cp.ravel().tolist()
    result['coll_gvm'] = cg.ravel().tolist()
    result['steps'] = [int(fr['timestep']) for fr in frames[::3]]
    queue.put(result)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the digital twin backends on a recorded run")
    parser.add_argument("--recording", required=True, help="Imagination recording folder (frame_*.npz, scene.npz)")
    parser.add_argument("--mesh", default=None, help="Obstacle mesh (default: the path stored in the recording)")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--out", default="benchmark_twins", help="Output prefix for the .md and .csv tables")
    args = parser.parse_args()

    scene = dict(np.load(os.path.join(args.recording, 'scene.npz')))
    mesh = args.mesh or str(scene.get('obs84_mesh_path', ''))
    if not os.path.exists(mesh):
        raise FileNotFoundError(f"obstacle mesh '{mesh}' not found: pass --mesh")

    ctx = mp.get_context('spawn')
    results: List[Dict[str, Any]] = []
    for variant in args.variants:
        queue = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(variant, args.recording, mesh, queue))
        proc.start()
        try:
            results.append(queue.get(timeout=900))
        except Exception as e:
            print(f"[benchmark] variant '{variant}' failed: {e}")
        proc.join()

    if not results:
        return
    ref = results[0]
    rows = []
    for r in results:
        clearance_diff = np.abs(np.array(r['clearance']) - np.array(ref['clearance'])) * 100
        rows.append({
            'variant': r['variant'],
            'free space reported inside [%]': f"{r['free_inside_pct']:.1f}",
            'distance error median / p95 [mm]': f"{r['dist_err_median_mm']:.1f} / {r['dist_err_p95_mm']:.1f}",
            f"clearance vs {ref['variant']} mean / max [cm]": f"{clearance_diff.mean():.1f} / {clearance_diff.max():.1f}",
            f"C_P corr vs {ref['variant']}": f"{_corr(np.array(r['coll_p']), np.array(ref['coll_p'])):.2f}",
            f"GVM corr vs {ref['variant']}": f"{_corr(np.array(r['coll_gvm']), np.array(ref['coll_gvm'])):.2f}",
            'cost batch 72x15 [ms]': f"{r['cost_ms']:.1f}",
            'probe query [us]': f"{r['probe_us']:.0f}",
        })
    header = list(rows[0].keys())
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row[h] for h in header) + " |" for row in rows]
    md_text = "\n".join(md)
    print(md_text)
    with open(args.out + ".md", "w") as f:
        f.write(md_text + "\n")
    with open(args.out + ".csv", "w") as f:
        f.write(",".join(header) + "\n")
        f.writelines(",".join(row[h] for h in header) + "\n" for row in rows)
    with open(args.out + "_clearance.json", "w") as f:
        json.dump({r['variant']: dict(steps=r['steps'], clearance=r['clearance']) for r in results}, f)
    print(f"saved {args.out}.md, {args.out}.csv, {args.out}_clearance.json")


if __name__ == "__main__":
    main()
