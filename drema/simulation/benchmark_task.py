"""
Task-level benchmark of the digital twin backends: the same RLBench task, tunnel motion and seeds are run in closed
loop with the MPC using either the PyBullet or the MuJoCo twin, and per-episode statistics are collected.

Everything runs in one process (no gRPC server): CoppeliaSim/RLBench, the twin and the controller built from the
same config as the server. Perception is emulated: the true tunnel pose plus Gaussian noise (measured on the
RecurGS logs) is delivered to the twin at camera rate with a fixed latency. Clearance and contacts are measured on
the real CoppeliaSim geometry.

Metrics per episode: success and time to success, minimum real clearance robot-tunnel, steps closer than 1.5 cm,
contact steps with the tunnel and the table, MPC solve time, safety stops, TCP path length, final TCP error.

Run in the simulation environment (RLBench + CoppeliaSim), e.g.:
    python -m drema.simulation.benchmark_task --backends pybullet mujoco --modes static single --seeds 0 1 2 \\
           --mesh assets/scanned_meshes/tunnel_obstacle_84.obj --out logs/task_benchmark
"""

import argparse
import csv
import os
import time
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation

MODES = ("static", "single", "double", "continuous")


def tunnel_y(task: Any, mode: str, step: int) -> float:
    """Tunnel lateral position for the motion modes of dynamic_drema_test_1 (same formulas as the task file)."""
    if mode == "static":
        y = task.lateral_start_y
    else:
        phase = step * task.oscillation_freq
        if mode == "single":
            phase = min(phase, np.pi)
        elif mode == "double":
            phase = min(phase, 2.0 * np.pi)
        y = task.target_pos[1] + task.amplitude * np.cos(phase)
    return float(np.clip(y, -0.40, 0.40))


def build_twin(backend: str, cfg: Any, offsets):
    twin_cfg = cfg.get_nested("digital_twin", {}) or {}
    common = dict(
        visualize=False,
        table_z=float(cfg.get_nested("perception.workspace.table_z_prior", 0.75)),
        kp_pos=float(twin_cfg.get("kp_pos", 250.0)),
        kd_pos=float(twin_cfg.get("kd_pos", 30.0)),
        kp_rot=float(twin_cfg.get("kp_rot", 15.0)),
        kd_rot=float(twin_cfg.get("kd_rot", 1.5)),
        tracking_mode=str(twin_cfg.get("tracking_mode", "constraint")),
        constraint_max_force=float(twin_cfg.get("constraint_max_force", 300.0)),
        sim_substeps=int(twin_cfg.get("sim_substeps", 2)),
        joint_angle_offsets=offsets,
        obstacle_convex_decomposition=bool(twin_cfg.get("obstacle_convex_decomposition", True)),
        vhacd_resolution=int(twin_cfg.get("vhacd_resolution", 100000)),
        vhacd_concavity=float(twin_cfg.get("vhacd_concavity", 0.05)),
    )
    if backend == "pybullet":
        from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
        return PyBulletDigitalTwin(robot_urdf_path=twin_cfg.get("urdf_path", "franka_panda/panda.urdf"), **common)
    from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
    return MuJoCoDigitalTwin(robot_model_path=twin_cfg.get("mujoco_model_path"), **common)


def run_episode(env: Any, task: Any, backend: str, mode: str, seed: int, args: Any, cfg: Any) -> Dict[str, Any]:
    from pyrep.objects.shape import Shape
    from pyrep.backend import sim
    from drema.communication.proto import drema_comm_pb2 as pb2
    from drema.controller.mpc_controller import MPCController
    import run_coppelia_client as rc

    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    task.reset()
    t_inst = task._task
    arm, gripper = task._robot.arm, task._robot.gripper
    tip, tunnel, pyrep = arm.get_tip(), t_inst.tunnel, env._pyrep
    robot_shapes = [Shape(h) for h in sim.simGetObjectsInTree(arm.get_handle(), sim.sim_object_shape_type, 0)]
    robot_shapes = [s for s in robot_shapes if s.is_respondable()]
    finger_shapes = [Shape(h) for h in sim.simGetObjectsInTree(gripper.get_handle(), sim.sim_object_shape_type, 0)]
    finger_shapes = [s for s in finger_shapes if s.is_respondable()]
    table_shapes = [Shape(n) for n in ("diningTable", "workspace") if Shape.exists(n)]

    task_step = {'n': 0}

    def tunnel_step():
        task_step['n'] += 1
        x, _, z = t_inst.init_tunnel_pos
        tunnel.set_position([x, tunnel_y(t_inst, mode, task_step['n']), z])

    t_inst.step = tunnel_step

    offsets = [float(x) for x in cfg.get_nested("robot.joint_angle_offsets", [0.0] * 7)]
    twin = build_twin(backend, cfg, offsets)
    twin.spawn_scanned_table(args.table_z, bounds=tuple(args.table_bounds))
    twin.spawn_scanned_mesh_obstacle(args.mesh, tuple(tunnel.get_position()), obj_id=84, name="tunnel_obstacle")
    controller = MPCController.from_config(cfg.get_nested("controller", {}), twin, offsets)
    client = rc.CoppeliaSimulationClient.__new__(rc.CoppeliaSimulationClient)
    client.keyframe_idx, client.keyframe_label, client.keyframe_funnel = 0, "", None
    base = np.array(arm.joints[0].get_position()) - [0.0, 0.0, 0.333]
    dt = args.dt

    pending: List[tuple] = []
    rec = dict(success_step=None, min_clearance=np.inf, steps_within_1p5cm=0, tunnel_contact_steps=0,
               table_contact_steps=0, safety_stops=0, path_length=0.0)
    solve_ms: List[float] = []
    prev_tcp = None
    robot_loaded = False
    for step in range(args.start_delay + args.max_steps):
        # perception emulation: capture at camera rate, delivery after the latency
        if step % args.camera_every == 0:
            noisy = np.asarray(tunnel.get_position()) + rng.normal(0.0, args.pos_noise, 3) * np.array([0.4, 1.0, 0.4])
            quat = Rotation.from_rotvec(np.radians(rng.normal(0.0, args.rot_noise_deg, 3))).as_quat()
            pending.append((step + args.latency_steps, noisy, quat, (step + 1) * dt))
        while pending and pending[0][0] <= step:
            _, pos, quat, stamp = pending.pop(0)
            twin.sync_object_pose(84, tuple(pos), tuple(quat), timestamp=stamp)

        active = step >= args.start_delay and rec['success_step'] is None
        if active:
            q = list(arm.get_joint_positions())
            qd = list(arm.get_joint_velocities())
            ee_pose = list(tip.get_pose())
            target_pose = client._active_keyframe_pose(t_inst.get_keyframes(), ee_pose)
            if not robot_loaded:
                twin.load_robot(tuple(base), joint_positions=q)
                robot_loaded = True
            else:
                twin.sync_robot_state(q)
            twin.step()
            state = pb2.RobotState(timestep=step, joint_positions=q, joint_velocities=qd, ee_pose=ee_pose,
                                   task_active=True, target_available=True, target_pose=target_pose,
                                   robot_base_pos=list(base), gripper_open=float(gripper.get_open_amount()[0]))
            if client.keyframe_funnel is not None:
                state.approach_funnel = bool(client.keyframe_funnel)
            t0 = time.perf_counter()
            action = controller.compute_action(robot_state=state, target_goal=np.asarray(target_pose[:3], np.float32))
            solve_ms.append((time.perf_counter() - t0) * 1000.0)
            rec['safety_stops'] += int(action.safety_stop)
            arm.set_joint_target_velocities([0.0] * 7 if action.safety_stop else list(action.joint_velocities))
        else:
            arm.set_joint_target_velocities([0.0] * 7)
        t_inst.step()
        pyrep.step()

        if step >= args.start_delay:
            clearance = min(s.check_distance(tunnel) for s in robot_shapes)
            rec['min_clearance'] = min(rec['min_clearance'], clearance)
            rec['steps_within_1p5cm'] += int(clearance < 0.015)
            rec['tunnel_contact_steps'] += int(any(s.check_collision(tunnel) for s in robot_shapes))
            rec['table_contact_steps'] += int(any(s.check_collision(t) for s in finger_shapes for t in table_shapes))
            tcp = np.asarray(tip.get_position())
            if prev_tcp is not None:
                rec['path_length'] += float(np.linalg.norm(tcp - prev_tcp))
            prev_tcp = tcp
            if rec['success_step'] is None and t_inst.success()[0]:
                rec['success_step'] = step - args.start_delay
                break

    twin.shutdown()
    final_err = float(np.linalg.norm(np.asarray(tip.get_position()) - np.asarray(t_inst.target.get_position())))
    return dict(
        backend=backend, mode=mode, seed=seed,
        success=int(rec['success_step'] is not None),
        time_to_success_s=(rec['success_step'] * dt) if rec['success_step'] is not None else np.nan,
        min_clearance_cm=rec['min_clearance'] * 100.0,
        steps_within_1p5cm=rec['steps_within_1p5cm'],
        tunnel_contact_steps=rec['tunnel_contact_steps'],
        table_contact_steps=rec['table_contact_steps'],
        safety_stops=rec['safety_stops'],
        path_length_m=rec['path_length'],
        final_tcp_error_cm=final_err * 100.0,
        solve_mean_ms=float(np.mean(solve_ms)) if solve_ms else np.nan,
        solve_p95_ms=float(np.percentile(solve_ms, 95)) if solve_ms else np.nan,
    )


def summarize(rows: List[Dict[str, Any]]) -> str:
    def ms(values, fmt="{:.1f}"):
        values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if not values.size:
            return "-"
        return (fmt + " +/- " + fmt).format(values.mean(), values.std())

    header = ["backend", "mode", "episodes", "success rate", "time to success [s]", "min real clearance [cm]",
              "worst clearance [cm]", "episodes with tunnel contact", "table contact steps", "safety stops",
              "path length [m]", "solve mean / p95 [ms]"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key in sorted({(r['backend'], r['mode']) for r in rows}, key=lambda k: (MODES.index(k[1]), k[0])):
        group = [r for r in rows if (r['backend'], r['mode']) == key]
        lines.append("| " + " | ".join([
            key[0], key[1], str(len(group)),
            f"{100.0 * np.mean([r['success'] for r in group]):.0f}%",
            ms([r['time_to_success_s'] for r in group]),
            ms([r['min_clearance_cm'] for r in group]),
            f"{min(r['min_clearance_cm'] for r in group):.1f}",
            str(sum(1 for r in group if r['tunnel_contact_steps'] > 0)),
            ms([r['table_contact_steps'] for r in group], "{:.0f}"),
            ms([r['safety_stops'] for r in group], "{:.0f}"),
            ms([r['path_length_m'] for r in group], "{:.2f}"),
            f"{np.nanmean([r['solve_mean_ms'] for r in group]):.0f} / {np.nanmean([r['solve_p95_ms'] for r in group]):.0f}",
        ]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Same task, same seeds, PyBullet vs MuJoCo twin: episode statistics")
    parser.add_argument("--config", default="configs/drema_default.yaml")
    parser.add_argument("--backends", nargs="+", default=["pybullet", "mujoco"], choices=["pybullet", "mujoco"])
    parser.add_argument("--modes", nargs="+", default=["static", "single"], choices=MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--mesh", required=True, help="Tunnel mesh used by the twin (e.g. the reconstructed one)")
    parser.add_argument("--max_steps", type=int, default=300, help="Control steps per episode after the start delay")
    parser.add_argument("--start_delay", type=int, default=0,
                        help="Steps the tunnel moves before the robot starts (the interactive client waits for 'start')")
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation / control period [s]")
    parser.add_argument("--camera_every", type=int, default=2, help="Perception update every N steps (client default)")
    parser.add_argument("--latency_steps", type=int, default=4, help="Perception latency in steps (~0.2 s in the logs)")
    parser.add_argument("--pos_noise", type=float, default=0.004, help="Perceived position noise std [m] (along y)")
    parser.add_argument("--rot_noise_deg", type=float, default=2.0, help="Perceived orientation noise std [deg]")
    parser.add_argument("--table_z", type=float, default=0.749)
    parser.add_argument("--table_bounds", type=float, nargs=4, default=[-0.482, 0.691, -0.549, 0.549])
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--out", default="logs/task_benchmark", help="Output prefix (.csv per episode, .md summary)")
    args = parser.parse_args()

    from drema.config import load_config
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.action_modes.action_mode import ActionMode
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.tasks import DynamicDremaTest1

    cfg = load_config(args.config)
    obs = ObservationConfig()
    obs.set_all(False)
    env = Environment(action_mode=ActionMode(JointVelocity(), Discrete()), obs_config=obs, headless=args.headless,
                      robot_setup='panda')
    env.launch()
    task = env.get_task(DynamicDremaTest1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows: List[Dict[str, Any]] = []
    try:
        for mode in args.modes:
            for seed in args.seeds:
                for backend in args.backends:
                    row = run_episode(env, task, backend, mode, seed, args, cfg)
                    rows.append(row)
                    print(f"[task benchmark] {backend:8s} {mode:10s} seed {seed}: success {row['success']} | "
                          f"t {row['time_to_success_s']:.2f}s | min clearance {row['min_clearance_cm']:.1f} cm | "
                          f"tunnel contacts {row['tunnel_contact_steps']} | solve {row['solve_mean_ms']:.0f} ms", flush=True)
                    with open(args.out + ".csv", "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
    finally:
        env.shutdown()

    table = summarize(rows)
    with open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    print(f"saved {args.out}.csv and {args.out}.md")


if __name__ == "__main__":
    main()
