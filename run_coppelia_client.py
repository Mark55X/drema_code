#!/usr/bin/env python
"""
CoppeliaSim / RLBench Closed-Loop Client (run_coppelia_client.py)

Runs an RLBench task (default: dynamic_drema_test_1 with oscillating tunnel obstacle) as the DREMA environment:
- streams multi-camera RGB-D with exact segmentation to DREMA, after a 360° orbital scan at t=0;
- interactive CLI: type 'start' to engage task execution, 'stop'/'pause' to halt, 'reset' to restart, 'quit' to exit;
- --sync_mode stepped (lockstep) or realtime (wall-clock continuous).
The loop itself is drema.client.runner.ClosedLoopRunner, shared with the laboratory environment (run_lab_client.py).
"""

import argparse
import os
import sys

# Ensure paths are set (do not shadow installed PyRep with source tree)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../RLBench")))

from drema.client import JointCommandExecutor
from drema.client.coppelia import CoppeliaEnvironment
from drema.client.runner import ClosedLoopRunner


def parse_args():
    parser = argparse.ArgumentParser(description="CoppeliaSim / RLBench Closed-Loop Client for DREMA")
    parser.add_argument("--server_address", type=str, default="localhost:50051", help="DREMA gRPC server address (default: localhost:50051)")
    parser.add_argument("--task", type=str, default="dynamic_drema_test_1", help="Task name (default: dynamic_drema_test_1)")
    parser.add_argument("--sync_mode", type=str, choices=["stepped", "realtime"], default="realtime", help="Simulation sync mode: 'stepped' or 'realtime' (default: realtime)")
    parser.add_argument("--cam_fps", type=float, default=10.0, help="Camera sensor capture and streaming frequency in Hz (default: 10)")
    parser.add_argument("--ctrl_fps", type=float, default=None,
                        help="Wall-clock control loop frequency in Hz (default: 1 / simulation timestep, i.e. real-time pacing)")
    parser.add_argument("--max_action_age", type=float, default=0.20,
                        help="Max age in seconds of the last action replayed when a control request times out (default: 0.20)")
    parser.add_argument("--reachability_radius", type=float, default=0.95, help="Robot maximum reachable radius in meters (default: 0.95)")
    parser.add_argument("--scan_resolution", type=int, nargs=2, default=[1280, 720], metavar=("WIDTH", "HEIGHT"),
                        help="Orbital scan camera resolution [width, height] (default: 1280 720)")
    parser.add_argument("--label_non_physical_shapes", action="store_true",
                        help="Also label shapes outside the physics (spawn areas, painted targets): perception may "
                             "then take them for objects (default: only shapes that are, or belong to, a respondable body)")
    parser.add_argument("--scan_steps", type=int, default=50, help="Number of orbital rotation steps at t=0 (default: 50)")
    parser.add_argument("--scan_chunk_size", type=int, default=2, help="Number of views per gRPC transmission chunk (default: 2)")
    parser.add_argument("--scan_chunk_timeout", type=float, default=180.0, help="Timeout in seconds per chunk transmission (default: 180.0)")
    parser.add_argument("--headless", action="store_true", help="Run CoppeliaSim in headless mode (no GUI window)")
    parser.add_argument("--ping_timeout", type=float, default=1.5, help="gRPC ping timeout in seconds (default: 1.5)")
    parser.add_argument("--ping_retries", type=int, default=3, help="Consecutive failed pings before holding (default: 3)")
    parser.add_argument("--force_scan", action="store_true", default=False,
                        help="Reset the DREMA server episode and scan, even if the server already has a scene")
    parser.add_argument("--reset_timeout", type=float, default=30.0, help="Timeout [s] of the server episode reset")
    parser.add_argument("--cam_resolution", type=int, nargs=2, default=[256, 256], metavar=("WIDTH", "HEIGHT"),
                        help="Streaming camera resolution [width, height] (default: 256 256)")
    parser.add_argument("--log_interval_actions", type=int, default=10,
                        help="Print control action telemetry every N steps (default: 10)")
    parser.add_argument("--record_dir", type=str, default=None,
                        help="Record simulation video frames (for render_imagination_video.py) into a timestamped subfolder")
    parser.add_argument("--record_camera", type=str, default="showcase", choices=["showcase", "front", "wrist", "overhead"],
                        help="Recording viewpoint: 'showcase' (three-quarter view of arm, table and target) or the pose "
                             "of a streaming camera (default: showcase)")
    parser.add_argument("--record_fov", type=float, default=None, help="Recording field of view [deg] (default: 60)")
    parser.add_argument("--record_eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Recording camera position in world frame (overrides the viewpoint)")
    parser.add_argument("--record_look_at", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Point the recording camera looks at (with --record_eye)")
    parser.add_argument("--record_resolution", type=int, nargs=2, default=[960, 540], metavar=("WIDTH", "HEIGHT"),
                        help="Recording resolution (default: 960 540)")
    parser.add_argument("--record_every", type=int, default=1, help="Record every N simulation steps (default: 1)")
    parser.add_argument("--roles_config", type=str, default="configs/roles_env.yaml",
                        help="Environment-side object roles config ('' disables config / VLM / human providers)")
    parser.add_argument("--keyframes", type=str, default="auto", choices=list(CoppeliaEnvironment.keyframe_modes),
                        help="Keyframe source: task keyframes, RLBench demo keyposes (PerAct keypoint discovery), "
                             "an external policy (--policy), or none (default: auto)")
    parser.add_argument("--policy", type=str, default=None,
                        help="External keyframe policy 'module:function' returning act(obs) -> PerAct action vector")
    parser.add_argument("--no_settle_gripper", action="store_true", default=False,
                        help="Actuate the gripper on the first step inside the keyframe tolerance, without waiting for the arm to settle")
    parser.add_argument("--no_settle_approach", action="store_true", default=False,
                        help="Approach poses of a grasp / release keep the free-space tolerance")
    parser.add_argument("--actuation", type=str, default="position", choices=list(JointCommandExecutor.MODES),
                        help="Arm actuation: 'position' integrates the MPC joint velocities into a joint reference "
                             "(holds against gravity, default); 'velocity' sends raw joint velocity targets")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = CoppeliaEnvironment(
        task_name=args.task,
        headless=args.headless,
        reachability_radius=args.reachability_radius,
        scan_resolution=args.scan_resolution,
        label_non_physical_shapes=args.label_non_physical_shapes,
        cam_resolution=args.cam_resolution,
        record_dir=args.record_dir,
        record_camera=args.record_camera,
        record_resolution=args.record_resolution,
        record_every=args.record_every,
        record_fov=args.record_fov,
        record_eye=args.record_eye,
        record_look_at=args.record_look_at,
        actuation=args.actuation
    )
    runner = ClosedLoopRunner(
        env,
        server_address=args.server_address,
        sync_mode=args.sync_mode,
        cam_fps=args.cam_fps,
        ctrl_fps=args.ctrl_fps,
        scan_steps=args.scan_steps,
        scan_chunk_size=args.scan_chunk_size,
        scan_chunk_timeout=args.scan_chunk_timeout,
        ping_timeout=args.ping_timeout,
        ping_max_retries=args.ping_retries,
        force_scan=args.force_scan,
        reset_timeout=args.reset_timeout,
        log_interval_actions=args.log_interval_actions,
        max_action_age=args.max_action_age,
        roles_config=args.roles_config or None,
        keyframes=args.keyframes,
        policy=args.policy,
        settle_gripper=not args.no_settle_gripper,
        settle_approach=not args.no_settle_approach
    )
    runner.run()
