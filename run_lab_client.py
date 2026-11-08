#!/usr/bin/env python
"""
Laboratory Closed-Loop Client (run_lab_client.py)

Runs the real FR3 and the calibrated lab cameras as the DREMA environment, through the lab_bridges processes on the
robot PC (docs/lab_environment.md). Same loop and console as run_coppelia_client.py: 'start', 'stop', 'scan',
'reset', 'quit' and the object-role commands. Without --actuate nothing is sent to the robot (shadow mode).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drema.client.lab import LabEnvironment
from drema.client.runner import ClosedLoopRunner
from drema.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Laboratory (real FR3) Closed-Loop Client for DREMA")
    parser.add_argument("--config", type=str, default="configs/environments/lab.yaml", help="Lab environment configuration")
    parser.add_argument("--server_address", type=str, default="localhost:50051", help="DREMA gRPC server address (default: localhost:50051)")
    parser.add_argument("--actuate", action="store_true",
                        help="Send the arm and gripper commands to the robot (default: shadow mode, the robot does not move)")
    parser.add_argument("--cam_fps", type=float, default=10.0, help="Camera streaming frequency in Hz (default: 10)")
    parser.add_argument("--max_action_age", type=float, default=0.20,
                        help="Max age in seconds of the last action replayed when a control request times out (default: 0.20)")
    parser.add_argument("--scan_chunk_size", type=int, default=2, help="Number of views per gRPC transmission chunk (default: 2)")
    parser.add_argument("--scan_chunk_timeout", type=float, default=180.0, help="Timeout in seconds per chunk transmission (default: 180.0)")
    parser.add_argument("--ping_timeout", type=float, default=1.5, help="gRPC ping timeout in seconds (default: 1.5)")
    parser.add_argument("--ping_retries", type=int, default=3, help="Consecutive failed pings before holding (default: 3)")
    parser.add_argument("--force_scan", action="store_true", default=False,
                        help="Reset the DREMA server episode and scan, even if the server already has a scene")
    parser.add_argument("--reset_timeout", type=float, default=30.0, help="Timeout [s] of the server episode reset")
    parser.add_argument("--log_interval_actions", type=int, default=10, help="Print control action telemetry every N steps (default: 10)")
    parser.add_argument("--roles_config", type=str, default="configs/roles_env.yaml",
                        help="Environment-side object roles config ('' disables config / VLM / human providers)")
    parser.add_argument("--keyframes", type=str, default="auto", choices=list(LabEnvironment.keyframe_modes),
                        help="Keyframe source: a YAML file (lab.keyframes.file), an external policy (--policy), or none "
                             "(default: auto)")
    parser.add_argument("--policy", type=str, default=None,
                        help="External keyframe policy 'module:function' returning act(obs) -> PerAct action vector")
    parser.add_argument("--no_settle_gripper", action="store_true", default=False,
                        help="Actuate the gripper on the first step inside the keyframe tolerance, without waiting for the arm to settle")
    parser.add_argument("--no_settle_approach", action="store_true", default=False,
                        help="Approach poses of a grasp / release keep the free-space tolerance")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config).get("lab", {})
    env = LabEnvironment(cfg, actuate=args.actuate)
    runner = ClosedLoopRunner(
        env,
        server_address=args.server_address,
        sync_mode="realtime",
        cam_fps=args.cam_fps,
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
