#!/usr/bin/env python
"""
Client-side pieces on a real RLBench task (headless CoppeliaSim, no DREMA server): demo keyframes found with
PerAct's keypoint discovery and the non-blocking RLBench gripper. The arm is teleported to each keypoint's joint
configuration (the MPC is not involved), the gripper runs through RLBenchGripperController, and the task success
condition is checked at the end.

Usage (sim_env): python scratch/test_rlbench_keyframes.py [task_name]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.environ.get("RLBENCH_PATH", "../RLBench")))

from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from drema.client import RLBenchGripperController, discover_keypoints, keyframes_from_demo


def main(task_name: str) -> bool:
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)
    env = Environment(ActionMode(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
                      obs_config=obs_config, headless=True, robot_setup='panda')
    env.launch()
    try:
        task = env.get_task(task_file_to_task_class(task_name))
        # The RLBench fork calls callable_each_step(reset=True) after a failed attempt, so it must not be None.
        demo = task.get_demos(1, live_demos=True, callable_each_step=lambda *a, **k: None)[0]
        kps = discover_keypoints(demo)
        frames = keyframes_from_demo(demo)
        print(f"{task_name}: demo of {len(demo)} steps, keypoints {kps}")
        for k, i in zip(frames, kps):
            print(f"  {k.name}: pos {np.round(k.pose[:3], 3).tolist()} grip={'open' if k.gripper_open else 'close'}"
                  f" ignore_collisions={k.ignore_collisions}")
        task.reset_to_demo(demo)

        robot, pyrep = task._robot, env._pyrep
        gripper = RLBenchGripperController(robot, task._task)
        for k, i in zip(frames, kps):
            robot.arm.set_joint_positions(demo[i].joint_positions, disable_dynamics=True)
            robot.arm.set_joint_target_positions(demo[i].joint_positions)
            pyrep.step()
            tip = robot.arm.get_tip().get_pose()
            err = np.linalg.norm(tip[:3] - np.asarray(k.pose[:3]))
            if (k.gripper_open > 0.5) != gripper.is_open():
                gripper.command(k.gripper_open)
                steps = 0
                while not gripper.done and steps < 500:
                    gripper.step()
                    pyrep.step()
                    task._task.step()
                    steps += 1
                print(f"  at {k.name}: tip error {err * 1000:.1f} mm, gripper {'opened' if gripper.is_open() else 'closed'}"
                      f" in {steps} steps, grasped {[o.get_name() for o in robot.gripper.get_grasped_objects()]}")
        success, _ = task._task.success()
        print(f"{task_name}: success after replaying the keyframes = {success}")
        return bool(success)
    finally:
        env.shutdown()


if __name__ == "__main__":
    ok = main(sys.argv[1] if len(sys.argv) > 1 else "pick_and_lift")
    sys.exit(0 if ok else 1)
