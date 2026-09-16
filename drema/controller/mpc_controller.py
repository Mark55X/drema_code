#!/usr/bin/env python
"""
Model Predictive Control (MPC) / Motion Planner for DREMA.

Currently implements a modular stub controller returning standard test joint velocities
for closed-loop testing, with safety collision monitoring against the PyBullet Digital Twin.
Structured to be replaced by the complete MPC formulation.
"""

import time
import numpy as np
from typing import Optional, List, Dict, Tuple

from ..communication.proto import drema_comm_pb2


class MPCController:
    """
    MPC Controller interface for Franka Panda 7-DOF arm.
    """

    def __init__(
        self,
        num_joints: int = 7,
        max_joint_velocity: float = 0.15,  # rad/s safety limit for test trials
        safety_collision_distance: float = 0.03  # meters: stop threshold
    ):
        self.num_joints = num_joints
        self.max_joint_velocity = max_joint_velocity
        self.safety_collision_distance = safety_collision_distance

        # Default target position on table for test trials [x, y, z]
        self.default_target_pos = np.array([0.25, 0.0, 0.77], dtype=np.float32)

        # Internal state
        self.step_counter = 0

    def compute_action(
        self,
        robot_state: drema_comm_pb2.RobotState,
        digital_twin=None,
        target_goal: Optional[np.ndarray] = None
    ) -> drema_comm_pb2.ControlAction:
        """
        Calculates joint velocity action given the current robot state and PyBullet digital twin.

        :param robot_state: Current state from CoppeliaSim (joint angles, velocities, ee_pose).
        :param digital_twin: PyBulletDigitalTwin instance for collision evaluation.
        :param target_goal: Optional 3D position [x, y, z] of target.
        :return: ControlAction message with 7 joint velocities.
        """
        n_joints = len(robot_state.joint_positions) if len(robot_state.joint_positions) > 0 else self.num_joints
        curr_q = np.array(robot_state.joint_positions) if len(robot_state.joint_positions) == n_joints else np.zeros(n_joints)

        # 1. If CLI user has not pressed 'start', hold position (idle mode)
        if not robot_state.task_active:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * n_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=False,
                status_message="IDLE: Waiting for CLI 'start' command"
            )

        # 2. Check collision safety in PyBullet Digital Twin
        min_dist = float('inf')
        if digital_twin is not None and hasattr(digital_twin, 'get_min_obstacle_distance'):
            try:
                min_dist = digital_twin.get_min_obstacle_distance()
            except Exception:
                pass

        if min_dist < self.safety_collision_distance:
            # Collision warning: decelerate / halt
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * n_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=True,
                status_message=f"SAFETY HALT: Obstacle too close in Digital Twin ({min_dist:.3f}m < {self.safety_collision_distance:.3f}m)"
            )

        # 3. Standard test action: gentle approach towards target (or periodic exploration)
        # TODO: Replace with full MPC optimization formulation
        self.step_counter += 1
        test_velocities = np.zeros(n_joints, dtype=np.float32)

        active_target = target_goal if target_goal is not None else self.default_target_pos
        target_str = f"[{active_target[0]:.2f}, {active_target[1]:.2f}, {active_target[2]:.2f}]"

        if target_goal is not None and len(curr_q) >= 4:
            # Proportional orienting: direct base joint (joint 0) towards target azimuth
            target_yaw = np.arctan2(active_target[1], active_target[0])
            yaw_err = target_yaw - curr_q[0]
            test_velocities[0] = np.clip(0.3 * yaw_err, -0.05, 0.05)

            # Gentle forward extension
            test_velocities[1] = 0.02 * np.cos(self.step_counter * 0.05)
            test_velocities[3] = -0.02 * np.abs(np.sin(self.step_counter * 0.05))
        else:
            omega = 0.05
            test_velocities[0] = 0.04 * np.sin(self.step_counter * omega)
            test_velocities[1] = 0.02 * np.cos(self.step_counter * omega)
            test_velocities[3] = -0.03 * np.abs(np.sin(self.step_counter * omega))

        # Clamp to safety limits
        test_velocities = np.clip(test_velocities, -self.max_joint_velocity, self.max_joint_velocity)

        return drema_comm_pb2.ControlAction(
            timestamp=time.time(),
            timestep=robot_state.timestep,
            joint_velocities=test_velocities.tolist(),
            gripper_action=robot_state.gripper_open,
            safety_stop=False,
            status_message=f"TEST MPC: Active (goal: {target_str}, min obs dist: {min_dist:.3f}m)"
        )

    def reset(self):
        """Resets controller internal states."""
        self.step_counter = 0
