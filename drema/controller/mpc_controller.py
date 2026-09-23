#!/usr/bin/env python
"""
Model Predictive Control (MPC) Controller for DREMA.

Implements the MP-PMPPI (Motion-Primitive Guided Parallel MPPI) controller
combining:
- Lelai Zhou et al. (IEEE Transactions on Robotics 2025):
  Parallel MPPI with Greedy, Sensitive, and Judge strategies, GVM-SDF cost,
  and Gaussian Mixture Regression (GMR) policy fusion.
- Mathisen et al. (arXiv 2026):
  Motion-Primitive Guided sampling-based optimization.
- Marco Stefani (2026):
  Hybrid sampling matrix U_t, selective primitive filtering, and exact PyBullet
  mesh-based GVM-SDF modulation without spherical link approximations.
"""

import time
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

from ..communication.proto import drema_comm_pb2
from .franka_kinematics import FrankaKinematics
from .mp_pmppi_engine import MPPMPPIEngine


class MPCController:
    """
    MP-PMPPI Controller interface for Franka Panda 7-DOF manipulator.
    Directly interfaces with CoppeliaSim via gRPC and PyBullet Digital Twin.
    """

    def __init__(
        self,
        num_joints: int = 7,
        max_joint_velocity: float = 0.50,         # rad/s execution limit
        safety_collision_distance: float = 0.015, # Stop distance threshold (meters)
        horizon: int = 15,                        # MPPI prediction horizon H
        dt: float = 0.05,                         # Planning timestep dt (s)
        num_samples_per_planner: int = 24,        # Stochastic samples per strategy
        top_k: int = 12,                          # Candidates evaluated by the Judge
        max_joint_acc: float = 0.50               # Joint acceleration saturation limit (rad/s^2)
    ):
        self.num_joints = num_joints
        self.max_joint_velocity = max_joint_velocity
        self.safety_collision_distance = safety_collision_distance

        # Kinematics & MP-PMPPI engine
        self.kinematics = FrankaKinematics()
        self.engine = MPPMPPIEngine(
            kinematics=self.kinematics,
            horizon=horizon,
            dt=dt,
            num_samples_per_planner=num_samples_per_planner,
            top_k=top_k,
            max_joint_acc=max_joint_acc,
            lambda_param=1.0,
            beta_param=0.8,
            alpha_param=1.0,
            alpha_mu=0.8,
            alpha_sigma=0.2
        )

        # Default fallback target position on workspace table [x, y, z]
        self.default_target_pos = np.array([0.25, 0.0, 0.77], dtype=np.float32)

        # Tracking state
        self.step_counter = 0
        self.last_base_synced = False

    def compute_action(
        self,
        robot_state: drema_comm_pb2.RobotState,
        digital_twin=None,
        target_goal: Optional[np.ndarray] = None
    ) -> drema_comm_pb2.ControlAction:
        """
        Calculates optimal joint velocity action using MP-PMPPI given current state
        and PyBullet digital twin environment.

        :param robot_state: Current state from CoppeliaSim (joint angles, velocities, ee_pose).
        :param digital_twin: PyBulletDigitalTwin instance for collision evaluation.
        :param target_goal: Optional 3D position [x, y, z] of target.
        :return: ControlAction message with 7 joint velocities.
        """
        n_joints = len(robot_state.joint_positions) if len(robot_state.joint_positions) > 0 else self.num_joints
        curr_q = (
            np.array(robot_state.joint_positions, dtype=np.float32)
            if len(robot_state.joint_positions) == n_joints
            else np.zeros(n_joints, dtype=np.float32)
        )
        curr_qd = (
            np.array(robot_state.joint_velocities, dtype=np.float32)
            if len(robot_state.joint_velocities) == n_joints
            else np.zeros(n_joints, dtype=np.float32)
        )

        # Synchronize robot base position in kinematics module once received
        if not self.last_base_synced and len(robot_state.robot_base_pos) >= 3:
            self.kinematics.base_position = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)
            self.last_base_synced = True

        # 1. Idle mode: hold position if CLI user has not triggered 'start'
        if not robot_state.task_active:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * n_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=False,
                status_message="IDLE: Waiting for CLI 'start' command"
            )

        # 2. Critical Safety Pre-check: Immediate emergency halt if in hard collision
        min_dist = float('inf')
        if digital_twin is not None and hasattr(digital_twin, 'get_min_obstacle_distance'):
            try:
                min_dist = digital_twin.get_min_obstacle_distance()
            except Exception:
                min_dist = float('inf')

        if min_dist < self.safety_collision_distance:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * n_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=True,
                status_message=f"SAFETY HALT: Obstacle collision threshold breached ({min_dist:.3f}m < {self.safety_collision_distance:.3f}m)"
            )

        # 3. Determine Active Target Pose (PerAct keyframe or target_goal)
        active_target_pos = None
        if target_goal is not None and len(target_goal) >= 3:
            active_target_pos = np.array(target_goal[:3], dtype=np.float32)
        elif robot_state.target_available and len(robot_state.target_pose) >= 3:
            active_target_pos = np.array(robot_state.target_pose[:3], dtype=np.float32)
        else:
            active_target_pos = self.default_target_pos.copy()

        # Target orientation (if available in target_pose)
        active_target_rot = None
        if robot_state.target_available and len(robot_state.target_pose) >= 7:
            # Quaternion [x, y, z, w] to rotation matrix
            qx, qy, qz, qw = robot_state.target_pose[3:7]
            active_target_rot = self._quat_to_rot_matrix(qx, qy, qz, qw)

        # 4. Execute MP-PMPPI Real-Time Optimization (Algorithm 1)
        qd_optimal, diag = self.engine.solve(
            q_current=curr_q,
            qd_current=curr_qd,
            target_pos=active_target_pos,
            target_rot=active_target_rot,
            digital_twin=digital_twin
        )

        # 5. Project onto physical safety joint velocity limits
        qd_clamped = np.clip(qd_optimal, -self.max_joint_velocity, self.max_joint_velocity)

        self.step_counter += 1

        # Format informative telemetry status message
        weights = diag.get('weights', {})
        w_greedy = weights.get('greedy', 0.5)
        w_sensi = weights.get('sensitive', 0.5)
        calc_t = diag.get('calc_time_ms', 0.0)
        tgt_str = f"[{active_target_pos[0]:.2f}, {active_target_pos[1]:.2f}, {active_target_pos[2]:.2f}]"

        status_msg = (
            f"MP-PMPPI | w_grd: {w_greedy:.2f}, w_sns: {w_sensi:.2f} | "
            f"tgt: {tgt_str} | min_d: {min_dist:.3f}m | {calc_t:.1f}ms"
        )

        return drema_comm_pb2.ControlAction(
            timestamp=time.time(),
            timestep=robot_state.timestep,
            joint_velocities=qd_clamped.tolist(),
            gripper_action=robot_state.gripper_open,
            safety_stop=False,
            status_message=status_msg
        )

    def reset(self):
        """Resets controller internal states and distributions."""
        self.step_counter = 0
        self.engine.reset()
        self.last_base_synced = False

    @staticmethod
    def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        """Converts quaternion (x, y, z, w) to 3x3 rotation matrix."""
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm > 1e-6:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)]
        ], dtype=np.float32)
