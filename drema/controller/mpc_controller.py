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
    Directly interfaces with environment via gRPC and Digital Twin.
    """

    def __init__(
        self,
        digital_twin: Optional[Any] = None,
        num_joints: int = 7,
        max_joint_velocity: float = 1.0,          # rad/s limit shared by rollouts, primitives and command
        safety_collision_distance: float = 0.0,   # Physical penetration limit (<= 0 is hard contact)
        horizon: int = 15,                        # MPPI prediction horizon H
        dt: float = 0.05,                         # Planning timestep dt (s)
        num_samples_per_planner: int = 24,        # Stochastic samples per strategy
        top_k: int = 12,                          # Candidates evaluated by the Judge
        max_joint_acc: float = 2.0,               # Joint acceleration saturation limit (rad/s^2)
        alpha_pos: float = 1.0,                   # Position error weight (Zhou et al. Eq. 27)
        alpha_rot: float = 0.25,                  # Orientation error weight (Zhou et al. Eq. 27)
        convergence_radius_xi: float = 0.15,      # Sparse reward radius on ||q - q_des|| [rad] (Eq. 28)
        convergence_radius_xi_cart: float = 0.03, # Sparse reward radius on the Cartesian fallback (Eq. 27)
        approach_funnel_weight: float = 300.0,    # Approach cone cost weight (0 disables)
        approach_funnel_activation_radius: float = 0.15,
        approach_cone_half_angle_deg: float = 30.0,
        approach_cone_base_radius: float = 0.01,
        sigma_1: float = 0.015,                   # Inscribed hard safety collision margin (m)
        sigma_2: float = 0.035,                   # Inflation potential radius (m)
        kappa: float = 120.0,                     # Descending potential slope
        adaptive_goal_margin: bool = False,       # Strict Zhou et al. Eq. 21 without ad-hoc tapering
        evade_min_distance: float = 0.01,
        evade_max_distance: float = 0.35,
        evade_ttc_threshold: float = 2.0,
        evade_min_speed: float = 0.03,
        evade_imminent_distance: float = 0.06,
        evade_retreat_speed: float = 0.15,
        evade_lift_speed: float = 0.12,
        joint_angle_offsets: Optional[List[float]] = None  # theta_DH = q_measured + offset (see FrankaKinematics)
    ):
        self.digital_twin = digital_twin
        self.num_joints = num_joints
        self.max_joint_velocity = max_joint_velocity
        self.velocity_clip_events = 0
        self.safety_collision_distance = safety_collision_distance
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot

        # Kinematics & MP-PMPPI engine
        self.kinematics = FrankaKinematics(base_position=None, joint_angle_offsets=joint_angle_offsets)
        self.engine = MPPMPPIEngine(
            kinematics=self.kinematics,
            horizon=horizon,
            dt=dt,
            num_samples_per_planner=num_samples_per_planner,
            top_k=top_k,
            max_joint_acc=max_joint_acc,
            max_joint_velocity=max_joint_velocity,
            lambda_param=1.0,
            beta_param=0.8,
            alpha_param=1.0,
            alpha_mu=0.8,
            alpha_sigma=0.2,
            alpha_pos=alpha_pos,
            alpha_rot=alpha_rot,
            convergence_radius_xi=convergence_radius_xi,
            convergence_radius_xi_cart=convergence_radius_xi_cart,
            approach_funnel_weight=approach_funnel_weight,
            approach_funnel_activation_radius=approach_funnel_activation_radius,
            approach_cone_half_angle_deg=approach_cone_half_angle_deg,
            approach_cone_base_radius=approach_cone_base_radius,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            kappa=kappa,
            adaptive_goal_margin=adaptive_goal_margin,
            evade_min_distance=evade_min_distance,
            evade_max_distance=evade_max_distance,
            evade_ttc_threshold=evade_ttc_threshold,
            evade_min_speed=evade_min_speed,
            evade_imminent_distance=evade_imminent_distance,
            evade_retreat_speed=evade_retreat_speed,
            evade_lift_speed=evade_lift_speed
        )

    @classmethod
    def from_config(cls, ctrl_cfg: Dict[str, Any], digital_twin: Optional[Any] = None,
                    joint_angle_offsets: Optional[List[float]] = None) -> "MPCController":
        """Controller built from the 'controller' section of the DREMA config (shared by the server and the benchmarks)."""
        c = ctrl_cfg or {}
        return cls(
            digital_twin=digital_twin,
            num_joints=7,
            max_joint_velocity=float(c.get("max_joint_velocity", 1.0)),
            horizon=int(c.get("horizon", 15)),
            dt=float(c.get("dt", 0.05)),
            num_samples_per_planner=int(c.get("num_samples_per_planner", 24)),
            top_k=int(c.get("top_k", 12)),
            max_joint_acc=float(c.get("max_joint_acc", 2.0)),
            alpha_pos=float(c.get("alpha_pos", 1.0)),
            alpha_rot=float(c.get("alpha_rot", 0.25)),
            convergence_radius_xi=float(c.get("convergence_radius_xi", 0.15)),
            convergence_radius_xi_cart=float(c.get("convergence_radius_xi_cart", 0.03)),
            approach_funnel_weight=float(c.get("approach_funnel_weight", 300.0)),
            approach_funnel_activation_radius=float(c.get("approach_funnel_activation_radius", 0.15)),
            approach_cone_half_angle_deg=float(c.get("approach_cone_half_angle_deg", 30.0)),
            approach_cone_base_radius=float(c.get("approach_cone_base_radius", 0.01)),
            sigma_1=float(c.get("sigma_1", 0.02)),
            sigma_2=float(c.get("sigma_2", 0.05)),
            kappa=float(c.get("kappa", 15.0)),
            adaptive_goal_margin=bool(c.get("adaptive_goal_margin", False)),
            evade_min_distance=float(c.get("evade_min_distance", 0.01)),
            evade_max_distance=float(c.get("evade_max_distance", 0.35)),
            evade_ttc_threshold=float(c.get("evade_ttc_threshold", 2.0)),
            evade_min_speed=float(c.get("evade_min_speed", 0.03)),
            evade_imminent_distance=float(c.get("evade_imminent_distance", 0.06)),
            evade_retreat_speed=float(c.get("evade_retreat_speed", 0.15)),
            evade_lift_speed=float(c.get("evade_lift_speed", 0.12)),
            joint_angle_offsets=joint_angle_offsets
        )

    def set_digital_twin(self, digital_twin: Any) -> None:
        """Sets or updates the Digital Twin reference."""
        self.digital_twin = digital_twin

    def compute_action(
        self,
        robot_state: drema_comm_pb2.RobotState,
        digital_twin: Optional[Any] = None,
        target_goal: Optional[np.ndarray] = None
    ) -> drema_comm_pb2.ControlAction:
        """
        Calculates optimal joint velocity action using MP-PMPPI given current state
        and Digital Twin environment.

        :param robot_state: Current state from environment (joint angles, velocities, ee_pose).
        :param digital_twin: Optional Digital Twin instance override (defaults to self.digital_twin).
        :param target_goal: Optional 3D position [x, y, z] of target.
        :return: ControlAction message with 7 joint velocities.
        """
        if digital_twin is None:
            digital_twin = self.digital_twin
        # 1. Early input validation: ensure strict 7-joint Franka Panda compatibility
        if len(robot_state.joint_positions) != self.num_joints:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * self.num_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=True,
                status_message=f"INVALID STATE: joint_positions length ({len(robot_state.joint_positions)}) != {self.num_joints}"
            )

        curr_q = np.array(robot_state.joint_positions, dtype=np.float32)
        curr_qd = (
            np.array(robot_state.joint_velocities, dtype=np.float32)
            if len(robot_state.joint_velocities) == self.num_joints
            else np.zeros(self.num_joints, dtype=np.float32)
        )

        # Synchronize robot base position in kinematics module once received
        if self.kinematics.base_position is None and len(robot_state.robot_base_pos) >= 3:
            self.kinematics.base_position = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)

        # 2. Idle mode: hold position if CLI user has not triggered 'start'
        if not robot_state.task_active:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * self.num_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=False,
                status_message="IDLE: Waiting for CLI 'start' command"
            )

        # 3. Critical Safety Check: Only trigger emergency halt on actual physical collision/penetration.
        # We do NOT prematurely freeze if an obstacle is nearby, because the MP-PMPPI
        # Sensitive planner and evasive primitives proactively steer around it!
        min_dist = float('inf')
        if digital_twin is not None and hasattr(digital_twin, 'get_min_obstacle_distance'):
            try:
                min_dist = digital_twin.get_min_obstacle_distance()
            except Exception as e:
                print(f"[MPC CONTROLLER WARNING] get_min_obstacle_distance error: {e}")
                min_dist = float('inf')

        if min_dist <= self.safety_collision_distance:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * self.num_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=True,
                status_message=f"SAFETY HALT: Active physical collision detected in Digital Twin (d = {min_dist:.4f}m)"
            )

        # 4. Resolve Active Target Pose (target_goal argument or robot_state.target_pose)
        active_target_pos = None
        if target_goal is not None and len(target_goal) >= 3:
            active_target_pos = np.array(target_goal[:3], dtype=np.float32)
        elif robot_state.target_available and len(robot_state.target_pose) >= 3:
            active_target_pos = np.array(robot_state.target_pose[:3], dtype=np.float32)

        # If no active target is available, hold position cleanly without inventing coordinates
        if active_target_pos is None:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * self.num_joints,
                gripper_action=robot_state.gripper_open,
                safety_stop=False,
                status_message="WAIT: Task active but waiting for target goal (holding position)"
            )

        # Target orientation (if available in target_pose)
        active_target_rot = None
        if robot_state.target_available and len(robot_state.target_pose) >= 7:
            # Quaternion [x, y, z, w] to rotation matrix
            qx, qy, qz, qw = robot_state.target_pose[3:7]
            active_target_rot = self._quat_to_rot_matrix(qx, qy, qz, qw)
        approach_funnel = (not robot_state.HasField('approach_funnel')) or robot_state.approach_funnel

        # 5. Execute MP-PMPPI Real-Time Optimization (Algorithm 1)
        qd_optimal, diag = self.engine.solve(
            q_current=curr_q,
            qd_current=curr_qd,
            target_pos=active_target_pos,
            target_rot=active_target_rot,
            digital_twin=digital_twin,
            approach_funnel=approach_funnel
        )

        # 6. Safety net: the engine already plans within the same bound, so a clip here
        #    signals an inconsistency and is counted in the telemetry.
        qd_clamped = np.clip(qd_optimal, -self.max_joint_velocity, self.max_joint_velocity)
        clipped = bool(np.any(np.abs(qd_clamped - qd_optimal) > 1e-6))
        self.velocity_clip_events += int(clipped)

        # 7. Format informative telemetry status message
        weights = diag.get('weights', {})
        w_greedy = weights.get('greedy', 0.5)
        w_sensi = weights.get('sensitive', 0.5)
        calc_t = diag.get('calc_time_ms', 0.0)
        tgt_str = f"[{active_target_pos[0]:.2f}, {active_target_pos[1]:.2f}, {active_target_pos[2]:.2f}]"
        lim_warns = diag.get('joint_limit_warnings', [])
        lim_str = f" | lim_warn: {','.join(lim_warns)}" if lim_warns else ""
        ik_str = "" if diag.get('q_des_available', True) else " | IK: FAIL"
        vclip_str = f" | VCLIP x{self.velocity_clip_events}" if clipped else ""
        top_cand = diag.get('top_candidate', '')
        top_w = diag.get('top_weight', 0.0)
        prim_w = diag.get('primitive_weight_sum', 0.0)
        mode_str = f" | top: {top_cand} ({top_w*100:.0f}%, prims: {prim_w*100:.0f}%)" if top_cand else ""

        top_c = diag.get('top_costs', {})
        appr_c = diag.get('appr_costs', {})
        cost_str = ""
        if top_c:
            cost_str = f" | C[{top_cand}]={top_c.get('total', 0.0):.2f}(g:{top_c.get('goal', 0.0):.2f},s:{top_c.get('safe', 0.0):.2f},c:{top_c.get('coll', 0.0):.2f},f:{top_c.get('funnel', 0.0):.2f})"
            if appr_c:
                cost_str += f" vs C[appr]={appr_c.get('total', 0.0):.2f}(g:{appr_c.get('goal', 0.0):.2f},s:{appr_c.get('safe', 0.0):.2f})"

        timings = diag.get('timings', {})
        t_coll = timings.get('coll_ms', 0.0)
        t_samp = timings.get('samples_ms', 0.0)
        t_opt = timings.get('opt_ms', 0.0)
        time_breakdown_str = f" [Coll: {t_coll:.0f}ms, Roll: {t_samp:.0f}ms, Opt: {t_opt:.0f}ms]"

        status_msg = (
            f"MP-PMPPI | w_grd: {w_greedy:.2f}, w_sns: {w_sensi:.2f}{mode_str}{cost_str} | "
            f"tgt: {tgt_str} | min_d: {min_dist:.3f}m | {calc_t:.1f}ms{time_breakdown_str}{lim_str}{ik_str}{vclip_str}"
        )

        self.last_diagnostics = diag

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
        self.engine.reset()

    @staticmethod
    def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        """Converts quaternion (x, y, z, w) to 3x3 rotation matrix using FrankaKinematics (scipy)."""
        return FrankaKinematics.quat_to_rot_matrix(np.array([x, y, z, w], dtype=np.float32))
