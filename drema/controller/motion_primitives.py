#!/usr/bin/env python
"""
Motion Primitive Library for Franka Panda Manipulator (MP-PMPPI).

Synthesizes concepts from:
- Mathisen et al. (MP-MPPI, 2026), Section 2.3: Ingestion of structured Motion Primitives into MPPI.
- Marco Stefani (2026), Section 3:
    3.1 Geometric Bottlenecks & Narrow Passages: Vertical linear insertion/retract.
    3.2 High-Dimensional Joint Coordination: Cartesian primitives mapped via Jacobian.
    3.3 Myopic Obstacle Avoidance vs. Broad Maneuvers: Broad parabolic and lateral bypass arcs.
- Equations (13)-(14): Hybrid sampling matrix U_t and selective primitive filtering.
"""

import numpy as np
from typing import List, Optional, Tuple

from .franka_kinematics import FrankaKinematics


class MotionPrimitiveLibrary:
    """
    Generates and manages task-specific and exploratory motion primitives (U_p)
    in the joint acceleration space over lookahead horizon H.
    """

    def __init__(
        self,
        kinematics: FrankaKinematics,
        horizon: int = 20,
        dt: float = 0.05,
        max_joint_acc: float = 0.5  # rad/s^2 (matching Zhou et al. Table I / Isaac Gym)
    ):
        """
        :param kinematics: FrankaKinematics instance for Jacobian and FK computations.
        :param horizon: Prediction horizon H steps.
        :param dt: Timestep delta (seconds).
        :param max_joint_acc: Maximum joint acceleration limit.
        """
        self.kin = kinematics
        self.H = horizon
        self.dt = dt
        self.max_acc = max_joint_acc

    def generate_primitives(
        self,
        q_current: np.ndarray,
        qd_current: np.ndarray,
        target_pos: Optional[np.ndarray] = None,
        target_rot: Optional[np.ndarray] = None,
        obstacles: Optional[List[dict]] = None
    ) -> np.ndarray:
        """
        Generates the full library of N_p motion primitives:
        U_p = [u_{mp, 1}, u_{mp, 2}, ..., u_{mp, N_p}] in R^{N_p x H x 7}.

        :param q_current: Current robot joint angles (shape: [7]).
        :param qd_current: Current robot joint velocities (shape: [7]).
        :param target_pos: Optional 3D position [x, y, z] of target goal in world frame.
        :param target_rot: Optional 3x3 target rotation matrix.
        :param obstacles: Optional list of dynamic obstacles with positions for reactive steering.
        :return: Array of shape [N_p, H, 7] containing joint acceleration control sequences.
        """
        primitives: List[np.ndarray] = []

        ee_pos, ee_rot = self.kin.forward_kinematics_ee(q_current)
        z_ee = ee_rot[:, 2]  # Gripper approach vector (local Z)
        x_ee = ee_rot[:, 0]  # Local X
        y_ee = ee_rot[:, 1]  # Local Y

        # ---------------------------------------------------------------------
        # 1. Linear Cartesian Approach Primitive (Target Seeking)
        # ---------------------------------------------------------------------
        if target_pos is not None:
            dir_to_goal = target_pos - ee_pos
            dist_to_goal = np.linalg.norm(dir_to_goal)
            if dist_to_goal > 1e-4:
                unit_dir = dir_to_goal / dist_to_goal
                v_cart_des = unit_dir * min(0.15, dist_to_goal / (self.H * self.dt))
                u_approach = self._project_cartesian_linear_velocity(q_current, qd_current, v_cart_des)
                primitives.append(u_approach)

                # Accelerated approach variant (higher speed)
                v_cart_fast = unit_dir * min(0.30, dist_to_goal / (self.H * self.dt * 0.5))
                u_fast = self._project_cartesian_linear_velocity(q_current, qd_current, v_cart_fast)
                primitives.append(u_fast)

        # ---------------------------------------------------------------------
        # 2. Vertical Insertion & Retraction (Bottleneck & Peg-in-Hole)
        # ---------------------------------------------------------------------
        # Primitive: Vertical descent (-Z_world or +Z_ee)
        v_descend = np.array([0.0, 0.0, -0.08], dtype=np.float32)
        primitives.append(self._project_cartesian_linear_velocity(q_current, qd_current, v_descend))

        # Primitive: Vertical retract (+Z_world) to safely disengage
        v_lift = np.array([0.0, 0.0, 0.10], dtype=np.float32)
        primitives.append(self._project_cartesian_linear_velocity(q_current, qd_current, v_lift))

        # ---------------------------------------------------------------------
        # 3. Lateral, Upward & Reactive Obstacle Evasion Primitives (Bypass Obstacles)
        # Marco Stefani Section 3.3 & 6.2: "Broad pre-calculated motion primitives
        # allow evaluating macro-maneuvers that bypass obstacles entirely."
        # ---------------------------------------------------------------------
        # Sidestep Left (+Y world)
        v_left = np.array([0.0, 0.12, 0.0], dtype=np.float32)
        primitives.append(self._project_cartesian_linear_velocity(q_current, qd_current, v_left))

        # Sidestep Right (-Y world)
        v_right = np.array([0.0, -0.12, 0.0], dtype=np.float32)
        primitives.append(self._project_cartesian_linear_velocity(q_current, qd_current, v_right))

        # Parabolic Upward Sweep (Arc: +Z upward while advancing forward)
        primitives.append(self._create_parabolic_arc(q_current, qd_current, forward_speed=0.08, arc_height=0.12))

        # Reactive Obstacle Evasion Primitive (Marco Stefani Section 6.2):
        # Generates a repulsive velocity vector directed away from closest dynamic obstacle
        if obstacles:
            for obs in obstacles:
                obs_pos = np.array(obs.get('position', [0, 0, 0]), dtype=np.float32)
                vec_away = ee_pos - obs_pos
                dist_obs = np.linalg.norm(vec_away)
                if 0.01 < dist_obs < 0.40:
                    v_evade = (vec_away / dist_obs) * 0.15 # 15 cm/s evasive retreat
                    primitives.append(self._project_cartesian_linear_velocity(q_current, qd_current, v_evade))
                    # Also an upward bypass arc over the obstacle
                    v_over = (vec_away / dist_obs) * 0.08 + np.array([0.0, 0.0, 0.12], dtype=np.float32)
                    primitives.append(self._project_cartesian_linear_velocity(q_current, qd_current, v_over))
                    break

        # ---------------------------------------------------------------------
        # 4. Pure Screwing / Rotational Primitive (Local Z-axis spin)
        # ---------------------------------------------------------------------
        w_screw = z_ee * 0.5  # 0.5 rad/s pure rotation around tool axis
        primitives.append(self._project_cartesian_twist(q_current, qd_current, v_linear=np.zeros(3), w_angular=w_screw))

        # ---------------------------------------------------------------------
        # 5. Deceleration / Brake Primitive (Safe Stop / Hold)
        # ---------------------------------------------------------------------
        primitives.append(self._create_braking_primitive(qd_current))

        # Stack into numpy array: [N_p, H, 7]
        U_p = np.array(primitives, dtype=np.float32)
        # Clamp to acceleration limits
        U_p = np.clip(U_p, -self.max_acc, self.max_acc)
        return U_p

    def _project_cartesian_linear_velocity(
        self,
        q_init: np.ndarray,
        qd_init: np.ndarray,
        v_cart_des: np.ndarray
    ) -> np.ndarray:
        """
        Projects a constant Cartesian velocity vector into joint acceleration sequence
        over H steps using forward integration and Jacobian pseudo-inverse.
        """
        u_seq = np.zeros((self.H, 7), dtype=np.float32)
        q = q_init.copy()
        qd = qd_init.copy()

        for h in range(self.H):
            J = self.kin.geometric_jacobian(q, link_idx=-1)[:3, :]  # 3x7 linear Jacobian
            # Damped Least Squares: J^T (J J^T + lambda^2 I)^(-1)
            JJT = J @ J.T
            J_dls = J.T @ np.linalg.inv(JJT + 0.01 * np.eye(3, dtype=np.float32))

            # Desired joint velocity
            qd_des = J_dls @ v_cart_des
            # Required joint acceleration to reach qd_des
            qdd = (qd_des - qd) / self.dt
            qdd = np.clip(qdd, -self.max_acc, self.max_acc)

            u_seq[h] = qdd
            qd = qd + qdd * self.dt
            q = np.clip(q + qd * self.dt, self.kin.Q_MIN, self.kin.Q_MAX)

        return u_seq

    def _project_cartesian_twist(
        self,
        q_init: np.ndarray,
        qd_init: np.ndarray,
        v_linear: np.ndarray,
        w_angular: np.ndarray
    ) -> np.ndarray:
        """
        Projects a 6D spatial twist [v; w] through full 6x7 Jacobian.
        """
        u_seq = np.zeros((self.H, 7), dtype=np.float32)
        twist = np.concatenate([v_linear, w_angular])
        q = q_init.copy()
        qd = qd_init.copy()

        for h in range(self.H):
            J = self.kin.geometric_jacobian(q, link_idx=-1)  # 6x7
            JJT = J @ J.T
            J_dls = J.T @ np.linalg.inv(JJT + 0.02 * np.eye(6, dtype=np.float32))

            qd_des = J_dls @ twist
            qdd = (qd_des - qd) / self.dt
            qdd = np.clip(qdd, -self.max_acc, self.max_acc)

            u_seq[h] = qdd
            qd = qd + qdd * self.dt
            q = np.clip(q + qd * self.dt, self.kin.Q_MIN, self.kin.Q_MAX)

        return u_seq

    def _create_parabolic_arc(
        self,
        q_init: np.ndarray,
        qd_init: np.ndarray,
        forward_speed: float = 0.08,
        arc_height: float = 0.12
    ) -> np.ndarray:
        """
        Generates a parabolic arc trajectory: lifting upward in the first half of horizon,
        then descending in the second half while advancing forward.
        """
        u_seq = np.zeros((self.H, 7), dtype=np.float32)
        q = q_init.copy()
        qd = qd_init.copy()

        mid_step = self.H / 2.0
        for h in range(self.H):
            # Parabolic vertical velocity profile
            vz = arc_height * (1.0 - (h / mid_step))
            v_cart = np.array([forward_speed, 0.0, vz], dtype=np.float32)

            J = self.kin.geometric_jacobian(q, link_idx=-1)[:3, :]
            JJT = J @ J.T
            J_dls = J.T @ np.linalg.inv(JJT + 0.01 * np.eye(3, dtype=np.float32))

            qd_des = J_dls @ v_cart
            qdd = (qd_des - qd) / self.dt
            qdd = np.clip(qdd, -self.max_acc, self.max_acc)

            u_seq[h] = qdd
            qd = qd + qdd * self.dt
            q = np.clip(q + qd * self.dt, self.kin.Q_MIN, self.kin.Q_MAX)

        return u_seq

    def _create_braking_primitive(self, qd_init: np.ndarray) -> np.ndarray:
        """
        Generates an active braking control sequence: smoothly decelerating joint speeds to 0.
        """
        u_seq = np.zeros((self.H, 7), dtype=np.float32)
        qd = qd_init.copy()

        for h in range(self.H):
            # Proportional braking acceleration
            qdd = -qd / (self.dt * 2.0)
            qdd = np.clip(qdd, -self.max_acc, self.max_acc)
            u_seq[h] = qdd
            qd = qd + qdd * self.dt

        return u_seq
