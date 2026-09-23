#!/usr/bin/env python
"""
Franka Panda Kinematics & Differential Kinematics Module.

Provides:
- Analytical Forward Kinematics (FK) for all 7 links and end-effector.
- Geometric Jacobian J(q) in R^{6x7} for end-effector and intermediate links.
- Cartesian link velocities: Vel(link_k) = J_v^{(k)}(q) * q_dot.
- Damped Least-Squares (DLS) Inverse Kinematics solver.
- Franka Panda physical joint, velocity, and acceleration limits.

References:
- Zhou et al. (IEEE T-RO 2025), Eq. (23)-(24): Forward kinematics and Cartesian link velocities.
- Standard Franka Emika Panda DH parameters and geometric model.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class FrankaKinematics:
    """
    Kinematics model and Jacobian calculator for 7-DOF Franka Emika Panda.
    """

    # Franka Panda joint position limits [rad]
    Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32)

    # Franka Panda joint velocity limits [rad/s]
    QD_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=np.float32)

    # Maximum joint acceleration limits [rad/s^2] (conservatively configured for smooth MPC sampling)
    QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=np.float32)

    # Modified Denavit-Hartenberg (MDH) parameters for Franka Panda:
    # [a_{i-1}, alpha_{i-1}, d_i, theta_offset]
    # Frames: 0 to 7 + flange + EE
    MDH_PARAMS = [
        (0.0,      0.0,         0.333, 0.0),        # Joint 1
        (0.0,     -np.pi / 2.0, 0.0,   0.0),        # Joint 2
        (0.0,      np.pi / 2.0, 0.316, 0.0),        # Joint 3
        (0.0825,   np.pi / 2.0, 0.0,   0.0),        # Joint 4
        (-0.0825, -np.pi / 2.0, 0.384, 0.0),        # Joint 5
        (0.0,      np.pi / 2.0, 0.0,   0.0),        # Joint 6
        (0.088,    np.pi / 2.0, 0.107, 0.0),        # Joint 7
    ]
    # End-Effector / Gripper Flange offset from Joint 7
    EE_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)

    def __init__(self, base_position: Optional[np.ndarray] = None):
        """
        :param base_position: 3D coordinates [x, y, z] of the robot base in world frame.
        """
        self.base_position = np.array(base_position, dtype=np.float32) if base_position is not None else np.zeros(3, dtype=np.float32)

    def _dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Computes standard Modified DH 4x4 homogeneous transformation matrix."""
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        cos_al = np.cos(alpha)
        sin_al = np.sin(alpha)

        return np.array([
            [cos_th,          -sin_th,          0.0,            a],
            [sin_th * cos_al,  cos_th * cos_al, -sin_al, -sin_al * d],
            [sin_th * sin_al,  cos_th * sin_al,  cos_al,  cos_al * d],
            [0.0,              0.0,              0.0,            1.0]
        ], dtype=np.float32)

    def forward_kinematics_all(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Computes forward kinematics for all 7 joint frames and the end-effector.

        :param q: Joint angles array of length 7 [rad].
        :return: List of 4x4 homogeneous transformation matrices relative to robot base:
                 [T_base, T_link1, T_link2, ..., T_link7, T_ee]
        """
        transforms = []
        T_current = np.eye(4, dtype=np.float32)
        transforms.append(T_current.copy())  # Base frame (T_0)

        for i in range(7):
            a, alpha, d, th_offset = self.MDH_PARAMS[i]
            theta = float(q[i]) + th_offset
            T_i = self._dh_transform(a, alpha, d, theta)
            T_current = T_current @ T_i
            transforms.append(T_current.copy())

        # End-effector transform (flange + gripper offset)
        T_ee = T_current.copy()
        T_ee[:3, 3] += T_current[:3, :3] @ self.EE_OFFSET
        transforms.append(T_ee)

        return transforms

    def forward_kinematics_ee(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Cartesian position and rotation matrix of the end-effector.

        :param q: Joint angles array of length 7 [rad].
        :return: (position [x, y, z], rotation_matrix 3x3) in world coordinates.
        """
        all_T = self.forward_kinematics_all(q)
        T_ee = all_T[-1]
        pos_base = T_ee[:3, 3]
        pos_world = pos_base + self.base_position
        rot_matrix = T_ee[:3, :3]
        return pos_world, rot_matrix

    def geometric_jacobian(self, q: np.ndarray, link_idx: int = -1) -> np.ndarray:
        """
        Computes the 6x7 geometric Jacobian matrix J(q) = [J_linear; J_angular].

        :param q: Joint angles array of length 7 [rad].
        :param link_idx: Index of frame to compute Jacobian for (-1 for end-effector, 1..7 for joints).
        :return: 6x7 Jacobian matrix.
        """
        all_T = self.forward_kinematics_all(q)
        target_T = all_T[link_idx]
        p_target = target_T[:3, 3]

        J = np.zeros((6, 7), dtype=np.float32)

        # Number of active joints contributing to this link
        max_joint = 7 if link_idx == -1 or link_idx >= 7 else link_idx

        for i in range(max_joint):
            T_prev = all_T[i]
            z_i = T_prev[:3, 2]  # Z-axis of joint i
            p_i = T_prev[:3, 3]  # Origin of joint i

            # Linear velocity part: J_v = z_i x (p_target - p_i)
            J[:3, i] = np.cross(z_i, p_target - p_i)
            # Angular velocity part: J_w = z_i
            J[3:, i] = z_i

        return J

    def compute_cartesian_velocity(self, q: np.ndarray, qd: np.ndarray, link_idx: int = -1) -> np.ndarray:
        """
        Computes Cartesian linear velocity Vel(pt) = J_v(q) * q_dot.
        Corresponds to Eq. (24) and Section 5.1 in Zhou et al. (IEEE T-RO 2025).

        :param q: Joint positions [rad] (shape: [7]).
        :param qd: Joint velocities [rad/s] (shape: [7]).
        :return: 3D Cartesian linear velocity vector [vx, vy, vz].
        """
        J = self.geometric_jacobian(q, link_idx=link_idx)
        vel_linear = J[:3, :] @ qd
        return vel_linear

    def solve_dls_ik(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        damping: float = 0.05,
        max_iters: int = 35,
        pos_tolerance: float = 2e-3,
        rot_weight: float = 0.2
    ) -> Tuple[np.ndarray, bool]:
        """
        Solves Inverse Kinematics using Damped Least-Squares (DLS):
        dq = J^T * (J * J^T + lambda^2 * I)^(-1) * error

        :param q_init: Initial joint configuration (shape: [7]).
        :param target_pos: Target 3D position [x, y, z] in world frame.
        :param target_rot: Optional target 3x3 rotation matrix.
        :param damping: Damping factor lambda to prevent singularity instability.
        :param max_iters: Maximum iteration steps.
        :param pos_tolerance: Target position error threshold in meters.
        :param rot_weight: Weight of orientation error relative to position error.
        :return: (q_sol, success_flag)
        """
        q = np.array(q_init, dtype=np.float32).copy()
        target_p_rel = target_pos - self.base_position

        for _ in range(max_iters):
            all_T = self.forward_kinematics_all(q)
            T_ee = all_T[-1]
            current_p = T_ee[:3, 3]
            pos_err = target_p_rel - current_p

            if np.linalg.norm(pos_err) < pos_tolerance:
                return q, True

            J = self.geometric_jacobian(q, link_idx=-1)

            if target_rot is not None:
                # Compute orientation error via matrix logarithm / skew-symmetric cross
                current_R = T_ee[:3, :3]
                rot_err = 0.5 * (
                    np.cross(current_R[:, 0], target_rot[:, 0]) +
                    np.cross(current_R[:, 1], target_rot[:, 1]) +
                    np.cross(current_R[:, 2], target_rot[:, 2])
                )
                dx = np.concatenate([pos_err, rot_weight * rot_err])
            else:
                # Position-only IK (first 3 rows of J)
                J = J[:3, :]
                dx = pos_err

            # DLS formula: J^T * (J * J^T + lambda^2 * I)^(-1) * dx
            JJT = J @ J.T
            damped_inv = np.linalg.inv(JJT + (damping ** 2) * np.eye(J.shape[0], dtype=np.float32))
            dq = J.T @ (damped_inv @ dx)

            # Step update with limit clipping
            q = np.clip(q + dq, self.Q_MIN, self.Q_MAX)

        # Check final position error
        curr_p, _ = self.forward_kinematics_ee(q)
        success = np.linalg.norm(target_pos - curr_p) < pos_tolerance * 5.0
        return q, success

    def batch_forward_kinematics_ee(self, q_batch: np.ndarray) -> np.ndarray:
        """
        Fast forward kinematics of end-effector positions for an entire batch of trajectories.

        :param q_batch: Array of shape [K, H, 7] containing joint trajectories.
        :return: Array of shape [K, H, 3] containing end-effector positions in world coordinates.
        """
        K, H, _ = q_batch.shape
        ee_positions = np.zeros((K, H, 3), dtype=np.float32)

        for k in range(K):
            for h in range(H):
                pos, _ = self.forward_kinematics_ee(q_batch[k, h])
                ee_positions[k, h] = pos

        return ee_positions
