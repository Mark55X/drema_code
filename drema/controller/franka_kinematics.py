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
from typing import Tuple, List, Optional, Dict, Union
from scipy.spatial.transform import Rotation


class FrankaKinematics:
    """
    Kinematics model and Jacobian calculator for 7-DOF Franka Emika Panda.
    """

    # -------------------------------------------------------------------------
    # Franka Panda Joint Position Limits [rad] (Q_MIN, Q_MAX)
    # Source (Official Franka Control Interface / franka_ros joint_limits.yaml):
    # https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml
    # Official FCI Manual: https://frankaemika.github.io/docs/
    # -------------------------------------------------------------------------
    Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Franka Panda Joint Velocity Limits [rad/s] (QD_MAX)
    # Source (Official franka_ros joint_limits.yaml & panda.urdf):
    # https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml
    # Joint 1-4: 2.1750 rad/s (125 deg/s) | Joint 5-7: 2.6100 rad/s (150 deg/s)
    # -------------------------------------------------------------------------
    QD_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Maximum Joint Acceleration Limits [rad/s^2] (QDD_MAX)
    # Source (Official Franka Control Interface / FCI Specifications):
    # https://frankaemika.github.io/docs/control_parameters.html
    # Note: In Zhou et al. (IEEE T-RO 2025, Table I) sampling is performed in
    # acceleration space with bounds [-0.5, 0.5] rad/s^2 for smooth MPC sampling.
    QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Canonical Franka Panda Ready / Home Pose [rad] (Q_REST)
    # [0.0, -pi/4, 0.0, -3*pi/4, 0.0, pi/2, pi/4]
    # Standard rest configuration for nullspace posture and IK regularization.
    # -------------------------------------------------------------------------
    Q_REST = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Modified Denavit-Hartenberg (MDH) Parameters for Franka Panda:
    # [a_{i-1}, alpha_{i-1}, d_i, theta_offset]
    # Frames: 0 to 7 + flange + EE
    # Source: Claudio Gaz et al., "Dynamic Identification of the Franka Emika
    # Panda Robot with Retrieval of Feasible Parameters Using Penalty-Based
    # Optimization", IEEE Robotics and Automation Letters (RA-L), 2019.
    # DOI: https://doi.org/10.1109/LRA.2019.2931248
    # GitHub Repository: https://github.com/marcocognetti/FrankaEmikaPandaDynModel
    # -------------------------------------------------------------------------
    MDH_PARAMS = [
        (0.0,      0.0,         0.333, 0.0),        # Joint 1 (d1 = 0.333m)
        (0.0,     -np.pi / 2.0, 0.0,   0.0),        # Joint 2
        (0.0,      np.pi / 2.0, 0.316, 0.0),        # Joint 3 (d3 = 0.316m)
        (0.0825,   np.pi / 2.0, 0.0,   0.0),        # Joint 4 (a4 = 0.0825m)
        (-0.0825, -np.pi / 2.0, 0.384, 0.0),        # Joint 5 (a5 = -0.0825m, d5 = 0.384m)
        (0.0,      np.pi / 2.0, 0.0,   0.0),        # Joint 6
        (0.088,    np.pi / 2.0, 0.107, 0.0),        # Joint 7 (a7 = 0.088m, d7 = 0.107m)
    ]

    # -------------------------------------------------------------------------
    # End-Effector / Gripper TCP Offset from Joint 7 Flange [m], in the flange frame.
    # Fitted to CoppeliaSim Panda_tip over 41 random configurations (0.6 mm mean residual,
    # once the joint-4 zero offset below is applied).
    # The EE orientation is the flange orientation: target orientations are expressed in this
    # convention. CoppeliaSim's Panda_tip frame is the flange frame rotated by 135 deg about z.
    # -------------------------------------------------------------------------
    EE_OFFSET = np.array([0.00139, 0.00144, 0.11003], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Joint-angle zero offsets of the CoppeliaSim/RLBench Panda model [rad]:
    # theta_DH = q_measured + offset. Panda_joint4 is 3.92 deg away from the Franka DH zero
    # (fitted on joint axes, joint origins and Panda_tip; other joints < 0.05 deg).
    # A real Franka arm uses zero offsets.
    # -------------------------------------------------------------------------
    COPPELIA_JOINT_ANGLE_OFFSETS = np.array([0.0, 0.0, 0.0, 0.0683, 0.0, 0.0, 0.0], dtype=np.float64)

    # -------------------------------------------------------------------------
    # Precomputed Craig MDH screw transform along X_{i-1} [Rot_X(alpha) * Trans_X(a)]
    # for exact geometric Jacobian calculation (each joint i rotates around the Z-axis
    # of frame i-1 after the alpha_{i-1} twist and a_{i-1} offset).
    # -------------------------------------------------------------------------
    _MDH_RX_TX = [
        np.array([
            [1.0, 0.0, 0.0, a],
            [0.0, np.cos(al), -np.sin(al), 0.0],
            [0.0, np.sin(al), np.cos(al), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        for a, al, d, th in MDH_PARAMS
    ]

    def __init__(self, base_position: Optional[np.ndarray] = None, joint_angle_offsets: Optional[np.ndarray] = None):
        """
        :param base_position: 3D coordinates [x, y, z] of the robot base in world frame.
        :param joint_angle_offsets: Per-joint zero offsets [rad] added to the measured joint angles
                                    before the DH chain (theta_DH = q + offset). Joint limits,
                                    commands and IK solutions stay in the measured joint space.
        """
        self.base_position = np.array(base_position, dtype=np.float32) if base_position is not None else None
        self.joint_angle_offsets = (
            np.asarray(joint_angle_offsets, dtype=np.float64).reshape(7)
            if joint_angle_offsets is not None else np.zeros(7, dtype=np.float64)
        )
        self.MDH_PARAMS = [
            (a, alpha, d, th + float(off))
            for (a, alpha, d, th), off in zip(FrankaKinematics.MDH_PARAMS, self.joint_angle_offsets)
        ]

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
        base_offset = self.base_position if self.base_position is not None else np.zeros(3, dtype=np.float32)
        pos_world = pos_base + base_offset
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
            # In Craig MDH, joint i rotation is about the Z-axis of frame i-1 after Rot_X(alpha_{i-1}) * Trans_X(a_{i-1}):
            T_prime = all_T[i] @ self._MDH_RX_TX[i]
            z_i = T_prime[:3, 2]  # Z-axis of joint i
            p_i = T_prime[:3, 3]  # Origin of joint i

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
        max_iters: int = 100,
        pos_tolerance: float = 1e-3,
        rot_weight: float = 0.5,
        rot_tolerance: float = np.radians(1.0),
        nullspace_gain: float = 0.5,
        max_step: float = 0.2,
        accept_pos: float = 5e-3,
        accept_rot: float = np.radians(3.0)
    ) -> Tuple[np.ndarray, bool]:
        """
        Solves Inverse Kinematics with Damped Least-Squares (Nakamura & Hanafusa / Levenberg-Marquardt):
        dq = J^+ e + (I - J^+ J) k (Q_REST - q),  J^+ = J^T (J J^T + lambda^2 I)^(-1).

        It uses the same kinematic model as the MPPI costs, so the joint-space guidance center
        q_{des,t} (Zhou et al. Eq. 26) is consistent with the Cartesian goal term. The null-space
        posture term resolves the redundancy towards Q_REST, which keeps q_des continuous across
        control cycles when the solver is warm-started.

        :param q_init: Initial joint configuration (shape: [7]).
        :param target_pos: Target 3D position [x, y, z] in world frame.
        :param target_rot: Optional target 3x3 rotation matrix of the EE (flange orientation convention).
        :param damping: Damping factor lambda to prevent singularity instability.
        :param max_iters: Maximum iteration steps.
        :param pos_tolerance: Position error [m] at which iterations stop.
        :param rot_weight: Weight [m/rad] of the orientation error rows relative to the position rows.
        :param rot_tolerance: Orientation error [rad] at which iterations stop.
        :param nullspace_gain: Gain of the null-space attraction towards Q_REST. The posture is projected
                               with the exact (undamped) null-space projector: the damped one leaks into
                               the task and would stall the solver centimeters away from the target.
        :param max_step: Maximum joint change per iteration [rad].
        :param accept_pos: Position error [m] below which the final solution is reported as successful.
        :param accept_rot: Orientation error [rad] below which the final solution is reported as successful.
        :return: (q_sol, success_flag)
        """
        q = np.clip(np.asarray(q_init, dtype=np.float64), self.Q_MIN, self.Q_MAX)
        base_offset = self.base_position if self.base_position is not None else np.zeros(3)
        target_p_rel = np.asarray(target_pos, dtype=np.float64) - base_offset
        target_R = np.asarray(target_rot, dtype=np.float64) if target_rot is not None else None
        local_ee = self.EE_OFFSET.astype(np.float64)[None]
        q_rest = self.Q_REST.astype(np.float64)

        def errors(q_eval):
            T_all = self._batch_link_transforms(q_eval[None])
            J, p = self._point_jacobian_from_transforms(T_all, np.array([7]), local_ee)
            pos_err = target_p_rel - p[0]
            if target_R is None:
                return J[0, :3], pos_err, pos_err, 0.0
            rot_err = Rotation.from_matrix(target_R @ T_all[0, 7, :3, :3].T).as_rotvec()
            task_J = np.vstack([J[0, :3], rot_weight * J[0, 3:]])
            return task_J, np.concatenate([pos_err, rot_weight * rot_err]), pos_err, float(np.linalg.norm(rot_err))

        for _ in range(max_iters):
            J, e, pos_err, rot_err_norm = errors(q)
            if np.linalg.norm(pos_err) < pos_tolerance and rot_err_norm < rot_tolerance:
                return q.astype(np.float32), True

            J_pinv = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(J.shape[0]))
            nullspace = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J
            dq = J_pinv @ e + nullspace @ (nullspace_gain * (q_rest - q))
            dq *= min(1.0, max_step / max(np.max(np.abs(dq)), 1e-12))
            q = np.clip(q + dq, self.Q_MIN, self.Q_MAX)

        _, _, pos_err, rot_err_norm = errors(q)
        success = np.linalg.norm(pos_err) < accept_pos and rot_err_norm < accept_rot
        return q.astype(np.float32), bool(success)

    @staticmethod
    def rot_matrix_to_quat(R: np.ndarray) -> np.ndarray:
        """
        Converts a 3x3 rotation matrix to a normalized quaternion [qx, qy, qz, qw] (scalar-last)
        using scipy.spatial.transform.Rotation.
        """
        return Rotation.from_matrix(R).as_quat().astype(np.float32)

    @staticmethod
    def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
        """
        Converts a normalized quaternion [qx, qy, qz, qw] (scalar-last) to a 3x3 rotation matrix
        using scipy.spatial.transform.Rotation.
        """
        q_arr = np.asarray(q, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(q_arr))
        if norm > 1e-6:
            q_arr = q_arr / norm
        return Rotation.from_quat(q_arr).as_matrix().astype(np.float32)

    def batch_forward_kinematics_ee(
        self,
        q_batch: np.ndarray,
        return_orientations: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fast forward kinematics of end-effector positions (and optionally quaternions)
        for an entire batch of trajectories.

        :param q_batch: Array of shape [K, H, 7] containing joint trajectories.
        :param return_orientations: If True, also returns normalized quaternions [K, H, 4].
        :return: If return_orientations is False:
                     ee_positions: [K, H, 3]
                 If return_orientations is True:
                     (ee_positions [K, H, 3], ee_quaternions [K, H, 4])
        """
        K, H, _ = q_batch.shape
        T_all = self._batch_link_transforms(np.asarray(q_batch).reshape(-1, 7))
        T = T_all[:, 7]

        rot = T[:, :3, :3]
        base_offset = self.base_position if self.base_position is not None else np.zeros(3)
        pos = T[:, :3, 3] + rot @ self.EE_OFFSET.astype(np.float64) + base_offset
        ee_positions = pos.reshape(K, H, 3).astype(np.float32)

        if return_orientations:
            ee_quats = Rotation.from_matrix(rot).as_quat().reshape(K, H, 4).astype(np.float32)
            return ee_positions, ee_quats
        return ee_positions

    def _batch_link_transforms(self, q_flat: np.ndarray) -> np.ndarray:
        """
        Vectorized MDH chain for N configurations.

        :param q_flat: Joint angles of shape [N, 7].
        :return: Base-frame transforms of shape [N, 8, 4, 4]: [T_base, T_link1, ..., T_link7].
        """
        q_flat = np.asarray(q_flat, dtype=np.float64)
        N = q_flat.shape[0]
        T_all = np.empty((N, 8, 4, 4), dtype=np.float64)
        T_all[:, 0] = np.eye(4)
        T_i = np.zeros((N, 4, 4), dtype=np.float64)
        T_i[:, 3, 3] = 1.0
        for i, (a, alpha, d, th_offset) in enumerate(self.MDH_PARAMS):
            cos_th = np.cos(q_flat[:, i] + th_offset)
            sin_th = np.sin(q_flat[:, i] + th_offset)
            cos_al, sin_al = np.cos(alpha), np.sin(alpha)
            T_i[:, 0, 0] = cos_th
            T_i[:, 0, 1] = -sin_th
            T_i[:, 0, 3] = a
            T_i[:, 1, 0] = sin_th * cos_al
            T_i[:, 1, 1] = cos_th * cos_al
            T_i[:, 1, 2] = -sin_al
            T_i[:, 1, 3] = -sin_al * d
            T_i[:, 2, 0] = sin_th * sin_al
            T_i[:, 2, 1] = cos_th * sin_al
            T_i[:, 2, 2] = cos_al
            T_i[:, 2, 3] = cos_al * d
            T_all[:, i + 1] = T_all[:, i] @ T_i
        return T_all

    def batch_geometric_jacobian_ee(self, q_flat: np.ndarray) -> np.ndarray:
        """
        End-effector geometric Jacobians [J_linear; J_angular] for N configurations,
        identical to geometric_jacobian(q, link_idx=-1) applied to each row.

        :param q_flat: Joint angles of shape [N, 7].
        :return: Jacobians of shape [N, 6, 7].
        """
        N = np.asarray(q_flat).shape[0]
        J, _ = self.batch_point_jacobian(
            q_flat,
            link_idx=np.full(N, 7, dtype=np.int64),
            local_points=np.tile(self.EE_OFFSET.astype(np.float64), (N, 1))
        )
        return J

    def batch_link_linear_velocities(self, q_flat: np.ndarray, qd_flat: np.ndarray) -> np.ndarray:
        """
        Linear velocities of the link 1..7 frame origins and of the end-effector for N configurations,
        identical to compute_cartesian_velocity(q, qd, link_idx) for link_idx = 1..7 and -1.

        :param q_flat: Joint angles of shape [N, 7].
        :param qd_flat: Joint velocities of shape [N, 7].
        :return: Velocities of shape [N, 8, 3]; slot l-1 is link l, slot 7 is the end-effector.
        """
        q_flat = np.asarray(q_flat, dtype=np.float64)
        qd_flat = np.asarray(qd_flat, dtype=np.float64)
        N = q_flat.shape[0]
        T_all = self._batch_link_transforms(q_flat)
        origin = np.zeros((N, 3), dtype=np.float64)
        ee_offset = np.tile(self.EE_OFFSET.astype(np.float64), (N, 1))
        V = np.empty((N, 8, 3), dtype=np.float64)
        for slot in range(8):
            link = min(slot + 1, 7)
            J, _ = self._point_jacobian_from_transforms(
                T_all, np.full(N, link, dtype=np.int64), ee_offset if slot == 7 else origin
            )
            V[:, slot] = (J[:, :3, :] @ qd_flat[..., None])[..., 0]
        return V

    def batch_point_jacobian(
        self,
        q_flat: np.ndarray,
        link_idx: np.ndarray,
        local_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Geometric Jacobians of points rigidly attached to robot links, for N configurations.
        Only joints 1..link_idx move a point on link `link_idx`; the other columns are zero.

        :param q_flat: Joint angles of shape [N, 7].
        :param link_idx: Link carrying each point, shape [N], values in 1..7 (7 includes flange and hand).
        :param local_points: Point coordinates in the frame of their link, shape [N, 3].
        :return: (Jacobians [N, 6, 7], point positions in the robot base frame [N, 3]).
        """
        return self._point_jacobian_from_transforms(self._batch_link_transforms(q_flat), link_idx, local_points)

    def _point_jacobian_from_transforms(
        self,
        T_all: np.ndarray,
        link_idx: np.ndarray,
        local_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = T_all.shape[0]
        link_idx = np.asarray(link_idx, dtype=np.int64)
        T_link = T_all[np.arange(N), link_idx]
        p = T_link[:, :3, 3] + (T_link[:, :3, :3] @ np.asarray(local_points, dtype=np.float64)[..., None])[..., 0]

        J = np.zeros((N, 6, 7), dtype=np.float64)
        for i in range(7):
            moves = i < link_idx
            T_prime = T_all[:, i] @ self._MDH_RX_TX[i].astype(np.float64)
            z_i = T_prime[:, :3, 2]
            J[:, :3, i] = np.where(moves[:, None], np.cross(z_i, p - T_prime[:, :3, 3]), 0.0)
            J[:, 3:, i] = np.where(moves[:, None], z_i, 0.0)
        return J, p

    def world_to_link_point(self, q: np.ndarray, link_idx: int, point_world: np.ndarray) -> np.ndarray:
        """Expresses a world-frame point in the frame of link `link_idx` (1..7) at configuration q."""
        T = self._batch_link_transforms(np.asarray(q, dtype=np.float64)[None, :])[0, link_idx]
        base_offset = self.base_position if self.base_position is not None else np.zeros(3)
        p_base = np.asarray(point_world, dtype=np.float64) - base_offset
        return T[:3, :3].T @ (p_base - T[:3, 3])
