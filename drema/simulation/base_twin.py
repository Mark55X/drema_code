#!/usr/bin/env python
"""
Abstract Base Interface for Physics Digital Twins in DREMA Suite.
Defines the required simulation contract for physics backends (e.g., PyBullet, Isaac Sim).
"""

import functools
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from scipy.spatial.transform import Rotation

from ..roles.types import FULL_CONTACT, ObjectCollisionPolicy



# Default physical / visual rendering attributes across all digital twin backends
DEFAULT_TABLE_COLOR: Tuple[float, float, float, float] = (0.82, 0.82, 0.82, 1.0)
DEFAULT_OBSTACLE_COLOR: Tuple[float, float, float, float] = (0.2, 0.45, 0.85, 1.0)


def synchronized(method):
    """
    Runs a twin method under the twin's re-entrant state lock.
    The perception worker and the gRPC control threads share one physics state; the MPC
    collision rollout temporarily teleports robot and obstacles, so interleaved calls
    would observe or overwrite those transient poses.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.state_lock:
            return method(self, *args, **kwargs)
    return wrapper


def aggregate_gvm_costs(
    Q_eval: np.ndarray,
    QD_eval: np.ndarray,
    rec_k: List[int],
    rec_e: List[int],
    rec_group: List[int],
    rec_vel_slot: List[int],
    rec_d: List[float],
    rec_normal: List[Any],
    rec_v_obs: List[Any],
    n_groups: int,
    sigma_1: float,
    sigma_2: float,
    kappa: float,
    rho: float,
    kin_helper: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized C_P and GVM-SDF costs (Zhou et al. Eq. 10, 21, 25) from the closest-point records of a
    batch of configurations Q_eval / QD_eval [K, E, 7]. Each record carries its configuration (k, e),
    an aggregation group (max over the points of a group, then sum over groups), the velocity slot of
    kin_helper.batch_link_linear_velocities, the distance, the obstacle-to-robot normal and the obstacle velocity.

    :return: (coll_p, coll_gvm) of shape [K, E].
    """
    K, E, _ = Q_eval.shape
    coll_p = np.zeros((K * E, n_groups), dtype=np.float64)
    coll_gvm = np.zeros((K * E, n_groups), dtype=np.float64)
    if rec_d:
        d = np.asarray(rec_d, dtype=np.float64)
        phi = np.where(d < sigma_1, 1.0, np.where(d <= sigma_2, np.exp(-kappa * (d - sigma_1)), 0.0))
        keep = phi > 1e-4
        if np.any(keep):
            phi = phi[keep]
            ke = (np.asarray(rec_k)[keep] * E + np.asarray(rec_e)[keep])
            group = np.asarray(rec_group)[keep]
            normal = np.asarray(rec_normal, dtype=np.float64).reshape(-1, 3)[keep]
            v_obs = np.asarray(rec_v_obs, dtype=np.float64).reshape(-1, 3)[keep]

            V = kin_helper.batch_link_linear_velocities(Q_eval.reshape(-1, 7), QD_eval.reshape(-1, 7))
            v_rel = V[ke, np.asarray(rec_vel_slot)[keep]] - v_obs
            vel_mag = np.linalg.norm(v_rel, axis=-1)
            norm_mag = np.linalg.norm(normal, axis=-1)
            valid = (vel_mag > 1e-4) & (norm_mag > 1e-6)
            cos_theta = np.where(
                valid, np.sum(normal * v_rel, axis=-1) / np.maximum(norm_mag * vel_mag, 1e-12), 0.0
            )
            gvm = phi * (1.0 + vel_mag * (1.0 - rho * cos_theta))

            np.maximum.at(coll_p, (ke, group), phi)
            np.maximum.at(coll_gvm, (ke, group), gvm)
    return (coll_p.sum(axis=1).reshape(K, E).astype(np.float32),
            coll_gvm.sum(axis=1).reshape(K, E).astype(np.float32))


class BaseDigitalTwin(ABC):
    """
    Abstract Base Class for Digital Twin simulation engines.
    Any concrete physics backend (PyBullet, Isaac Sim, MuJoCo) must implement this interface.
    """

    def __init__(self, joint_angle_offsets: Optional[List[float]] = None):
        """
        :param joint_angle_offsets: Per-joint zero offsets [rad] between the measured joint angles of
                                    the controlled robot and the twin's (Franka DH) robot model:
                                    q_twin = q_measured + offset. Every joint value crossing the twin
                                    interface is in the measured space.
        """
        self.state_lock = threading.RLock()
        self.joint_angle_offsets = (
            np.asarray(joint_angle_offsets, dtype=np.float64).reshape(7)
            if joint_angle_offsets is not None else np.zeros(7, dtype=np.float64)
        )
        # Diagnostics only (e.g. drema.visualization): when True, compute_trajectory_collision_costs keeps the
        # robot-side closest points of its last call in last_collision_points.
        self.record_collision_points = False
        self.last_collision_points: Optional[Dict[str, np.ndarray]] = None

        # Object roles (drema.roles): per-object collision policies and objects held by the gripper,
        # with the TCP->object transform and the kinematics (measured joint space) that moves them.
        self.object_policies: Dict[int, ObjectCollisionPolicy] = {}
        self.table_policy = ObjectCollisionPolicy()
        self.attachments: Dict[int, np.ndarray] = {}
        self._attach_kin: Any = None
        self._surface_points: Dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # Object roles (shared by every backend)
    # -------------------------------------------------------------------------

    def object_ids(self) -> List[int]:
        return list(getattr(self, 'tracked_objects', {}).keys())

    def is_legacy_target(self, obj_id: int) -> bool:
        return bool(getattr(self, 'tracked_objects', {}).get(obj_id, {}).get('is_target', False))

    def set_object_policies(self, policies: Dict[int, ObjectCollisionPolicy],
                            table_policy: Optional[ObjectCollisionPolicy] = None) -> None:
        with self.state_lock:
            self.object_policies = dict(policies)
            self.table_policy = table_policy if table_policy is not None else ObjectCollisionPolicy()

    def object_policy(self, obj_id: int) -> ObjectCollisionPolicy:
        if obj_id in self.attachments:
            return ObjectCollisionPolicy(hand_contact=True, arm_contact=True, held=True)
        policy = self.object_policies.get(obj_id)
        if policy is not None:
            return policy
        return FULL_CONTACT if self.is_legacy_target(obj_id) else ObjectCollisionPolicy()

    def _object_surface_points(self, mesh_path: str, max_points: int = 2000) -> np.ndarray:
        """Vertices of an object mesh in its body frame (perception exports meshes centred on the body origin)."""
        pts = self._surface_points.get(mesh_path)
        if pts is None:
            import trimesh
            pts = np.asarray(trimesh.load(mesh_path, force='mesh').vertices, dtype=np.float64)
            if len(pts) > max_points:
                pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(np.int64)]
            self._surface_points[mesh_path] = pts
        return pts

    def object_body_points(self, obj_id: int) -> Optional[np.ndarray]:
        """Surface points of a tracked object in its body frame, or None without a mesh."""
        mesh_path = getattr(self, 'tracked_objects', {}).get(obj_id, {}).get('mesh_path')
        if not mesh_path:
            return None
        try:
            return self._object_surface_points(mesh_path)
        except Exception:
            return None

    def _object_world_points(self, obj_id: int) -> Optional[np.ndarray]:
        local = self.object_body_points(obj_id)
        pose = self.tracked_object_pose(obj_id)
        if local is None or pose is None or not len(local):
            return None
        return local @ Rotation.from_quat(pose[1]).as_matrix().T + np.asarray(pose[0])

    def objects_near_point(self, point: np.ndarray, radius: float, include_table: bool = False) -> Dict[int, float]:
        """
        Objects whose surface lies within `radius` of a world point, as {obj_id: distance}; the table, when
        requested and present, is reported with id -1 (distance to its top plane).
        """
        point = np.asarray(point, dtype=np.float64).reshape(3)
        near: Dict[int, float] = {}
        with self.state_lock:
            for obj_id in getattr(self, 'tracked_objects', {}):
                world = self._object_world_points(obj_id)
                if world is None:
                    continue
                d = float(np.min(np.linalg.norm(world - point, axis=1)))
                if d <= radius:
                    near[obj_id] = d
            if include_table and getattr(self, 'table_id', -1) >= 0:
                d_table = max(0.0, float(point[2]) - float(getattr(self, 'table_z', 0.0)))
                if d_table <= radius:
                    near[-1] = d_table
        return near

    def objects_near_geometry(
        self,
        radius: float,
        boxes: List[Tuple[np.ndarray, np.ndarray]] = (),
        points: Optional[np.ndarray] = None,
        include_table: bool = False,
        exclude: Any = (),
        signed: bool = False
    ) -> Dict[int, float]:
        """
        Objects whose surface lies within `radius` of a world geometry made of oriented boxes
        [(T_world_box 4x4, half_extents)] and a point cloud [N, 3], as {obj_id: distance}; the table is id -1.
        With `signed`, surface points inside a box count as minus their depth (how far the boxes overlap the object).
        """
        from scipy.spatial import cKDTree
        tree = cKDTree(points) if points is not None and len(points) else None
        near: Dict[int, float] = {}
        with self.state_lock:
            for obj_id in getattr(self, 'tracked_objects', {}):
                if obj_id in exclude:
                    continue
                world = self._object_world_points(obj_id)
                if world is None:
                    continue
                d = float('inf')
                for T, half in boxes:
                    q = np.abs((world - T[:3, 3]) @ T[:3, :3]) - half
                    dist = np.linalg.norm(np.maximum(q, 0.0), axis=1)
                    if signed:
                        dist = dist + np.minimum(np.max(q, axis=1), 0.0)
                    d = min(d, float(np.min(dist)))
                if tree is not None:
                    d = min(d, float(np.min(tree.query(world)[0])))
                if d <= radius:
                    near[obj_id] = d
            if include_table and getattr(self, 'table_id', -1) >= 0:
                lows = [float(np.min((T[:3, :3] @ (np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1)
                                                    * np.asarray(half)[:, None]))[2] + T[2, 3]))
                        for T, half in boxes]
                if points is not None and len(points):
                    lows.append(float(np.min(points[:, 2])))
                if lows:
                    d_table = max(0.0, min(lows) - float(getattr(self, 'table_z', 0.0)))
                    if d_table <= radius:
                        near[-1] = d_table
        return near

    def object_velocities(self) -> Dict[int, np.ndarray]:
        """Estimated linear velocity [m/s] of each tracked object (zero without a motion estimate)."""
        return {int(o['id']): np.asarray(o.get('velocity', (0.0, 0.0, 0.0)), dtype=np.float64)
                for o in self.get_tracked_obstacles_info()}

    def object_speeds(self) -> Dict[int, float]:
        """Estimated linear speed [m/s] of each tracked object (0 without a motion estimate)."""
        return {obj_id: float(np.linalg.norm(v)) for obj_id, v in self.object_velocities().items()}

    def tracked_object_pose(self, obj_id: int) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Pose last set by tracking (or by the gripper when held), not the physics body lagging behind it."""
        return self.get_object_pose(obj_id)

    def attach_object(self, obj_id: int, tcp_T_obj: np.ndarray, kinematics: Any) -> None:
        """Rigidly attaches a tracked object to the TCP: it follows the robot instead of perception."""
        with self.state_lock:
            self.attachments[obj_id] = np.asarray(tcp_T_obj, dtype=np.float64)
            self._attach_kin = kinematics
            self._on_attach(obj_id)

    def detach_object(self, obj_id: int) -> None:
        with self.state_lock:
            if self.attachments.pop(obj_id, None) is None:
                return
            # The pose history while held came from the robot, not from perception.
            predictor = getattr(self, 'predictor', None)
            if predictor is not None and hasattr(predictor, 'filters'):
                predictor.filters.pop(obj_id, None)
            self._on_detach(obj_id)

    def _on_attach(self, obj_id: int) -> None:
        pass

    def _on_detach(self, obj_id: int) -> None:
        pass

    def attached_poses(self, q_flat: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        World poses of the held objects for a batch of measured joint configurations [N, 7]:
        {obj_id: (positions [N, 3], quaternions xyzw [N, 4])}.
        """
        if not self.attachments or self._attach_kin is None:
            return {}
        q_flat = np.asarray(q_flat, dtype=np.float64).reshape(-1, 7)
        pos, quat = self._attach_kin.batch_forward_kinematics_ee(q_flat[None], return_orientations=True)
        R_tcp = Rotation.from_quat(quat[0].astype(np.float64)).as_matrix()
        p_tcp = pos[0].astype(np.float64)
        out = {}
        for obj_id, T in self.attachments.items():
            R_obj = R_tcp @ T[:3, :3]
            p_obj = p_tcp + R_tcp @ T[:3, 3]
            out[obj_id] = (p_obj, Rotation.from_matrix(R_obj).as_quat())
        return out

    def sync_gripper(self, open_amount: float) -> None:
        """Mirrors the measured gripper opening (1 open, 0 closed) on the twin fingers."""
        pass

    @abstractmethod
    def load_robot(
        self,
        base_position: Tuple[float, float, float],
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        joint_positions: Optional[List[float]] = None
    ) -> bool:
        """Loads or updates the manipulator arm at the specified world base pose. Returns True on success."""
        pass

    @abstractmethod
    def sync_robot_state(
        self,
        joint_positions: List[float],
        joint_velocities: Optional[List[float]] = None
    ) -> None:
        """Synchronizes the physical twin robot joints with current kinematic state."""
        pass

    @abstractmethod
    def spawn_scanned_table(
        self,
        table_z: float,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        mesh_file_path: Optional[str] = None,
        color: Tuple[float, float, float, float] = DEFAULT_TABLE_COLOR
    ) -> int:
        """Spawns the tabletop support structure discovered by initial 3D scan."""
        pass

    @abstractmethod
    def spawn_scanned_mesh_obstacle(
        self,
        mesh_path: str,
        initial_pos: Tuple[float, float, float],
        initial_quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str = "obstacle",
        mass: float = 1.0,
        color: Optional[Union[List[float], Tuple[float, ...]]] = None,
        obj_id: Optional[int] = None,
        **kwargs
    ) -> int:
        """Spawns an extracted Marching Cubes surface mesh obstacle into the physics engine."""
        pass

    @abstractmethod
    def sync_object_pose(
        self,
        obj_id: int,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float, float]
    ) -> None:
        """Updates rigid body position and orientation estimated by SE(3) tracking."""
        pass

    def get_object_pose(
        self,
        obj_id: int
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """Retrieves rigid body position and orientation for the given object ID."""
        return None

    @abstractmethod
    def step_simulation(self) -> None:
        """Advances physical simulation step."""
        pass

    def step(self) -> None:
        """Alias for step_simulation()."""
        self.step_simulation()

    def get_tracked_obstacles_info(self) -> List[Dict[str, Any]]:
        """
        Retrieves list of tracked dynamic obstacle dictionaries.
        Each dictionary contains:
            - 'id': integer obstacle identifier
            - 'position': (x, y, z) tuple
            - 'orientation': (x, y, z, w) tuple
            - 'name': str name
            - 'is_target': bool flag indicating if object is the target to manipulate
        """
        return []

    def get_obstacle_proximity(self, max_distance: float) -> List[Dict[str, Any]]:
        """
        Closest surface points between each robot link and each non-target obstacle, at the
        current robot configuration. One entry per (obstacle, link) pair closer than max_distance:
            - 'obj_id': obstacle identifier
            - 'link_idx': kinematic link carrying the robot point (1..7; flange and hand map to 7)
            - 'point_world': closest point on the robot surface, world frame
            - 'normal': unit vector from the obstacle surface towards the robot
            - 'distance': signed distance in meters (negative when penetrating)
            - 'velocity': estimated obstacle linear velocity from the predictor
        """
        return []

    def calculate_inverse_kinematics(
        self,
        target_pos: Tuple[float, float, float],
        target_quat: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """
        Calculates inverse kinematics solution for Franka Panda end-effector.
        Returns 7-element numpy array of joint angles, or None if unavailable/unconverged.
        """
        return None

    def compute_trajectory_collision_costs(
        self,
        Q: np.ndarray,
        QD: np.ndarray,
        sigma_1: float,
        sigma_2: float,
        kappa: float,
        rho: float,
        kin_helper: Any,
        dt: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates collision clearance and GVM (Gradient-Velocity Modulated) costs for
        a batch of candidate trajectories Q (K x H x 7) and QD (K x H x 7).
        
        Returns:
            coll_p: (K x H) float32 numpy array of static proximity/penetration costs
            coll_gvm: (K x H) float32 numpy array of dynamic approach-velocity modulated costs
        """
        K, H, _ = Q.shape
        return np.zeros((K, H), dtype=np.float32), np.zeros((K, H), dtype=np.float32)

    @abstractmethod
    def reset(self) -> None:
        """Resets the simulation environment and cleans up dynamic bodies."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleans up physics server and closes visualizer."""
        pass

    def close(self) -> None:
        """Alias for shutdown()."""
        self.shutdown()

