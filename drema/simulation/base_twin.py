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

