#!/usr/bin/env python
"""
Abstract Base Interface for Physics Digital Twins in DREMA Suite.
Defines the required simulation contract for physics backends (e.g., PyBullet, Isaac Sim).
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any, Union


# Default physical / visual rendering attributes across all digital twin backends
DEFAULT_TABLE_COLOR: Tuple[float, float, float, float] = (0.82, 0.82, 0.82, 1.0)
DEFAULT_OBSTACLE_COLOR: Tuple[float, float, float, float] = (0.2, 0.45, 0.85, 1.0)


class BaseDigitalTwin(ABC):
    """
    Abstract Base Class for Digital Twin simulation engines.
    Any concrete physics backend (PyBullet, Isaac Sim, MuJoCo) must implement this interface.
    """

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
