#!/usr/bin/env python
"""
Base Perception Interface for DREMA Suite.
Defines data structures and abstract contract for 3D dynamic perception pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


@dataclass
class DiscoveredObstacle:
    """Represents a physical obstacle extracted from the 3D scene scan."""
    oid: int
    name: str
    mesh_path: str
    initial_pos: Tuple[float, float, float]
    initial_quat: Tuple[float, float, float, float]
    dims: Tuple[float, float, float]
    rgb: List[float]
    is_target: bool = False
    pybullet_body_id: int = -1


@dataclass
class InitialScanResult:
    """Result of initial 360 scene reconstruction."""
    table_z: float
    table_bounds: Tuple[float, float, float, float]
    active_workspace_bounds: Dict[str, float]
    grid_origin: Tuple[float, float, float]
    grid_dim: Tuple[int, int, int]
    voxel_size: float
    discovered_obstacles: List[DiscoveredObstacle] = field(default_factory=list)
    raw_gaussians_count: int = 0
    retained_gaussians_count: int = 0
    restored_from_cache: bool = False


@dataclass
class StreamingUpdateResult:
    """Result of per-frame dynamic 3D perception update."""
    timestep: int
    active_gaussians_count: int
    tracked_object_poses: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]] = field(default_factory=dict)
    latency_ms: float = 0.0


class BasePerceptionModule(ABC):
    """
    Abstract interface for 3D perception and tracking backends.
    Allows seamlessly swapping VG-Mapping + RecurGS with other 3D reconstruction/tracking modules.
    """

    @abstractmethod
    def process_initial_scan(
        self,
        scan_frames: List[Dict[str, Any]],
        semantic_labels: Dict[str, int],
        robot_base_pos: np.ndarray,
        digital_twin: Optional[Any] = None,
        reachability_radius: Optional[float] = None
    ) -> InitialScanResult:
        """Processes initial 360 scan frames or restores from pre-computed cache."""
        pass

    @abstractmethod
    def update_streaming_frame(
        self,
        timestep: int,
        camera_views: Dict[str, Dict[str, Any]],
        robot_state: Optional[Dict[str, Any]] = None,
        digital_twin: Optional[Any] = None
    ) -> StreamingUpdateResult:
        """Processes incoming multi-camera streaming frames and updates dynamic 3DGS & tracking."""
        pass

    @abstractmethod
    def get_viser_splats_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Returns centers, covariances, rgbs, and opacities formatted for Viser 3D Web Visualizer."""
        pass

    def get_viser_surface_voxels(self) -> Optional[Dict[str, np.ndarray]]:
        """Optional method returning discrete TSDF surface voxels for Viser visualization."""
        return None

    @abstractmethod
    def save_cache(self, cache_dir: str) -> bool:
        """Saves reconstructed initial scene state to disk."""
        pass

    @abstractmethod
    def load_cache(
        self,
        cache_dir: str,
        digital_twin: Optional[Any] = None
    ) -> Optional[InitialScanResult]:
        """Loads reconstructed initial scene state from disk."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets perception state for a new episode."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Releases GPU tensors and pipeline resources."""
        pass
