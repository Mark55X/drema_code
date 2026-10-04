from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TrajectoryPrediction:
    """
    Standard container for predicted multi-step obstacle future states.
    All arrays have length H (horizon steps) along the first dimension.
    """
    positions: np.ndarray          # Shape: [H, 3] in world frame [m]
    velocities: np.ndarray         # Shape: [H, 3] linear velocities [m/s]
    orientations: np.ndarray       # Shape: [H, 4] quaternions [x, y, z, w]
    angular_velocities: np.ndarray # Shape: [H, 3] angular velocities [rad/s]
    timestamps: np.ndarray         # Shape: [H] future absolute timestamps [s]


class BaseObstaclePredictor(ABC):
    """
    Generic Abstract Interface for Dynamic Obstacle Trajectory Predictors.
    
    Agnostic to forecasting paradigm (Kalman filtering, Neural ODEs, Diffusion,
    Physics rollouts, or Ground-Truth Oracle).
    Any future prediction architecture should inherit from this base class.
    """

    @abstractmethod
    def update_obstacle_pose(
        self,
        obj_id: Union[int, str],
        position: Union[np.ndarray, List[float], Tuple[float, ...]],
        orientation: Union[np.ndarray, List[float], Tuple[float, ...]],
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Updates internal dynamic state of the given obstacle with a new 6D observation.
        :param obj_id: Identifier of the tracked obstacle.
        :param position: 3D position [x, y, z] in world coordinates.
        :param orientation: 4D quaternion [x, y, z, w].
        :param timestamp: Sensor observation timestamp (or None for current time).
        :return: True if observation was accepted, False if rejected as outlier.
        """
        pass

    @abstractmethod
    def predict_obstacle_trajectory(
        self,
        obj_id: Union[int, str],
        horizon: int,
        dt: float,
        latency_comp_sec: float = 0.0
    ) -> Optional[TrajectoryPrediction]:
        """
        Forecasts future states for the designated obstacle across H lookahead steps.
        :param obj_id: Target obstacle ID.
        :param horizon: Number of prediction steps H.
        :param dt: Time duration per step [s].
        :param latency_comp_sec: Extra lookahead time to compensate for pipeline latency.
        :return: TrajectoryPrediction object or None if object is untracked.
        """
        pass

    @abstractmethod
    def predict_all(
        self,
        horizon: int,
        dt: float,
        latency_comp_sec: float = 0.0
    ) -> Dict[Union[int, str], TrajectoryPrediction]:
        """
        Forecasts future states for ALL actively tracked obstacles across horizon H.
        :return: Dictionary mapping obj_id -> TrajectoryPrediction.
        """
        pass

    @abstractmethod
    def get_estimated_state(self, obj_id: Union[int, str]) -> Optional[Dict[str, np.ndarray]]:
        """
        Returns the instantaneous filtered state dictionary for an obstacle
        (e.g., 'position', 'velocity', 'acceleration', 'orientation', 'angular_velocity').
        """
        pass

    def prune_stale_obstacles(self, timeout_sec: Optional[float] = None) -> List[Union[int, str]]:
        """Removes tracked obstacles that have ceased receiving sensory updates."""
        return []

    @abstractmethod
    def reset(self) -> None:
        """Clears all tracked obstacle states."""
        pass
