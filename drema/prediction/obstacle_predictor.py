"""
Re-export module for backward compatibility.
The architecture is cleanly decomposed into:
- drema.prediction.base_predictor: Abstract interface BaseObstaclePredictor & TrajectoryPrediction container.
- drema.prediction.kalman_predictor: Kalman / Kinematic Bayesian filter on SE(3) x R^6.
"""

from .base_predictor import BaseObstaclePredictor, TrajectoryPrediction
from .kalman_predictor import (
    SingleObjectKinematicFilter,
    KalmanObstaclePredictor,
    ObstacleTrajectoryPredictor,
)

__all__ = [
    "BaseObstaclePredictor",
    "TrajectoryPrediction",
    "SingleObjectKinematicFilter",
    "KalmanObstaclePredictor",
    "ObstacleTrajectoryPredictor",
]
