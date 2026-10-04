"""
DREMA Dynamic Obstacle Trajectory Prediction Module.

Architectural Decomposition:
- Base Interface: `BaseObstaclePredictor` in `drema.prediction.base_predictor`
- Data Container: `TrajectoryPrediction` in `drema.prediction.base_predictor`
- Implementations:
    * `KalmanObstaclePredictor` (Bayesian Kinematic Filter on SE(3) x R^6) in `drema.prediction.kalman_predictor`
    * Future predictors (Neural ODE, DDPM, Diffusion, Oracle) inherit from `BaseObstaclePredictor`.
"""

from .base_predictor import (
    BaseObstaclePredictor,
    TrajectoryPrediction,
)
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
