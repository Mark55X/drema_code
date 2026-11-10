# Simulation package for DREMA
from .base_twin import BaseDigitalTwin, DEFAULT_TABLE_COLOR, DEFAULT_OBSTACLE_COLOR
from .pybullet_digital_twin import PyBulletDigitalTwin
from .mujoco_digital_twin import MuJoCoDigitalTwin
from .mujoco_warp_twin import MuJoCoWarpDigitalTwin


def create_digital_twin(engine: str = "pybullet", **kwargs) -> BaseDigitalTwin:
    """
    Factory function to instantiate the requested Physics Digital Twin backend.

    :param engine: Simulation backend ("pybullet", "mujoco" or "mujoco_warp").
    :param kwargs: Backend specific configuration parameters.
    :return: Concrete instance of BaseDigitalTwin.
    """
    eng = engine.lower()
    if eng == "pybullet":
        return PyBulletDigitalTwin(**kwargs)
    elif eng == "mujoco":
        return MuJoCoDigitalTwin(**kwargs)
    elif eng == "mujoco_warp":
        return MuJoCoWarpDigitalTwin(**kwargs)
    else:
        raise ValueError(f"Unsupported digital twin engine: '{engine}' (expected 'pybullet', 'mujoco' or 'mujoco_warp')")


__all__ = [
    "BaseDigitalTwin",
    "PyBulletDigitalTwin",
    "MuJoCoDigitalTwin",
    "MuJoCoWarpDigitalTwin",
    "create_digital_twin",
    "DEFAULT_TABLE_COLOR",
    "DEFAULT_OBSTACLE_COLOR"
]
