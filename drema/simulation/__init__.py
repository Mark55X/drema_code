# Simulation package for DREMA
from .base_twin import BaseDigitalTwin, DEFAULT_TABLE_COLOR, DEFAULT_OBSTACLE_COLOR
from .pybullet_digital_twin import PyBulletDigitalTwin
from .mujoco_digital_twin import MuJoCoDigitalTwin


def create_digital_twin(engine: str = "pybullet", **kwargs) -> BaseDigitalTwin:
    """
    Factory function to instantiate the requested Physics Digital Twin backend.

    :param engine: Simulation backend ("pybullet" or "mujoco").
    :param kwargs: Backend specific configuration parameters.
    :return: Concrete instance of BaseDigitalTwin.
    """
    eng = engine.lower()
    if eng == "pybullet":
        return PyBulletDigitalTwin(**kwargs)
    elif eng == "mujoco":
        return MuJoCoDigitalTwin(**kwargs)
    else:
        raise ValueError(f"Unsupported digital twin engine: '{engine}' (expected 'pybullet' or 'mujoco')")


__all__ = [
    "BaseDigitalTwin",
    "PyBulletDigitalTwin",
    "MuJoCoDigitalTwin",
    "create_digital_twin",
    "DEFAULT_TABLE_COLOR",
    "DEFAULT_OBSTACLE_COLOR"
]
