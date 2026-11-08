#!/usr/bin/env python
"""
What the closed-loop runner (drema/client/runner.py) needs from a robot environment: sensors, robot, task.

The runner owns the DREMA server connection, the scan / streaming / control schedule, the keyframe sequencing, the
object roles and the console; an environment owns the cameras, the arm and gripper, the clock and the task. One loop
iteration of the runner is one `advance()` of the environment, `dt` seconds of environment time.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .keyframes import KeyframeSource

# One initial scan view: (name, rgb uint8 HxWx3, depth float32 HxW [m], cam-to-world 4x4, K 3x3, near, far, mask)
ScanView = Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, Optional[np.ndarray]]


@dataclass
class RobotReading:
    joint_positions: List[float]
    joint_velocities: List[float]
    ee_pose: List[float]        # TCP [x, y, z, qx, qy, qz, qw] in the world frame, in the convention of the keyframes
    gripper_open: float         # measured opening, 1 fully open, 0 closed


@dataclass
class SceneScan:
    views: List[ScanView]
    joint_positions: List[float] = field(default_factory=list)


class Environment:
    """
    Camera frames are dicts {'rgb', 'depth', 'extrinsics' (cam-to-world), 'intrinsics', 'near_clipping',
    'far_clipping', 'mask'}; mask pixels hold the ids of `semantic_labels`, whose `label_categories` say robot, object
    or background, or None when the environment has no segmentation.
    `actuator` exposes apply(joint_velocities) / hold() / reset(); `gripper` exposes command(open_amount), step(),
    reset(), is_open(), done, busy and target, as drema.client.gripper.RLBenchGripperController.
    """

    client_tag = "Client"                # prefix of the runner's own messages
    log_tag = "ENVIRONMENT"              # prefix of the task / control messages
    sync_modes: Tuple[str, ...] = ("realtime",)
    keyframe_modes: Tuple[str, ...] = ("none",)

    task_name: str = ""
    task_description: str = ""
    dt: float = 0.05                     # environment time advanced by one loop iteration [s]
    reachability_radius: float = 0.95
    semantic_labels: Dict[str, int]
    label_categories: Dict[int, str]
    actuator: Any = None
    gripper: Any = None

    def launch(self) -> None:
        """Starts sensors, robot and task; afterwards task_description, labels and gripper are valid."""
        raise NotImplementedError

    def ready(self) -> None:
        """Called once the keyframes are prepared, before the first loop iteration: arm actuation, recorders."""

    def reset(self) -> None:
        """New episode: task, clock and gripper back to their initial state (the runner resets the actuator)."""
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    def describe(self) -> None:
        """Printed once when the loop starts (kinematics inspection, camera list)."""

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def timestamp(self) -> float:
        """Time [s] of the current loop iteration on the environment clock, > 0."""
        raise NotImplementedError

    def set_stream_rate(self, hz: float) -> None:
        """Rate [Hz] at which the runner will ask for streaming frames."""

    def capture_cameras(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def frames_timestamp(self, frames: Dict[str, Dict[str, Any]]) -> float:
        """Time of a set of streaming frames on the environment clock (their capture, if not this iteration)."""
        return self.timestamp()

    def capture_scan(self, num_steps: int) -> SceneScan:
        """Views of the initial scene scan. `num_steps` is the scan length requested on the command line."""
        raise NotImplementedError

    def robot_base_position(self) -> List[float]:
        raise NotImplementedError

    def read_robot(self) -> RobotReading:
        raise NotImplementedError

    def keyframe_source(self, mode: str, policy_spec: Optional[str]) -> Optional[KeyframeSource]:
        """Source of the episode keyframes for `mode` (one of keyframe_modes, 'auto' resolved by the environment)."""
        return None

    def fallback_target(self) -> Tuple[List[float], bool]:
        """Target pose of a task without keyframes: (pose, available)."""
        return [], False

    def observe(self) -> Any:
        """Observation handed to keyframe policies."""
        return None

    def advance(self, step: int, task_active: bool) -> None:
        """Advances the environment by dt after the action of this iteration was applied."""
        raise NotImplementedError

    def check_success(self) -> bool:
        return False


def clip_joint_velocities(v: Sequence[float], limit: Sequence[float]) -> np.ndarray:
    """Scales the whole vector so no joint exceeds its limit: the direction in joint space is kept."""
    v = np.asarray(v, dtype=np.float64)
    limit = np.asarray(limit, dtype=np.float64)
    ratio = float(np.max(np.abs(v) / np.maximum(limit, 1e-9))) if v.size else 0.0
    return v / ratio if ratio > 1.0 else v
