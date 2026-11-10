#!/usr/bin/env python
"""
Per-pixel labels of the laboratory cameras, in the form the server expects from every environment: an int32 mask of
label ids, with semantic_labels (name -> id) and label_categories (id -> robot / object / background).

- Robot: rendered from the FR3 model at the measured joints into each calibrated camera (collision geometry, a bit
  larger than the arm: robot pixels are not left in the map).
- Objects: an ObjectSegmenter. None is implemented yet (planned: SAM3 prompts on the scan, a video tracker after),
  so the server receives robot labels only.
"""

import os
import threading
from typing import Dict, Optional, Sequence

import numpy as np

BACKGROUND_ID = 0
ROBOT_ID = 1
ROBOT_LABEL = "fr3_robot"


class ObjectSegmenter:
    """
    Object instance masks with ids stable over the episode. `start` sees the scan frames and returns the objects
    found (name -> id, ids above ROBOT_ID); `segment` labels one streaming frame of a camera. A pixel the segmenter
    leaves at BACKGROUND_ID is static scene for the server.
    """

    def start(self, frames: Dict[str, Dict]) -> Dict[str, int]:
        return {}

    def segment(self, camera: str, rgb: np.ndarray, depth: np.ndarray) -> Optional[np.ndarray]:
        return None

    def physics(self) -> Dict[int, Dict]:
        """Object id -> {'mass' [kg], 'friction', 'static'} from the model that estimates them; {} when unknown."""
        return {}

    def reset(self) -> None:
        pass


def opengl_projection(K: np.ndarray, width: int, height: int, near: float, far: float) -> np.ndarray:
    """OpenGL projection of a pinhole camera (OpenCV K, pixel centres at integer coordinates, rows from the top)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([
        [2.0 * fx / width, 0.0, 1.0 - 2.0 * (cx + 0.5) / width, 0.0],
        [0.0, 2.0 * fy / height, 2.0 * (cy + 0.5) / height - 1.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ])


CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


class RobotMaskRenderer:
    """FR3 silhouette in a camera, rendered with PyBullet's CPU renderer (no display, no GPU)."""

    def __init__(self, urdf_path: str, joint_names: Sequence[str], finger_joint_names: Sequence[str],
                 base_position: Sequence[float] = (0.0, 0.0, 0.0), dilation_px: int = 2):
        import pybullet as p
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"Robot mask URDF not found: {urdf_path}")
        self.p = p
        self.client = p.connect(p.DIRECT)
        self.robot = p.loadURDF(urdf_path, basePosition=list(base_position), useFixedBase=True,
                                flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, physicsClientId=self.client)
        index = {p.getJointInfo(self.robot, j, physicsClientId=self.client)[1].decode(): j
                 for j in range(p.getNumJoints(self.robot, physicsClientId=self.client))}
        missing = [n for n in list(joint_names) + list(finger_joint_names) if n not in index]
        if missing:
            raise ValueError(f"Joints {missing} not in {urdf_path}")
        self.arm_joints = [index[n] for n in joint_names]
        self.finger_joints = [index[n] for n in finger_joint_names]
        self.dilation_px = int(dilation_px)

    def set_state(self, q: Sequence[float], finger_width: Optional[float]) -> None:
        p = self.p
        for j, v in zip(self.arm_joints, q):
            p.resetJointState(self.robot, j, float(v), physicsClientId=self.client)
        if finger_width is not None:
            for j in self.finger_joints:
                p.resetJointState(self.robot, j, 0.5 * float(finger_width), physicsClientId=self.client)

    def render(self, c2w: np.ndarray, K: np.ndarray, width: int, height: int,
               near: float = 0.05, far: float = 4.0) -> np.ndarray:
        """Boolean HxW robot mask of an OpenCV camera (c2w: camera-to-world of the optical frame)."""
        p = self.p
        view = np.linalg.inv(np.asarray(c2w, dtype=np.float64) @ CV_TO_GL)
        proj = opengl_projection(np.asarray(K, dtype=np.float64), width, height, near, far)
        _, _, _, _, seg = p.getCameraImage(width, height, viewMatrix=view.T.reshape(-1).tolist(),
                                           projectionMatrix=proj.T.reshape(-1).tolist(),
                                           renderer=p.ER_TINY_RENDERER, physicsClientId=self.client)
        mask = np.asarray(seg, dtype=np.int64).reshape(height, width) == self.robot
        if self.dilation_px > 0 and mask.any():
            import cv2
            k = 2 * self.dilation_px + 1
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        return mask

    def close(self) -> None:
        try:
            self.p.disconnect(self.client)
        except Exception:
            pass


class LabSegmentation:
    """Combines robot rendering and object segmenter into the label mask of each frame."""

    def __init__(self, robot_renderer: Optional[RobotMaskRenderer], objects: Optional[ObjectSegmenter] = None):
        self.robot_renderer = robot_renderer
        self.objects = objects or ObjectSegmenter()
        self.object_labels: Dict[str, int] = {}
        # Camera threads label concurrently; one PyBullet client and one segmenter serve them all
        self._lock = threading.Lock()

    @property
    def semantic_labels(self) -> Dict[str, int]:
        labels = dict(self.object_labels)
        if self.robot_renderer is not None:
            labels[ROBOT_LABEL] = ROBOT_ID
        return labels

    @property
    def label_categories(self) -> Dict[int, str]:
        categories = {i: 'object' for i in self.object_labels.values()}
        if self.robot_renderer is not None:
            categories[ROBOT_ID] = 'robot'
        return categories

    @property
    def object_physics(self) -> Dict[int, Dict]:
        return dict(self.objects.physics())

    def start(self, frames: Dict[str, Dict]) -> None:
        self.object_labels = dict(self.objects.start(frames))

    def label(self, camera: str, frame: Dict, q: Sequence[float], finger_width: Optional[float]) -> Optional[np.ndarray]:
        """int32 label mask of one frame, or None with neither robot model nor object segmenter."""
        h, w = frame['depth'].shape[:2]
        with self._lock:
            mask = self.objects.segment(camera, frame['rgb'], frame['depth'])
            if mask is None and self.robot_renderer is None:
                return None
            mask = np.zeros((h, w), dtype=np.int32) if mask is None else np.asarray(mask, dtype=np.int32)
            if self.robot_renderer is not None:
                self.robot_renderer.set_state(q, finger_width)
                robot = self.robot_renderer.render(frame['extrinsics'], frame['intrinsics'], w, h)
                mask[robot] = ROBOT_ID
        return mask

    def reset(self) -> None:
        self.objects.reset()
        self.object_labels = {}

    def close(self) -> None:
        if self.robot_renderer is not None:
            self.robot_renderer.close()
