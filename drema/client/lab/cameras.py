#!/usr/bin/env python
"""
Laboratory RGB-D cameras: any number, each behind its own bridge (lab_bridges.zed_camera or lab_bridges.ros2_camera)
and with its own calibration, in the OpenCV optical convention (x right, y down, z forward, positive focal lengths):
the server back-projects with K as given, so no axis flip is needed.

Mounts:
- fixed: the calibration is T_base_camera (Drema2's configs/extraction/calibrations format, camera -> robot base);
- wrist: the calibration is T_link_camera, composed every frame with the pose of `link` at the measured joints
  ('ee': the hand pose the robot reports, 'flange': link 8 of the kinematic model).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lab_bridges.ros2_camera import Ros2CameraClient
from lab_bridges.zed_camera import ZedCameraClient


def load_calibration(path: str) -> np.ndarray:
    """4x4 row-major camera pose, '#' comments allowed; the rotation must be orthonormal."""
    T = np.loadtxt(path, comments="#", dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"Calibration {path}: expected a 4x4 matrix, got {T.shape}")
    R = T[:3, :3]
    if np.abs(R @ R.T - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1.0) > 1e-3:
        raise ValueError(f"Calibration {path}: the rotation is not orthonormal")
    return T


def resize_frame(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray,
                 resolution: Optional[Sequence[Optional[int]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resizes to [width, height] (height None: aspect kept); depth by nearest neighbour, K with pixel centres kept."""
    import cv2
    h0, w0 = depth.shape[:2]
    if not resolution:
        return rgb, depth, K
    w = int(resolution[0])
    h = int(resolution[1]) if len(resolution) > 1 and resolution[1] else int(round(h0 * w / w0))
    if (w, h) == (w0, h0):
        return rgb, depth, K
    sx, sy = w / w0, h / h0
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if sx < 1.0 else cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    K = np.array(K, dtype=np.float64)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return rgb, depth, K


class LabCamera:

    SOURCES = ("zed", "ros2")
    MOUNTS = ("fixed", "wrist")

    def __init__(self, cfg: Dict[str, Any]):
        self.name = str(cfg["name"])
        self.source = str(cfg.get("source", "zed"))
        if self.source not in self.SOURCES:
            raise ValueError(f"Camera {self.name}: source '{self.source}' not in {self.SOURCES}")
        self.mount = str(cfg.get("mount", "fixed"))
        if self.mount not in self.MOUNTS:
            raise ValueError(f"Camera {self.name}: mount '{self.mount}' not in {self.MOUNTS}")
        self.link = str(cfg.get("link", "ee"))
        self.calibration_path = str(cfg["calibration"])
        self.T_calibration = load_calibration(self.calibration_path)
        self.stream_resolution = cfg.get("stream_resolution")
        self.scan_resolution = cfg.get("scan_resolution")
        near, far = cfg.get("depth_range", [0.1, 3.0])
        self.near, self.far = float(near), float(far)
        host, port = str(cfg.get("host", "127.0.0.1")), int(cfg["port"])
        timeout = float(cfg.get("timeout_s", 5.0))
        # Learnt from the first full-size frame: a ZED then resizes in the SDK even with a null height
        self.native_size: Optional[Tuple[int, int]] = None
        self.client = ZedCameraClient(host, port, timeout) if self.source == "zed" else Ros2CameraClient(host, port, timeout)

    def describe(self) -> str:
        where = "fixed" if self.mount == "fixed" else f"wrist ({self.link})"
        return (f"{self.name}: {self.source} bridge {self.client.host}:{self.client.port}, {where}, "
                f"calibration {self.calibration_path}, stream {self.stream_resolution}, scan {self.scan_resolution}")

    def pose(self, T_world_base: np.ndarray, link_poses: Dict[str, np.ndarray]) -> np.ndarray:
        if self.mount == "fixed":
            return T_world_base @ self.T_calibration
        if self.link not in link_poses:
            raise RuntimeError(f"Camera {self.name}: no pose of link '{self.link}'")
        return link_poses[self.link] @ self.T_calibration

    def grab(self, resolution: Optional[Sequence[Optional[int]]]) -> Optional[Dict[str, Any]]:
        """{'rgb', 'depth' [m, 0 = invalid], 'intrinsics', 'near_clipping', 'far_clipping', 'stamp'} or None."""
        w = int(resolution[0]) if resolution else None
        h = int(resolution[1]) if resolution and len(resolution) > 1 and resolution[1] else None
        if w and not h and self.native_size is not None:
            h = int(round(self.native_size[1] * w / self.native_size[0]))
        frame = self.client.grab(w, h) if self.source == "zed" else self.client.grab()
        if frame is None:
            return None
        rgb = np.ascontiguousarray(frame["rgb"][..., :3]).astype(np.uint8)
        depth = np.asarray(frame["depth"], dtype=np.float32)
        if self.native_size is None and not (self.source == "zed" and w and h):
            self.native_size = (depth.shape[1], depth.shape[0])
        rgb, depth, K = resize_frame(rgb, depth, frame["intrinsics"], resolution)
        depth = np.where((depth > self.near) & (depth < self.far), depth, 0.0).astype(np.float32)
        return {'rgb': rgb, 'depth': depth, 'intrinsics': K.astype(np.float32),
                'near_clipping': self.near, 'far_clipping': self.far, 'stamp': frame.get("stamp")}

    def close(self) -> None:
        self.client.close()


def build_cameras(configs: List[Dict[str, Any]]) -> List[LabCamera]:
    cameras = [LabCamera(c) for c in configs]
    names = [c.name for c in cameras]
    if len(set(names)) != len(names):
        raise ValueError(f"Camera names must be unique: {names}")
    return cameras
