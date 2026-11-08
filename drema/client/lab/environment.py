#!/usr/bin/env python
"""
Laboratory environment: the real FR3 with its Franka Hand and the calibrated RGB-D cameras of configs/environments/
lab.yaml, behind the lab_bridges processes on the robot PC.

World frame = robot base frame translated by world.robot_base_position (the server models the base as a translation
only). The initial scan is what the cameras see now, with the arm where it is: `frames_per_camera` frames of every
camera; how many views that gives is up to the setup. The clock is the wall clock: one loop iteration lasts 1 /
control_hz, paced by the runner (realtime only).

Streaming frames are captured off the control loop, one thread per camera at the streaming rate: a camera that
blocks for its next exposure must not delay the arm commands. Each frame is posed and labelled with the joints read
right after its grab and carries its capture time, which becomes the frame timestamp sent to the server.
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..environment import Environment, RobotReading, SceneScan
from ..keyframes import Keyframe, KeyframeSource, PolicyKeyframeSource, StaticKeyframeSource
from .cameras import LabCamera, build_cameras
from .robot import LabArmActuator, LabGripper, LabRobot
from .segmentation import LabSegmentation, RobotMaskRenderer

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def repo_path(path: Optional[str]) -> Optional[str]:
    """A relative path from the working directory, else from the repository root."""
    if not path or os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(REPO, path)


@dataclass
class LabObservation:
    """What a keyframe policy receives: camera frames (stream resolution, world-frame poses) and robot state."""
    cameras: Dict[str, Dict[str, Any]]
    robot: Optional[RobotReading]
    gripper_open: float
    task_description: str


def load_keyframe_file(path: str) -> List[Keyframe]:
    """YAML list of keyframes (Keyframe.from_dict fields) under 'keyframes'; poses in the world frame."""
    with open(repo_path(path)) as f:
        data = yaml.safe_load(f) or {}
    entries = data.get("keyframes", data) if isinstance(data, dict) else data
    frames = [Keyframe.from_dict(d, i) for i, d in enumerate(entries or [])]
    for kf in frames:
        if len(kf.pose) != 7:
            raise ValueError(f"Keyframe '{kf.name}' in {path}: pose must be [x, y, z, qx, qy, qz, qw]")
    return frames


class CameraWorker(threading.Thread):
    """Grabs one camera at the streaming rate; `take` returns a frame once, `peek` the latest without consuming it."""

    def __init__(self, env: "LabEnvironment", camera: LabCamera, period: float):
        super().__init__(name=f"camera-{camera.name}", daemon=True)
        self.env = env
        self.camera = camera
        self.period = float(period)
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._fresh = False
        self._last_warning = 0.0

    def run(self) -> None:
        while not self.stopped.is_set():
            start = time.monotonic()
            if not self.paused.is_set():
                try:
                    frame = self.env.grab_frame(self.camera, "stream_resolution")
                    if frame is not None:
                        with self._lock:
                            self._latest, self._fresh = frame, True
                except Exception as e:
                    if start - self._last_warning > 2.0:
                        self._last_warning = start
                        print(f"[{self.env.client_tag}] [Warning] Camera {self.camera.name}: {e}")
            self.stopped.wait(max(0.005, self.period - (time.monotonic() - start)))

    def take(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self._fresh:
                return None
            self._fresh = False
            return self._latest

    def peek(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest


class LabEnvironment(Environment):

    client_tag = "LabClient"
    log_tag = "LAB ENVIRONMENT"
    sync_modes = ("realtime",)
    keyframe_modes = ("auto", "file", "policy", "none")

    def __init__(self, cfg: Dict[str, Any], actuate: bool = False):
        self.cfg = cfg
        self.actuate = bool(actuate)
        self.task_name = str(cfg.get("task_name", "lab"))
        self.task_description = str(cfg.get("task_description", ""))
        self.dt = 1.0 / float(cfg.get("control_hz", 20.0))
        self.reachability_radius = float(cfg.get("reachability_radius", 0.855))
        world = cfg.get("world", {}) or {}
        self.base_position = [float(x) for x in world.get("robot_base_position", [0.0, 0.0, 0.0])]
        self.T_world_base = np.eye(4)
        self.T_world_base[:3, 3] = self.base_position
        self.scan_cfg = cfg.get("scan", {}) or {}
        self.keyframes_cfg = cfg.get("keyframes", {}) or {}
        self.semantic_labels: Dict[str, int] = {}
        self.label_categories: Dict[int, str] = {}
        self._t0 = time.monotonic()
        self.cameras: List[LabCamera] = []
        self.workers: List[CameraWorker] = []
        self.robot: Optional[LabRobot] = None
        self.segmentation: Optional[LabSegmentation] = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def launch(self) -> None:
        robot_cfg = self.cfg.get("robot", {}) or {}
        with open(repo_path(robot_cfg.get("profile", "configs/robots/fr3_real.yaml"))) as f:
            profile = yaml.safe_load(f) or {}
        self.robot = LabRobot(robot_cfg, self.base_position, profile)
        cameras_cfg = [{**c, "calibration": repo_path(c.get("calibration"))} for c in self.cfg.get("cameras", []) or []]
        self.cameras = build_cameras(cameras_cfg)
        if not self.cameras:
            raise ValueError("No cameras in the lab configuration")

        mask_cfg = (self.cfg.get("segmentation", {}) or {}).get("robot_mask", {}) or {}
        renderer = None
        if bool(mask_cfg.get("enabled", True)):
            renderer = RobotMaskRenderer(repo_path(mask_cfg.get("urdf", "assets/franka_fr3/fr3_mask.urdf")),
                                         self.robot.joint_names, self.robot.finger_joint_names,
                                         self.base_position, int(mask_cfg.get("dilation_px", 2)))
        self.segmentation = LabSegmentation(renderer)
        self._update_labels()

        self.gripper = LabGripper(self.robot, robot_cfg.get("gripper", {}) or {}, self.actuate)
        self.actuator = LabArmActuator(self.robot, robot_cfg.get("actuation", {}) or {}, self.dt, self.actuate)
        print(f"[{self.client_tag}] Lab environment '{self.task_name}': {len(self.cameras)} camera(s), "
              f"robot mask {'on' if renderer else 'off'}, object segmentation: none")

    def ready(self) -> None:
        reading = self.robot.read()
        self.gripper.reset()
        self.actuator.reset()
        print(f"[{self.client_tag}] Robot state OK: q = {np.round(reading.joint_positions, 3).tolist()}, "
              f"gripper {reading.gripper_open:.2f}")
        if self.actuate:
            print(f"[{self.client_tag}] ACTUATION ENABLED: '{self.actuator.mode}' commands go to the robot "
                  f"(command bridge {self.robot.commands.host}:{self.robot.commands.port})")
        else:
            print(f"[{self.client_tag}] Actuation disabled (shadow mode): commands are computed, nothing is sent "
                  f"to the robot. Run with --actuate to move it.")

    def set_stream_rate(self, hz: float) -> None:
        for worker in self.workers:
            worker.stopped.set()
        self.workers = [CameraWorker(self, cam, 1.0 / float(hz)) for cam in self.cameras]
        for worker in self.workers:
            worker.start()

    def reset(self) -> None:
        self._t0 = time.monotonic()
        self.segmentation.reset()
        self._update_labels()
        try:
            self.robot.read()
        except Exception as e:
            print(f"[{self.client_tag}] [Warning] Robot state unavailable at reset: {e}")

    def shutdown(self) -> None:
        for worker in self.workers:
            worker.stopped.set()
        for worker in self.workers:
            worker.join(timeout=2.0)
        if self.actuator is not None:
            try:
                self.actuator.hold()
            except Exception:
                pass
        for cam in self.cameras:
            cam.close()
        if self.segmentation is not None:
            self.segmentation.close()
        if self.robot is not None:
            self.robot.close()

    def describe(self) -> None:
        print("\n" + "=" * 60)
        print("  [LAB SETUP]")
        print(f"  • World frame: robot base at {self.base_position}")
        for cam in self.cameras:
            print(f"  • Camera {cam.describe()}")
        print(f"  • Hand pose: {self.robot.ee_source}; arm actuation: {self.actuator.mode} "
              f"({'ENABLED' if self.actuate else 'shadow'})")
        print("=" * 60 + "\n")

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.robot.q_min, self.robot.q_max

    def timestamp(self) -> float:
        return self._clock(time.monotonic())

    def _clock(self, monotonic: float) -> float:
        return monotonic - self._t0 + self.dt

    def _update_labels(self) -> None:
        self.semantic_labels = self.segmentation.semantic_labels
        self.label_categories = self.segmentation.label_categories

    # -------------------------------------------------------------------------
    # Sensing
    # -------------------------------------------------------------------------

    def grab_frame(self, camera: LabCamera, resolution_key: str, label: bool = True) -> Optional[Dict[str, Any]]:
        """One frame, posed (and labelled) at the joints read right after the grab, with its capture time."""
        frame = camera.grab(getattr(camera, resolution_key))
        if frame is None:
            return None
        captured = time.monotonic()
        reading, width = self.robot.sample()
        frame['extrinsics'] = camera.pose(self.T_world_base, self.robot.link_poses(reading)).astype(np.float32)
        frame['joints'], frame['finger_width'], frame['captured'] = list(reading.joint_positions), width, captured
        if label:
            self._label(camera.name, frame)
        return frame

    def _label(self, camera: str, frame: Dict[str, Any]) -> None:
        frame['mask'] = self.segmentation.label(camera, frame, frame['joints'], frame['finger_width'])

    def capture_cameras(self) -> Dict[str, Dict[str, Any]]:
        """The frames captured since the last call (streaming), or one synchronous grab before the workers start."""
        if not self.workers:
            frames = {}
            for cam in self.cameras:
                try:
                    frame = self.grab_frame(cam, "stream_resolution")
                    if frame is not None:
                        frames[cam.name] = frame
                except Exception as e:
                    print(f"[{self.client_tag}] [Warning] Camera {cam.name}: {e}")
            return frames
        return {w.camera.name: f for w in self.workers for f in [w.take()] if f is not None}

    def frames_timestamp(self, frames: Dict[str, Dict[str, Any]]) -> float:
        times = [f['captured'] for f in frames.values() if 'captured' in f]
        return self._clock(float(np.mean(times))) if times else self.timestamp()

    def capture_scan(self, num_steps: int) -> SceneScan:
        n = max(1, int(self.scan_cfg.get("frames_per_camera", 1)))
        interval = float(self.scan_cfg.get("interval_s", 0.2))
        for worker in self.workers:
            worker.paused.set()
        try:
            reading = self.robot.read()
            captures = []
            for k in range(n):
                if k:
                    time.sleep(interval)
                frames = {}
                for cam in self.cameras:
                    try:
                        frame = self.grab_frame(cam, "scan_resolution", label=False)
                        if frame is not None:
                            frames[cam.name] = frame
                    except Exception as e:
                        print(f"[{self.client_tag}] [Warning] Camera {cam.name}: {e}")
                captures.append(frames)
        finally:
            for worker in self.workers:
                worker.paused.clear()
        first = {}
        for frames in captures:
            for name, frame in frames.items():
                first.setdefault(name, frame)
        if not first:
            raise RuntimeError("no camera delivered a frame")
        # The object segmenter picks its objects on the raw scan frames before any frame is labelled
        self.segmentation.start(first)
        self._update_labels()
        views = []
        for k, frames in enumerate(captures):
            for name, f in frames.items():
                self._label(name, f)
                views.append((f"{name}_scan{k}", f['rgb'], f['depth'], f['extrinsics'], f['intrinsics'],
                              f['near_clipping'], f['far_clipping'], f['mask']))
        print(f"[{self.client_tag}] Scan: {n} frame(s) x {len(first)} camera(s) = {len(views)} views")
        return SceneScan(views=views, joint_positions=list(reading.joint_positions))

    def robot_base_position(self) -> List[float]:
        return list(self.base_position)

    def read_robot(self) -> RobotReading:
        return self.robot.read()

    # -------------------------------------------------------------------------
    # Task
    # -------------------------------------------------------------------------

    def keyframe_source(self, mode: str, policy_spec: Optional[str]) -> Optional[KeyframeSource]:
        path = self.keyframes_cfg.get("file")
        if mode == "auto":
            mode = "policy" if policy_spec else ("file" if path else "none")
        if mode == "file":
            if not path:
                raise ValueError("keyframes 'file' needs keyframes.file in the lab configuration")
            frames = load_keyframe_file(path)
            print(f"[{self.client_tag}] Keyframes from {path}: " + ", ".join(k.name for k in frames))
            return StaticKeyframeSource(frames)
        if mode == "policy":
            return PolicyKeyframeSource.load(policy_spec, self)
        return None

    def observe(self) -> LabObservation:
        """Latest frame of every camera (not consumed: streaming still sends it) and the last robot reading."""
        if self.workers:
            cameras = {w.camera.name: f for w in self.workers for f in [w.peek()] if f is not None}
        else:
            cameras = self.capture_cameras()
        reading = self.robot.last
        return LabObservation(cameras=cameras, robot=reading,
                              gripper_open=reading.gripper_open if reading is not None else 1.0,
                              task_description=self.task_description)

    def advance(self, step: int, task_active: bool) -> None:
        self.gripper.step()
