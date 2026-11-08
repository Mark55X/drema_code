#!/usr/bin/env python
"""
The laboratory FR3 seen through the bridges: state from lab_bridges.ros2_joints, arm and gripper commands to
lab_bridges.ros2_command.

Without actuation (the default) nothing is sent to the robot: the environment streams, the server plans and the
commands are only computed, so perception and control can be watched on the real scene with the arm still.
"""

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from lab_bridges.ros2_command import FR3_Q_MAX, FR3_Q_MIN, CommandClient
from lab_bridges.ros2_joints import JointStateClient

from ...controller.franka_kinematics import FrankaKinematics
from ..environment import RobotReading, clip_joint_velocities


def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose[3:7]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


class LabRobot:

    def __init__(self, cfg: Dict[str, Any], base_position: Sequence[float], profile: Dict[str, Any]):
        self.joint_names = list(cfg.get("joint_names", [f"fr3_joint{i}" for i in range(1, 8)]))
        self.finger_joint_names = list(cfg.get("finger_joint_names", ["fr3_finger_joint1", "fr3_finger_joint2"]))
        self.max_width = float(cfg.get("gripper_max_width", 0.08))
        self.ee_source = str(cfg.get("ee_pose", "measured"))
        self.max_state_age = float(cfg.get("max_state_age_s", 0.25))
        self.base_position = np.asarray(base_position, dtype=np.float64)
        limits = profile.get("joint_limits") or {}
        self.q_min = np.asarray(limits.get("q_min", FR3_Q_MIN), dtype=np.float64)
        self.q_max = np.asarray(limits.get("q_max", FR3_Q_MAX), dtype=np.float64)
        tcp_rotation = profile.get("tcp_rotation_xyz_deg")
        self.R_tcp = (Rotation.from_euler('xyz', tcp_rotation, degrees=True).as_matrix()
                      if tcp_rotation is not None else np.eye(3))
        self.kinematics = FrankaKinematics(base_position=self.base_position.astype(np.float32),
                                           joint_angle_offsets=profile.get("joint_angle_offsets"),
                                           ee_offset=profile.get("tcp_offset"))
        self.flange_kinematics = FrankaKinematics(base_position=self.base_position.astype(np.float32),
                                                  joint_angle_offsets=profile.get("joint_angle_offsets"),
                                                  ee_offset=[0.0, 0.0, 0.0])
        bridge = cfg.get("joints_bridge", {}) or {}
        self.joints = JointStateClient(str(bridge.get("host", "127.0.0.1")), int(bridge.get("port", 5003)),
                                       float(bridge.get("timeout_s", 1.0)))
        cmd = cfg.get("command_bridge", {}) or {}
        self.commands = CommandClient(str(cmd.get("host", "127.0.0.1")), int(cmd.get("port", 5005)),
                                      float(cmd.get("timeout_s", 0.5)))
        self.last: Optional[RobotReading] = None
        self.finger_width: Optional[float] = None
        self._ee_warned = False

    def sample(self) -> Tuple[RobotReading, Optional[float]]:
        """(reading, finger width [m] or None) from the bridge; safe from any thread, stores nothing."""
        s = self.joints.get_robot_state()
        if s.get("age") is not None and float(s["age"]) > self.max_state_age:
            raise RuntimeError(f"joint state {float(s['age']) * 1000:.0f} ms old (max {self.max_state_age * 1000:.0f})")
        if self.ee_source == "measured" and s.get("ee_age") is not None and float(s["ee_age"]) > self.max_state_age:
            raise RuntimeError(f"hand pose {float(s['ee_age']) * 1000:.0f} ms old (max {self.max_state_age * 1000:.0f})")
        index = {n: i for i, n in enumerate(s["names"])}
        missing = [n for n in self.joint_names if n not in index]
        if missing:
            raise RuntimeError(f"joints {missing} not in the joint state ({s['names']})")
        q = [float(s["positions"][index[n]]) for n in self.joint_names]
        velocities = s.get("velocities") or []
        dq = [float(velocities[index[n]]) if index[n] < len(velocities) else 0.0 for n in self.joint_names]
        width = self.finger_width
        if all(n in index for n in self.finger_joint_names):
            width = float(sum(s["positions"][index[n]] for n in self.finger_joint_names))
        gripper_open = 1.0 if width is None else float(np.clip(width / self.max_width, 0.0, 1.0))
        return RobotReading(q, dq, self._ee_pose(q, s.get("ee_pose")), gripper_open), width

    def read(self) -> RobotReading:
        """The control loop's reading, kept as `last` for the actuator and the gripper."""
        self.last, self.finger_width = self.sample()
        return self.last

    def _ee_pose(self, q, measured) -> list:
        """Hand pose in the world frame: the robot's own (O_T_EE) when it publishes it, else the kinematic model."""
        if self.ee_source == "measured" and measured is not None and len(measured) >= 7:
            return [float(measured[i] + self.base_position[i]) for i in range(3)] + [float(x) for x in measured[3:7]]
        if self.ee_source == "measured" and not self._ee_warned:
            self._ee_warned = True
            print("[LabRobot] [Warning] No measured hand pose from the bridge: using the kinematic model (the server "
                  "self-check then only checks the profile against itself)")
        p, R = self.kinematics.forward_kinematics_ee(np.asarray(q, dtype=np.float64))
        quat = Rotation.from_matrix(np.asarray(R, dtype=np.float64) @ self.R_tcp).as_quat()
        return [float(x) for x in p] + [float(x) for x in quat]

    def link_poses(self, reading: RobotReading) -> Dict[str, np.ndarray]:
        """World poses of the links a wrist camera can be calibrated to."""
        p, R = self.flange_kinematics.forward_kinematics_ee(np.asarray(reading.joint_positions, dtype=np.float64))
        flange = np.eye(4)
        flange[:3, :3], flange[:3, 3] = R, p
        return {"ee": pose_matrix(reading.ee_pose), "flange": flange}

    def close(self) -> None:
        self.joints.close()
        self.commands.close()


class LabArmActuator:
    """
    'position' (bridge --interface position): the MPC velocities integrated into a joint reference that stays within
    max_lead of the measured joints, as drema.client.actuation in CoppeliaSim. 'velocity': the velocities themselves.
    Both scaled down as a whole to max_velocity per joint and kept off the joint limits.
    """

    MODES = ("position", "velocity")

    def __init__(self, robot: LabRobot, cfg: Dict[str, Any], dt: float, enabled: bool):
        self.robot = robot
        self.mode = str(cfg.get("interface", "position"))
        if self.mode not in self.MODES:
            raise ValueError(f"Actuation interface '{self.mode}' not in {self.MODES}")
        self.dt = float(dt)
        self.max_lead = float(cfg.get("max_lead", 0.06))
        self.v_max = np.asarray(cfg.get("max_velocity", [0.5] * 7), dtype=np.float64)
        margin = float(cfg.get("limit_margin", 0.05))
        self.q_lo, self.q_hi = robot.q_min + margin, robot.q_max - margin
        self.enabled = bool(enabled)
        self.q_ref: Optional[np.ndarray] = None
        self._holding = True
        self._last_error = 0.0
        self.last_command: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.q_ref = None
        self._holding = True
        self.hold()

    def apply(self, joint_velocities: Sequence[float]) -> None:
        v = clip_joint_velocities(joint_velocities, self.v_max)
        q = np.asarray(self.robot.last.joint_positions, dtype=np.float64)
        if self.mode == "velocity":
            towards_limit = ((q <= self.q_lo) & (v < 0)) | ((q >= self.q_hi) & (v > 0))
            v[towards_limit] = 0.0
            self.last_command = v
        else:
            if self.q_ref is None or self._holding:
                self.q_ref = q.copy()
            ref = np.clip(self.q_ref + v * self.dt, q - self.max_lead, q + self.max_lead)
            # Past the limit margin the reference may stay where it is, it only cannot go further out
            self.q_ref = np.clip(ref, np.minimum(self.q_lo, self.q_ref), np.maximum(self.q_hi, self.q_ref))
            self.last_command = self.q_ref.copy()
        self._holding = False
        self._send(lambda: self.robot.commands.command(self.last_command))

    def hold(self) -> None:
        if not self._holding:
            self._holding = True
            self.q_ref = None
        self._send(self.robot.commands.hold)

    def _send(self, call) -> None:
        if not self.enabled:
            return
        try:
            call()
        except Exception as e:
            # The bridge watchdog stops the arm when commands stop arriving
            if time.time() - self._last_error > 2.0:
                self._last_error = time.time()
                print(f"[LabArm] [Warning] Command bridge: {e}")


class LabGripper:
    """Open / close through the command bridge, with the interface of RLBenchGripperController."""

    def __init__(self, robot: LabRobot, cfg: Dict[str, Any], enabled: bool):
        self.robot = robot
        self.enabled = bool(enabled)
        self.open_threshold = float(cfg.get("open_width_threshold", 0.07))
        self.min_motion_s = float(cfg.get("min_motion_s", 0.3))
        self.timeout_s = float(cfg.get("timeout_s", 5.0))
        self.poll_s = float(cfg.get("poll_s", 0.1))
        self.busy = False
        self.target = 1.0
        self._started = 0.0
        self._last_poll = 0.0
        self.reset()

    def is_open(self) -> bool:
        return self.target > 0.5

    @property
    def done(self) -> bool:
        return not self.busy

    def command(self, open_amount: float) -> None:
        target = 1.0 if open_amount > 0.5 else 0.0
        if target == self.target and not self.busy:
            return
        self.target = target
        self.busy = True
        self._started = time.monotonic()
        if self.enabled:
            try:
                self.robot.commands.gripper(target > 0.5)
            except Exception as e:
                print(f"[LabGripper] [Warning] Gripper command failed: {e}")
                self.busy = False

    def step(self) -> None:
        if not self.busy:
            return
        now = time.monotonic()
        elapsed = now - self._started
        if not self.enabled:
            self.busy = elapsed < self.min_motion_s
            return
        if elapsed < self.min_motion_s or now - self._last_poll < self.poll_s:
            return
        self._last_poll = now
        try:
            state = self.robot.commands.status().get("gripper", {})
        except Exception:
            state = {}
        if not state.get("moving", True) or elapsed > self.timeout_s:
            self.busy = False
            if state.get("last_result") not in (None, "success"):
                print(f"[LabGripper] Gripper {'open' if self.is_open() else 'close'}: {state.get('last_result')}")

    def reset(self) -> None:
        self.busy = False
        width = self.robot.finger_width
        self.target = 1.0 if width is None or width > self.open_threshold else 0.0
