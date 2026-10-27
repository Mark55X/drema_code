#!/usr/bin/env python
"""
Execution of the MPC joint velocity commands on the simulated arm.

CoppeliaSim joint motors in velocity mode lose a roughly constant ~0.02 rad/s on the gravity-loaded joints
(shoulder, elbow): the small commands the MPC issues near a keyframe are not executed at all, and the arm settles
1-3 cm below the target. In position mode the command is integrated into a joint reference tracked by the simulator
position controller, which holds the arm against gravity like the internal controller of a real Franka does.
"""

from typing import Any, Optional, Sequence

import numpy as np

from ..controller.franka_kinematics import FrankaKinematics


class JointCommandExecutor:

    MODES = ("position", "velocity")

    def __init__(self, arm: Any, mode: str = "position", dt: float = 0.05, max_lead: float = 0.06,
                 q_min: Optional[Sequence[float]] = None, q_max: Optional[Sequence[float]] = None):
        """
        :param mode: "position" (integrated reference, default) or "velocity" (raw joint velocity targets).
        :param dt: Simulation time [s] advanced between two commands.
        :param max_lead: Max distance [rad] of the reference ahead of the measured joints. Measured lag at the 1 rad/s
            bound is 0.02-0.04 rad, so normal motion is unaffected; against an obstacle it bounds the pushing force.
        """
        if mode not in self.MODES:
            raise ValueError(f"Unknown actuation mode '{mode}' (expected one of {self.MODES})")
        self.arm = arm
        self.mode = mode
        self.dt = float(dt)
        self.max_lead = float(max_lead)
        self.q_min = np.asarray(q_min if q_min is not None else FrankaKinematics.Q_MIN, dtype=np.float64)
        self.q_max = np.asarray(q_max if q_max is not None else FrankaKinematics.Q_MAX, dtype=np.float64)
        self.q_ref: Optional[np.ndarray] = None
        self._holding = True

    def reset(self) -> None:
        """Call after every scene reset or demo recording: they change the arm control loop and configuration."""
        if self.mode == "position":
            self.arm.set_control_loop_enabled(True)
        else:
            self.arm.set_control_loop_enabled(False)
            self.arm.set_motor_locked_at_zero_velocity(True)
        self.q_ref = np.asarray(self.arm.get_joint_positions(), dtype=np.float64)
        self._holding = True

    def apply(self, joint_velocities: Sequence[float]) -> None:
        if self.mode == "velocity":
            self.arm.set_joint_target_velocities(list(joint_velocities))
            return
        q = np.asarray(self.arm.get_joint_positions(), dtype=np.float64)
        if self.q_ref is None:
            self.q_ref = q.copy()
        ref = self.q_ref + np.asarray(joint_velocities, dtype=np.float64) * self.dt
        self.q_ref = np.clip(np.clip(ref, q - self.max_lead, q + self.max_lead), self.q_min, self.q_max)
        self._holding = False
        self.arm.set_joint_target_positions(self.q_ref.tolist())

    def hold(self) -> None:
        """Keeps the arm still (idle, pause, safety stop); the reference is anchored once to the measured joints."""
        if self.mode == "velocity":
            self.arm.set_joint_target_velocities([0.0] * len(self.arm.joints))
            return
        if not self._holding or self.q_ref is None:
            self.q_ref = np.asarray(self.arm.get_joint_positions(), dtype=np.float64)
            self._holding = True
        self.arm.set_joint_target_positions(self.q_ref.tolist())
