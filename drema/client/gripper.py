#!/usr/bin/env python
"""
Non-blocking RLBench gripper, one simulation step at a time, with the semantics of the RLBench Discrete gripper
action mode used by PerAct (attach graspable objects before closing, detach before opening, let released objects
drop for a few steps). The arm stays under MPC control while the fingers move.
"""

from typing import Any


class RLBenchGripperController:

    def __init__(self, robot: Any, task: Any, velocity: float = 0.2, settle_steps: int = 10,
                 attach_grasped_objects: bool = True):
        self.gripper = robot.gripper
        self.task = task
        self.velocity = float(velocity)
        self.settle_steps = int(settle_steps)
        self.attach_grasped_objects = bool(attach_grasped_objects)
        self.busy = False
        self._settle_left = 0
        self.target = 1.0 if self._measured_open() else 0.0

    def _measured_open(self) -> bool:
        return all(x > 0.9 for x in self.gripper.get_open_amount())

    def is_open(self) -> bool:
        """Last commanded state: fingers closed on a wide object (e.g. a jar lid) stay above the 0.9 threshold."""
        return self.target > 0.5

    @property
    def done(self) -> bool:
        return not self.busy

    def command(self, open_amount: float) -> None:
        target = 1.0 if open_amount > 0.5 else 0.0
        if target == self.target and not self.busy:
            return
        self.target = target
        if target == 0.0 and self.attach_grasped_objects:
            for obj in self.task.get_graspable_objects():
                self.gripper.grasp(obj)
        else:
            self.gripper.release()
        self.busy = True
        self._settle_left = self.settle_steps if target == 1.0 else 0

    def step(self) -> None:
        """Call once per simulation step, before pyrep.step()."""
        if not self.busy:
            return
        if self.gripper.actuate(self.target, velocity=self.velocity):
            if self._settle_left > 0:
                self._settle_left -= 1
            else:
                self.busy = False

    def reset(self) -> None:
        self.busy = False
        self._settle_left = 0
        self.target = 1.0 if self._measured_open() else 0.0
