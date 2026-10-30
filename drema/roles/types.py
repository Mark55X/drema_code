#!/usr/bin/env python
"""
Object roles shared by the environment (who may ask what) and the server (what the twin and the MPC enforce).

Two independent axes per object id (the CoppeliaSim handle, same id used by perception):
- permission: semantic, decided in the environment by pluggable providers (config, VLM, human);
- interaction state: physical (free / held), decided by the server from the gripper and the twin.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional


class Permission(IntEnum):
    """Values match drema_comm.proto ObjectPermission."""
    UNSPECIFIED = 0
    AVOID = 1
    TOUCH = 2
    FRAGILE = 3

    @classmethod
    def parse(cls, value: Any) -> "Permission":
        if isinstance(value, Permission):
            return value
        if isinstance(value, int):
            return cls(value)
        key = str(value).strip().upper()
        aliases = {"OBSTACLE": "AVOID", "AVOIDED": "AVOID", "CONTACT": "TOUCH", "TARGET": "TOUCH",
                   "MANIPULABLE": "TOUCH", "NEVER_TOUCH": "FRAGILE"}
        return cls[aliases.get(key, key)]


class Source(IntEnum):
    """Values match drema_comm.proto RoleSource."""
    UNSPECIFIED = 0
    CONFIG = 1
    VLM = 2
    HUMAN = 3


# Higher = more restrictive.
RESTRICTIVENESS = {Permission.TOUCH: 0, Permission.AVOID: 1, Permission.FRAGILE: 2}


def relaxes(new: Permission, current: Permission) -> bool:
    """True when `new` allows more contact than `current`."""
    return RESTRICTIVENESS.get(new, 1) < RESTRICTIVENESS.get(current, 1)


@dataclass
class RoleAssignment:
    object_id: int
    permission: Permission
    source: Source
    reason: str = ""
    extra_margin: float = 0.0

    def to_proto(self, pb2) -> Any:
        return pb2.ObjectRoleAssignment(
            object_id=int(self.object_id), permission=int(self.permission), source=int(self.source),
            reason=self.reason, extra_margin=float(self.extra_margin)
        )

    @classmethod
    def from_proto(cls, msg) -> "RoleAssignment":
        return cls(int(msg.object_id), Permission(msg.permission), Source(msg.source), msg.reason,
                   float(msg.extra_margin))


@dataclass
class ObjectCollisionPolicy:
    """
    What the twin enforces for one object, in the collision costs, the evasive primitives and the safety halt.
    hand_contact / arm_contact exempt the robot hand (hand + fingers) / the other links from the pair;
    held objects move with the TCP and are checked against the other obstacles as part of the hand.
    """
    hand_contact: bool = False
    arm_contact: bool = False
    extra_margin: float = 0.0
    held: bool = False
    hand_margin: float = 0.0    # added to extra_margin for the hand only; negative = the hand may come closer
    held_contact: bool = False  # objects held by the gripper may touch this one (e.g. a lid placed on its jar)

    def pair_allowed(self, is_hand: bool) -> bool:
        """True when the (robot part, object) pair must be kept collision free."""
        if self.held:
            return False
        return not (self.hand_contact if is_hand else self.arm_contact)

    def pair_margin(self, is_hand: bool) -> Optional[float]:
        """Distance offset of the (robot part, object) pair in the collision costs; None when the pair is exempt."""
        if not self.pair_allowed(is_hand):
            return None
        return self.extra_margin + (self.hand_margin if is_hand else 0.0)

    def held_margin(self) -> Optional[float]:
        """Distance offset of the (held object, object) pair; None when the pair is exempt."""
        return None if self.held_contact else self.pair_margin(True)


FULL_CONTACT = ObjectCollisionPolicy(hand_contact=True, arm_contact=True)
