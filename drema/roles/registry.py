#!/usr/bin/env python
"""
Server-side resolution of object roles into the collision policies enforced by the digital twin.

Arbitration of the environment permissions (per object):
- HUMAN wins over everything and is the only source allowed to relax (TOUCH);
- CONFIG and VLM may only restrict (AVOID, FRAGILE): a TOUCH from them is dropped;
- otherwise the most restrictive of CONFIG / VLM applies; with no assignment the object is a default obstacle.

On top of the permissions, two server-side mechanisms:
- policy contact: local (hand and fingers only, objects within `policy_contact_radius` of the keypose) and
  temporary (the active keyframe only). On a grasp keyframe (gripper commanded closed) the hand may enclose those
  objects; on an approach keyframe (ignore_collisions with the gripper open, e.g. a pre-grasp) they stay obstacles
  for the hand with a margin reduced by `near_contact_margin`, so it may come close but not run into them. Objects
  with an explicit AVOID / FRAGILE are never touched this way;
- held state: when the gripper closes on a touchable object it is attached to the TCP in the twin, and released
  when the gripper opens again. A released object stays touchable by the hand until the TCP is
  `release_clearance` away from it (the open fingers still surround it right after the release).
"""

import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .types import FULL_CONTACT, ObjectCollisionPolicy, Permission, RoleAssignment, Source, relaxes

TABLE_ID = -1


class ObjectRoleRegistry:

    # Change of the measured open amount that counts as finger motion
    GRIPPER_MOTION_EPS = 0.02

    def __init__(
        self,
        permissions_enabled: bool = True,
        touch_exempts_arm: bool = False,
        fragile_extra_margin: float = 0.03,
        policy_contact_enabled: bool = True,
        policy_contact_on_grasp: bool = True,
        policy_contact_on_approach: bool = True,
        policy_contact_radius: float = 0.05,
        policy_contact_table: bool = True,
        near_contact_margin: float = 0.03,
        grasp_enabled: bool = True,
        grasp_radius: float = 0.04,
        release_clearance: float = 0.03,
        keep_released_touchable: bool = True,
        gripper_closed_threshold: float = 0.9,
        gripper_open_threshold: float = 0.9,
        gripper_settle_steps: int = 3,
        verbose: bool = True
    ):
        """
        :param permissions_enabled: Apply the permissions sent by the environment (config rules, VLM, human).
        :param touch_exempts_arm: A human TOUCH also exempts the arm links, not only hand and fingers.
        :param fragile_extra_margin: Extra clearance [m] added to the margins of FRAGILE objects.
        :param policy_contact_on_grasp: A keyframe with the gripper commanded closed counts as contact-allowed
            (the fingers must surround the object), even without ignore_collisions.
        :param policy_contact_on_approach: ignore_collisions with the gripper open lets the hand come close
            (margin reduced by near_contact_margin).
        :param policy_contact_radius: Objects whose surface is within this distance [m] of the keypose TCP.
        :param policy_contact_table: The table also becomes touchable by the hand when it lies at the keypose.
        :param near_contact_margin: Reduction [m] of the hand margins on approach keyframes.
        :param grasp_radius: An object closer than this [m] to the TCP when the gripper closes becomes held.
        :param release_clearance: TCP distance [m] from a released object beyond which it is an obstacle again.
        :param keep_released_touchable: Enables the released state; when False a released object is an obstacle at once.
        :param gripper_closed_threshold / gripper_open_threshold: Measured open amount below / above which the
            gripper counts as closed / open.
        :param gripper_settle_steps: Consecutive requests with the gripper closed before attaching.
        """
        self.permissions_enabled = bool(permissions_enabled)
        self.touch_exempts_arm = bool(touch_exempts_arm)
        self.fragile_extra_margin = float(fragile_extra_margin)
        self.policy_contact_enabled = bool(policy_contact_enabled)
        self.policy_contact_on_grasp = bool(policy_contact_on_grasp)
        self.policy_contact_on_approach = bool(policy_contact_on_approach)
        self.policy_contact_radius = float(policy_contact_radius)
        self.policy_contact_table = bool(policy_contact_table)
        self.near_contact_margin = float(near_contact_margin)
        self.grasp_enabled = bool(grasp_enabled)
        self.grasp_radius = float(grasp_radius)
        self.release_clearance = float(release_clearance)
        self.keep_released_touchable = bool(keep_released_touchable)
        self.gripper_closed_threshold = float(gripper_closed_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.gripper_settle_steps = max(1, int(gripper_settle_steps))
        self.verbose = bool(verbose)

        self._lock = threading.Lock()
        self.roles_revision = -1
        self.assignments: Dict[int, Dict[Source, RoleAssignment]] = {}
        self.policy_contact: Set[int] = set()
        self.policy_contact_grasp = False
        self.held: Dict[int, np.ndarray] = {}
        self.released: Set[int] = set()
        self._closed_count = 0
        self._last_keyframe_index: Optional[int] = None
        self._rejected_logged: Set[Tuple[int, int]] = set()
        self._last_measured: Optional[float] = None
        self._close_ref: Optional[float] = None
        self._held_open_amount: Optional[float] = None
        self.status: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, roles_cfg: Dict[str, Any]) -> "ObjectRoleRegistry":
        c = roles_cfg or {}
        pc = c.get("policy_contact", {}) or {}
        gr = c.get("grasp", {}) or {}
        return cls(
            permissions_enabled=bool((c.get("permissions", {}) or {}).get("enabled", True)),
            touch_exempts_arm=bool(c.get("touch_exempts_arm", False)),
            fragile_extra_margin=float(c.get("fragile_extra_margin", 0.03)),
            policy_contact_enabled=bool(pc.get("enabled", True)),
            policy_contact_on_grasp=bool(pc.get("on_grasp_keyframes", True)),
            policy_contact_on_approach=bool(pc.get("on_approach_keyframes", True)),
            policy_contact_radius=float(pc.get("radius", 0.05)),
            policy_contact_table=bool(pc.get("include_table", True)),
            near_contact_margin=float(pc.get("near_margin", 0.03)),
            grasp_enabled=bool(gr.get("enabled", True)),
            grasp_radius=float(gr.get("radius", 0.04)),
            release_clearance=float(gr.get("release_clearance", 0.03)),
            keep_released_touchable=bool(gr.get("released_state", True)),
            gripper_closed_threshold=float(gr.get("closed_threshold", 0.9)),
            gripper_open_threshold=float(gr.get("open_threshold", 0.9)),
            gripper_settle_steps=int(gr.get("settle_steps", 3)),
            verbose=bool(c.get("verbose", True))
        )

    def reset(self, digital_twin: Any = None) -> None:
        with self._lock:
            if digital_twin is not None:
                for obj_id in list(self.held):
                    digital_twin.detach_object(obj_id)
            self.roles_revision = -1
            self.assignments.clear()
            self.policy_contact.clear()
            self.held.clear()
            self.released.clear()
            self._closed_count = 0
            self._last_keyframe_index = None
            self._last_measured = None
            self._close_ref = None
            self._held_open_amount = None
            self._rejected_logged.clear()
            self.status = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[ROLE REGISTRY] {msg}")

    def update(self, robot_state: Any, digital_twin: Any, kinematics: Any) -> None:
        """Applies the role snapshot and the keyframe flags of a RobotState, then pushes the policies to the twin."""
        if digital_twin is None or not hasattr(digital_twin, "set_object_policies"):
            return
        with self._lock:
            if kinematics.base_position is None and len(robot_state.robot_base_pos) >= 3:
                kinematics.base_position = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)
            if robot_state.roles_revision != self.roles_revision:
                self._apply_snapshot(robot_state.roles_revision, robot_state.role_assignments)

            object_ids = set(digital_twin.object_ids())
            q = np.asarray(robot_state.joint_positions, dtype=np.float64)
            tcp_T = self._tcp_transform(kinematics, q) if len(q) == 7 else None

            self._update_policy_contact(robot_state, digital_twin)
            if tcp_T is not None:
                self._update_held(robot_state, digital_twin, kinematics, tcp_T, object_ids)

            policies, status = {}, []
            for obj_id in sorted(object_ids):
                policy, role, decided_by = self._resolve(obj_id, digital_twin)
                policies[obj_id] = policy
                status.append({"object_id": obj_id, "role": role, "decided_by": decided_by,
                               "hand_contact_allowed": policy.hand_contact or policy.held})
            table_policy = self._policy_contact_policy() if TABLE_ID in self.policy_contact else ObjectCollisionPolicy()
            digital_twin.set_object_policies(policies, table_policy)
            self.status = status

    def status_protos(self, pb2) -> List[Any]:
        return [pb2.ObjectRoleStatus(object_id=s["object_id"], role=s["role"], decided_by=s["decided_by"],
                                     hand_contact_allowed=s["hand_contact_allowed"]) for s in self.status]

    def summary(self) -> str:
        special = [f"{s['object_id']}:{s['role']}" for s in self.status if s["role"] != "OBSTACLE"]
        table = (f" table:{'TARGET' if self.policy_contact_grasp else 'NEAR'}"
                 if TABLE_ID in self.policy_contact else "")
        return ("roles: " + ", ".join(special) + table) if (special or table) else "roles: all obstacles"

    def _apply_snapshot(self, revision: int, assignments_pb) -> None:
        self.roles_revision = int(revision)
        self.assignments = {}
        if not self.permissions_enabled:
            if len(assignments_pb):
                self._log(f"Role snapshot rev {revision} ignored: environment permissions disabled")
            return
        for msg in assignments_pb:
            a = RoleAssignment.from_proto(msg)
            if a.permission == Permission.UNSPECIFIED:
                continue
            if a.permission == Permission.TOUCH and a.source != Source.HUMAN:
                key = (a.object_id, int(a.source))
                if key not in self._rejected_logged:
                    self._rejected_logged.add(key)
                    self._log(f"TOUCH on object {a.object_id} from {a.source.name} rejected: only a human may relax")
                continue
            self.assignments.setdefault(a.object_id, {})[a.source] = a
        self._log(f"Role snapshot rev {revision}: " + (", ".join(
            f"{oid}<-{'/'.join(f'{s.name}:{a.permission.name}' for s, a in per.items())}"
            for oid, per in sorted(self.assignments.items())) or "no assignments"))

    def _permission(self, obj_id: int) -> Tuple[Permission, str, float]:
        """(effective permission, deciding source, extra margin); UNSPECIFIED = default obstacle."""
        per = self.assignments.get(obj_id, {})
        if Source.HUMAN in per:
            a = per[Source.HUMAN]
            return a.permission, "HUMAN", a.extra_margin
        best: Optional[RoleAssignment] = None
        for src in (Source.CONFIG, Source.VLM):
            a = per.get(src)
            if a is not None and (best is None or relaxes(best.permission, a.permission)):
                best = a
        if best is None:
            return Permission.UNSPECIFIED, "DEFAULT", 0.0
        return best.permission, best.source.name, best.extra_margin

    def _resolve(self, obj_id: int, digital_twin: Any) -> Tuple[ObjectCollisionPolicy, str, str]:
        if obj_id in self.held:
            return ObjectCollisionPolicy(hand_contact=True, arm_contact=True, held=True), "HELD", "GRIPPER"
        if obj_id in self.released:
            return ObjectCollisionPolicy(hand_contact=True), "RELEASED", "GRIPPER"
        if digital_twin.is_legacy_target(obj_id):
            return FULL_CONTACT, "TARGET", "PERCEPTION"
        permission, decided_by, margin = self._permission(obj_id)
        if permission == Permission.TOUCH:
            return (ObjectCollisionPolicy(hand_contact=True, arm_contact=self.touch_exempts_arm),
                    "TARGET", decided_by)
        if permission == Permission.FRAGILE:
            return (ObjectCollisionPolicy(extra_margin=margin if margin > 0 else self.fragile_extra_margin),
                    "FRAGILE", decided_by)
        if permission == Permission.UNSPECIFIED and obj_id in self.policy_contact:
            return self._policy_contact_policy(), ("TARGET" if self.policy_contact_grasp else "NEAR"), "POLICY"
        return ObjectCollisionPolicy(), "OBSTACLE", decided_by

    def _policy_contact_policy(self) -> ObjectCollisionPolicy:
        if self.policy_contact_grasp:
            return ObjectCollisionPolicy(hand_contact=True)
        return ObjectCollisionPolicy(hand_margin=-self.near_contact_margin)

    def _update_policy_contact(self, robot_state: Any, digital_twin: Any) -> None:
        keyframe_index = int(robot_state.keyframe_index)
        grasp = (self.policy_contact_on_grasp and robot_state.HasField("gripper_command")
                 and robot_state.gripper_command < 0.5)
        approach = (self.policy_contact_on_approach and robot_state.HasField("ignore_collisions")
                    and robot_state.ignore_collisions)
        allows = grasp or approach
        self.policy_contact_grasp = bool(grasp)
        if not (self.policy_contact_enabled and allows and robot_state.target_available
                and len(robot_state.target_pose) >= 3):
            if self.policy_contact:
                self._log(f"Policy contact cleared (keyframe {keyframe_index})")
            self.policy_contact = set()
            self._last_keyframe_index = keyframe_index
            return

        keypose = np.asarray(robot_state.target_pose[:3], dtype=np.float64)
        near = digital_twin.objects_near_point(keypose, self.policy_contact_radius, include_table=self.policy_contact_table)
        new_set = {oid for oid in near if oid not in self.held}
        # Objects stay touchable for the whole keyframe even when the hand pushes them out of the radius.
        if keyframe_index == self._last_keyframe_index:
            new_set |= self.policy_contact
        if new_set != self.policy_contact:
            self._log(f"Policy contact (keyframe {keyframe_index}, {'grasp' if grasp else 'approach'}): hand near "
                      f"{['table' if o == TABLE_ID else o for o in sorted(new_set)] or 'nothing'}")
        self.policy_contact = new_set
        self._last_keyframe_index = keyframe_index

    @staticmethod
    def _tcp_transform(kinematics: Any, q: np.ndarray) -> np.ndarray:
        pos, rot = kinematics.forward_kinematics_ee(q)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def _update_held(self, robot_state: Any, digital_twin: Any, kinematics: Any, tcp_T: np.ndarray,
                     object_ids: Set[int]) -> None:
        if not self.grasp_enabled or not robot_state.HasField("gripper_command"):
            return
        commanded_open = robot_state.gripper_command >= 0.5
        if self.released:
            near = digital_twin.objects_near_point(tcp_T[:3, 3], self.release_clearance)
            for obj_id in [o for o in self.released if o not in near or o not in object_ids]:
                self.released.discard(obj_id)
                self._log(f"Hand clear of released object {obj_id}: obstacle again")
        measured_open = float(robot_state.gripper_open)
        last_measured = self._last_measured if self._last_measured is not None else measured_open
        self._last_measured = measured_open

        # Thresholds alone fail on wide objects (a jar lid keeps the fingers above 0.9 open), so closing and
        # opening are recognised from the finger motion relative to where it started.
        if commanded_open:
            self._close_ref = None
            opened = (self._held_open_amount is not None
                      and measured_open > self._held_open_amount + self.GRIPPER_MOTION_EPS) \
                or measured_open >= max(self.gripper_open_threshold, 0.99)
            for obj_id in list(self.held):
                if obj_id not in object_ids or opened:
                    digital_twin.detach_object(obj_id)
                    del self.held[obj_id]
                    if obj_id in object_ids and self.keep_released_touchable:
                        self.released.add(obj_id)
                    self._log(f"Object {obj_id} released")
            if not self.held:
                self._held_open_amount = None
            self._closed_count = 0
            return

        for obj_id in [o for o in self.held if o not in object_ids]:
            digital_twin.detach_object(obj_id)
            del self.held[obj_id]
        if self._close_ref is None:
            self._close_ref = measured_open
        moved = measured_open < self._close_ref - self.GRIPPER_MOTION_EPS
        settled = abs(measured_open - last_measured) < 1e-3 or measured_open < self.gripper_closed_threshold
        self._closed_count = self._closed_count + 1 if (moved and settled) else 0
        if self.held or self._closed_count < self.gripper_settle_steps:
            return

        candidates = []
        for obj_id, d in digital_twin.objects_near_point(tcp_T[:3, 3], self.grasp_radius).items():
            if obj_id == TABLE_ID:
                continue
            policy, _, _ = self._resolve(obj_id, digital_twin)
            if policy.hand_contact:
                candidates.append((d, obj_id))
        if not candidates:
            return
        _, obj_id = min(candidates)
        pose = digital_twin.tracked_object_pose(obj_id)
        if pose is None:
            return
        obj_T = np.eye(4)
        obj_T[:3, :3] = Rotation.from_quat(pose[1]).as_matrix()
        obj_T[:3, 3] = pose[0]
        tcp_T_obj = np.linalg.inv(tcp_T) @ obj_T
        digital_twin.attach_object(obj_id, tcp_T_obj, kinematics)
        self.held[obj_id] = tcp_T_obj
        self._held_open_amount = measured_open
        self.policy_contact.discard(obj_id)
        self._log(f"Object {obj_id} grasped: attached to the TCP "
                  f"(offset {np.linalg.norm(tcp_T_obj[:3, 3]) * 100:.1f} cm)")
