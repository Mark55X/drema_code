#!/usr/bin/env python
"""
Server-side resolution of object roles into the collision policies enforced by the digital twin.

Arbitration of the environment permissions (per object):
- HUMAN wins over everything and is the only source allowed to relax (TOUCH);
- CONFIG and VLM may only restrict (AVOID, FRAGILE): a TOUCH from them is dropped;
- otherwise the most restrictive of CONFIG / VLM applies; with no assignment the object is a default obstacle.

On top of the permissions, two server-side mechanisms:
- policy contact: local (hand and fingers only, objects in the contact zone: within a tolerance of the hand
  footprint placed at the keypose) and temporary (the active keyframe only), on keyframes with the gripper commanded
  closed or with ignore_collisions. On a grasp keyframe the objects between the fingers become TARGET (the hand may
  enclose them); the other objects of the zone stay obstacles for the hand with a margin reduced by
  `near_contact_margin` (NEAR), so it may come close but not run into them. On ignore_collisions keyframes with an
  object held, its supports are TARGET instead, so the hand holding it and the held object may touch them: the
  objects within the zone tolerance of it at the keypose that the keyframe moves it towards (the jar under its lid)
  or keeps it resting on (a lid screwed on its jar), the closest one and any other in contact (the floor and the
  wall of a shelf). The held object may touch the other NEAR objects only with `policy_contact_held`. Objects faster than `policy_contact_max_speed` never enter the zone and lose it
  for the rest of the keyframe when they start moving; objects with an explicit AVOID / FRAGILE never enter it.
  An object of the zone that the hand (or what it holds) still touches when the zone changes keeps its TARGET /
  NEAR policy until the hand is `policy_contact_zone_tolerance` away from it (the table under a cube just lifted),
  also when the new zone holds it with a lower policy (a neighbour grasped along with the cube, NEAR on the lift).
  Push contact: when an ignore_collisions keyframe starts with the hand empty, the first object the hand footprint
  runs into within `policy_push_reach` of its straight way to the keypose becomes TARGET for the keyframe (the block
  of slide_block, which lies where the push starts, not at the keypose); only that one, and only when at rest and
  without an explicit permission. It is not revoked while it moves with the hand;
- held state: when the gripper closes on a touchable object it is attached to the TCP in the twin, and released
  when the gripper opens again. A released object stays touchable by the hand until the TCP is
  `release_clearance` away from it (the open fingers still surround it right after the release).
"""

import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .types import FULL_CONTACT, ObjectCollisionPolicy, Permission, RoleAssignment, Source, relaxes

TABLE_ID = -1

# Franka hand at the TCP (union of the PyBullet and MuJoCo twin hands), in a frame with z along the approach and y
# along the finger opening: center, half extents. The finger and grip boxes reach the TCP, where the keyframe expects
# the grasped object.
PANDA_HAND_FOOTPRINT = {
    "finger_axis_yaw_deg": -45.0,
    "palm": {"center": [0.0, -0.004, -0.1075], "half_extents": [0.032, 0.104, 0.0685]},
    "fingers": {"center": [0.0, -0.002, -0.034], "half_extents": [0.011, 0.067, 0.037]},
    "grip": {"center": [0.0, -0.002, -0.034], "half_extents": [0.011, 0.038, 0.037]},
}


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
        policy_contact_zone_tolerance: float = 0.035,
        policy_contact_grip_tolerance: float = 0.01,
        policy_contact_table: bool = True,
        policy_contact_max_speed: float = 0.03,
        policy_contact_held: bool = False,
        policy_contact_hold: bool = True,
        policy_support_enabled: bool = True,
        policy_support_table: bool = True,
        policy_support_tolerance: float = 0.01,
        policy_push_enabled: bool = True,
        policy_push_reach: float = 0.035,
        policy_push_step: float = 0.005,
        policy_push_tolerance: float = 0.005,
        hand_footprint: Optional[Dict[str, Any]] = None,
        near_contact_margin: float = 0.03,
        grasp_enabled: bool = True,
        grasp_radius: float = 0.04,
        release_clearance: float = 0.03,
        keep_released_touchable: bool = True,
        gripper_closed_threshold: float = 0.9,
        gripper_open_threshold: float = 0.9,
        gripper_settle_steps: int = 3,
        gripper_empty_threshold: float = 0.05,
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
        :param policy_contact_zone_tolerance: Objects within this distance [m] of the palm or fingers placed at the
            keypose (or of a held object there) form the contact zone.
        :param policy_contact_grip_tolerance: On grasp keyframes, objects within this distance [m] of the space
            between the fingers at the keypose become TARGET.
        :param policy_contact_table: The table also enters the contact zone.
        :param policy_contact_max_speed: Objects faster than this [m/s] are never part of the contact zone.
        :param policy_contact_held: A held object may touch every NEAR object of the zone, not only its support.
        :param policy_support_enabled: On ignore_collisions keyframes with an object held, its supports at the
            keypose are TARGET for the keyframe.
        :param policy_support_table: The table may be a support.
        :param policy_support_tolerance: Contact distance [m] of a support, and how much closer to the held object
            it must get over the keyframe when not in contact at its start.
        :param policy_contact_hold: Objects still touched by the hand when the zone changes keep their zone policy
            until the hand is clear of them.
        :param policy_push_enabled: On ignore_collisions keyframes started with the hand empty, the first object the
            hand runs into on its way to the keypose may be pushed (TARGET for the keyframe).
        :param policy_push_reach: Only an object hit within this travel [m] from the start of the keyframe.
        :param policy_push_step: Sweep step [m] of the hand footprint along the way to the keypose.
        :param policy_push_tolerance: The footprint hits an object within this distance [m] of it, when also closer
            than at the start by as much (the motion goes towards the object, not past it).
        :param hand_footprint: Hand boxes in the TCP frame (see PANDA_HAND_FOOTPRINT).
        :param near_contact_margin: Reduction [m] of the hand margins on approach keyframes.
        :param grasp_radius: An object closer than this [m] to the TCP when the gripper closes becomes held.
        :param release_clearance: TCP distance [m] from a released object beyond which it is an obstacle again.
        :param keep_released_touchable: Enables the released state; when False a released object is an obstacle at once.
        :param gripper_closed_threshold / gripper_open_threshold: Measured open amount below / above which the
            gripper counts as closed / open.
        :param gripper_settle_steps: Consecutive requests with the gripper closed before attaching (clients that
            do not report the gripper actuator state).
        :param gripper_empty_threshold: Open amount below which fingers that stopped closing hold nothing.
        """
        self.permissions_enabled = bool(permissions_enabled)
        self.touch_exempts_arm = bool(touch_exempts_arm)
        self.fragile_extra_margin = float(fragile_extra_margin)
        self.policy_contact_enabled = bool(policy_contact_enabled)
        self.policy_contact_on_grasp = bool(policy_contact_on_grasp)
        self.policy_contact_on_approach = bool(policy_contact_on_approach)
        self.policy_contact_zone_tolerance = float(policy_contact_zone_tolerance)
        self.policy_contact_grip_tolerance = float(policy_contact_grip_tolerance)
        self.policy_contact_table = bool(policy_contact_table)
        self.policy_contact_max_speed = float(policy_contact_max_speed)
        self.policy_contact_held = bool(policy_contact_held)
        self.policy_contact_hold = bool(policy_contact_hold)
        self.policy_support_enabled = bool(policy_support_enabled)
        self.policy_support_table = bool(policy_support_table)
        self.policy_support_tolerance = float(policy_support_tolerance)
        self.policy_push_enabled = bool(policy_push_enabled)
        self.policy_push_reach = float(policy_push_reach)
        self.policy_push_step = max(1e-3, float(policy_push_step))
        self.policy_push_tolerance = float(policy_push_tolerance)
        fp = dict(PANDA_HAND_FOOTPRINT)
        fp.update(hand_footprint or {})
        self.hand_frame_R = Rotation.from_euler("z", float(fp["finger_axis_yaw_deg"]), degrees=True).as_matrix()
        self.hand_boxes = {k: (np.asarray(fp[k]["center"], dtype=np.float64),
                               np.asarray(fp[k]["half_extents"], dtype=np.float64))
                           for k in ("palm", "fingers", "grip")}
        self.near_contact_margin = float(near_contact_margin)
        self.grasp_enabled = bool(grasp_enabled)
        self.grasp_radius = float(grasp_radius)
        self.release_clearance = float(release_clearance)
        self.keep_released_touchable = bool(keep_released_touchable)
        self.gripper_closed_threshold = float(gripper_closed_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.gripper_settle_steps = max(1, int(gripper_settle_steps))
        self.gripper_empty_threshold = float(gripper_empty_threshold)
        self.verbose = bool(verbose)

        self._lock = threading.Lock()
        self.roles_revision = -1
        self.assignments: Dict[int, Dict[Source, RoleAssignment]] = {}
        self.policy_contact: Set[int] = set()
        self.policy_contact_target: Set[int] = set()
        self._policy_contact_revoked: Set[int] = set()
        self._contact_hold: Set[int] = set()
        self.push_targets: Set[int] = set()
        self._pushed: Set[int] = set()
        self._support_start: Optional[Dict[int, float]] = None
        self.held: Dict[int, np.ndarray] = {}
        self.released: Set[int] = set()
        self._closed_count = 0
        self._last_keyframe_index: Optional[int] = None
        self._rejected_logged: Set[Tuple[int, int]] = set()
        self._last_measured: Optional[float] = None
        self._close_ref: Optional[float] = None
        self._held_open_amount: Optional[float] = None
        self._grasp_candidates: Set[int] = set()
        self._closing = False
        self.status: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, roles_cfg: Dict[str, Any]) -> "ObjectRoleRegistry":
        c = roles_cfg or {}
        pc = c.get("policy_contact", {}) or {}
        gr = c.get("grasp", {}) or {}
        pp = pc.get("push_contact", {}) or {}
        sc = pc.get("support_contact", {}) or {}
        return cls(
            permissions_enabled=bool((c.get("permissions", {}) or {}).get("enabled", True)),
            touch_exempts_arm=bool(c.get("touch_exempts_arm", False)),
            fragile_extra_margin=float(c.get("fragile_extra_margin", 0.03)),
            policy_contact_enabled=bool(pc.get("enabled", True)),
            policy_contact_on_grasp=bool(pc.get("on_grasp_keyframes", True)),
            policy_contact_on_approach=bool(pc.get("on_approach_keyframes", True)),
            policy_contact_zone_tolerance=float(pc.get("zone_tolerance", 0.035)),
            policy_contact_grip_tolerance=float(pc.get("grip_tolerance", 0.01)),
            policy_contact_table=bool(pc.get("include_table", True)),
            policy_contact_max_speed=float(pc.get("max_speed", 0.03)),
            policy_contact_held=bool(pc.get("held_contact", False)),
            policy_support_enabled=bool(sc.get("enabled", True)),
            policy_support_table=bool(sc.get("include_table", True)),
            policy_support_tolerance=float(sc.get("tolerance", 0.01)),
            policy_contact_hold=bool(pc.get("hold_until_clear", True)),
            policy_push_enabled=bool(pp.get("enabled", True)),
            policy_push_reach=float(pp.get("reach", 0.035)),
            policy_push_step=float(pp.get("step", 0.005)),
            policy_push_tolerance=float(pp.get("tolerance", 0.005)),
            hand_footprint=pc.get("hand_footprint"),
            near_contact_margin=float(pc.get("near_margin", 0.03)),
            grasp_enabled=bool(gr.get("enabled", True)),
            grasp_radius=float(gr.get("radius", 0.04)),
            release_clearance=float(gr.get("release_clearance", 0.03)),
            keep_released_touchable=bool(gr.get("released_state", True)),
            gripper_closed_threshold=float(gr.get("closed_threshold", 0.9)),
            gripper_open_threshold=float(gr.get("open_threshold", 0.9)),
            gripper_settle_steps=int(gr.get("settle_steps", 3)),
            gripper_empty_threshold=float(gr.get("empty_threshold", 0.05)),
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
            self.policy_contact_target.clear()
            self._policy_contact_revoked.clear()
            self._contact_hold.clear()
            self.push_targets.clear()
            self._pushed.clear()
            self._support_start = None
            self.held.clear()
            self.released.clear()
            self._closed_count = 0
            self._last_keyframe_index = None
            self._last_measured = None
            self._close_ref = None
            self._held_open_amount = None
            self._grasp_candidates = set()
            self._closing = False
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

            v_tcp = self._tcp_velocity(kinematics, q, robot_state.joint_velocities, tcp_T)
            self._update_policy_contact(robot_state, digital_twin, tcp_T, v_tcp)
            if tcp_T is not None:
                self._update_held(robot_state, digital_twin, kinematics, tcp_T, object_ids)

            policies, status = {}, []
            for obj_id in sorted(object_ids):
                policy, role, decided_by = self._resolve(obj_id, digital_twin)
                policies[obj_id] = policy
                status.append({"object_id": obj_id, "role": role, "decided_by": decided_by,
                               "hand_contact_allowed": policy.hand_contact or policy.held})
            table_policy = self._zone_policy(TABLE_ID) if TABLE_ID in self.policy_contact else ObjectCollisionPolicy()
            digital_twin.set_object_policies(policies, table_policy)
            self.status = status

    def status_protos(self, pb2) -> List[Any]:
        return [pb2.ObjectRoleStatus(object_id=s["object_id"], role=s["role"], decided_by=s["decided_by"],
                                     hand_contact_allowed=s["hand_contact_allowed"]) for s in self.status]

    def summary(self) -> str:
        special = [f"{s['object_id']}:{s['role']}" for s in self.status if s["role"] != "OBSTACLE"]
        table = (f" table:{'TARGET' if TABLE_ID in self.policy_contact_target else 'NEAR'}"
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
            return (self._zone_policy(obj_id), "TARGET" if obj_id in self.policy_contact_target else "NEAR",
                    "POLICY")
        return ObjectCollisionPolicy(), "OBSTACLE", decided_by

    def _zone_policy(self, obj_id: int) -> ObjectCollisionPolicy:
        if obj_id in self.policy_contact_target:
            return ObjectCollisionPolicy(hand_contact=True)
        return ObjectCollisionPolicy(hand_margin=-self.near_contact_margin,
                                     held_contact=self.policy_contact_held and bool(self.held))

    def _hand_boxes_at(self, robot_state: Any, tcp_T: Optional[np.ndarray]) -> Optional[Dict[str, Tuple]]:
        """Hand boxes (T_world_box, half extents) at the keypose; the current TCP orientation without a target one."""
        key_T = np.eye(4)
        key_T[:3, 3] = robot_state.target_pose[:3]
        if len(robot_state.target_pose) >= 7:
            key_T[:3, :3] = Rotation.from_quat(robot_state.target_pose[3:7]).as_matrix()
        elif tcp_T is not None:
            key_T[:3, :3] = tcp_T[:3, :3]
        else:
            return None
        return self._hand_boxes(key_T)

    def _hand_boxes(self, key_T: np.ndarray) -> Dict[str, Tuple]:
        R_hand = key_T[:3, :3] @ self.hand_frame_R
        boxes = {}
        for name, (center, half) in self.hand_boxes.items():
            T = np.eye(4)
            T[:3, :3] = R_hand
            T[:3, 3] = key_T[:3, 3] + R_hand @ center
            boxes[name] = (T, half)
        boxes["keypose"] = key_T
        return boxes

    def _held_points_at(self, key_T: np.ndarray, digital_twin: Any) -> Optional[np.ndarray]:
        clouds = []
        for obj_id, tcp_T_obj in self.held.items():
            local = digital_twin.object_body_points(obj_id)
            if local is not None and len(local):
                T = key_T @ tcp_T_obj
                clouds.append(local @ T[:3, :3].T + T[:3, 3])
        return np.vstack(clouds) if clouds else None

    def _update_policy_contact(self, robot_state: Any, digital_twin: Any, tcp_T: Optional[np.ndarray],
                               v_tcp: np.ndarray) -> None:
        keyframe_index = int(robot_state.keyframe_index)
        same_keyframe = keyframe_index == self._last_keyframe_index
        self._last_keyframe_index = keyframe_index
        if not same_keyframe:
            self._policy_contact_revoked = set()
            self._support_start = None
        closed = (self.policy_contact_on_grasp and robot_state.HasField("gripper_command")
                  and robot_state.gripper_command < 0.5)
        approach = (self.policy_contact_on_approach and robot_state.HasField("ignore_collisions")
                    and robot_state.ignore_collisions)
        boxes = None
        if (self.policy_contact_enabled and (closed or approach) and robot_state.target_available
                and len(robot_state.target_pose) >= 3):
            boxes = self._hand_boxes_at(robot_state, tcp_T)

        grasp = closed and not self.held
        exclude = set(self.held)
        velocities = digital_twin.object_velocities()
        speeds = {o: float(np.linalg.norm(v)) for o, v in velocities.items()}
        if not same_keyframe:
            self.push_targets = set()
            if (self.policy_push_enabled and approach and boxes is not None and not self.held
                    and tcp_T is not None):
                self.push_targets = self._push_contact(boxes["keypose"], tcp_T, digital_twin, speeds, keyframe_index)
        pushed = self.push_targets | self._pushed
        moving = lambda o: self._moving(o, velocities, v_tcp, pushed)
        zone, target = set(), set()
        if boxes is not None:
            held_points = self._held_points_at(boxes["keypose"], digital_twin)
            zone = set(digital_twin.objects_near_geometry(
                self.policy_contact_zone_tolerance, [boxes["palm"], boxes["fingers"]], held_points,
                include_table=self.policy_contact_table, exclude=exclude))
            if grasp:
                target = set(digital_twin.objects_near_geometry(
                    self.policy_contact_grip_tolerance, [boxes["grip"]], None,
                    include_table=self.policy_contact_table, exclude=exclude))
            target |= self.push_targets
            if self.policy_support_enabled and approach and held_points is not None and tcp_T is not None:
                if self._support_start is None:
                    self._support_start = self._held_distances(self._held_points_at(tcp_T, digital_twin),
                                                               digital_twin, exclude, np.inf)
                target |= self._support_contact(held_points, digital_twin, exclude, moving, keyframe_index)
            # Objects stay in the zone for the whole keyframe even when the hand pushes them out of it.
            if same_keyframe:
                zone |= self.policy_contact - self._contact_hold
                target |= self.policy_contact_target - self._contact_hold
            zone = (zone | target) - exclude

        revoked = {o for o in zone if moving(o)}
        for obj_id in sorted(revoked - self._policy_contact_revoked):
            if obj_id in self.policy_contact:
                self._log(f"Object {obj_id} moving at {speeds[obj_id]:.3f} m/s: out of the contact zone "
                          f"until the next keyframe")
        self._policy_contact_revoked |= revoked
        zone -= self._policy_contact_revoked
        target &= zone

        hold = self._still_touched(((self.policy_contact - zone) | (self.policy_contact_target - target)) - exclude,
                                   digital_twin, tcp_T, moving)
        name = lambda o: "table" if o == TABLE_ID else o
        for obj_id in sorted(hold - self._contact_hold):
            self._log(f"Hand still touching {name(obj_id)} after the zone changed (keyframe {keyframe_index}): "
                      f"it stays {'TARGET' if obj_id in self.policy_contact_target else 'NEAR'} until the hand is clear")
        for obj_id in sorted(self._contact_hold - hold - zone - exclude):
            self._log(f"Hand clear of {name(obj_id)}")
        new_zone = zone | hold
        new_target = target | (hold & self.policy_contact_target)
        if boxes is None and self.policy_contact and not new_zone:
            self._log(f"Policy contact cleared (keyframe {keyframe_index})")
        elif boxes is not None and (zone != self.policy_contact - self._contact_hold
                                    or target != self.policy_contact_target - self._contact_hold):
            self._log(f"Contact zone (keyframe {keyframe_index}, {'grasp' if grasp else 'approach'}): "
                      f"target {[name(o) for o in sorted(target)] or 'none'}, "
                      f"near {[name(o) for o in sorted(zone - target)] or 'none'}")
        self._contact_hold = hold
        self._pushed = (self.push_targets & zone) | (self._pushed & hold)
        self.policy_contact, self.policy_contact_target = new_zone, new_target

    def _moving(self, obj_id: int, velocities: Dict[int, np.ndarray], v_tcp: np.ndarray, pushed: Set[int]) -> bool:
        """Faster than policy_contact_max_speed; a pushed object only when it does not move with the hand."""
        v = velocities.get(obj_id)
        if v is None or np.linalg.norm(v) <= self.policy_contact_max_speed:
            return False
        return obj_id not in pushed or np.linalg.norm(v - v_tcp) > self.policy_contact_max_speed

    def _push_contact(self, key_T: np.ndarray, tcp_T: np.ndarray, digital_twin: Any, speeds: Dict[int, float],
                      keyframe_index: int) -> Set[int]:
        """First object the hand footprint runs into on the straight way to the keypose, when it may be pushed."""
        travel = key_T[:3, 3] - tcp_T[:3, 3]
        length = float(np.linalg.norm(travel))
        if length < self.policy_push_step:
            return set()
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([tcp_T[:3, :3], key_T[:3, :3]])))
        radius = self.policy_push_reach + self.policy_push_tolerance

        def footprint_distances(s: float) -> Dict[int, float]:
            T = np.eye(4)
            T[:3, :3] = slerp([s / length]).as_matrix()[0]
            T[:3, 3] = tcp_T[:3, 3] + travel * (s / length)
            boxes = self._hand_boxes(T)
            return digital_twin.objects_near_geometry(radius, [boxes["palm"], boxes["fingers"]],
                                                      exclude=set(self.held), signed=True)

        start = footprint_distances(0.0)
        tol = self.policy_push_tolerance
        for s in np.arange(self.policy_push_step, min(self.policy_push_reach, length) + 1e-9, self.policy_push_step):
            hits = {o: d for o, d in footprint_distances(float(s)).items()
                    if d <= tol and d <= start.get(o, np.inf) - tol}
            if not hits:
                continue
            obj_id = min(hits, key=hits.get)
            permission = self._permission(obj_id)[0]
            if permission != Permission.UNSPECIFIED:
                reason = permission.name
            elif obj_id in self.released or digital_twin.is_legacy_target(obj_id):
                reason = "already touchable"
            elif speeds.get(obj_id, 0.0) > self.policy_contact_max_speed:
                reason = f"moving at {speeds[obj_id]:.3f} m/s"
            else:
                self._log(f"Push contact (keyframe {keyframe_index}): object {obj_id} ahead of the hand after "
                          f"{s * 100:.1f} cm, TARGET for the keyframe")
                return {obj_id}
            self._log(f"Push contact (keyframe {keyframe_index}): first object ahead {obj_id} is {reason}, "
                      f"nothing to push")
            return set()
        return set()

    def _held_distances(self, held_points: Optional[np.ndarray], digital_twin: Any, exclude: Set[int],
                        radius: float) -> Dict[int, float]:
        if held_points is None:
            return {}
        return digital_twin.objects_near_geometry(radius, [], held_points, exclude=exclude,
                                                  include_table=self.policy_contact_table and self.policy_support_table)

    def _support_contact(self, held_points: np.ndarray, digital_twin: Any, exclude: Set[int], moving: Any,
                         keyframe_index: int) -> Set[int]:
        """
        Supports of the held object placed at the keypose: objects the keyframe moves it towards or keeps it resting
        on; the closest one and any other in contact, when they may be touched.
        """
        tol = self.policy_support_tolerance
        start = self._support_start or {}
        near = self._held_distances(held_points, digital_twin, exclude, self.policy_contact_zone_tolerance)
        candidates = {o: d for o, d in near.items()
                      if d <= start.get(o, np.inf) - tol or (d <= tol and start.get(o, np.inf) <= tol)}
        if not candidates:
            return set()
        closest = min(candidates, key=candidates.get)
        name = lambda o: "table" if o == TABLE_ID else o
        supports = set()
        for obj_id in sorted(candidates, key=candidates.get):
            if obj_id != closest and candidates[obj_id] > tol:
                continue
            if obj_id != TABLE_ID and (self._permission(obj_id)[0] != Permission.UNSPECIFIED
                                       or obj_id in self.released or digital_twin.is_legacy_target(obj_id)):
                if obj_id == closest:
                    return set()
                continue
            if moving(obj_id) or obj_id in self._policy_contact_revoked:
                continue
            supports.add(obj_id)
            if obj_id not in self.policy_contact_target:
                how = "approached" if candidates[obj_id] <= start.get(obj_id, np.inf) - tol else "resting on it"
                self._log(f"Support contact (keyframe {keyframe_index}): {name(obj_id)} under the held object "
                          f"({candidates[obj_id] * 100:.1f} cm, {how}), TARGET for the keyframe")
        return supports

    def _tcp_velocity(self, kinematics: Any, q: np.ndarray, qd: Any, tcp_T: Optional[np.ndarray]) -> np.ndarray:
        qd = np.asarray(qd, dtype=np.float64)
        if tcp_T is None or len(qd) != 7:
            return np.zeros(3)
        eps = 1e-3
        return (self._tcp_transform(kinematics, q + eps * qd)[:3, 3] - tcp_T[:3, 3]) / eps

    def _still_touched(self, candidates: Set[int], digital_twin: Any, tcp_T: Optional[np.ndarray],
                       moving: Any) -> Set[int]:
        """Objects among the candidates that the hand, or what it holds, is still within the zone tolerance of."""
        if not (self.policy_contact_hold and candidates) or tcp_T is None:
            return set()
        boxes = self._hand_boxes(tcp_T)
        held_points = self._held_points_at(tcp_T, digital_twin)
        near = digital_twin.objects_near_geometry(
            self.policy_contact_zone_tolerance, [boxes["palm"], boxes["fingers"]], held_points,
            include_table=TABLE_ID in candidates, exclude=set(self.held))
        return {o for o in candidates if o in near and not moving(o)}

    @staticmethod
    def _tcp_transform(kinematics: Any, q: np.ndarray) -> np.ndarray:
        pos, rot = kinematics.forward_kinematics_ee(q)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def _touchable_near_tcp(self, digital_twin: Any, tcp_T: np.ndarray) -> Dict[int, float]:
        near = {}
        for obj_id, d in digital_twin.objects_near_point(tcp_T[:3, 3], self.grasp_radius).items():
            if obj_id != TABLE_ID and self._resolve(obj_id, digital_twin)[0].hand_contact:
                near[obj_id] = d
        return near

    def _update_held(self, robot_state: Any, digital_twin: Any, kinematics: Any, tcp_T: np.ndarray,
                     object_ids: Set[int]) -> None:
        if not self.grasp_enabled:
            return
        # gripper_command is the keyframe intent (already "open" while carrying towards a place keyframe); the
        # actuator fields say what the fingers do. Clients without them fall back to the intent.
        actuator = robot_state.HasField("gripper_actuated") and robot_state.HasField("gripper_moving")
        if actuator:
            commanded_open = robot_state.gripper_actuated >= 0.5
            moving = bool(robot_state.gripper_moving)
        elif robot_state.HasField("gripper_command"):
            commanded_open = robot_state.gripper_command >= 0.5
            moving = None
        else:
            return
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
            self._grasp_candidates = set()
            self._closing = False
            opened = (moving is False
                      or (self._held_open_amount is not None
                          and measured_open > self._held_open_amount + self.GRIPPER_MOTION_EPS)
                      or measured_open >= max(self.gripper_open_threshold, 0.99))
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
        if self.held:
            return

        if actuator:
            # One grasp check per closing motion, on the objects touchable while the fingers were closing: the
            # keyframe may already have advanced (and its policy contact with it) when the fingers stop.
            if moving:
                self._closing = True
                self._grasp_candidates |= set(self._touchable_near_tcp(digital_twin, tcp_T))
                return
            if not self._closing:
                return
            self._closing = False
            if measured_open < self.gripper_empty_threshold:
                self._log("Gripper closed on nothing")
                return
            near = digital_twin.objects_near_point(tcp_T[:3, 3], self.grasp_radius)
            candidates = [(near[o], o) for o in self._grasp_candidates if o in near and o in object_ids]
        else:
            if self._close_ref is None:
                self._close_ref = measured_open
            moved = measured_open < self._close_ref - self.GRIPPER_MOTION_EPS
            settled = abs(measured_open - last_measured) < 1e-3 or measured_open < self.gripper_closed_threshold
            self._closed_count = self._closed_count + 1 if (moved and settled) else 0
            if self._closed_count < self.gripper_settle_steps:
                return
            candidates = [(d, o) for o, d in self._touchable_near_tcp(digital_twin, tcp_T).items()]
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
        self.policy_contact_target.discard(obj_id)
        self._log(f"Object {obj_id} grasped: attached to the TCP "
                  f"(offset {np.linalg.norm(tcp_T_obj[:3, 3]) * 100:.1f} cm)")
