#!/usr/bin/env python
"""
Object roles end to end, without CoppeliaSim: arbitration, policy contact, grasp / release, FRAGILE margins,
Panda_tip -> EE target conversion, keyframe sequencing, PerAct keypoint discovery and VLM grounding,
on both digital twins.

Run from the repository root: python scratch/test_object_roles.py
"""

import os
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath("."))

from drema.communication.proto import drema_comm_pb2 as pb
from drema.controller.franka_kinematics import FrankaKinematics
from drema.roles import ObjectCollisionPolicy, ObjectRoleRegistry, Permission, RoleAssignment, Source
from drema.roles.registry import PANDA_HAND_FOOTPRINT
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin

BASE = np.array([0.0, 0.0, 0.75])
Q0 = FrankaKinematics.Q_REST.astype(np.float64)


def cube_mesh(size: float, folder: str) -> str:
    path = os.path.join(folder, f"cube_{int(size * 1000)}mm.obj")
    mesh = trimesh.creation.box(extents=[size] * 3).subdivide().subdivide().subdivide()
    mesh.export(path)
    return path


def settle(twin, obj_id, pos, n=12):
    """Pose updates of an object at rest, so the motion estimate forgets a teleport."""
    t0 = time.time()
    for i in range(n):
        twin.sync_object_pose(obj_id, tuple(pos), (0, 0, 0, 1), timestamp=t0 + 0.05 * i)


def costs(twin, kin, q=Q0):
    Q = np.tile(q.astype(np.float32), (1, 5, 1))
    coll_p, _ = twin.compute_trajectory_collision_costs(Q, np.zeros_like(Q), 0.015, 0.035, 120.0, 0.5, kin, dt=0.05)
    return float(coll_p.sum())


def state(q=Q0, gripper_open=1.0, gripper_command=None, ignore_collisions=None, keyframe_index=0,
          target=None, assignments=(), revision=0):
    rs = pb.RobotState(joint_positions=list(q), gripper_open=gripper_open, task_active=True,
                       robot_base_pos=list(BASE), keyframe_index=keyframe_index, roles_revision=revision)
    if target is not None:
        rs.target_pose.extend(list(target))
        rs.target_available = True
    if gripper_command is not None:
        rs.gripper_command = gripper_command
    if ignore_collisions is not None:
        rs.ignore_collisions = ignore_collisions
    rs.role_assignments.extend(a.to_proto(pb) for a in assignments)
    return rs


def run_twin(name, twin, folder):
    print(f"\n========== {name} ==========")
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    z_tool = R_tcp[:, 2]
    twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
    twin.spawn_scanned_table(table_z=0.75, bounds=(-0.5, 1.0, -0.6, 0.6))
    cube4 = cube_mesh(0.04, folder)
    grasp_pos = tcp + 0.03 * z_tool
    twin.spawn_scanned_mesh_obstacle(cube4, tuple(grasp_pos), name="cube", obj_id=101)
    twin.sync_robot_state(list(Q0))

    base_cost = costs(twin, kin)
    print(f"default (cube 1 cm below the fingertips): cost {base_cost:.3f}, min_d {twin.get_min_obstacle_distance():.4f}")
    assert base_cost > 0.0, "a cube at the fingertips must be penalized by default"

    twin.set_object_policies({101: ObjectCollisionPolicy(hand_contact=True)})
    c = costs(twin, kin)
    print(f"hand_contact on the cube: cost {c:.3f}, min_d {twin.get_min_obstacle_distance():.4f}")
    # the arm links are still checked against the cube, they are just far from it
    assert c == 0.0 and twin.get_min_obstacle_distance() > 0.05
    twin.set_object_policies({})

    T_all = kin.forward_kinematics_all(Q0)
    elbow = T_all[4][:3, 3] + BASE
    twin.spawn_scanned_mesh_obstacle(cube4, tuple(elbow), name="elbow_box", obj_id=202)
    twin.set_object_policies({101: ObjectCollisionPolicy(hand_contact=True), 202: ObjectCollisionPolicy(hand_contact=True)})
    c_arm = costs(twin, kin)
    twin.set_object_policies({101: ObjectCollisionPolicy(hand_contact=True),
                              202: ObjectCollisionPolicy(hand_contact=True, arm_contact=True)})
    c_arm_free = costs(twin, kin)
    print(f"box at the elbow with hand-only contact: cost {c_arm:.3f}; with arm contact too: {c_arm_free:.3f}")
    assert c_arm > 0.0 and c_arm_free == 0.0, "hand contact must not exempt the arm links"
    twin.remove_object(202) if hasattr(twin, "remove_object") else None

    registry = ObjectRoleRegistry(verbose=True)
    # VLM / CONFIG may not relax, a human may
    registry.update(state(revision=1, assignments=[RoleAssignment(101, Permission.TOUCH, Source.VLM)]), twin, kin)
    assert registry.status[0]["role"] == "OBSTACLE", registry.status
    registry.update(state(revision=2, assignments=[RoleAssignment(101, Permission.FRAGILE, Source.VLM),
                                                   RoleAssignment(101, Permission.TOUCH, Source.HUMAN)]), twin, kin)
    assert registry.status[0]["role"] == "TARGET" and registry.status[0]["decided_by"] == "HUMAN", registry.status
    # explicit restriction blocks the policy contact
    registry.update(state(revision=3, assignments=[RoleAssignment(101, Permission.AVOID, Source.CONFIG)],
                          target=tcp, ignore_collisions=True, keyframe_index=1), twin, kin)
    assert registry.status[0]["role"] == "OBSTACLE", registry.status
    # approach keyframe (ignore_collisions, gripper open): the hand may come close but not run into the object
    registry.update(state(revision=4, target=tcp, ignore_collisions=True, keyframe_index=1), twin, kin)
    assert registry.status[0]["role"] == "NEAR" and registry.status[0]["decided_by"] == "POLICY", registry.status
    c_near_close = costs(twin, kin)
    # inside the palm (the open fingers leave an 8 cm gap at the TCP, so a 4 cm cube there touches nothing)
    twin.sync_object_pose(101, tuple(tcp - 0.07 * z_tool), (0, 0, 0, 1))
    c_near_into = costs(twin, kin)
    # grasp keyframe (gripper commanded closed): the hand may enclose it
    registry.update(state(revision=4, target=tcp, gripper_command=0.0, keyframe_index=1), twin, kin)
    c_grasp_into = costs(twin, kin)
    print(f"approach keyframe: cube 1 cm away cost {c_near_close:.3f}, cube inside the palm {c_near_into:.3f}; "
          f"grasp keyframe: {c_grasp_into:.3f}")
    # in the palm the cube also reaches link7, which stays checked: only the hand is exempt on a grasp keyframe
    assert c_near_close == 0.0 and c_near_into > c_grasp_into
    assert registry.status[0]["role"] == "TARGET"
    settle(twin, 101, grasp_pos)
    # next keyframe without contact: the fingers still surround the cube, so it stays TARGET until the hand leaves
    registry.update(state(revision=4, target=tcp, ignore_collisions=False, keyframe_index=2), twin, kin)
    assert registry.status[0]["role"] == "TARGET", registry.status

    # grasp keyframe: gripper commanded closed at the keypose. While the arm approaches the fingers are still open
    # and nothing may be attached; then they close on a wide object (open amount stays above 0.9, like a jar lid)
    for _ in range(4):
        registry.update(state(revision=4, target=tcp, gripper_command=0.0, gripper_open=1.0, keyframe_index=3), twin, kin)
    assert registry.status[0]["role"] == "TARGET", registry.status
    for g in (0.97, 0.94, 0.93, 0.93, 0.93, 0.93):
        registry.update(state(revision=4, target=tcp, gripper_command=0.0, gripper_open=g, keyframe_index=3), twin, kin)
    assert registry.status[0]["role"] == "HELD", registry.status
    q1 = Q0.copy()
    q1[0] += 0.4
    twin.sync_robot_state(list(q1))
    pos, _ = twin.get_object_pose(101)
    tcp1, R1 = kin.forward_kinematics_ee(q1)
    expected = tcp1 + 0.03 * R1[:, 2]
    print(f"held cube follows the TCP: error {np.linalg.norm(np.asarray(pos) - expected) * 1000:.2f} mm")
    assert np.linalg.norm(np.asarray(pos) - expected) < 2e-3
    twin.sync_object_pose(101, (0.0, 0.0, 0.0), (0, 0, 0, 1))  # perception of a held object is ignored
    assert np.linalg.norm(np.asarray(twin.get_object_pose(101)[0]) - expected) < 2e-3

    # the held cube is checked against the other obstacles as part of the hand
    below = expected + 0.06 * R1[:, 2]
    twin.spawn_scanned_mesh_obstacle(cube4, tuple(below), name="under", obj_id=303)
    twin.sync_robot_state(list(q1))
    registry.update(state(q=q1, revision=4, gripper_command=0.0, gripper_open=0.93, keyframe_index=4), twin, kin)
    c_held = costs(twin, kin, q1)
    registry.update(state(q=q1, revision=5, gripper_command=0.0, gripper_open=0.93, keyframe_index=4,
                          assignments=[RoleAssignment(303, Permission.TOUCH, Source.HUMAN)]), twin, kin)
    c_held_touch = costs(twin, kin, q1)
    print(f"held cube 2 cm above another box: cost {c_held:.3f}; with that box touchable: {c_held_touch:.3f}")
    assert c_held > 0.0 and c_held_touch == 0.0

    # place keyframe: open commanded while still moving with the fingers closed -> still held; released once they open
    registry.update(state(q=q1, revision=5, gripper_command=1.0, gripper_open=0.93, keyframe_index=5), twin, kin)
    assert any(s["role"] == "HELD" for s in registry.status), registry.status
    registry.update(state(q=q1, revision=5, gripper_command=1.0, gripper_open=0.97, keyframe_index=5), twin, kin)
    first = {s["object_id"]: s["role"] for s in registry.status}[101]
    assert first == "RELEASED", registry.status  # still between the open fingers
    registry.update(state(q=Q0, revision=5, gripper_command=1.0, gripper_open=1.0, keyframe_index=6), twin, kin)
    then = {s["object_id"]: s["role"] for s in registry.status}[101]
    print(f"after release: {first} while the fingers surround it, {then} once the hand moved away")
    assert then == "OBSTACLE", registry.status
    twin.remove_object(303)

    # FRAGILE: an object 5 cm from the fingertips is beyond sigma_2, unless it is fragile
    twin.sync_robot_state(list(Q0))
    registry.update(state(revision=6), twin, kin)
    twin.sync_object_pose(101, tuple(tcp + 0.07 * z_tool), (0, 0, 0, 1))
    c_far = costs(twin, kin)
    registry.update(state(revision=7, assignments=[RoleAssignment(101, Permission.FRAGILE, Source.VLM)]), twin, kin)
    c_fragile = costs(twin, kin)
    print(f"cube 5 cm away: cost {c_far:.3f}; FRAGILE: {c_fragile:.3f}")
    assert c_far == 0.0 and c_fragile > 0.0
    print(f"{name}: OK")


def test_tip_conversion():
    from run_drema_dynamic_system import DremaDynamicSystem
    kin = FrankaKinematics(base_position=BASE)
    pos, R_fk = kin.forward_kinematics_ee(Q0)
    R_tip = R_fk @ Rotation.from_euler("z", 135, degrees=True).as_matrix()
    q_tip = Rotation.from_matrix(R_tip).as_quat()
    rs = pb.RobotState(joint_positions=list(Q0), ee_pose=list(pos) + list(q_tip), target_available=True,
                       target_pose=list(pos) + list(q_tip), target_frame=pb.FRAME_TIP, robot_base_pos=list(BASE))
    fake = SimpleNamespace(mpc_controller=SimpleNamespace(kinematics=kin), _tip_offset_R=None, yaw_symmetry=True, _yaw_choice={})
    out = DremaDynamicSystem._target_in_flange_frame(fake, rs)
    err = np.degrees(Rotation.from_matrix(Rotation.from_quat(out.target_pose[3:7]).as_matrix() @ R_fk.T).magnitude())
    print(f"\nPanda_tip target at the current tip pose -> EE target error vs FK: {err:.4f} deg")
    assert err < 1e-3 and out.target_frame == pb.FRAME_FLANGE


def test_keyframes_and_vlm():
    from drema.client.keyframes import (Keyframe, KeyframeSequencer, StaticKeyframeSource, discover_keypoints,
                                        keyframes_from_demo)
    from drema.roles.providers import HumanRoleProvider, RoleManager
    from drema.roles.vlm import KeywordVLM, SceneObject

    def obs(g, v, ic=False):
        return SimpleNamespace(gripper_open=g, joint_velocities=np.full(7, v), gripper_pose=np.array([0.3, 0, 0.9, 0, 1, 0, 0.0]),
                               ignore_collisions=ic)
    demo = [obs(1, 1)] * 5 + [obs(1, 0)] * 6 + [obs(1, 1)] * 5 + [obs(0, 0, True)] * 3 + [obs(0, 1)] * 5 + [obs(0, 0)]
    kps = discover_keypoints(demo)
    frames = keyframes_from_demo(demo)
    print(f"\nkeypoints {kps} -> {[(k.name, k.gripper_open, k.ignore_collisions) for k in frames]}")
    assert any(k.gripper_open == 0.0 for k in frames) and frames[0].frame == "tip"

    class FakeGripper:
        def __init__(self): self.open, self.busy = True, False
        def is_open(self): return self.open
        def command(self, g): self.open, self.busy = g > 0.5, True
        @property
        def done(self): return not self.busy

    kf = [Keyframe([0.3, 0, 1.0, 0, 1, 0, 0], "pre", "tip", 1.0, False, 0.01, 10, 10),
          Keyframe([0.3, 0, 0.9, 0, 1, 0, 0], "grasp", "tip", 0.0, True, 0.01, 10, 10)]
    seq, grip = KeyframeSequencer(StaticKeyframeSource(kf)), FakeGripper()
    for ee in ([0.3, 0, 1.0, 0, 1, 0, 0], [0.3, 0, 0.9, 0, 1, 0, 0]):
        seq.current(lambda: None)
        seq.step(ee, grip)
    assert seq.index == 1 and seq.phase == "gripper" and not grip.open
    grip.busy = False
    seq.current(lambda: None); seq.step([0.3, 0, 0.9, 0, 1, 0, 0], grip); seq.current(lambda: None)
    assert seq.finished
    # a 90 deg yaw error about the gripper axis blocks a tip-frame keyframe
    q_yaw = (Rotation.from_quat([0, 1, 0, 0]) * Rotation.from_euler("z", 90, degrees=True)).as_quat()
    from drema.client.keyframes import keyframe_reached
    assert not keyframe_reached(kf[0], [0.3, 0, 1.0] + list(q_yaw))

    objects = [SceneObject(11, "red_cup"), SceneObject(12, "blue_cup"), SceneObject(13, "jar_lid")]
    vlm = KeywordVLM()
    g = vlm.ground_command("you can touch the red cup", objects, None)
    assert g.object_id == 11 and g.permission == "touch", g
    g = vlm.ground_command("do not touch the cup", objects, None)
    assert g.ambiguous and set(g.candidates) == {11, 12}, g

    mgr, logs = RoleManager(), []
    human = HumanRoleProvider(vlm, mgr, min_confidence=0.95, log=logs.append)
    names = {o.object_id: o.name for o in objects}
    human._apply_grounded(vlm.ground_command("you can grab the jar lid", objects, None), names)
    assert human.pending is not None and not mgr.snapshot()[1], "a low-confidence TOUCH must wait for confirmation"
    human.confirm(True)
    human._apply_grounded(vlm.ground_command("avoid the blue cup", objects, None), names)
    rev, snap = mgr.snapshot()
    print("human roles:", [(a.object_id, a.permission.name, a.source.name) for a in snap], "|", logs[-1])
    assert {(a.object_id, a.permission) for a in snap} == {(13, Permission.TOUCH), (12, Permission.AVOID)}


def test_config_switches(folder):
    """Each mechanism switched off in the roles config leaves the object a plain obstacle."""
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    twin = MuJoCoDigitalTwin(visualize=False, table_z=0.75)
    twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
    twin.spawn_scanned_mesh_obstacle(cube_mesh(0.04, folder), tuple(tcp + 0.03 * R_tcp[:, 2]), name="cube", obj_id=101)

    def role(cfg, rs):
        reg = ObjectRoleRegistry.from_config(cfg)
        reg.update(rs, twin, kin)
        return reg.status[0]["role"]

    human_touch = state(revision=1, assignments=[RoleAssignment(101, Permission.TOUCH, Source.HUMAN)])
    approach = state(target=tcp, ignore_collisions=True, keyframe_index=1)
    grasp = state(target=tcp, gripper_command=0.0, keyframe_index=1)
    results = {
        "permissions on / off": (role({}, human_touch), role({"permissions": {"enabled": False}}, human_touch)),
        "approach on / off": (role({}, approach), role({"policy_contact": {"on_approach_keyframes": False}}, approach)),
        "grasp keyframe on / off": (role({}, grasp), role({"policy_contact": {"on_grasp_keyframes": False}}, grasp)),
        "policy contact on / off": (role({}, grasp), role({"policy_contact": {"enabled": False}}, grasp)),
    }
    print("\nconfig switches:", results)
    assert results == {"permissions on / off": ("TARGET", "OBSTACLE"), "approach on / off": ("NEAR", "OBSTACLE"),
                       "grasp keyframe on / off": ("TARGET", "OBSTACLE"), "policy contact on / off": ("TARGET", "OBSTACLE")}

    reg = ObjectRoleRegistry.from_config({"grasp": {"released_state": False}})
    for g in (1.0, 0.97, 0.94, 0.93, 0.93, 0.93, 0.93):
        reg.update(state(target=tcp, gripper_command=0.0, gripper_open=g, keyframe_index=1), twin, kin)
    assert reg.status[0]["role"] == "HELD", reg.status
    reg.update(state(gripper_command=1.0, gripper_open=1.0, keyframe_index=2), twin, kin)
    print("released_state off: after release ->", reg.status[0]["role"])
    assert reg.status[0]["role"] == "OBSTACLE", reg.status


def test_gripper_actuator_state(folder):
    """Grasp / release follow the gripper actuator, not the keyframe intent (close_jar, put_groceries cases)."""
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    twin = MuJoCoDigitalTwin(visualize=False, table_z=0.75)
    twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
    twin.spawn_scanned_mesh_obstacle(cube_mesh(0.04, folder), tuple(tcp + 0.03 * R_tcp[:, 2]), name="lid", obj_id=101)
    far = tcp + np.array([0.0, 0.0, 0.3])

    def st(actuated, moving, g, kf, target=tcp, command=0.0):
        rs = state(target=target, gripper_command=command, gripper_open=g, keyframe_index=kf)
        rs.gripper_actuated = actuated
        rs.gripper_moving = moving
        return rs

    reg = ObjectRoleRegistry()
    for _ in range(4):  # moving to the grasp keyframe, fingers still open
        reg.update(st(1.0, False, 1.0, 1), twin, kin)
    assert reg.status[0]["role"] == "TARGET", reg.status
    # wide object: the fingers barely move, and they stop only after the next keyframe is active
    reg.update(st(0.0, True, 0.99, 1), twin, kin)
    reg.update(st(0.0, False, 0.985, 2, target=far), twin, kin)
    print("\nlid closed on, keyframe advanced before the fingers stopped ->", reg.status[0]["role"])
    assert reg.status[0]["role"] == "HELD", reg.status
    # carrying towards a place keyframe whose intent is "open"; the measured opening fluctuates
    for g in (0.985, 1.0, 0.99, 1.0):
        reg.update(st(0.0, False, g, 3, target=far, command=1.0), twin, kin)
    print("carried with keyframe intent open ->", reg.status[0]["role"])
    assert reg.status[0]["role"] == "HELD", reg.status
    reg.update(st(1.0, True, 0.985, 3, target=far, command=1.0), twin, kin)
    assert reg.status[0]["role"] == "HELD", reg.status
    reg.update(st(1.0, False, 1.0, 3, target=far, command=1.0), twin, kin)
    print("gripper actually opened ->", reg.status[0]["role"])
    assert reg.status[0]["role"] == "RELEASED", reg.status

    reg = ObjectRoleRegistry()
    reg.update(st(0.0, True, 0.5, 1), twin, kin)
    reg.update(st(0.0, False, 0.0, 1), twin, kin)
    print("closed on nothing ->", reg.status[0]["role"])
    assert reg.status[0]["role"] == "TARGET", reg.status


def test_contact_zone(folder):
    """
    Contact zone = hand footprint at the keypose: a neighbour beside the palm (beyond any TCP radius) is NEAR, not
    TARGET; a moving object never enters the zone; a held object may touch what lies under it at a place keyframe.
    """
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    z_tool = R_tcp[:, 2]
    R_hand = R_tcp @ Rotation.from_euler("z", PANDA_HAND_FOOTPRINT["finger_axis_yaw_deg"], degrees=True).as_matrix()
    fingers = PANDA_HAND_FOOTPRINT["fingers"]
    cube4 = cube_mesh(0.04, folder)
    for twin_cls in (MuJoCoDigitalTwin, PyBulletDigitalTwin):
        name = twin_cls.__name__
        twin = twin_cls(visualize=False, table_z=0.75)
        twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
        twin.sync_robot_state(list(Q0))
        # at the fingertips, 1 cm past a finger along the finger axis and below the palm: ~10 cm from the TCP
        side_pos = tcp + R_hand @ np.array([0.0, fingers["center"][1] + fingers["half_extents"][1] + 0.01 + 0.02, 0.0])
        twin.spawn_scanned_mesh_obstacle(cube4, tuple(side_pos), name="neighbour", obj_id=202)
        settle(twin, 202, side_pos)
        approach = state(target=tcp, ignore_collisions=True, keyframe_index=1)

        off = ObjectRoleRegistry.from_config({"policy_contact": {"enabled": False}})
        off.update(approach, twin, kin)
        c_off = costs(twin, kin)
        reg = ObjectRoleRegistry()
        reg.update(approach, twin, kin)
        role_side = {s["object_id"]: s["role"] for s in reg.status}[202]
        c_zone = costs(twin, kin)
        print(f"\n[{name}] neighbour 1 cm beside a fingertip: zone role {role_side}, cost {c_zone:.3f} "
              f"(policy contact off: {c_off:.3f})")
        assert role_side == "NEAR" and c_off > 0.0 and c_zone == 0.0

        # the target between the fingers is TARGET on a grasp keyframe, the neighbour stays NEAR
        grasp_pos = tcp + 0.03 * z_tool
        twin.spawn_scanned_mesh_obstacle(cube4, tuple(grasp_pos), name="target", obj_id=101)
        settle(twin, 101, grasp_pos)
        reg = ObjectRoleRegistry()
        reg.update(state(target=tcp, gripper_command=0.0, keyframe_index=2), twin, kin)
        roles = {s["object_id"]: s["role"] for s in reg.status}
        print(f"[{name}] grasp keyframe: {roles}")
        assert roles == {101: "TARGET", 202: "NEAR"}, roles

        # a moving object never enters the zone; one that starts moving leaves it until the next keyframe
        t0 = time.time()
        for i in range(12):
            twin.sync_object_pose(202, tuple(side_pos + 0.005 * (i - 11) * z_tool), (0, 0, 0, 1), timestamp=t0 + 0.05 * i)
        speed = twin.object_speeds()[202]
        reg = ObjectRoleRegistry()
        reg.update(state(target=tcp, ignore_collisions=True, keyframe_index=3), twin, kin)
        moving_role = {s["object_id"]: s["role"] for s in reg.status}[202]
        print(f"[{name}] neighbour moving at {speed:.3f} m/s: {moving_role}")
        assert speed > 0.03 and moving_role == "OBSTACLE"
        settle(twin, 202, side_pos)
        reg.update(state(target=tcp, ignore_collisions=True, keyframe_index=3), twin, kin)
        assert {s["object_id"]: s["role"] for s in reg.status}[202] == "OBSTACLE", "revoked for the keyframe"
        reg.update(state(target=tcp, ignore_collisions=True, keyframe_index=4), twin, kin)
        back = {s["object_id"]: s["role"] for s in reg.status}[202]
        print(f"[{name}] at rest again, next keyframe: {back}")
        assert back == "NEAR"
        twin.remove_object(202)

        # missed grasp: the keyframe moves on with the fingers still around the target -> TARGET until the hand leaves
        above = tcp - 0.15 * z_tool
        q_away = Q0.copy()
        q_away[0] += 0.4
        for hold in (True, False):
            reg = ObjectRoleRegistry.from_config({"policy_contact": {"hold_until_clear": hold}})
            reg.update(state(target=tcp, gripper_command=0.0, keyframe_index=7), twin, kin)
            reg.update(state(target=above, gripper_command=0.0, keyframe_index=8), twin, kin)
            around = {s["object_id"]: s["role"] for s in reg.status}[101]
            reg.update(state(q=q_away, target=above, gripper_command=0.0, keyframe_index=8), twin, kin)
            left = {s["object_id"]: s["role"] for s in reg.status}[101]
            print(f"[{name}] keyframe advanced around the target (hold {'on' if hold else 'off'}): {around}, "
                  f"hand moved away: {left}")
            assert (around, left) == (("TARGET", "OBSTACLE") if hold else ("OBSTACLE", "OBSTACLE"))

        # place keyframe: the held cube sinks 1 cm into the box it is placed on (as the lid on the jar)
        results = {}
        for held_contact in (False, True):
            settle(twin, 101, grasp_pos)
            reg = ObjectRoleRegistry.from_config({"policy_contact": {"held_contact": held_contact,
                                                                     "support_contact": {"enabled": False}}})
            for actuated, moving, g in ((1.0, False, 1.0), (0.0, True, 0.99), (0.0, False, 0.95)):
                rs = state(target=tcp, gripper_command=0.0, gripper_open=g, keyframe_index=5)
                rs.gripper_actuated, rs.gripper_moving = actuated, moving
                reg.update(rs, twin, kin)
            assert {s["object_id"]: s["role"] for s in reg.status}[101] == "HELD", reg.status
            under = grasp_pos + (0.04 - 0.01) * z_tool
            twin.spawn_scanned_mesh_obstacle(cube4, tuple(under), name="jar", obj_id=303)
            settle(twin, 303, under)
            rs = state(target=tcp, gripper_command=1.0, gripper_open=0.95, ignore_collisions=True, keyframe_index=6)
            rs.gripper_actuated, rs.gripper_moving = 0.0, False
            reg.update(rs, twin, kin)
            roles = {s["object_id"]: s["role"] for s in reg.status}
            results[held_contact] = (roles[303], costs(twin, kin), twin.get_min_obstacle_distance())
            if held_contact:
                # next keyframe, no contact zone, the held cube still sunk into the box: the box stays NEAR
                rs = state(target=tcp - 0.15 * z_tool, gripper_command=1.0, gripper_open=0.95, keyframe_index=7)
                rs.gripper_actuated, rs.gripper_moving = 0.0, False
                reg.update(rs, twin, kin)
                after = {s["object_id"]: s["role"] for s in reg.status}[303]
                print(f"[{name}] keyframe after the place, held cube still on the box: {after}, cost {costs(twin, kin):.3f}")
                assert after == "NEAR" and costs(twin, kin) == 0.0
            reg.reset(twin)
            twin.remove_object(303)
        print(f"[{name}] held cube 1 cm into the box below: held_contact off {results[False]}, on {results[True]}")
        assert results[False][0] == "NEAR" and results[False][1] > 0.0 and results[False][2] < 0.0
        assert results[True][0] == "NEAR" and results[True][1] == 0.0 and results[True][2] > 0.0
        twin.shutdown()


def test_push_contact(folder):
    """
    ignore_collisions keyframe started with the hand empty: only the first object ahead of the hand on its way to
    the keypose becomes TARGET (the block of slide_block); a small cube beside the hand, the one behind the block,
    an AVOID block and a block moving on its own are not pushed; a pushed block moving with the hand stays TARGET.
    """
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    R_hand = R_tcp @ Rotation.from_euler("z", PANDA_HAND_FOOTPRINT["finger_axis_yaw_deg"], degrees=True).as_matrix()
    fingers = PANDA_HAND_FOOTPRINT["fingers"]
    push_dir = R_hand[:, 0]
    key = tcp + 0.20 * push_dir
    # 1 cm ahead of the fingers along the push, a 2 cm cube 1 cm beside a finger, a cube behind the block
    block_pos = tcp + R_hand @ np.array([fingers["half_extents"][0] + 0.01 + 0.02, 0.0, 0.0])
    side_pos = tcp + R_hand @ np.array([0.0, fingers["center"][1] + fingers["half_extents"][1] + 0.01 + 0.01, 0.0])
    behind_pos = block_pos + 0.045 * push_dir
    cube4, cube2 = cube_mesh(0.04, folder), cube_mesh(0.02, folder)

    J = np.stack([(kin.forward_kinematics_ee(Q0 + 1e-4 * np.eye(7)[i])[0] - tcp) / 1e-4 for i in range(7)], axis=1)
    qd_with_block = np.linalg.pinv(J) @ (0.1 * push_dir)

    def push_state(kf, ignore_collisions=True, qd=None, **kw):
        rs = state(target=key, gripper_command=0.0, ignore_collisions=ignore_collisions, keyframe_index=kf, **kw)
        if qd is not None:
            rs.joint_velocities.extend(list(qd))
        return rs

    def roles(reg):
        return {s["object_id"]: s["role"] for s in reg.status}

    for twin_cls in (MuJoCoDigitalTwin, PyBulletDigitalTwin):
        name = twin_cls.__name__
        twin = twin_cls(visualize=False, table_z=0.75)
        twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
        twin.sync_robot_state(list(Q0))
        for obj_id, mesh, pos in ((401, cube4, block_pos), (402, cube2, side_pos), (403, cube4, behind_pos)):
            twin.spawn_scanned_mesh_obstacle(mesh, tuple(pos), name=f"obj{obj_id}", obj_id=obj_id)
            settle(twin, obj_id, pos)

        off = ObjectRoleRegistry.from_config({"policy_contact": {"push_contact": {"enabled": False}}})
        off.update(push_state(1), twin, kin)
        reg = ObjectRoleRegistry()
        reg.update(push_state(1), twin, kin)
        no_ic = ObjectRoleRegistry()
        no_ic.update(push_state(1, ignore_collisions=False), twin, kin)
        print(f"\n[{name}] push keyframe: {roles(reg)} | push contact off: {roles(off)} | "
              f"no ignore_collisions: {roles(no_ic)}")
        assert roles(reg) == {401: "TARGET", 402: "OBSTACLE", 403: "OBSTACLE"}, roles(reg)
        assert roles(off)[401] == "OBSTACLE" and roles(no_ic)[401] == "OBSTACLE"

        avoid = ObjectRoleRegistry()
        avoid.update(push_state(1, assignments=[RoleAssignment(401, Permission.AVOID, Source.HUMAN)], revision=1),
                     twin, kin)
        print(f"[{name}] block AVOID: {roles(avoid)}")
        assert roles(avoid)[401] == "OBSTACLE" and roles(avoid)[403] != "TARGET"

        # the pushed block moves at 10 cm/s: with the hand it stays TARGET, with the hand still it is revoked
        t0 = time.time()
        for i in range(12):
            twin.sync_object_pose(401, tuple(block_pos + 0.005 * (i - 11) * push_dir), (0, 0, 0, 1),
                                  timestamp=t0 + 0.05 * i)
        reg.update(push_state(1, qd=qd_with_block), twin, kin)
        with_hand = roles(reg)[401]
        reg.update(push_state(1, qd=np.zeros(7)), twin, kin)
        hand_still = roles(reg)[401]
        print(f"[{name}] pushed block at {twin.object_speeds()[401]:.3f} m/s: moving with the hand {with_hand}, "
              f"hand still {hand_still}")
        assert (with_hand, hand_still) == ("TARGET", "OBSTACLE")

        # a block already moving on its own when the keyframe starts is not pushed, nor the cube behind it
        moving = ObjectRoleRegistry()
        moving.update(push_state(2), twin, kin)
        print(f"[{name}] block moving when the keyframe starts: {roles(moving)}")
        assert roles(moving)[401] == "OBSTACLE" and roles(moving)[403] != "TARGET"
        twin.shutdown()


def test_support_contact(folder):
    """
    ignore_collisions keyframe with a cube held and sunk 1 cm into the box it is put on (the lid on its jar): the box
    is TARGET, so neither the hand nor the cube trips the safety halt on it; a neighbour of the cube stays NEAR and the
    cube may not touch it; an AVOID box, a keyframe without ignore_collisions or the switch off leave the box NEAR.
    """
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    z_tool = R_tcp[:, 2]
    R_hand = R_tcp @ Rotation.from_euler("z", PANDA_HAND_FOOTPRINT["finger_axis_yaw_deg"], degrees=True).as_matrix()
    grasp_pos = tcp + 0.03 * z_tool
    box_pos = grasp_pos + (0.04 - 0.01) * z_tool
    neighbour_pos = grasp_pos + 0.055 * R_hand[:, 0]
    cube4 = cube_mesh(0.04, folder)

    def roles(reg):
        return {s["object_id"]: s["role"] for s in reg.status}

    def grasp(reg, twin):
        for actuated, moving, g in ((1.0, False, 1.0), (0.0, True, 0.99), (0.0, False, 0.95)):
            rs = state(target=tcp, gripper_command=0.0, gripper_open=g, keyframe_index=5)
            rs.gripper_actuated, rs.gripper_moving = actuated, moving
            reg.update(rs, twin, kin)
        assert roles(reg)[501] == "HELD", reg.status

    def place(reg, twin, ignore_collisions=True, gripper_command=1.0, **kw):
        rs = state(target=tcp, gripper_command=gripper_command, gripper_open=0.95, ignore_collisions=ignore_collisions,
                   keyframe_index=6, **kw)
        rs.gripper_actuated, rs.gripper_moving = 0.0, False
        reg.update(rs, twin, kin)

    for twin_cls in (MuJoCoDigitalTwin, PyBulletDigitalTwin):
        name = twin_cls.__name__
        twin = twin_cls(visualize=False, table_z=0.75)
        twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
        twin.sync_robot_state(list(Q0))
        twin.spawn_scanned_mesh_obstacle(cube4, tuple(grasp_pos), name="cube", obj_id=501)
        settle(twin, 501, grasp_pos)

        cases = {"support": ({}, {}), "support off": ({"support_contact": {"enabled": False}}, {}),
                 "no ignore_collisions": ({}, {"ignore_collisions": False, "gripper_command": 0.0}),
                 "box AVOID": ({}, {"assignments": [RoleAssignment(502, Permission.AVOID, Source.HUMAN)],
                                    "revision": 1})}
        results = {}
        for case, (cfg, kw) in cases.items():
            settle(twin, 501, grasp_pos)
            reg = ObjectRoleRegistry.from_config({"policy_contact": cfg})
            grasp(reg, twin)
            twin.spawn_scanned_mesh_obstacle(cube4, tuple(box_pos), name="box", obj_id=502)
            twin.spawn_scanned_mesh_obstacle(cube4, tuple(neighbour_pos), name="neighbour", obj_id=503)
            settle(twin, 502, box_pos)
            settle(twin, 503, neighbour_pos)
            place(reg, twin, **kw)
            r = roles(reg)
            results[case] = (r[502], r[503], twin.object_policy(503).held_margin() is not None,
                             round(twin.get_min_obstacle_distance(), 4))
            reg.reset(twin)
            twin.remove_object(502)
            twin.remove_object(503)
        for case, res in results.items():
            print(f"[{name}] cube held 1 cm into the box, {case}: box {res[0]}, neighbour {res[1]}, "
                  f"cube kept off the neighbour {res[2]}, halt distance {res[3]}")
        assert results["support"][:3] == ("TARGET", "NEAR", True) and results["support"][3] > 0.0
        assert results["support off"][0] == "NEAR" and results["support off"][3] < 0.0
        assert results["no ignore_collisions"][0] == "NEAR" and results["no ignore_collisions"][3] < 0.0
        assert results["box AVOID"][0] == "OBSTACLE" and results["box AVOID"][1] == "NEAR"
        twin.shutdown()


def test_support_motion(folder):
    """
    Supports follow the motion of the held object: a box it is lowered onto is a support, the neighbour beside it is
    not; lifting it off a box makes neither the box nor the neighbour a support; resting on a box and against a wall
    makes both supports, the neighbour 1.5 cm away stays NEAR.
    """
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    z_tool = R_tcp[:, 2]
    R_hand = R_tcp @ Rotation.from_euler("z", PANDA_HAND_FOOTPRINT["finger_axis_yaw_deg"], degrees=True).as_matrix()
    grasp_pos = tcp + 0.03 * z_tool
    side = R_hand[:, 0]
    cube4 = cube_mesh(0.04, folder)

    def roles(reg):
        return {s["object_id"]: s["role"] for s in reg.status}

    # (keypose shift along the approach, box gap below the held cube [m], extra objects {id: (side, lateral gap [m])})
    cases = {
        "lowered onto the box": (0.02, 0.02, {603: (1, 0.015)}),
        "lifted off the box": (-0.03, -0.01, {603: (1, 0.015)}),
        "resting on a box, against a wall": (0.0, -0.01, {603: (1, 0.015), 604: (-1, 0.005)}),
    }
    expected = {
        "lowered onto the box": {602: "TARGET", 603: "NEAR"},
        "lifted off the box": {602: "NEAR", 603: "NEAR"},
        "resting on a box, against a wall": {602: "TARGET", 603: "NEAR", 604: "TARGET"},
    }
    for twin_cls in (MuJoCoDigitalTwin, PyBulletDigitalTwin):
        name = twin_cls.__name__
        twin = twin_cls(visualize=False, table_z=0.75)
        twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
        twin.sync_robot_state(list(Q0))
        twin.spawn_scanned_mesh_obstacle(cube4, tuple(grasp_pos), name="cube", obj_id=601)
        for case, (shift, gap, extra) in cases.items():
            settle(twin, 601, grasp_pos)
            reg = ObjectRoleRegistry()
            for actuated, moving, g in ((1.0, False, 1.0), (0.0, True, 0.99), (0.0, False, 0.95)):
                rs = state(target=tcp, gripper_command=0.0, gripper_open=g, keyframe_index=5)
                rs.gripper_actuated, rs.gripper_moving = actuated, moving
                reg.update(rs, twin, kin)
            assert roles(reg)[601] == "HELD", reg.status
            placed = {602: grasp_pos + (0.04 + gap) * z_tool}
            placed.update({oid: grasp_pos + sign * (0.04 + g) * side for oid, (sign, g) in extra.items()})
            for oid, pos in placed.items():
                twin.spawn_scanned_mesh_obstacle(cube4, tuple(pos), name=f"obj{oid}", obj_id=oid)
                settle(twin, oid, pos)
            rs = state(target=tcp + shift * z_tool, gripper_command=0.0, gripper_open=0.95, ignore_collisions=True,
                       keyframe_index=6)
            rs.gripper_actuated, rs.gripper_moving = 0.0, False
            reg.update(rs, twin, kin)
            got = {oid: roles(reg)[oid] for oid in placed}
            print(f"[{name}] {case}: {got}")
            assert got == expected[case], (case, got)
            reg.reset(twin)
            for oid in placed:
                twin.remove_object(oid)
        twin.shutdown()


def test_keyframe_settling():
    """
    Settling: the gripper closes, and the precise approach pose before it is left, only once the arm stops closing
    in; without settling both happen on the first step inside the tolerance. Demo keyframes before a gripper change
    are precise.
    """
    from drema.client.keyframes import Keyframe, KeyframeSequencer, StaticKeyframeSource, keyframes_from_demo

    def obs(g, v, ic=False):
        return SimpleNamespace(gripper_open=g, joint_velocities=np.full(7, v), gripper_pose=np.array([0.3, 0, 0.9, 0, 1, 0, 0.0]),
                               ignore_collisions=ic)
    demo = [obs(1, 1)] * 5 + [obs(1, 0)] * 6 + [obs(1, 1)] * 5 + [obs(0, 0, True)] * 3 + [obs(0, 1)] * 5 + [obs(0, 0)]
    frames = keyframes_from_demo(demo)
    flags = [(k.gripper_open, k.precise) for k in frames]
    print(f"\ndemo keyframes (gripper, precise): {flags}")
    assert all(k.precise == ((n.gripper_open > 0.5) != (k.gripper_open > 0.5)) for k, n in zip(frames, frames[1:]))
    assert any(k.precise for k in frames) and not frames[-1].precise

    class FakeGripper:
        def __init__(self): self.open, self.busy = True, False
        def is_open(self): return self.open
        def command(self, g): self.open, self.busy = g > 0.5, True
        @property
        def done(self): return not self.busy

    def run(settle):
        pre = Keyframe([0.3, 0, 1.0, 0, 1, 0, 0], "pre", "tip", 1.0, False, 0.02, 10, 10)
        pre.precise = True
        grasp = Keyframe([0.3, 0, 0.9, 0, 1, 0, 0], "grasp", "tip", 0.0, True, 0.01, 10, 10)
        seq = KeyframeSequencer(StaticKeyframeSource([pre, grasp]), log=lambda m: None,
                                settle_on_gripper_change=settle, settle_on_precise_approach=settle)
        grip, steps = FakeGripper(), {}
        # 1.5 cm off the pre-grasp (inside its free-space tolerance), creeping to 0.6 cm and stopping there; then
        # 0.9 cm off the grasp keypose (inside its tolerance), creeping to 0.5 cm and stopping there
        for t in range(200):
            target_z = 1.0 if seq.index == 0 else 0.9
            k = t - steps.get(seq.index, t)
            steps.setdefault(seq.index, t)
            off = 0.015 - min(k, 18) * 0.0005 if seq.index == 0 else 0.009 - min(k, 8) * 0.0005
            seq.current(lambda: None)
            seq.step([0.3 + off, 0, target_z, 0, 1, 0, 0], grip)
            if seq.phase == "gripper":
                return {"pre left after": steps[1] - steps[0], "gripper closed after": t - steps[1],
                        "grasp offset [cm]": round(off * 100, 2)}
        return None

    fast, slow = run(False), run(True)
    print(f"no settling: {fast} | settling: {slow}")
    assert fast == {"pre left after": 1, "gripper closed after": 0, "grasp offset [cm]": 0.9}
    assert slow["pre left after"] > 10 and slow["gripper closed after"] > 10 and slow["grasp offset [cm]"] == 0.5

    # an approach pose the arm cannot get closer than 1.4 cm to is still left once the arm has settled there
    pre = Keyframe([0.3, 0, 1.0, 0, 1, 0, 0], "pre", "tip", 1.0, False, 0.02, 10, 10)
    pre.precise = True
    seq = KeyframeSequencer(StaticKeyframeSource([pre]), log=lambda m: None, settle_on_precise_approach=True)
    for t in range(100):
        seq.current(lambda: None)
        seq.step([0.3 + 0.015 - min(t, 2) * 0.0005, 0, 1.0, 0, 1, 0, 0], FakeGripper())
        if seq.index == 1:
            break
    print(f"approach pose blocked at 1.4 cm: left after {t + 1} steps")
    assert seq.index == 1 and t < 20


def test_hold_over_zone(folder):
    """
    A neighbour between the fingers on the grasp (TARGET) that the held cube still touches on the lift stays TARGET
    although the lift zone holds it as NEAR: no halt on the perception overlap; NEAR again once the hand is away.
    """
    kin = FrankaKinematics(base_position=BASE)
    tcp, R_tcp = kin.forward_kinematics_ee(Q0)
    z_tool = R_tcp[:, 2]
    R_hand = R_tcp @ Rotation.from_euler("z", PANDA_HAND_FOOTPRINT["finger_axis_yaw_deg"], degrees=True).as_matrix()
    grasp_pos = tcp + 0.03 * z_tool
    neighbour_pos = grasp_pos + (0.04 - 0.003) * R_hand[:, 1]
    cube4 = cube_mesh(0.04, folder)
    q_away = Q0.copy()
    q_away[0] += 0.4

    def roles(reg):
        return {s["object_id"]: s["role"] for s in reg.status}

    for twin_cls in (MuJoCoDigitalTwin, PyBulletDigitalTwin):
        name = twin_cls.__name__
        twin = twin_cls(visualize=False, table_z=0.75)
        twin.load_robot(base_position=tuple(BASE), joint_positions=list(Q0))
        twin.sync_robot_state(list(Q0))
        for oid, pos in ((701, grasp_pos), (702, neighbour_pos)):
            twin.spawn_scanned_mesh_obstacle(cube4, tuple(pos), name=f"obj{oid}", obj_id=oid)
            settle(twin, oid, pos)
        reg = ObjectRoleRegistry()
        for actuated, moving, g in ((1.0, False, 1.0), (0.0, True, 0.99), (0.0, False, 0.95)):
            rs = state(target=tcp, gripper_command=0.0, gripper_open=g, ignore_collisions=True, keyframe_index=5)
            rs.gripper_actuated, rs.gripper_moving = actuated, moving
            reg.update(rs, twin, kin)
        grasped = roles(reg)
        lift = state(target=tcp - 0.055 * z_tool, gripper_command=0.0, gripper_open=0.95, ignore_collisions=True,
                     keyframe_index=6)
        lift.gripper_actuated, lift.gripper_moving = 0.0, False
        reg.update(lift, twin, kin)
        lifted, halt = roles(reg), twin.get_min_obstacle_distance()
        away = state(q=q_away, target=tcp - 0.055 * z_tool, gripper_command=0.0, gripper_open=0.95,
                     ignore_collisions=True, keyframe_index=6)
        away.gripper_actuated, away.gripper_moving = 0.0, False
        reg.update(away, twin, kin)
        print(f"[{name}] grasp {grasped} | lift, held cube still on the neighbour {lifted} (halt distance {halt:.4f}) "
              f"| hand away {roles(reg)}")
        assert grasped[702] == "TARGET" and lifted == {701: "HELD", 702: "TARGET"} and halt > 0.0
        assert roles(reg)[702] != "TARGET"
        reg.reset(twin)
        twin.shutdown()


if __name__ == "__main__":
    folder = tempfile.mkdtemp(prefix="drema_roles_")
    run_twin("MuJoCo twin", MuJoCoDigitalTwin(visualize=False, table_z=0.75), folder)
    run_twin("PyBullet twin", PyBulletDigitalTwin(visualize=False, table_z=0.75), folder)
    test_tip_conversion()
    test_keyframes_and_vlm()
    test_keyframe_settling()
    test_config_switches(folder)
    test_gripper_actuator_state(folder)
    test_contact_zone(folder)
    test_push_contact(folder)
    test_support_contact(folder)
    test_support_motion(folder)
    test_hold_over_zone(folder)
    print("\nALL OBJECT ROLE TESTS PASSED")
