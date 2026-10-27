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
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath("."))

from drema.communication.proto import drema_comm_pb2 as pb
from drema.controller.franka_kinematics import FrankaKinematics
from drema.roles import ObjectCollisionPolicy, ObjectRoleRegistry, Permission, RoleAssignment, Source
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin

BASE = np.array([0.0, 0.0, 0.75])
Q0 = FrankaKinematics.Q_REST.astype(np.float64)


def cube_mesh(size: float, folder: str) -> str:
    path = os.path.join(folder, f"cube_{int(size * 1000)}mm.obj")
    mesh = trimesh.creation.box(extents=[size] * 3).subdivide().subdivide().subdivide()
    mesh.export(path)
    return path


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
    twin.sync_object_pose(101, tuple(grasp_pos), (0, 0, 0, 1))
    registry.update(state(revision=4, target=tcp, ignore_collisions=False, keyframe_index=2), twin, kin)
    assert registry.status[0]["role"] == "OBSTACLE"

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


if __name__ == "__main__":
    folder = tempfile.mkdtemp(prefix="drema_roles_")
    run_twin("MuJoCo twin", MuJoCoDigitalTwin(visualize=False, table_z=0.75), folder)
    run_twin("PyBullet twin", PyBulletDigitalTwin(visualize=False, table_z=0.75), folder)
    test_tip_conversion()
    test_keyframes_and_vlm()
    test_config_switches(folder)
    print("\nALL OBJECT ROLE TESTS PASSED")
