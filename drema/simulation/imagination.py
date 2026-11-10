"""
Physics imagination of the digital twin: the present scene rebuilt as a physical MuJoCo model (objects as free
bodies with their mass and friction, the robot with its actuators, the held object welded to the hand or grasped by
the fingers) and batched rollouts of many joint-target sequences over randomized physical worlds.

Two rollout backends give the same physics: MuJoCo Warp (GPU, many worlds per kernel; the CPU device is only for
tests) and mujoco.rollout (C, one thread per world). World w of a request is sample w // R in physical world w % R;
world 0 of every sample keeps the nominal parameters.
"""

import itertools
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
FINGER_OPENING = 0.04
_MODEL_VERSIONS = itertools.count()
# A CUDA graph capture must not interleave with kernels another thread launches on the device (MPPI queries).
WARP_LOCK = threading.Lock()


@dataclass
class ImaginedObject:
    obj_id: int
    name: str
    parts: List[str]
    pos: np.ndarray                 # world position [m]
    quat_wxyz: np.ndarray
    velocity: np.ndarray            # linear velocity estimate [m/s]
    mass: float
    friction: float
    static: bool
    held: bool


@dataclass
class SceneSnapshot:
    """Everything the imagination model needs from the present-state twin, copied under its lock."""
    robot_model_path: str
    base_pos: np.ndarray
    base_quat_wxyz: np.ndarray
    hand_body: str
    q: np.ndarray                   # arm joints in the twin model space (measured + offsets)
    qd: np.ndarray
    finger_opening: float           # per finger [m]
    joint_angle_offsets: np.ndarray
    objects: List[ImaginedObject]
    table_bounds: Optional[Tuple[float, float, float, float]] = None
    table_mesh: Optional[str] = None
    table_z: float = 0.75


@dataclass
class ImaginationRequest:
    """
    arm_targets: [S, T, 7] joint position targets in the measured joint space, one per control step;
    gripper: [S, T] open amount in [0, 1] per control step (None: closed on a grasped object, otherwise as now).
    Targets are interpolated linearly over each control step, starting from the current configuration.
    """
    arm_targets: np.ndarray
    control_dt: float
    gripper: Optional[np.ndarray] = None
    worlds_per_sample: int = 1
    record_every: int = 0           # control steps between recorded states (0: final state only)
    seed: Optional[int] = None


@dataclass
class ImaginationResult:
    object_ids: List[int]
    times: np.ndarray               # [n_rec] seconds from the start
    object_pos: np.ndarray          # [S, R, n_rec, O, 3]
    object_quat: np.ndarray         # [S, R, n_rec, O, 4] xyzw
    arm_q: np.ndarray               # [S, R, n_rec, 7] measured joint space
    valid: np.ndarray               # [S, R] no NaN and no buffer overflow
    mass_scale: np.ndarray          # [R, O]
    friction_scale: np.ndarray      # [R, O]
    initial_pos: np.ndarray         # [O, 3]
    initial_quat: np.ndarray        # [O, 4] xyzw
    backend: str = ""
    wall_time_s: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

    def displacement(self) -> np.ndarray:
        """Final horizontal displacement of every object [S, R, O] (m)."""
        return np.linalg.norm(self.object_pos[:, :, -1, :, :2] - self.initial_pos[None, None, :, :2], axis=-1)

    def tilt_deg(self) -> np.ndarray:
        """Final angle between each object's z axis and its initial z axis [S, R, O] (deg)."""
        z0 = Rotation.from_quat(self.initial_quat).as_matrix()[:, :, 2]
        S, R, _, O, _ = self.object_quat.shape
        z = Rotation.from_quat(self.object_quat[:, :, -1].reshape(-1, 4)).as_matrix()[:, :, 2].reshape(S, R, O, 3)
        return np.degrees(np.arccos(np.clip(np.sum(z * z0[None, None], axis=-1), -1.0, 1.0)))


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


class ImaginationModel:
    """Physical MuJoCo model of a snapshot, with the index maps used to write states and read outcomes."""

    def __init__(self, snap: SceneSnapshot, held_object: str, timestep: Optional[float],
                 exclude_pairs: Sequence[Tuple[int, int]] = (), grasp_force: float = 70.0):
        self.held_mode = held_object
        self.exclude_pairs = tuple(exclude_pairs)
        spec = mujoco.MjSpec.from_file(snap.robot_model_path)
        spec.meshdir = os.path.join(os.path.dirname(os.path.abspath(snap.robot_model_path)), spec.meshdir)
        for g in list(spec.geoms):
            if g.classname.name == "visual":
                spec.delete(g)
        used = {g.meshname for g in spec.geoms if g.meshname}
        for mesh in list(spec.meshes):
            if (mesh.name or os.path.splitext(os.path.basename(mesh.file))[0]) not in used:
                spec.delete(mesh)
        if timestep:
            spec.option.timestep = float(timestep)
        # Thin convex parts of scanned meshes give near-singular inertias that blow up under contact.
        spec.compiler.boundmass = 1e-3
        spec.compiler.boundinertia = 1e-6
        link0 = spec.body("link0")
        if link0 is None:
            raise ValueError(f"[IMAGINATION] No body 'link0' in {snap.robot_model_path}")
        R_base = Rotation.from_quat(_wxyz_to_xyzw(snap.base_quat_wxyz))
        link0.pos = (snap.base_pos + R_base.apply(np.asarray(link0.pos))).tolist()
        q_link0 = R_base * Rotation.from_quat(_wxyz_to_xyzw(np.asarray(link0.quat, dtype=np.float64)))
        x, y, z, w = q_link0.as_quat()
        link0.quat = [w, x, y, z]

        world = spec.worldbody
        world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], friction=[1, 0.005, 0.0001])
        if snap.table_bounds is not None:
            x0, x1, y0, y1 = snap.table_bounds
            world.add_geom(name="table_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[0.5 * abs(x1 - x0), 0.5 * abs(y1 - y0), 0.5 * snap.table_z],
                           pos=[0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * snap.table_z], friction=[0.5, 0.005, 0.0001])
        elif snap.table_mesh is not None and os.path.exists(snap.table_mesh):
            spec.add_mesh(name="table_mesh", file=os.path.abspath(snap.table_mesh))
            world.add_geom(name="table_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="table_mesh",
                           friction=[0.5, 0.005, 0.0001])

        self.object_ids = [o.obj_id for o in snap.objects]
        self.static = {o.obj_id: bool(o.static and not o.held) for o in snap.objects}
        for o in snap.objects:
            body = world.add_body(name=f"obj_{o.obj_id}", pos=o.pos.tolist(), quat=o.quat_wxyz.tolist())
            if not self.static[o.obj_id]:
                body.add_freejoint(name=f"obj_{o.obj_id}_free")
            for k, part in enumerate(o.parts):
                spec.add_mesh(name=f"obj_{o.obj_id}_mesh_{k}", file=os.path.abspath(part))
                body.add_geom(name=f"obj_{o.obj_id}_geom_{k}", type=mujoco.mjtGeom.mjGEOM_MESH,
                              meshname=f"obj_{o.obj_id}_mesh_{k}", mass=max(o.mass, 1e-4) / len(o.parts),
                              friction=[o.friction, 0.005, 0.0001])
        for a, b in self.exclude_pairs:
            spec.add_exclude(bodyname1=f"obj_{a}", bodyname2=f"obj_{b}")
        self.held_ids = [o.obj_id for o in snap.objects if o.held]
        if held_object == "weld":
            for oid in self.held_ids:
                spec.add_equality(name=f"held_{oid}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1=snap.hand_body, name2=f"obj_{oid}")

        self.model = spec.compile()
        self.version = next(_MODEL_VERSIONS)
        m = self.model
        self.hand_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, snap.hand_body)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        if min(jid) < 0:
            raise ValueError(f"[IMAGINATION] Robot model without joints {ARM_JOINTS}")
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in jid])
        fid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in FINGER_JOINTS]
        self.finger_qadr = np.array([m.jnt_qposadr[j] for j in fid if j >= 0], dtype=np.int64)
        self.arm_act = np.full(7, -1)
        self.gripper_act: List[int] = []
        finger_set = set(j for j in fid if j >= 0)
        for a in range(m.nu):
            trn = m.actuator_trnid[a, 0]
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                if trn in jid:
                    self.arm_act[jid.index(trn)] = a
                elif trn in finger_set:
                    self.gripper_act.append(a)
            elif m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
                adr, num = m.tendon_adr[trn], m.tendon_num[trn]
                if any(m.wrap_objid[adr + i] in finger_set for i in range(num)):
                    self.gripper_act.append(a)
        if np.any(self.arm_act < 0):
            raise ValueError("[IMAGINATION] The robot model has no position actuator on every arm joint")
        self.gripper_range = np.array([m.actuator_ctrlrange[a] for a in self.gripper_act]).reshape(-1, 2)
        finger_bodies = {int(m.jnt_bodyid[j]) for j in finger_set}
        self.finger_collides = any((m.geom_contype[g] or m.geom_conaffinity[g]) and int(m.geom_bodyid[g]) in finger_bodies
                                   for g in range(m.ngeom))

        self.obj_body = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{oid}") for oid in self.object_ids],
                                 dtype=np.int64)
        self.obj_qadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] if m.body_jntnum[b] else -1 for b in self.obj_body],
                                 dtype=np.int64)
        self.obj_dadr = np.array([m.jnt_dofadr[m.body_jntadr[b]] if m.body_jntnum[b] else -1 for b in self.obj_body],
                                 dtype=np.int64)
        self.obj_geoms = [np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == b], dtype=np.int64)
                          for b in self.obj_body]
        self.weld_eq = {oid: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"held_{oid}") for oid in self.held_ids
                        if held_object == "weld"}
        if held_object == "grasp" and grasp_force > 0:
            self._stiffen_gripper(float(grasp_force))
        self._compiled_mass = m.body_mass.copy()
        self._compiled_inertia = m.body_inertia.copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_inertia = m.body_inertia.copy()
        self.nominal_friction = m.geom_friction.copy()

    def _stiffen_gripper(self, grasp_force: float) -> None:
        """
        Position-servo fingers squeeze with kp * (opening - target): soft model grippers hold a few newtons on an
        object, while the real hand clamps with its grasp force. The servo saturates at grasp_force (shared by the
        gripper actuators) within 5 mm of closing.
        """
        m = self.model
        if not self.gripper_act:
            return
        force = grasp_force / len(self.gripper_act)
        for a in self.gripper_act:
            kp = -float(m.actuator_biasprm[a, 1])
            if kp <= 0.0:
                continue
            gain = max(1.0, force / (0.005 * kp))
            m.actuator_gainprm[a, 0] *= gain
            m.actuator_biasprm[a, 1] *= gain
            m.actuator_biasprm[a, 2] *= np.sqrt(gain)
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = [-force, force]

    def set_nominal(self, snap: SceneSnapshot) -> None:
        """Nominal mass and friction of this request (the model is cached across requests of the same scene)."""
        m = self.model
        for i, o in enumerate(snap.objects):
            b = self.obj_body[i]
            scale = max(o.mass, 1e-4) / self._compiled_mass[b] if self._compiled_mass[b] > 0 else 1.0
            m.body_mass[b] = self._compiled_mass[b] * scale
            m.body_inertia[b] = self._compiled_inertia[b] * scale
            m.geom_friction[self.obj_geoms[i], 0] = o.friction
        self.nominal_mass = m.body_mass.copy()
        self.nominal_inertia = m.body_inertia.copy()
        self.nominal_friction = m.geom_friction.copy()

    def initial_qpos(self, snap: SceneSnapshot) -> Tuple[np.ndarray, np.ndarray]:
        m = self.model
        qpos = m.qpos0.copy()
        qvel = np.zeros(m.nv)
        qpos[self.arm_qadr] = snap.q
        qvel[self.arm_dadr] = snap.qd
        qpos[self.finger_qadr] = snap.finger_opening
        for i, o in enumerate(snap.objects):
            if self.obj_qadr[i] >= 0:
                qpos[self.obj_qadr[i]:self.obj_qadr[i] + 3] = o.pos
                qpos[self.obj_qadr[i] + 3:self.obj_qadr[i] + 7] = o.quat_wxyz
                if not o.held:
                    qvel[self.obj_dadr[i]:self.obj_dadr[i] + 3] = o.velocity
        return qpos, qvel

    def deep_pairs(self, snap: SceneSnapshot, depth: float) -> Tuple[Tuple[int, int], ...]:
        """Object pairs interpenetrating deeper than depth [m] in the snapshot (wrong pose estimates)."""
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:], d.qvel[:] = self.initial_qpos(snap)
        mujoco.mj_forward(m, d)
        body_obj = {int(b): oid for b, oid in zip(self.obj_body, self.object_ids)}
        pairs = set()
        for c in d.contact[:d.ncon]:
            a = body_obj.get(int(m.geom_bodyid[c.geom1]))
            b = body_obj.get(int(m.geom_bodyid[c.geom2]))
            if a is not None and b is not None and a != b and c.dist < -depth:
                pairs.add((min(a, b), max(a, b)))
        return tuple(sorted(pairs))

    def set_welds(self, qpos: np.ndarray) -> None:
        """Weld of each held object = its current pose in the hand frame (anchor at the object origin)."""
        if not self.weld_eq:
            return
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        R_h, p_h = d.xmat[self.hand_body].reshape(3, 3), d.xpos[self.hand_body]
        for oid, eq in self.weld_eq.items():
            b = self.obj_body[self.object_ids.index(oid)]
            rel_pos = R_h.T @ (d.xpos[b] - p_h)
            rel_quat = np.zeros(4)
            q_h_inv = np.zeros(4)
            mujoco.mju_negQuat(q_h_inv, d.xquat[self.hand_body])
            mujoco.mju_mulQuat(rel_quat, q_h_inv, d.xquat[b])
            m.eq_data[eq, :] = 0.0
            m.eq_data[eq, 3:6] = rel_pos
            m.eq_data[eq, 6:10] = rel_quat
            m.eq_data[eq, 10] = 1.0

    def control_table(self, snap: SceneSnapshot, req: ImaginationRequest, held_any: bool) -> Tuple[np.ndarray, int]:
        """[S, steps, nu] actuator controls; arm targets interpolated linearly over each control step."""
        m = self.model
        S, T, _ = req.arm_targets.shape
        sub = max(1, int(round(req.control_dt / m.opt.timestep)))
        targets = np.asarray(req.arm_targets, dtype=np.float64) + snap.joint_angle_offsets[None, None]
        prev = np.concatenate([np.broadcast_to(snap.q, (S, 1, 7)), targets[:, :-1]], axis=1)
        frac = (np.arange(1, sub + 1) / sub)[None, None, :, None]
        arm = (prev[:, :, None] + frac * (targets - prev)[:, :, None]).reshape(S, T * sub, 7)
        ctrl = np.zeros((S, T * sub, m.nu))
        ctrl[:, :, self.arm_act] = arm
        if self.gripper_act:
            if req.gripper is not None:
                amount = np.repeat(np.clip(np.asarray(req.gripper, dtype=np.float64), 0.0, 1.0), sub, axis=1)
            else:
                now = 0.0 if (held_any and self.held_mode == "grasp") else snap.finger_opening / FINGER_OPENING
                amount = np.full((S, T * sub), now)
            lo, hi = self.gripper_range[:, 0], self.gripper_range[:, 1]
            ctrl[:, :, self.gripper_act] = lo[None, None] + amount[..., None] * (hi - lo)[None, None]
        return ctrl, sub


class MujocoRolloutBackend:
    """mujoco.rollout over CPU threads; the physical worlds run one after the other on the same model."""

    name = "mujoco_c"

    def __init__(self, threads: int):
        self.threads = max(1, int(threads))
        self._datas: Dict[int, List[mujoco.MjData]] = {}

    def run(self, im: ImaginationModel, mass: np.ndarray, inertia: np.ndarray, friction: np.ndarray,
            qpos0: np.ndarray, qvel0: np.ndarray, ctrl: np.ndarray, record_steps: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        from mujoco import rollout
        m = im.model
        R = mass.shape[0]
        N, steps, _ = ctrl.shape
        key = im.version
        if key not in self._datas:
            self._datas = {key: [mujoco.MjData(m) for _ in range(self.threads)]}
        datas = self._datas[key]
        spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        nstate = mujoco.mj_stateSize(m, spec)
        out = np.zeros((N, len(record_steps), m.nq))
        valid = np.ones(N, dtype=bool)
        saved = (m.body_mass.copy(), m.body_inertia.copy(), m.geom_friction.copy())
        try:
            for r in range(R):
                worlds = np.arange(r, N, R)
                m.body_mass[:] = mass[r]
                m.body_inertia[:] = inertia[r]
                m.geom_friction[:] = friction[r]
                d = datas[0]
                x = np.zeros((len(worlds), nstate))
                for i, w in enumerate(worlds):
                    mujoco.mj_resetData(m, d)
                    d.qpos[:] = qpos0[w]
                    d.qvel[:] = qvel0[w]
                    mujoco.mj_getState(m, d, x[i], spec)
                start = 0
                ok = np.ones(len(worlds), dtype=bool)
                for j, stop in enumerate(record_steps):
                    if stop > start:
                        state, _ = rollout.rollout(m, datas, x, ctrl[worlds, start:stop], persistent_pool=True)
                        x = state[:, -1].copy()
                        start = stop
                    # MuJoCo resets a diverging world (bad acceleration) and carries on from qpos0: time restarts.
                    ok &= np.abs(x[:, 0] - stop * m.opt.timestep) < 0.5 * m.opt.timestep
                    out[worlds, j] = x[:, 1:1 + m.nq]
                valid[worlds] = ok & np.all(np.isfinite(x), axis=1)
        finally:
            m.body_mass[:], m.body_inertia[:], m.geom_friction[:] = saved
        return out, valid


class MjWarpRolloutBackend:
    """MuJoCo Warp: all worlds stepped together; on CUDA one step is a captured graph."""

    name = "mjwarp"

    def __init__(self, device: str, nconmax: int, njmax: int):
        import warp as wp
        import mujoco_warp as mjw
        self.wp, self.mjw = wp, mjw
        self.device = device
        self.nconmax, self.njmax = int(nconmax), int(njmax)
        self._stream = wp.Stream(device) if wp.get_device(device).is_cuda else None
        self._model_key = None
        self._mw = None
        self._data_key = None
        self._dw = None
        self._graph = None
        self.overflow_reported = False

        @wp.kernel
        def _set_ctrl(table: wp.array3d(dtype=float), t: wp.array(dtype=int), ctrl: wp.array2d(dtype=float)):
            w, j = wp.tid()
            ctrl[w, j] = table[w, t[0], j]

        @wp.kernel
        def _advance(t: wp.array(dtype=int)):
            t[0] = t[0] + 1

        self._set_ctrl, self._advance = _set_ctrl, _advance

    def _model(self, im: ImaginationModel, R: int):
        key = (im.version, R)
        if key != self._model_key:
            with self.wp.ScopedDevice(self.device):
                self._mw = self.mjw.put_model(im.model, batch_sizes={"body_mass": R, "body_inertia": R,
                                                                     "geom_friction": R})
                self._mw.opt.warn_overflow = 0
            self._model_key = key
            self._data_key = None
        return self._mw

    def _data(self, im: ImaginationModel, N: int, steps: int, scale: int):
        key = (im.version, N, steps, scale)
        if key != self._data_key:
            wp = self.wp
            with wp.ScopedDevice(self.device):
                self._dw = self.mjw.make_data(im.model, nworld=N, nconmax=self.nconmax * scale, njmax=self.njmax * scale)
                self._table = wp.zeros((N, steps, im.model.nu), dtype=float)
                self._t = wp.zeros(1, dtype=int)
            self._graph = None
            self._data_key = key
        return self._dw

    def _reset(self, mw, dw) -> None:
        self.mjw.reset_data(mw, dw)
        dw.overflow.zero_()

    def _step(self, mw, dw, N: int, nu: int) -> None:
        wp = self.wp
        if self._graph is not None:
            wp.capture_launch(self._graph)
            return
        wp.launch(self._set_ctrl, dim=(N, nu), inputs=[self._table, self._t, dw.ctrl], device=self.device)
        self.mjw.step(mw, dw)
        wp.launch(self._advance, dim=1, inputs=[self._t], device=self.device)

    def run(self, im: ImaginationModel, mass: np.ndarray, inertia: np.ndarray, friction: np.ndarray,
            qpos0: np.ndarray, qvel0: np.ndarray, ctrl: np.ndarray, record_steps: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        wp = self.wp
        m = im.model
        R = mass.shape[0]
        N, steps, nu = ctrl.shape
        for scale in (1, 2):
            mw = self._model(im, R)
            with wp.ScopedDevice(self.device), (wp.ScopedStream(self._stream) if self._stream else nullcontext()):
                mw.body_mass.assign(mass.astype(np.float32))
                mw.body_inertia.assign(inertia.astype(np.float32))
                mw.geom_friction.assign(friction.astype(np.float32))
                if m.neq:
                    eq = np.zeros((1, m.neq, mw.eq_data.dtype._length_), dtype=np.float32)
                    n = min(eq.shape[2], m.eq_data.shape[1])
                    eq[0, :, :n] = m.eq_data[:, :n]
                    mw.eq_data.assign(eq)
                dw = self._data(im, N, steps, scale)
                self._reset(mw, dw)
                dw.qpos.assign(qpos0.astype(np.float32))
                dw.qvel.assign(qvel0.astype(np.float32))
                self._table.assign(ctrl.astype(np.float32))
                self._t.zero_()
                if self._graph is None and wp.get_device(self.device).is_cuda:
                    self._step(mw, dw, N, nu)
                    with WARP_LOCK, wp.ScopedCapture() as capture:
                        self._step(mw, dw, N, nu)
                    self._graph = capture.graph
                    self._reset(mw, dw)
                    dw.qpos.assign(qpos0.astype(np.float32))
                    dw.qvel.assign(qvel0.astype(np.float32))
                    self._t.zero_()
                out = np.zeros((N, len(record_steps), m.nq))
                start = 0
                for j, stop in enumerate(record_steps):
                    for _ in range(stop - start):
                        self._step(mw, dw, N, nu)
                    start = stop
                    out[:, j] = dw.qpos.numpy()
                overflow = dw.overflow.numpy() != 0
            if not overflow.any() or scale == 2:
                break
        if overflow.any() and not self.overflow_reported:
            print(f"[IMAGINATION WARNING] contact/constraint buffers overflowed in {int(overflow.sum())} worlds even "
                  f"with doubled buffers: increase imagination.nconmax_per_world / njmax_per_world")
            self.overflow_reported = True
        valid = ~overflow & np.all(np.isfinite(out[:, -1]), axis=1)
        return out, valid


class Imagination:
    """Builds and caches the imagination model of a twin snapshot and runs the requested rollouts."""

    def __init__(self, backend: str = "mujoco_c", device: str = "cpu", held_object: str = "weld",
                 timestep: Optional[float] = None, cpu_threads: int = 4, nconmax_per_world: int = 128,
                 njmax_per_world: int = 1500, mass_scale: Sequence[float] = (0.7, 1.3),
                 friction_scale: Sequence[float] = (0.7, 1.3), position_noise: float = 0.0,
                 penetration_exclude_depth: float = 0.005, grasp_force: float = 70.0):
        if held_object not in ("weld", "grasp"):
            raise ValueError(f"[IMAGINATION] held_object must be 'weld' or 'grasp', got '{held_object}'")
        self.held_object = held_object
        self.timestep = timestep
        self.mass_scale = tuple(float(x) for x in mass_scale)
        self.friction_scale = tuple(float(x) for x in friction_scale)
        self.position_noise = float(position_noise)
        self.penetration_exclude_depth = float(penetration_exclude_depth)
        self.grasp_force = float(grasp_force)
        if backend == "mjwarp":
            self.backend = MjWarpRolloutBackend(device, nconmax_per_world, njmax_per_world)
        elif backend == "mujoco_c":
            self.backend = MujocoRolloutBackend(cpu_threads)
        else:
            raise ValueError(f"[IMAGINATION] Unknown backend '{backend}' (expected 'mjwarp' or 'mujoco_c')")
        self.device = device if backend == "mjwarp" else "cpu"
        self._key = None
        self._base: Optional[ImaginationModel] = None
        self._model: Optional[ImaginationModel] = None
        self._grasp_warned = False
        self._excluded_reported: set = set()

    def _scene_key(self, snap: SceneSnapshot, mode: str) -> tuple:
        return (snap.robot_model_path, tuple(np.round(snap.base_pos, 6)), tuple(np.round(snap.base_quat_wxyz, 6)),
                snap.table_bounds, snap.table_mesh, snap.table_z, mode,
                tuple((o.obj_id, tuple(o.parts), bool(o.static), bool(o.held)) for o in snap.objects))

    def model_for(self, snap: SceneSnapshot) -> ImaginationModel:
        """
        Model of the scene, cached until objects, held objects or table change. Object pairs that interpenetrate
        deeper than penetration_exclude_depth at the start do not collide with each other: such overlaps come from
        wrong pose estimates, and the contact solver would push the objects apart violently.
        """
        mode = self.held_object
        key = self._scene_key(snap, mode)
        if key != self._key:
            base = ImaginationModel(snap, mode, self.timestep, grasp_force=self.grasp_force)
            if mode == "grasp" and any(o.held for o in snap.objects) and not base.finger_collides:
                if not self._grasp_warned:
                    print("[IMAGINATION WARNING] held_object 'grasp' but the robot fingers have no collision geometry "
                          "in this model: the held object is welded instead")
                    self._grasp_warned = True
                base = ImaginationModel(snap, "weld", self.timestep)
            self._base, self._model, self._key = base, base, key
        pairs = self._base.deep_pairs(snap, self.penetration_exclude_depth) if self.penetration_exclude_depth > 0 else ()
        if pairs != self._model.exclude_pairs:
            self._model = ImaginationModel(snap, self._base.held_mode, self.timestep, pairs,
                                           grasp_force=self.grasp_force) if pairs else self._base
            new = set(pairs) - self._excluded_reported
            if new:
                print(f"[IMAGINATION] object pairs interpenetrating more than {1000 * self.penetration_exclude_depth:.0f} mm "
                      f"do not collide with each other: {sorted(new)}")
                self._excluded_reported |= new
        self._model.set_nominal(snap)
        return self._model

    def run(self, snap: SceneSnapshot, req: ImaginationRequest) -> ImaginationResult:
        t0 = time.perf_counter()
        im = self.model_for(snap)
        m = im.model
        S, T, _ = np.asarray(req.arm_targets).shape
        R = max(1, int(req.worlds_per_sample))
        N = S * R
        O = len(im.object_ids)
        rng = np.random.default_rng(req.seed)

        mass_scale = np.ones((R, O))
        friction_scale = np.ones((R, O))
        if R > 1:
            mass_scale[1:] = rng.uniform(*self.mass_scale, size=(R - 1, O))
            friction_scale[1:] = rng.uniform(*self.friction_scale, size=(R - 1, O))
        mass = np.repeat(im.nominal_mass[None], R, axis=0)
        inertia = np.repeat(im.nominal_inertia[None], R, axis=0)
        friction = np.repeat(im.nominal_friction[None], R, axis=0)
        for i in range(O):
            b = im.obj_body[i]
            mass[:, b] *= mass_scale[:, i]
            inertia[:, b] *= mass_scale[:, i, None]
            friction[:, im.obj_geoms[i], 0] *= friction_scale[:, i, None]

        qpos, qvel = im.initial_qpos(snap)
        im.set_welds(qpos)
        qpos0 = np.repeat(qpos[None], N, axis=0)
        qvel0 = np.repeat(qvel[None], N, axis=0)
        if self.position_noise > 0.0:
            noisy = np.arange(N) % R != 0
            for i, o in enumerate(snap.objects):
                if im.obj_qadr[i] >= 0 and not o.held:
                    qpos0[noisy, im.obj_qadr[i]:im.obj_qadr[i] + 2] += rng.normal(0.0, self.position_noise,
                                                                                (int(noisy.sum()), 2))

        held_any = any(o.held for o in snap.objects)
        ctrl_s, sub = im.control_table(snap, req, held_any)
        ctrl = np.repeat(ctrl_s, R, axis=0)
        steps = ctrl.shape[1]
        every = int(req.record_every)
        record_ctrl = list(range(every, T, every)) if every > 0 else []
        record_steps = [c * sub for c in record_ctrl] + [steps]

        out, valid = self.backend.run(im, mass, inertia, friction, qpos0, qvel0, ctrl, record_steps)

        n_rec = len(record_steps)
        obj_pos = np.zeros((N, n_rec, O, 3))
        obj_quat = np.zeros((N, n_rec, O, 4))
        for i, o in enumerate(snap.objects):
            a = im.obj_qadr[i]
            if a >= 0:
                obj_pos[:, :, i] = out[:, :, a:a + 3]
                obj_quat[:, :, i] = _wxyz_to_xyzw(out[:, :, a + 3:a + 7])
            else:
                obj_pos[:, :, i] = o.pos
                obj_quat[:, :, i] = _wxyz_to_xyzw(o.quat_wxyz)
        norm = np.linalg.norm(obj_quat, axis=-1, keepdims=True)
        obj_quat = np.where(norm > 1e-9, obj_quat / np.maximum(norm, 1e-12), obj_quat)
        arm_q = out[:, :, im.arm_qadr] - snap.joint_angle_offsets[None, None]
        return ImaginationResult(
            object_ids=list(im.object_ids),
            times=np.asarray(record_steps, dtype=np.float64) * m.opt.timestep,
            object_pos=obj_pos.reshape(S, R, n_rec, O, 3),
            object_quat=obj_quat.reshape(S, R, n_rec, O, 4),
            arm_q=arm_q.reshape(S, R, n_rec, 7),
            valid=valid.reshape(S, R),
            mass_scale=mass_scale,
            friction_scale=friction_scale,
            initial_pos=np.array([o.pos for o in snap.objects]).reshape(O, 3),
            initial_quat=_wxyz_to_xyzw(np.array([o.quat_wxyz for o in snap.objects]).reshape(O, 4)),
            backend=self.backend.name,
            wall_time_s=time.perf_counter() - t0,
            info={"worlds": N, "steps": steps, "timestep": float(m.opt.timestep), "device": self.device,
                  "held_object": im.held_mode, "excluded_pairs": list(im.exclude_pairs)},
        )
