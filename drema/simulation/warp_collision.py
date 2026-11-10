"""
MPPI collision queries on MuJoCo Warp, with the semantics of MuJoCoDigitalTwin._collision_records.

Every evaluated configuration (rollout k, step e) is one world with the robot, the forecast obstacle poses and the
held objects placed as in the C loop. Contacts with margin act as distance queries: a contact exists when the pair is
closer than its margin, and its dist is the signed distance mj_geomDistance returns (float32). Robot links and held
objects collide with the obstacles through contype/conaffinity bits. In the query model a held object is a free body
placed through its qpos: as a mocap body, like the obstacles, MuJoCo Warp would only report its penetrations.
Obstacle margins cover the largest per-pair margin and the records are then filtered per pair as in the C loop.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import mujoco

PROBE_BIT = 2


class WarpCollisionQuery:

    def __init__(self, device: str, contacts_per_world: int = 32):
        import warp as wp
        import mujoco_warp as mjw
        self.wp, self.mjw = wp, mjw
        self.device = device
        self.contacts_per_world = int(contacts_per_world)
        self._key = None
        self._mjm: Optional[mujoco.MjModel] = None
        self._mw = None
        self._mocap_of: Dict[int, int] = {}
        self._held_qadr: Dict[int, int] = {}
        self._data_key = None
        self._dw = None

    def _build(self, xml: str, present: mujoco.MjModel, probes: list, obstacles: list, held_ids: List[int]) -> None:
        spec = mujoco.MjSpec.from_string(xml)
        for oid in held_ids:
            body = spec.body(f"obs_{oid}")
            body.mocap = False
            body.add_freejoint()
        m = spec.compile()
        if m.ngeom != present.ngeom or np.any(m.geom_bodyid != present.geom_bodyid) \
                or np.any(m.geom_type != present.geom_type):
            raise RuntimeError("[WARP COLLISION] query model geoms differ from the twin model")
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
        m.geom_contype[[pr[0] for pr in probes]] = PROBE_BIT
        m.geom_conaffinity[[g for g, _ in obstacles]] = PROBE_BIT
        m.geom_margin[:] = 0.0
        m.geom_gap[:] = 0.0
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        self._mocap_of, self._held_qadr = {}, {}
        for b in range(m.nbody):
            name = m.body(b).name
            if not name.startswith("obs_"):
                continue
            try:
                oid = int(name.split("_")[1])
            except ValueError:
                continue
            if m.body_mocapid[b] >= 0:
                self._mocap_of[oid] = int(m.body_mocapid[b])
            elif oid in held_ids:
                self._held_qadr[oid] = int(m.jnt_qposadr[m.body_jntadr[b]])
        self._mjm = m
        with self.wp.ScopedDevice(self.device):
            self._mw = self.mjw.put_model(m)
        self._data_key = None

    def _data(self, nworld: int, cpw: int):
        key = (nworld, cpw)
        if key != self._data_key:
            with self.wp.ScopedDevice(self.device):
                self._dw = self.mjw.make_data(self._mjm, nworld=nworld, nconmax=cpw, njmax=1)
            self._data_key = key
        return self._dw

    def records(self, twin: Any, Q: np.ndarray, eval_steps: List[int], predictions: Dict[int, Any], probes: list,
                obstacles: list, allowed: np.ndarray, margins: np.ndarray, sigma_2: float,
                record_points: bool) -> Tuple[Dict[str, Any], float]:
        """Same records as MuJoCoDigitalTwin._collision_records, as arrays; plus the Warp time [s]."""
        import time
        wp, mjw = self.wp, self.mjw
        model, data = twin.model, twin.data
        K = Q.shape[0]
        E = len(eval_steps)
        W = K * E
        obstacle_geoms = [g for g, _ in obstacles]
        held_ids = sorted({pr[5] for pr in probes if pr[5] is not None})

        key = (twin.model_version, tuple(pr[0] for pr in probes), tuple(obstacle_geoms))
        if key != self._key:
            self._build(twin.query_xml(), model, probes, obstacles, held_ids)
            self._key = key
        mjm = self._mjm

        probe_of = np.full(model.ngeom, -1, dtype=np.int64)
        probe_of[[pr[0] for pr in probes]] = np.arange(len(probes))
        obst_of = np.full(model.ngeom, -1, dtype=np.int64)
        obst_of[obstacle_geoms] = np.arange(len(obstacle_geoms))
        geom_margin = np.zeros(model.ngeom, dtype=np.float32)
        reach = np.where(allowed, sigma_2 + margins, -np.inf)
        if len(probes) and len(obstacle_geoms):
            geom_margin[obstacle_geoms] = np.maximum(0.0, reach.max(axis=0))

        qpos = np.repeat(mjm.qpos0[None], W, axis=0)
        qpos[:, :model.nq] = data.qpos[None, :model.nq]
        qpos[:, :7] = Q[:, eval_steps].reshape(W, 7) + twin.joint_angle_offsets
        mocap_pos = np.repeat(data.mocap_pos[None], W, axis=0)
        mocap_quat = np.repeat(data.mocap_quat[None], W, axis=0)
        e_of_world = np.tile(np.arange(E), K)
        h = np.asarray(eval_steps)
        for obj_id, pred in predictions.items():
            if obj_id in twin.attachments:
                continue
            idx = twin._mocap_index(obj_id)
            if idx >= 0:
                mocap_pos[:, idx] = np.asarray(pred.positions, dtype=np.float64)[h][e_of_world]
                quat = np.asarray(pred.orientations, dtype=np.float64)[h]
                mocap_quat[:, idx] = np.concatenate([quat[:, 3:4], quat[:, :3]], axis=1)[e_of_world]
        q_mocap_pos = np.zeros((W, mjm.nmocap, 3))
        q_mocap_quat = np.zeros((W, mjm.nmocap, 4))
        for oid, qidx in self._mocap_of.items():
            pidx = twin._mocap_index(oid)
            q_mocap_pos[:, qidx] = mocap_pos[:, pidx]
            q_mocap_quat[:, qidx] = mocap_quat[:, pidx]
        if held_ids:
            for obj_id, (pos, quat) in twin.attached_poses(Q[:, eval_steps].reshape(-1, 7)).items():
                a = self._held_qadr.get(obj_id)
                if a is not None:
                    qpos[:, a:a + 3] = pos
                    qpos[:, a + 3:a + 7] = np.concatenate([quat[:, 3:4], quat[:, :3]], axis=1)

        from .imagination import WARP_LOCK
        t0 = time.perf_counter()
        cpw = self.contacts_per_world
        while True:
            dw = self._data(W, cpw)
            with WARP_LOCK, wp.ScopedDevice(self.device):
                self._mw.geom_margin.assign(geom_margin[None])
                dw.qpos.assign(qpos.astype(np.float32))
                if mjm.nmocap:
                    dw.mocap_pos.assign(q_mocap_pos.astype(np.float32))
                    dw.mocap_quat.assign(q_mocap_quat.astype(np.float32))
                mjw.kinematics(self._mw, dw)
                mjw.collision(self._mw, dw)
                n = int(dw.nacon.numpy()[0])
            if n <= dw.naconmax:
                break
            cpw = max(2 * cpw, int(np.ceil(1.25 * n / W)))
            self.contacts_per_world = cpw
        dist = dw.contact.dist.numpy()[:n].astype(np.float64)
        geom = dw.contact.geom.numpy()[:n]
        world = dw.contact.worldid.numpy()[:n]
        frame = dw.contact.frame.numpy()[:n]
        pos = dw.contact.pos.numpy()[:n].astype(np.float64)
        t_warp = time.perf_counter() - t0

        g0, g1 = geom[:, 0], geom[:, 1]
        probe_first = probe_of[g0] >= 0
        rg = np.where(probe_first, g0, g1)
        og = np.where(probe_first, g1, g0)
        ri, oi = probe_of[rg], obst_of[og]
        keep = (ri >= 0) & (oi >= 0)
        keep[keep] &= allowed[ri[keep], oi[keep]]
        d = dist - np.where(keep, margins[np.maximum(ri, 0), np.maximum(oi, 0)], 0.0)
        keep &= d < sigma_2
        idx = np.nonzero(keep)[0]
        # one record per (world, probe, obstacle geom): the closest contact, as mj_geomDistance returns
        order = idx[np.lexsort((d[idx], oi[idx], ri[idx], world[idx]))]
        pair_id = (world[order] * len(probes) + ri[order]) * len(obstacle_geoms) + oi[order]
        first = np.ones(len(order), dtype=bool)
        first[1:] = pair_id[1:] != pair_id[:-1]
        sel = order[first]

        n_vec = frame[sel, 0, :].astype(np.float64)
        pf = probe_first[sel]
        obs_to_robot = np.where(pf[:, None], -n_vec, n_vec) * np.abs(dist[sel])[:, None]
        robot_point = pos[sel] + np.where(pf[:, None], -1.0, 1.0) * n_vec * dist[sel, None] / 2.0
        w_sel = world[sel]
        k_sel, e_sel = w_sel // E, w_sel % E
        ri_sel, oi_sel = ri[sel], oi[sel]

        slots: Dict[int, int] = {}
        geom_slot = np.array([slots.setdefault(obj_id, len(slots)) for _, obj_id in obstacles], dtype=np.int64)
        rbody = np.array([pr[1] for pr in probes], dtype=np.int64)
        vel_slot = np.array([pr[2] for pr in probes], dtype=np.int64)
        obj_of_geom = np.array([obj_id for _, obj_id in obstacles], dtype=np.int64)
        v_table = np.zeros((E, len(obstacle_geoms), 3))
        for j, obj_id in enumerate(obj_of_geom):
            if obj_id >= 0 and obj_id in predictions:
                v_table[:, j] = np.asarray(predictions[obj_id].velocities, dtype=np.float64)[eval_steps]

        rec = {
            "k": k_sel, "e": e_sel,
            "group": geom_slot[oi_sel] * model.nbody + rbody[ri_sel],
            "slot": vel_slot[ri_sel],
            "d": d[sel],
            "normal": obs_to_robot,
            "v_obs": v_table[e_sel, oi_sel],
            "point": robot_point if record_points else np.zeros((0, 3)),
            "obj": obj_of_geom[oi_sel] if record_points else np.zeros(0, dtype=np.int64),
        }
        return rec, t_warp
