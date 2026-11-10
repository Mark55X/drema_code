"""
MuJoCo twin with the batched work on MuJoCo Warp.

The present state and the per-cycle queries (proximity, closest points, contact check) stay on MuJoCo C: one
world, answers needed at once. Two batched workloads go to Warp:
- the MPPI collision queries (one world per rollout configuration), same records as the C loop;
- the physics imagination (drema.simulation.imagination): rollouts of many joint-target sequences in randomized
  physical worlds, run synchronously or in a background thread.
With device 'auto' and no CUDA device, both use the C paths (MuJoCoDigitalTwin's loop and mujoco.rollout).
"""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

import numpy as np
import mujoco

from .mujoco_digital_twin import MuJoCoDigitalTwin
from .imagination import Imagination, ImaginationRequest, ImaginationResult, ImaginedObject, SceneSnapshot


def resolve_warp_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import warp as wp
        return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


class MuJoCoWarpDigitalTwin(MuJoCoDigitalTwin):

    def __init__(self, *args, warp_device: str = "auto", collision_backend: str = "auto",
                 contacts_per_world: int = 32, imagination: Optional[Dict[str, Any]] = None, **kwargs):
        """
        :param warp_device: Warp device ('auto': first CUDA device, else 'cpu').
        :param collision_backend: MPPI collision queries on 'mjwarp' or 'mujoco_c' ('auto': mjwarp on CUDA only).
        :param contacts_per_world: initial contact buffer per collision world (grown on overflow).
        :param imagination: digital_twin.imagination config (backend, held_object, timestep, cpu_threads,
            nconmax_per_world, njmax_per_world, default_mass, default_friction, randomization).
        """
        self.model_version = 0
        self._warp_query = None
        self._warp_failed = False
        self._imagination_engine: Optional[Imagination] = None
        self._imagination_pool: Optional[ThreadPoolExecutor] = None
        self._imagination_cfg = dict(imagination or {})
        self._contacts_per_world = int(contacts_per_world)
        super().__init__(*args, **kwargs)
        self.warp_device = resolve_warp_device(warp_device)
        on_gpu = self.warp_device.startswith("cuda")
        self.collision_backend = ("mjwarp" if on_gpu else "mujoco_c") if collision_backend == "auto" else collision_backend
        backend = str(self._imagination_cfg.get("backend", "auto"))
        self.imagination_backend = ("mjwarp" if on_gpu else "mujoco_c") if backend == "auto" else backend
        print(f"[MUJOCO WARP DIGITAL TWIN] Warp device: {self.warp_device} | MPPI collision queries: "
              f"{self.collision_backend} | physics imagination: {self.imagination_backend} "
              f"(held object: {self._imagination_cfg.get('held_object', 'weld')})")

    def _rebuild_model(self) -> None:
        super()._rebuild_model()
        self.model_version += 1

    def _mocap_index(self, obj_id: int) -> int:
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{obj_id}")
        if b < 0:
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs_{obj_id}_mocap")
        return int(self.model.body_mocapid[b]) if b >= 0 else -1

    def query_xml(self) -> str:
        with self.state_lock:
            return self._generate_mjcf_xml()

    # -------------------------------------------------------------------------
    # MPPI collision queries
    # -------------------------------------------------------------------------

    def _collision_backend_label(self) -> str:
        return "MuJoCo Warp" if self.collision_backend == "mjwarp" and not self._warp_failed else "MuJoCo"

    def _collision_records(self, Q, eval_steps, predictions, probes, obstacles, allowed, margins, sigma_2,
                           eval_data, record_points):
        if self.collision_backend != "mjwarp" or self._warp_failed:
            return super()._collision_records(Q, eval_steps, predictions, probes, obstacles, allowed, margins,
                                              sigma_2, eval_data, record_points)
        try:
            if self._warp_query is None:
                from .warp_collision import WarpCollisionQuery
                self._warp_query = WarpCollisionQuery(self.warp_device, self._contacts_per_world)
            return self._warp_query.records(self, Q, eval_steps, predictions, probes, obstacles, allowed, margins,
                                            sigma_2, record_points)
        except Exception as e:
            # The model can change between the XML and the compiled model (a spawn in between): this call uses C.
            print(f"[MUJOCO WARP DIGITAL TWIN WARNING] Warp collision query failed ({e!r}); using MuJoCo C for it")
            self._warp_query = None
            return super()._collision_records(Q, eval_steps, predictions, probes, obstacles, allowed, margins,
                                              sigma_2, eval_data, record_points)

    # -------------------------------------------------------------------------
    # Physics imagination
    # -------------------------------------------------------------------------

    def imagination_snapshot(self) -> SceneSnapshot:
        """The present scene as the imagination model needs it (copied under the twin lock)."""
        cfg = self._imagination_cfg
        default_mass = float(cfg.get("default_mass", 1.0))
        default_friction = float(cfg.get("default_friction", 0.5))
        with self.state_lock:
            if not self.robot_loaded or self.data is None:
                raise RuntimeError("[MUJOCO WARP DIGITAL TWIN] imagination needs the robot loaded in the twin")
            objects = []
            for oid, obj in self.tracked_objects.items():
                phys = self.object_physics.get(int(oid), {})
                mass = phys.get("mass")
                friction = phys.get("friction")
                objects.append(ImaginedObject(
                    obj_id=int(oid), name=str(obj.get("name", oid)),
                    parts=list(obj.get("collision_parts") or [obj["mesh_path"]]),
                    pos=np.array(obj["target_pos"], dtype=np.float64),
                    quat_wxyz=np.array(obj["target_quat"], dtype=np.float64),
                    velocity=np.array(self._estimated_velocity(oid), dtype=np.float64),
                    mass=float(mass) if mass and mass > 0 else default_mass,
                    friction=float(friction) if friction and friction > 0 else default_friction,
                    static=bool(phys.get("static", False)),
                    held=oid in self.attachments))
            return SceneSnapshot(
                robot_model_path=self.robot_model_path,
                base_pos=self.robot_base_pos.copy(),
                base_quat_wxyz=self.robot_base_quat.copy(),
                hand_body=self.hand_bodies[0],
                q=self.data.qpos[:7].copy(),
                qd=self.data.qvel[:7].copy(),
                finger_opening=float(self.finger_opening),
                joint_angle_offsets=self.joint_angle_offsets.copy(),
                objects=objects,
                table_bounds=tuple(self.table_bounds) if self.table_spawned and self.table_bounds is not None else None,
                table_mesh=self.table_mesh_path if self.table_spawned and self.table_bounds is None else None,
                table_z=float(self.table_z))

    def _engine(self) -> Imagination:
        if self._imagination_engine is None:
            cfg = self._imagination_cfg
            rnd = dict(cfg.get("randomization") or {})
            self._imagination_engine = Imagination(
                backend=self.imagination_backend, device=self.warp_device,
                held_object=str(cfg.get("held_object", "weld")), timestep=cfg.get("timestep"),
                cpu_threads=int(cfg.get("cpu_threads", 4)),
                nconmax_per_world=int(cfg.get("nconmax_per_world", 128)),
                njmax_per_world=int(cfg.get("njmax_per_world", 1500)),
                mass_scale=rnd.get("mass_scale", (0.7, 1.3)), friction_scale=rnd.get("friction_scale", (0.7, 1.3)),
                position_noise=float(rnd.get("position_noise", 0.0)),
                penetration_exclude_depth=float(cfg.get("penetration_exclude_depth", 0.005)),
                grasp_force=float(cfg.get("grasp_force", 70.0)))
        return self._imagination_engine

    def imagine(self, request: ImaginationRequest, snapshot: Optional[SceneSnapshot] = None) -> ImaginationResult:
        """Rolls out the request from the present scene (or the given snapshot) and returns the outcomes."""
        return self._engine().run(snapshot if snapshot is not None else self.imagination_snapshot(), request)

    def imagine_async(self, request: ImaginationRequest) -> "Future[ImaginationResult]":
        """Same as imagine, in a background thread; the scene is captured now."""
        snapshot = self.imagination_snapshot()
        if self._imagination_pool is None:
            self._imagination_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imagination")
        return self._imagination_pool.submit(self.imagine, request, snapshot)

    def shutdown(self) -> None:
        if self._imagination_pool is not None:
            self._imagination_pool.shutdown(wait=False, cancel_futures=True)
            self._imagination_pool = None
        super().shutdown()


__all__ = ["MuJoCoWarpDigitalTwin", "resolve_warp_device", "ImaginationRequest", "ImaginationResult"]
