#!/usr/bin/env python
"""
DREMA Dynamic System Orchestrator (run_drema_dynamic_system.py)

Agnostic & Modular Orchestrator:
- Decoupled architecture with BasePerceptionModule (e.g. VG-Mapping + RecurGS)
  and BaseDigitalTwin (e.g. PyBullet, Isaac Sim).
- Real-time gRPC communication with Simulation Environment or physical robot.
- Real-time 3D Viser Web Visualizer.
- Supports instant restoration of t=0 initial scan from disk cache (--cache / --load_cache).
- Fully configurable via configs/drema_default.yaml and CLI overrides.
"""

import os
import sys
import time
import queue
import signal
import threading
import argparse
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import trimesh
import viser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from drema.config import load_config, ConfigDict
from drema.communication.grpc_server import DremaGrpcServer
from drema.communication.grpc_client import unpack_camera_frame
from drema.communication.proto import drema_comm_pb2
from drema.simulation.base_twin import BaseDigitalTwin
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
from drema.perception.base_perception import BasePerceptionModule, InitialScanResult, StreamingUpdateResult
from drema.perception.vg_mapping_perception import VGMappingPerceptionModule
from drema.controller.mpc_controller import MPCController


def pointcloud_from_depth_and_camera_params(depth: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Exact equivalent of PyRep VisionSensor.pointcloud_from_depth_and_camera_params (0.0 error)."""
    h, w = depth.shape
    u = np.tile(np.arange(w), [h, 1]).astype(np.float32)
    v = np.tile(np.arange(h)[:, None], [1, w]).astype(np.float32)
    upc = np.stack([u, v, np.ones_like(u)], axis=-1)
    pc = upc * np.expand_dims(depth, -1)

    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T
    R_inv_C = np.matmul(R_inv, C)
    extrinsics_conv = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics_conv)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]

    pixel_coords = np.concatenate([pc, np.ones((h, w, 1))], -1)
    coords = np.reshape(pixel_coords, (h * w, -1))
    coords = np.transpose(coords, (1, 0))
    transformed_coords_vector = np.matmul(cam_proj_mat_inv, coords)
    transformed_coords_vector = np.transpose(transformed_coords_vector, (1, 0))
    return np.reshape(transformed_coords_vector, (h, w, 3))


class DremaDynamicSystem:
    """
    Main Orchestrator coordinating Perception, Digital Twin simulation,
    MPC Control, and real-time visualization.
    """

    def __init__(
        self,
        config: Optional[ConfigDict] = None,
        port: Optional[int] = None,
        visualize_digital_twin: Optional[bool] = None,
        visualize_viser: Optional[bool] = None,
        viser_port: Optional[int] = None,
        table_z_prior: Optional[float] = None,
        reachability_radius: Optional[float] = None,
        voxel_size: Optional[float] = None,
        device: Optional[str] = None,
        log_interval_actions: Optional[int] = None,
        **kwargs
    ):
        if config is None:
            config = load_config("configs/drema_default.yaml")

        if port is not None:
            config.set_nested("system.grpc_port", port)
        if visualize_digital_twin is not None:
            config.set_nested("digital_twin.gui", visualize_digital_twin)
        elif 'visualize_pybullet' in kwargs and kwargs['visualize_pybullet'] is not None:
            config.set_nested("digital_twin.gui", kwargs['visualize_pybullet'])
        if visualize_viser is not None:
            config.set_nested("system.viser.enabled", visualize_viser)
        if viser_port is not None:
            config.set_nested("system.viser.port", viser_port)
        if table_z_prior is not None:
            config.set_nested("perception.workspace.table_z_prior", table_z_prior)
        if reachability_radius is not None:
            config.set_nested("perception.workspace.reachability_radius", reachability_radius)
        if voxel_size is not None:
            config.set_nested("perception.voxel_size", voxel_size)
        if device is not None:
            config.set_nested("system.device", device)
        if log_interval_actions is not None:
            config.set_nested("controller.log_interval_actions", log_interval_actions)

        self.config = config
        self.port = int(config.get_nested("system.grpc_port", 50051))

        # Safe device resolution with automatic fallback to CPU if CUDA is unavailable
        raw_device = str(config.get_nested("system.device", "auto")).lower()
        if raw_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif raw_device == "cuda" and not torch.cuda.is_available():
            print("[DREMA DYNAMIC SYSTEM Notice] CUDA requested but not compiled/available in PyTorch. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = raw_device
        self.config.set_nested("system.device", self.device)

        # Viser configuration
        viser_cfg = config.get_nested("system.viser", {})
        self.visualize_viser = bool(viser_cfg.get("enabled", True))
        self.viser_port = int(viser_cfg.get("port", 8080))
        self.viser_server: Optional[viser.ViserServer] = None
        self.viser_handles: Dict[str, Any] = {}

        # Robot kinematic state from client environment
        self.robot_base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.robot_joint_positions: List[float] = []
        self.reachability_radius = float(config.get_nested("perception.workspace.reachability_radius", 0.95))

        print(f"==================================================")
        print(f"   DREMA Dynamic System Starting...               ")
        print(f"   Device: {self.device} | Port: {self.port}       ")
        print(f"==================================================")

        # 0. Initialize Real-Time Viser 3D Web Visualizer
        if self.visualize_viser:
            try:
                self.viser_server = viser.ViserServer(host="0.0.0.0", port=self.viser_port)
                print(f"[DREMA DYNAMIC SYSTEM] ✓ [Viser 3D] Real-time Web Visualizer active at http://localhost:{self.viser_port}")
            except Exception as e:
                print(f"[DREMA DYNAMIC SYSTEM] [Viser Notice] Could not start Viser on port {self.viser_port}: {e}")
                self.viser_server = None

        # 1. Initialize Modular Digital Twin Backend
        twin_engine = str(config.get_nested("digital_twin.engine", "pybullet")).lower()
        twin_gui = bool(config.get_nested("digital_twin.gui", True))
        table_z = float(config.get_nested("perception.workspace.table_z_prior", 0.75))
        urdf_path = config.get_nested("digital_twin.urdf_path", "franka_panda/panda.urdf")

        tracking_mode = str(config.get_nested("digital_twin.tracking_mode", "constraint")).lower()
        constraint_max_force = float(config.get_nested("digital_twin.constraint_max_force", 300.0))
        kp_pos = float(config.get_nested("digital_twin.kp_pos", 250.0))
        kd_pos = float(config.get_nested("digital_twin.kd_pos", 30.0))
        kp_rot = float(config.get_nested("digital_twin.kp_rot", 15.0))
        kd_rot = float(config.get_nested("digital_twin.kd_rot", 1.5))
        sim_substeps = int(config.get_nested("digital_twin.sim_substeps", 2))

        if twin_engine == "pybullet":
            self.digital_twin: BaseDigitalTwin = PyBulletDigitalTwin(
                visualize=twin_gui,
                table_z=table_z,
                robot_urdf_path=urdf_path,
                tracking_mode=tracking_mode,
                constraint_max_force=constraint_max_force,
                kp_pos=kp_pos,
                kd_pos=kd_pos,
                kp_rot=kp_rot,
                kd_rot=kd_rot,
                sim_substeps=sim_substeps
            )
            print(f"[DREMA DYNAMIC SYSTEM] ✓ Submodule 2 (Digital Twin: PyBullet, Tracking: {tracking_mode}) initialized.")
        elif twin_engine in ["mujoco", "mjx"]:
            mujoco_model_path = config.get_nested("digital_twin.mujoco_model_path", "assets/franka_panda/panda.xml")
            enable_mjx = bool(config.get_nested("digital_twin.enable_mjx", False)) or (twin_engine == "mjx")
            self.digital_twin: BaseDigitalTwin = MuJoCoDigitalTwin(
                visualize=twin_gui,
                table_z=table_z,
                robot_model_path=mujoco_model_path,
                tracking_mode=tracking_mode,
                constraint_max_force=constraint_max_force,
                kp_pos=kp_pos,
                kd_pos=kd_pos,
                kp_rot=kp_rot,
                kd_rot=kd_rot,
                sim_substeps=sim_substeps,
                enable_mjx=enable_mjx
            )
            print(f"[DREMA DYNAMIC SYSTEM] ✓ Submodule 2 (Digital Twin: MuJoCo (MJX: {enable_mjx}), Tracking: {tracking_mode}) initialized.")

        else:
            raise ValueError(f"[DREMA DYNAMIC SYSTEM] Unsupported digital twin engine: '{twin_engine}' (expected 'pybullet', 'mujoco')")

        # 2. Initialize Controller with Digital Twin reference
        ctrl_cfg = config.get_nested("controller", {})
        self.log_interval_actions = int(ctrl_cfg.get("log_interval_actions", 10))
        self.max_action_age_s = float(ctrl_cfg.get("max_action_age_s", 0.20))
        # Serializes MPC solves: gRPC runs RequestAction on a thread pool, and a client
        # timeout does not cancel the server-side handler.
        self._mpc_lock = threading.Lock()
        self._last_action: Optional[drema_comm_pb2.ControlAction] = None
        self._last_action_time = 0.0
        self.mpc_controller = MPCController(
            digital_twin=self.digital_twin,
            num_joints=7,
            max_joint_velocity=float(ctrl_cfg.get("max_joint_velocity", 0.50)),
            horizon=int(ctrl_cfg.get("horizon", 15)),
            dt=float(ctrl_cfg.get("dt", 0.05)),
            num_samples_per_planner=int(ctrl_cfg.get("num_samples_per_planner", 24)),
            top_k=int(ctrl_cfg.get("top_k", 12)),
            max_joint_acc=float(ctrl_cfg.get("max_joint_acc", 0.50)),
            alpha_pos=float(ctrl_cfg.get("alpha_pos", 1.0)),
            alpha_rot=float(ctrl_cfg.get("alpha_rot", 0.25)),
            sigma_1=float(ctrl_cfg.get("sigma_1", 0.02)),
            sigma_2=float(ctrl_cfg.get("sigma_2", 0.05)),
            kappa=float(ctrl_cfg.get("kappa", 15.0)),
            adaptive_goal_margin=bool(ctrl_cfg.get("adaptive_goal_margin", False)),
            evade_min_distance=float(ctrl_cfg.get("evade_min_distance", 0.01)),
            evade_max_distance=float(ctrl_cfg.get("evade_max_distance", 0.35)),
            evade_ttc_threshold=float(ctrl_cfg.get("evade_ttc_threshold", 2.0)),
            evade_min_speed=float(ctrl_cfg.get("evade_min_speed", 0.03)),
            evade_imminent_distance=float(ctrl_cfg.get("evade_imminent_distance", 0.06)),
            evade_retreat_speed=float(ctrl_cfg.get("evade_retreat_speed", 0.15)),
            evade_lift_speed=float(ctrl_cfg.get("evade_lift_speed", 0.12))
        )
        print(f"[DREMA DYNAMIC SYSTEM] ✓ Submodule 3 (MPC Controller) initialized (log_interval: {self.log_interval_actions} actions).")

        # 3. Initialize Modular Perception Backend
        perc_mod_name = str(config.get_nested("perception.module", "vg_mapping_recurgs")).lower()
        if perc_mod_name == "vg_mapping_recurgs":
            self.perception: BasePerceptionModule = VGMappingPerceptionModule(config=self.config)
            print(f"[DREMA DYNAMIC SYSTEM] ✓ Submodule 1 (Perception: VG-Mapping + RecurGS) successfully initialized on {self.device.upper()}.")
        else:
            raise ValueError(f"[DREMA DYNAMIC SYSTEM] Unsupported perception module: '{perc_mod_name}' (expected 'vg_mapping_recurgs')")

        # Frame queue & perception worker thread
        queue_size = int(config.get_nested("perception.queue_max_size", 5))
        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._perception_worker, daemon=True)
        self.worker_thread.start()

        # Initial 360-degree scan accumulation buffers
        self.initial_scan_ready = False
        self.accumulated_scan_frames: List[Dict[str, Any]] = []

        # Telemetry
        self.total_frames_processed = 0
        self.total_actions_served = 0
        self.dropped_frames_count = 0

        # 4. Initialize gRPC Server
        self.server = DremaGrpcServer(
            port=self.port,
            on_frame_callback=self.on_frame_received,
            on_action_callback=self.on_request_action,
            on_reset_callback=self.on_reset_episode,
            is_scan_ready_callback=lambda: self.initial_scan_ready
        )

    def _add_viser_obstacle_mesh(
        self,
        obj_name: str,
        idx: int,
        comp: trimesh.Trimesh,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    ):
        """Adds an extracted Marching Cubes obstacle surface mesh to the Viser 3D scene at its initial 3D pose."""
        if self.viser_server is not None and comp is not None:
            try:
                # Convert (x, y, z, w) quaternion from perception/PyBullet to Viser (w, x, y, z)
                if len(quat_xyzw) == 4:
                    qx, qy, qz, qw = quat_xyzw
                    wxyz = (float(qw), float(qx), float(qy), float(qz))
                else:
                    wxyz = (1.0, 0.0, 0.0, 0.0)

                pos = tuple(float(p) for p in position) if len(position) >= 3 else (0.0, 0.0, 0.0)

                self.viser_handles[f"mesh_{idx}"] = self.viser_server.scene.add_mesh_trimesh(
                    name=f"/marching_cubes/{obj_name}_{idx}",
                    mesh=comp,
                    position=pos,
                    wxyz=wxyz
                )
            except Exception as e:
                print(f"[Viser Warning] Failed to add obstacle mesh {obj_name}_{idx}: {e}")

    def _init_viser_voxel_grid(
        self,
        grid_origin: Tuple[float, float, float],
        grid_dim: Tuple[int, int, int]
    ):
        """Initializes Voxel Grid wireframe, footprint grid, coordinate frame, and GUI toggles in Viser."""
        if self.viser_server is None:
            return

        s_v = float(self.config.get_nested("perception.voxel_size", 0.01))
        nx_v, ny_v, nz_v = grid_dim
        gx, gy, gz = grid_origin
        ext_x = nx_v * s_v
        ext_y = ny_v * s_v
        ext_z = nz_v * s_v
        cx = gx + ext_x / 2.0
        cy = gy + ext_y / 2.0
        cz = gz + ext_z / 2.0

        # 1. Coordinate frame at voxel grid origin
        try:
            self.viser_handles['vg_base'] = self.viser_server.scene.add_frame(
                name="/voxel_grid/origin_frame",
                position=(gx, gy, gz),
                show_axes=True,
                axes_length=0.10,
                axes_radius=0.003
            )
        except Exception:
            pass

        # 2. Footprint grid on table surface
        try:
            self.viser_handles['vg_center'] = self.viser_server.scene.add_grid(
                name="/voxel_grid/table_plane",
                width=ext_x,
                height=ext_y,
                plane="xy",
                position=(cx, cy, gz)
            )
        except Exception:
            pass

        # 3. Dynamic Bounding Box wireframe
        try:
            box_mesh = trimesh.creation.box(extents=[ext_x, ext_y, ext_z])
            edges = box_mesh.edges_unique
            verts = box_mesh.vertices + np.array([cx, cy, cz])
            edge_points = verts[edges].reshape(-1, 3)

            self.viser_handles['vg_bbox'] = self.viser_server.scene.add_point_cloud(
                name="/voxel_grid/bbox_wireframe",
                points=edge_points,
                colors=np.tile(np.array([0.15, 0.70, 0.95]), (len(edge_points), 1)),
                point_size=0.005,
                point_shape="circle"
            )
        except Exception:
            pass

        # 4. Text label
        try:
            self.viser_handles['vg_label'] = self.viser_server.scene.add_label(
                name="/voxel_grid/label",
                text=f"TSDF Voxel Grid: {nx_v}x{ny_v}x{nz_v} ({s_v*100:.1f}cm)\nCenter: [{cx:.2f}, {cy:.2f}, {cz:.2f}]m",
                position=(cx, cy, gz + ext_z + 0.04)
            )
        except Exception:
            pass

        # 5. Sidebar Markdown info panel
        self.viser_server.gui.add_markdown(
            f"### DREMA Dynamic Voxel Workspace\n"
            f"- **Origin**: `[{gx:.3f}, {gy:.3f}, {gz:.3f}]` m\n"
            f"- **Dimensions**: `{nx_v} x {ny_v} x {nz_v}` voxels\n"
            f"- **Physical Size**: `{ext_x:.2f}m x {ext_y:.2f}m x {ext_z:.2f}m`\n"
            f"- **Resolution**: `{s_v * 100:.1f}` cm\n"
            f"- **Center**: `[{cx:.3f}, {cy:.3f}, {cz:.3f}]` m\n"
        )

        # 6. Interactive Visibility Checkboxes
        self.cb_bbox = self.viser_server.gui.add_checkbox("Show Voxel Grid BBox & Center", initial_value=True)
        @self.cb_bbox.on_update
        def _(_):
            is_vis = self.cb_bbox.value
            for k in ['vg_bbox', 'vg_base', 'vg_center', 'vg_label']:
                if k in self.viser_handles and self.viser_handles[k] is not None:
                    self.viser_handles[k].visible = is_vis

        self.cb_gaussians = self.viser_server.gui.add_checkbox("Show 3D Gaussians", initial_value=True)
        @self.cb_gaussians.on_update
        def _(_):
            if 'gaussians' in self.viser_handles and self.viser_handles['gaussians'] is not None:
                self.viser_handles['gaussians'].visible = self.cb_gaussians.value

        self.cb_meshes = self.viser_server.gui.add_checkbox("Show Obstacle Meshes", initial_value=True)
        @self.cb_meshes.on_update
        def _(_):
            for k, handle in self.viser_handles.items():
                if k.startswith("mesh_") and handle is not None:
                    handle.visible = self.cb_meshes.value

    def on_reset_episode(self, reset_req: drema_comm_pb2.ResetRequest) -> bool:
        """gRPC callback triggered when an episode resets."""
        print(f"[DREMA DYNAMIC SYSTEM] Resetting episode {reset_req.episode_index} for task {reset_req.task_name}...")
        self.initial_scan_ready = False
        self.accumulated_scan_frames.clear()
        self.robot_joint_positions.clear()
        self.perception.reset()
        self.digital_twin.reset()
        with self._mpc_lock:
            self.mpc_controller.reset()
            self._last_action = None
            self._last_action_time = 0.0
        if bool(self.config.get_nested("perception.cache.enabled", False)):
            self._try_startup_cache_restore()
        return True

    def _process_initial_scene_scan(self, semantic_labels: Optional[Dict[str, int]] = None) -> bool:
        """Processes initial 360 scene reconstruction or restores from pre-computed cache."""
        try:
            print("\n=======================================================")
            print(f"[DREMA DYNAMIC SYSTEM] Processing 360° Initial Scene Scan ({len(self.accumulated_scan_frames)} view point clouds)...")
            print("=======================================================")

            res: InitialScanResult = self.perception.process_initial_scan(
                scan_frames=self.accumulated_scan_frames,
                semantic_labels=semantic_labels or {},
                robot_base_pos=self.robot_base_pos,
                digital_twin=self.digital_twin,
                reachability_radius=self.reachability_radius
            )

            # Initialize Viser Voxel Grid
            self._init_viser_voxel_grid(grid_origin=res.grid_origin, grid_dim=res.grid_dim)

            # Add discovered obstacle meshes to Viser
            for obs in res.discovered_obstacles:
                if os.path.exists(obs.mesh_path):
                    try:
                        comp = trimesh.load(obs.mesh_path)
                        self._add_viser_obstacle_mesh(
                            obj_name=obs.name,
                            idx=obs.oid,
                            comp=comp,
                            position=obs.initial_pos,
                            quat_xyzw=obs.initial_quat
                        )
                    except Exception as e:
                        print(f"[Viser Warning] Failed to load mesh {obs.mesh_path}: {e}")

            # Update initial 3D Gaussians in Viser
            self._update_viser_gaussians()

            # Load Franka Panda in Digital Twin using REAL initial joint angles
            if len(self.robot_joint_positions) > 0 and len(self.robot_base_pos) >= 3:
                self.digital_twin.load_robot(
                    base_position=tuple(self.robot_base_pos.tolist()),
                    joint_positions=list(self.robot_joint_positions)
                )

            # Save raw scan frames to disk cache if enabled and frames are present
            if len(self.accumulated_scan_frames) > 0 and bool(self.config.get_nested("perception.cache.save_on_scan", False)):
                try:
                    cache_dir = self.config.get_nested("perception.cache.cache_dir", "cache/scene_init")
                    os.makedirs(cache_dir, exist_ok=True)
                    raw_frames_path = os.path.join(cache_dir, "scan_frames.pt")
                    torch.save({
                        'scan_frames': self.accumulated_scan_frames,
                        'semantic_labels': semantic_labels or {},
                        'robot_base_pos': self.robot_base_pos,
                        'robot_joint_positions': self.robot_joint_positions,
                        'reachability_radius': self.reachability_radius
                    }, raw_frames_path)
                    print(f"✓ [Perception Cache] Saved {len(self.accumulated_scan_frames)} raw scan frames to '{raw_frames_path}'")
                except Exception as e:
                    print(f"[Perception Cache Notice] Failed to save raw scan frames to disk: {e}")

            self.initial_scan_ready = True
            print(f"\n✓ [DREMA DYNAMIC SYSTEM] Initial Scene Setup Complete! Active Workspace Ready.")
            print("=======================================================\n")
            return True

        except Exception as e:
            print(f"\n[FATAL ERROR] [DREMA DYNAMIC SYSTEM] Failed to process initial scan: {e}")
            import traceback
            traceback.print_exc()
            return False

    def on_frame_received(self, obs: drema_comm_pb2.FrameObservation) -> Optional[drema_comm_pb2.StreamStatus]:
        """gRPC callback triggered when a camera frame arrives from the environment."""
        # Update robot base, reachability radius, and joints if transmitted by client
        if len(obs.robot_base_pos) >= 3:
            self.robot_base_pos = np.array(obs.robot_base_pos[:3], dtype=np.float32)
        if obs.reachability_radius > 0:
            self.reachability_radius = float(obs.reachability_radius)
        if len(obs.joint_positions) > 0:
            self.robot_joint_positions = list(obs.joint_positions)
            if hasattr(self.digital_twin, 'robot_id') and self.digital_twin.robot_id >= 0:
                self.digital_twin.sync_robot_state(self.robot_joint_positions)

        if obs.is_initial_scan:
            existing_names = {f['name'] for f in self.accumulated_scan_frames}
            for f in obs.cameras:
                name, rgb, depth, extrinsics, intrinsics, near_clip, far_clip, mask = unpack_camera_frame(f)
                if name in existing_names:
                    continue
                pcd = pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics)
                valid = (depth > near_clip) & (depth < far_clip)
                pts = pcd[valid]

                self.accumulated_scan_frames.append({
                    'name': name,
                    'rgb': rgb,
                    'depth': depth,
                    'extrinsics': extrinsics,
                    'intrinsics': intrinsics,
                    'near_clip': near_clip,
                    'far_clip': far_clip,
                    'mask': mask,
                    'point_cloud': pts
                })

            if obs.is_scan_finished:
                semantic_labels = dict(obs.semantic_labels) if obs.semantic_labels else None
                success = self._process_initial_scene_scan(semantic_labels=semantic_labels)
                return drema_comm_pb2.StreamStatus(
                    success=success,
                    message="Initial 360° scan processed and Digital Twin populated" if success else "Scan processing failed",
                    received_timestep=0,
                    initial_scan_ready=self.initial_scan_ready
                )
            return drema_comm_pb2.StreamStatus(
                success=True,
                message="Initial scan frame ingested",
                received_timestep=0,
                initial_scan_ready=False
            )

        # Standard real-time streaming frame: discard stale frame if queue is full
        if self.frame_queue.full():
            try:
                dropped_obs = self.frame_queue.get_nowait()
                self.dropped_frames_count += 1
                dropped_ts = getattr(dropped_obs, 'timestep', -1)
                print(f"[DREMA DYNAMIC SYSTEM] [Warning] Perception queue full (size {self.frame_queue.maxsize}). Dropping stale frame (timestep {dropped_ts}) | Total dropped: {self.dropped_frames_count}")
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(obs)
        except queue.Full:
            pass

        return drema_comm_pb2.StreamStatus(
            success=True,
            message="Frame queued",
            received_timestep=obs.timestep,
            initial_scan_ready=self.initial_scan_ready
        )

    def _perception_worker(self):
        """Background thread executing 3D reconstruction and SE(3) tracking via Perception module."""
        target_fps = float(self.config.get_nested("perception.frequency_hz", 10.0))
        target_period = 1.0 / max(1.0, target_fps)
        viser_decimation = int(self.config.get_nested("system.viser.update_decimation", 3))

        while not self.stop_event.is_set():
            try:
                obs = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            t0 = time.time()
            timestep = obs.timestep

            # Unpack all camera views
            camera_views = {}
            for f in obs.cameras:
                name, rgb, depth, extrinsics, intrinsics, near_clip, far_clip, mask = unpack_camera_frame(f)
                camera_views[name] = {
                    'rgb': rgb,
                    'depth': depth,
                    'extrinsics': extrinsics,
                    'intrinsics': intrinsics,
                    'near_clipping': near_clip,
                    'far_clipping': far_clip,
                    'mask': mask
                }

            # If observation contains no camera frames, sleep target period and avoid busy-waiting loop
            if len(camera_views) == 0:
                time.sleep(target_period)
                continue

            # Delegate to modular perception backend
            obs_ts = float(obs.timestamp) if hasattr(obs, 'timestamp') and obs.timestamp > 0 else None
            res: StreamingUpdateResult = self.perception.update_streaming_frame(
                timestep=timestep,
                camera_views=camera_views,
                robot_state={'base_pos': self.robot_base_pos, 'joints': self.robot_joint_positions},
                digital_twin=self.digital_twin,
                timestamp=obs_ts
            )

            # Update Viser mesh poses
            for oid, (new_pos, quat) in res.tracked_object_poses.items():
                mesh_handle = self.viser_handles.get(f"mesh_{oid}")
                if mesh_handle is not None:
                    mesh_handle.position = new_pos
                    mesh_handle.wxyz = (quat[3], quat[0], quat[1], quat[2])

            self.total_frames_processed += 1

            # Periodically refresh Viser 3D Gaussians
            if self.total_frames_processed % viser_decimation == 0:
                self._update_viser_gaussians()

            if self.total_frames_processed % 10 == 0:
                print(f"[DREMA DYNAMIC SYSTEM] [Dynamic Inference #{timestep:04d}] Active Gaussians: {res.active_gaussians_count:,} | Loop Latency: {res.latency_ms:.1f}ms | Tracked Objects: {len(res.tracked_object_poses)}")
                if hasattr(self.digital_twin, 'predictor') and self.digital_twin.predictor is not None:
                    with self.digital_twin.state_lock:
                        preds = self.digital_twin.predictor.predict_all(horizon=15, dt=0.05)
                        states = {oid: self.digital_twin.predictor.get_estimated_state(oid) for oid in res.tracked_object_poses.keys()}
                    for oid in res.tracked_object_poses.keys():
                        st = states[oid]
                        if st is not None:
                            name_o = self.perception.tracked_objects.get(oid, {}).get('name', f"Obj #{oid}")
                            vel = st['velocity']
                            speed = float(np.linalg.norm(vel))
                            pos = st['position']
                            pred_str = ""
                            if oid in preds and len(preds[oid].positions) > 0:
                                p_fut = preds[oid].positions[-1]
                                pred_str = f" -> Pred(+0.75s): [{p_fut[0]:.3f}, {p_fut[1]:.3f}, {p_fut[2]:.3f}]"
                            print(f"  └─ [PREDICTOR] {name_o} (ID {oid}): Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Vel: [{vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}]m/s (|v|={speed:.3f}m/s){pred_str}")

            self.frame_queue.task_done()

            # Frequency throttling
            elapsed = time.time() - t0
            sleep_time = target_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_viser_gaussians(self):
        """Updates the live 3D Gaussian Splats in Viser Web Visualizer."""
        if self.viser_server is None:
            return

        splats_data = self.perception.get_viser_splats_data()
        if splats_data is None:
            return

        try:
            h = self.viser_server.scene.add_gaussian_splats(
                name="/scene/gaussians",
                centers=splats_data['centers'],
                covariances=splats_data['covariances'],
                rgbs=splats_data['rgbs'],
                opacities=splats_data['opacities'],
                scale=1.0
            )
            if hasattr(self, 'cb_gaussians') and self.cb_gaussians is not None:
                h.visible = self.cb_gaussians.value
            self.viser_handles['gaussians'] = h
        except Exception:
            # Fallback to point cloud if add_gaussian_splats encounters an issue
            try:
                h = self.viser_server.scene.add_point_cloud(
                    name="/scene/gaussians",
                    points=splats_data['centers'],
                    colors=splats_data['rgbs'],
                    point_size=0.008,
                    point_shape="circle"
                )
                if hasattr(self, 'cb_gaussians') and self.cb_gaussians is not None:
                    h.visible = self.cb_gaussians.value
                self.viser_handles['gaussians'] = h
            except Exception:
                pass

    def on_request_action(self, robot_state: drema_comm_pb2.RobotState) -> drema_comm_pb2.ControlAction:
        """
        gRPC callback triggered when the environment client requests joint velocity command.
        Only one MPC solve runs at a time; requests arriving during a solve get the latest action.
        """
        if not self._mpc_lock.acquire(blocking=False):
            return self._busy_action(robot_state)
        try:
            action = self._solve_action(robot_state)
            self._last_action = action
            self._last_action_time = time.monotonic()
            return action
        finally:
            self._mpc_lock.release()

    def _busy_action(self, robot_state: drema_comm_pb2.RobotState) -> drema_comm_pb2.ControlAction:
        """Returns the last solved action while it is fresh, otherwise a zero-velocity hold."""
        last = self._last_action
        age = time.monotonic() - self._last_action_time
        if robot_state.task_active and last is not None and age <= self.max_action_age_s:
            busy = drema_comm_pb2.ControlAction()
            busy.CopyFrom(last)
            busy.status_message = f"MPC BUSY (replaying action from timestep {last.timestep}, age {age * 1000.0:.0f}ms) | {last.status_message}"
            return busy
        return drema_comm_pb2.ControlAction(
            timestamp=time.time(),
            timestep=robot_state.timestep,
            joint_velocities=[0.0] * max(len(robot_state.joint_positions), 7),
            gripper_action=robot_state.gripper_open,
            safety_stop=False,
            status_message="MPC BUSY: no fresh action available, holding position"
        )

    def _solve_action(self, robot_state: drema_comm_pb2.RobotState) -> drema_comm_pb2.ControlAction:
        self.total_actions_served += 1

        # Update robot base if provided
        if len(robot_state.robot_base_pos) >= 3:
            self.robot_base_pos = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)

        # 1. Update Digital Twin robot configuration
        needs_robot_load = (
            getattr(self.digital_twin, 'robot_id', -1) < 0
            or not getattr(self.digital_twin, 'robot_loaded', False)
        )
        if needs_robot_load and len(self.robot_base_pos) >= 3:
            self.digital_twin.load_robot(
                base_position=tuple(self.robot_base_pos.tolist()),
                joint_positions=list(robot_state.joint_positions) if len(robot_state.joint_positions) > 0 else None
            )
        elif len(robot_state.joint_positions) > 0:
            self.digital_twin.sync_robot_state(robot_state.joint_positions)

        # 2. Step physics forward
        self.digital_twin.step()

        # 3. Extract target goal
        target_goal = None
        if robot_state.target_available and len(robot_state.target_pose) >= 3:
            target_goal = np.array(robot_state.target_pose[:3], dtype=np.float32)

        # 4. Compute control action via MPC Controller (Digital Twin accessed internally)
        action = self.mpc_controller.compute_action(
            robot_state=robot_state,
            target_goal=target_goal
        )

        if self.total_actions_served <= 3 or (self.log_interval_actions > 0 and self.total_actions_served % self.log_interval_actions == 0):
            print(f"[DREMA DYNAMIC SYSTEM] Action #{self.total_actions_served:05d} (Timestep {robot_state.timestep:04d}) -> MPC: {action.status_message}")
            q_arr = list(robot_state.joint_positions)
            qd_arr = list(action.joint_velocities)
            q_str = "[" + ", ".join(f"{val:+.3f}" for val in q_arr) + "]" if q_arr else "[]"
            qd_str = "[" + ", ".join(f"{val:+.3f}" for val in qd_arr) + "]" if qd_arr else "[]"
            print(f"  └─ [TELEMETRY] q_pos  (J1..J7) [rad]:   {q_str}")
            print(f"  └─ [TELEMETRY] qd_cmd (J1..J7) [rad/s]: {qd_str}")
            diag = getattr(self.mpc_controller, 'last_diagnostics', {})
            t_info = diag.get('timings', {})
            if t_info:
                cdet = t_info.get('coll_details', {})
                cdet_str = ""
                if cdet:
                    b_name = cdet.get('backend', 'Sim')
                    b_ms = cdet.get('backend_ms', cdet.get('bullet_ms', 0.0))
                    cdet_str = f" [{b_name}: {b_ms:.1f}ms, GVM: {cdet.get('gvm_ms', 0.0):.1f}ms, Pts: {cdet.get('pts_count', 0)}]"
                print(f"  └─ [MPC TIMINGS] Coll: {t_info.get('coll_ms', 0.0):.1f}ms{cdet_str} | Rollouts: {t_info.get('samples_ms', 0.0):.1f}ms | Costs(FK): {t_info.get('cost_ms', 0.0):.1f}ms | IK: {t_info.get('ik_ms', 0.0):.1f}ms | Opt: {t_info.get('opt_ms', 0.0):.1f}ms")

        return action

    def _try_startup_cache_restore(self) -> bool:
        """
        Attempts to initialize the scene from disk cache.
        Supports:
        1. Local GPU reconstruction from saved raw scan frames (scan_frames.pt) in ~5s without network transfer.
        2. Instant restoration from pre-computed 3D Gaussians (scene_gaussians.pt) in <0.5s.
        """
        if not bool(self.config.get_nested("perception.cache.enabled", False)):
            return False

        cache_dir = self.config.get_nested("perception.cache.cache_dir", "cache/scene_init")
        g_path = os.path.join(cache_dir, "scene_gaussians.pt")
        frames_path = os.path.join(cache_dir, "scan_frames.pt")
        force_recompute = bool(self.config.get_nested("perception.cache.recompute_scan", False))

        # Mode A: Reconstruct locally on GPU from cached raw frames (if explicitly requested or if Gaussians missing)
        if (force_recompute or not os.path.exists(g_path)) and os.path.exists(frames_path):
            try:
                print(f"\n=======================================================")
                print(f"⚡ [DREMA DYNAMIC SYSTEM] Found cached raw frames in '{frames_path}'!")
                print(f"   Reconstructing scene locally on {self.device.upper()} (Zero SSH transfer)...")
                loaded = torch.load(frames_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, dict) and 'scan_frames' in loaded:
                    self.accumulated_scan_frames = loaded['scan_frames']
                    sem_labels = loaded.get('semantic_labels', {})
                    if 'robot_base_pos' in loaded and len(loaded['robot_base_pos']) >= 3:
                        self.robot_base_pos = np.array(loaded['robot_base_pos'], dtype=np.float32)
                    if 'robot_joint_positions' in loaded and len(loaded['robot_joint_positions']) > 0:
                        self.robot_joint_positions = list(loaded['robot_joint_positions'])
                    if 'reachability_radius' in loaded:
                        self.reachability_radius = float(loaded['reachability_radius'])
                elif isinstance(loaded, list):
                    self.accumulated_scan_frames = loaded
                    sem_labels = {}
                else:
                    return False

                # Temporarily disable perception cache loading so it re-runs reconstruction algorithm
                old_cache_enabled = self.perception.cache_enabled
                self.perception.cache_enabled = False
                success = self._process_initial_scene_scan(semantic_labels=sem_labels)
                self.perception.cache_enabled = old_cache_enabled
                return success
            except Exception as e:
                print(f"[DREMA DYNAMIC SYSTEM] [Cache Warning] Failed to reconstruct from raw frames cache: {e}")
                import traceback
                traceback.print_exc()

        # Mode B: Instant restoration from pre-computed 3D Gaussians & TSDF map
        if os.path.exists(g_path):
            try:
                print(f"\n[DREMA DYNAMIC SYSTEM] Restoring initial scene from cache '{cache_dir}' at startup...")
                return self._process_initial_scene_scan()
            except Exception as e:
                print(f"[DREMA DYNAMIC SYSTEM] [Cache Note] Startup cache restore skipped: {e}")

        return False

    def start(self):
        """Starts the DREMA gRPC Server."""
        self._try_startup_cache_restore()
        self.server.start()

    def stop(self):
        """Stops the gRPC server, worker thread, and visualizers."""
        self.stop_event.set()
        self.server.stop()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        if hasattr(self, 'perception') and self.perception is not None:
            self.perception.shutdown()
        if hasattr(self, 'digital_twin') and self.digital_twin is not None:
            self.digital_twin.shutdown()
        if self.viser_server is not None:
            try:
                self.viser_server.stop()
            except Exception:
                pass
        print(f"[DREMA DYNAMIC SYSTEM] ✓ DREMA Dynamic System cleanly stopped.")

    def spin(self):
        """Keeps server running until SIGINT/SIGTERM is received."""
        def handle_signal(sig, frame):
            print(f"[DREMA DYNAMIC SYSTEM] \nShutting down DREMA Dynamic System...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while not self.stop_event.is_set():
            time.sleep(1.0)


# Backwards compatibility alias
DremaDynamicSuite = DremaDynamicSystem


def parse_args():
    parser = argparse.ArgumentParser(description="Run DREMA Dynamic System")
    parser.add_argument("--config", type=str, default="configs/drema_default.yaml", help="Path to YAML configuration file (default: configs/drema_default.yaml)")

    # Networking & Visualization Overrides
    parser.add_argument("--port", type=int, default=None, help="gRPC Server port (overrides config)")
    parser.add_argument("--twin_engine", "--engine", choices=["pybullet", "mujoco", "mjx"], default=None,
                        help="Digital Twin Physics Engine: 'pybullet' or 'mujoco'/'mjx' (overrides config)")
    parser.add_argument("--enable_mjx", dest="enable_mjx", action="store_true", default=None,
                        help="Enable MJX GPU-accelerated parallel rollout bridge for MuJoCo")
    parser.add_argument("--no_mjx", dest="enable_mjx", action="store_false",
                        help="Disable MJX GPU acceleration (use C MuJoCo engine)")
    parser.add_argument("--visualize_twin", "--visualize_digital_twin", "--gui", dest="visualize_digital_twin", action="store_true", default=None, help="Open Digital Twin GUI window (PyBullet/MuJoCo)")
    parser.add_argument("--no_twin_gui", "--no_gui", dest="visualize_digital_twin", action="store_false", help="Disable Digital Twin GUI window")

    parser.add_argument("--visualize_pybullet", dest="visualize_digital_twin", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_pybullet_gui", dest="visualize_digital_twin", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--visualize_viser", action="store_true", default=None, help="Launch Viser Web Visualizer")
    parser.add_argument("--no_viser", dest="visualize_viser", action="store_false", help="Disable Viser Web Visualizer")
    parser.add_argument("--viser_port", type=int, default=None, help="Viser Web Visualizer port")

    # Cache Configuration
    parser.add_argument("--cache", "--load_cache", dest="load_cache", action="store_true", default=None, help="Restore initial scene scan from disk cache (fast startup)")
    parser.add_argument("--save_cache", dest="save_cache", action="store_true", default=None, help="Save initial scene scan results to disk cache")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for scene cache files (default: cache/scene_init)")
    parser.add_argument("--recompute_scan", "--reconstruct_from_frames", dest="recompute_scan", action="store_true", default=None,
                        help="Reconstruct initial scene from cached raw frames (scan_frames.pt) instead of loading precomputed Gaussians")

    # Perception & Tracking Overrides
    parser.add_argument("--closed_loop_avd", dest="closed_loop_avd", action="store_true", default=None, help="Enable closed-loop 3DGS rendering + AVD variation detection (paper Section III-B)")
    parser.add_argument("--no_closed_loop_avd", dest="closed_loop_avd", action="store_false", help="Disable closed-loop 3DGS rendering (bypass AVD)")
    parser.add_argument("--tau_s", type=float, default=None, help="SSIM threshold for AVD variation detection (paper: 0.6)")
    parser.add_argument("--raycast_stride", type=int, default=None, help="Pixel stride for raycast pruning (1 = full dense, 2 = 2x subsampled)")
    parser.add_argument("--tau_p", type=float, default=None, help="TSDF surface pruning threshold (e.g. 0.2)")
    parser.add_argument("--max_weight", type=float, default=None, help="TSDF maximum integration weight clamp (e.g. 3.0)")
    parser.add_argument("--safety_margin_factor", type=float, default=None, help="Stopping distance factor for raycast pruning (paper: 1.0)")
    parser.add_argument("--se3_iterations", type=int, default=None, help="Lie algebra SE(3) optimization iterations (overrides config)")
    parser.add_argument("--se3_icp_iterations", type=int, default=None, help="Coarse ICP iterations (overrides config)")
    parser.add_argument("--se3_subsample", type=int, default=None, help="Max subsampled points per object (overrides config)")
    parser.add_argument("--perception_fps", type=float, default=None, help="Target perception loop frequency in Hz (overrides config)")
    parser.add_argument("--ctrl_fps", type=float, default=None, help="Controller planning frequency in Hz (overrides config)")
    parser.add_argument("--voxel_size", type=float, default=None, help="TSDF voxel grid resolution in meters (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Computation device cuda/cpu (overrides config)")
    parser.add_argument("--enable_sgd", dest="enable_sgd", action="store_true", default=None,
                        help="Enable paper-compliant photometric SGD Adam optimization (paper Sec. III-B.3, Eq. 10)")
    parser.add_argument("--no_sgd", dest="enable_sgd", action="store_false",
                        help="Disable photometric SGD optimization (pure feedforward)")
    parser.add_argument("--sgd_steps", type=int, default=None,
                        help="Number of photometric SGD optimization steps per frame (default: 5 if enabled, 0 if disabled)")
    parser.add_argument("--log_interval_actions", type=int, default=None,
                        help="Telemetry print interval for served control actions (overrides config)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 1. Load base configuration from YAML
    cfg = load_config(args.config)

    # 2. Apply CLI overrides
    if args.port is not None:
        cfg.set_nested("system.grpc_port", args.port)
    if args.twin_engine is not None:
        cfg.set_nested("digital_twin.engine", args.twin_engine)
    if args.enable_mjx is not None:
        cfg.set_nested("digital_twin.enable_mjx", args.enable_mjx)
    if args.visualize_digital_twin is not None:

        cfg.set_nested("digital_twin.gui", args.visualize_digital_twin)
    if args.visualize_viser is not None:
        cfg.set_nested("system.viser.enabled", args.visualize_viser)
    if args.viser_port is not None:
        cfg.set_nested("system.viser.port", args.viser_port)
    if args.load_cache is not None:
        cfg.set_nested("perception.cache.enabled", args.load_cache)
    if args.save_cache is not None:
        cfg.set_nested("perception.cache.save_on_scan", args.save_cache)
    if args.cache_dir is not None:
        cfg.set_nested("perception.cache.cache_dir", args.cache_dir)
    if args.recompute_scan is not None:
        cfg.set_nested("perception.cache.recompute_scan", args.recompute_scan)
    if args.closed_loop_avd is not None:
        cfg.set_nested("perception.mapping.closed_loop_avd", args.closed_loop_avd)
    if args.tau_s is not None:
        cfg.set_nested("perception.mapping.tau_s", args.tau_s)
    if args.raycast_stride is not None:
        cfg.set_nested("perception.mapping.raycast_stride", args.raycast_stride)
    if args.tau_p is not None:
        cfg.set_nested("perception.mapping.tau_p", args.tau_p)
    if args.max_weight is not None:
        cfg.set_nested("perception.mapping.max_weight", args.max_weight)
    if args.safety_margin_factor is not None:
        cfg.set_nested("perception.mapping.safety_margin_factor", args.safety_margin_factor)
    if args.se3_iterations is not None:
        cfg.set_nested("perception.tracking.se3_iterations", args.se3_iterations)
    if args.se3_icp_iterations is not None:
        cfg.set_nested("perception.tracking.se3_icp_iterations", args.se3_icp_iterations)
    if args.se3_subsample is not None:
        cfg.set_nested("perception.tracking.subsample", args.se3_subsample)
    if args.perception_fps is not None:
        cfg.set_nested("perception.frequency_hz", args.perception_fps)
    if args.ctrl_fps is not None:
        cfg.set_nested("controller.frequency_hz", args.ctrl_fps)
    if args.voxel_size is not None:
        cfg.set_nested("perception.voxel_size", args.voxel_size)
    if args.device is not None:
        cfg.set_nested("system.device", args.device)
    if args.enable_sgd is not None:
        cfg.set_nested("perception.sgd.enabled", args.enable_sgd)
    if args.sgd_steps is not None:
        cfg.set_nested("perception.sgd.steps", args.sgd_steps)
    if args.log_interval_actions is not None:
        cfg.set_nested("controller.log_interval_actions", args.log_interval_actions)

    # 3. Instantiate and run orchestrator system
    system = DremaDynamicSystem(config=cfg)
    system.start()
    system.spin()
