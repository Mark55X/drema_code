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
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
import trimesh
import viser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from drema.config import load_config, ConfigDict
from drema.communication.grpc_server import DremaGrpcServer
from drema.communication.grpc_client import unpack_camera_frame, unpack_label_categories
from drema.communication.proto import drema_comm_pb2
from drema.simulation.base_twin import BaseDigitalTwin
from drema.simulation.pybullet_digital_twin import PyBulletDigitalTwin
from drema.simulation.mujoco_digital_twin import MuJoCoDigitalTwin
from drema.perception.base_perception import BasePerceptionModule, InitialScanResult, StreamingUpdateResult
from drema.perception.vg_mapping_perception import VGMappingPerceptionModule
from drema.controller.mpc_controller import MPCController
from drema.controller.global_ik import GlobalIK
from drema.controller.route_clearance import clearance_matrix, densify, route_violations
from drema.controller.transit_worker import TransitPlannerService
from drema.visualization.imagination import _link_skeletons
from drema.roles import ObjectRoleRegistry
from drema.visualization import build_visualization_hub

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _repo_path(path: str) -> str:
    """A model path relative to the repository root when it exists there; else as given (PyBullet's data path)."""
    candidate = os.path.join(REPO_ROOT, path)
    return candidate if not os.path.isabs(path) and not os.path.exists(path) and os.path.exists(candidate) else path


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

    # Joint-limit clearance [rad] below which an equivalent (180 deg) gripper yaw is preferred
    YAW_MIN_LIMIT_MARGIN = 0.2
    ROUTE_RESOLUTION = 0.04

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
        # Robot model of the twin: the robot profile's, unless digital_twin names one explicitly
        urdf_path = _repo_path(config.get_nested("digital_twin.urdf_path")
                               or config.get_nested("robot.twin.pybullet_urdf", "franka_panda/panda.urdf"))
        self.robot_urdf_path = urdf_path
        mujoco_model_path = _repo_path(config.get_nested("digital_twin.mujoco_model_path")
                                       or config.get_nested("robot.twin.mujoco_model",
                                                            "assets/mujoco_menagerie/franka_emika_panda/panda_upstream.xml"))
        hand_bodies = config.get_nested("robot.twin.mujoco_hand_bodies", None)
        joint_limits = config.get_nested("robot.joint_limits", None)
        self.joint_limits = {k: [float(x) for x in v] for k, v in joint_limits.items() if v is not None} \
            if joint_limits else None

        tracking_mode = str(config.get_nested("digital_twin.tracking_mode", "constraint")).lower()
        constraint_max_force = float(config.get_nested("digital_twin.constraint_max_force", 300.0))
        kp_pos = float(config.get_nested("digital_twin.kp_pos", 250.0))
        kd_pos = float(config.get_nested("digital_twin.kd_pos", 30.0))
        kp_rot = float(config.get_nested("digital_twin.kp_rot", 15.0))
        kd_rot = float(config.get_nested("digital_twin.kd_rot", 1.5))
        sim_substeps = int(config.get_nested("digital_twin.sim_substeps", 2))
        # Robot profile, shared by the controller kinematics and the twin robot model (theta_DH = q_measured + offset)
        self.joint_angle_offsets = [float(x) for x in config.get_nested("robot.joint_angle_offsets", [0.0] * 7)]
        tcp_offset = config.get_nested("robot.tcp_offset", None)
        self.tcp_offset = [float(x) for x in tcp_offset] if tcp_offset is not None else None
        tcp_rotation = config.get_nested("robot.tcp_rotation_xyz_deg", None)
        self.tcp_rotation_R = (Rotation.from_euler('xyz', [float(x) for x in tcp_rotation], degrees=True).as_matrix()
                               if tcp_rotation is not None else None)
        self.self_check_enabled = bool(config.get_nested("robot.self_check.enabled", True))
        self.self_check_max_position = float(config.get_nested("robot.self_check.max_position_error", 0.003))
        self.self_check_max_rotation = float(config.get_nested("robot.self_check.max_rotation_error_deg", 2.0))
        self.self_check_stop = str(config.get_nested("robot.self_check.on_failure", "stop")).lower() == "stop"
        self._self_check_result: Optional[bool] = None
        print(f"[DREMA DYNAMIC SYSTEM] Robot profile: {config.get_nested('robot.name', 'none')} "
              f"({config.get_nested('robot.profile', 'no profile file')}) | joint angle offsets [rad]: "
              f"{self.joint_angle_offsets} | TCP offset [m]: {self.tcp_offset or 'FrankaKinematics default'} | "
              f"joint limits: {'profile' if self.joint_limits else 'Panda (FrankaKinematics)'} | twin model: "
              f"{urdf_path if twin_engine == 'pybullet' else mujoco_model_path}")

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
                sim_substeps=sim_substeps,
                joint_angle_offsets=self.joint_angle_offsets,
                obstacle_convex_decomposition=bool(config.get_nested("digital_twin.obstacle_convex_decomposition", True)),
                vhacd_resolution=int(config.get_nested("digital_twin.vhacd_resolution", 100000)),
                vhacd_concavity=float(config.get_nested("digital_twin.vhacd_concavity", 0.05)),
                ee_offset=self.tcp_offset,
                joint_limits=self.joint_limits
            )
            print(f"[DREMA DYNAMIC SYSTEM] ✓ Submodule 2 (Digital Twin: PyBullet, Tracking: {tracking_mode}) initialized.")
        elif twin_engine in ["mujoco", "mjx"]:
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
                enable_mjx=enable_mjx,
                joint_angle_offsets=self.joint_angle_offsets,
                obstacle_convex_decomposition=bool(config.get_nested("digital_twin.obstacle_convex_decomposition", True)),
                vhacd_resolution=int(config.get_nested("digital_twin.vhacd_resolution", 100000)),
                vhacd_concavity=float(config.get_nested("digital_twin.vhacd_concavity", 0.05)),
                ee_offset=self.tcp_offset,
                joint_limits=self.joint_limits,
                hand_bodies=hand_bodies
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
        self.mpc_controller = MPCController.from_config(ctrl_cfg, self.digital_twin, self.joint_angle_offsets,
                                                        self.tcp_offset, self.joint_limits)
        print(f"[DREMA DYNAMIC SYSTEM] ✓ Submodule 3 (MPC Controller) initialized (log_interval: {self.log_interval_actions} actions).")

        # Object roles: environment permissions + keyframe policy contact + held objects -> twin collision policies
        roles_cfg = config.get_nested("roles", {}) or {}
        self.role_registry: Optional[ObjectRoleRegistry] = (
            ObjectRoleRegistry.from_config(roles_cfg) if bool(roles_cfg.get("enabled", True)) else None
        )
        # Panda_tip orientation relative to the FrankaKinematics EE (flange) orientation, measured online
        self._tip_offset_R: Optional[np.ndarray] = None
        self._yaw_choice: Dict[int, bool] = {}
        self.yaw_symmetry = bool(roles_cfg.get("yaw_symmetry", True))
        self.yaw_symmetry_held_limit = float(roles_cfg.get("yaw_symmetry_held_limit", 0.0))
        ik_cfg = ctrl_cfg.get("ik", {}) or {}
        self.global_ik: Optional[GlobalIK] = None
        self._keyframe_posture: Dict[int, Optional[np.ndarray]] = {}
        if str(ik_cfg.get("global_search", "off")).lower() == "geofik":
            try:
                self.global_ik = GlobalIK(self.mpc_controller.kinematics, int(ik_cfg.get("q7_samples", 33)))
            except ImportError as e:
                print(f"[DREMA DYNAMIC SYSTEM] panda_global_ik not available ({e}): keyframe postures from the local IK "
                      f"(pip install packages/panda_global_ik)")
        print(f"[DREMA DYNAMIC SYSTEM] Keyframe posture search: {'GeoFIK, global' if self.global_ik else 'local DLS'}")
        posture_cfg = ik_cfg.get("posture_selection", {}) or {}
        self.posture_scoring = bool(posture_cfg.get("enabled", True))
        self.posture_path_weight = float(posture_cfg.get("path_weight", 10.0))
        self.posture_candidates = int(posture_cfg.get("candidates_checked", 8))
        self.posture_clearance = float(posture_cfg.get("clearance", 0.02))
        rs = ik_cfg.get("transit_planner", {}) or {}
        steps_per_s = 1.0 / float(ctrl_cfg.get("dt", 0.05))
        self.route_enabled = bool(rs.get("enabled", False))
        self.route_trigger = float(rs.get("trigger_clearance", 0.01))
        sigma_2 = float(ctrl_cfg.get("sigma_2", 0.035))
        self.route_levels = [float(c) for c in (rs.get("clearance_levels") or [sigma_2, 0.025, 0.015])]
        self.route_check_steps = int(round(float(rs.get("check_interval_s", 0.5)) * steps_per_s))
        self.route_stall_steps = int(round(float(rs.get("stall_window_s", 2.0)) * steps_per_s))
        self.route_stall_progress = float(rs.get("stall_progress", 0.05))
        self.route_stall_min = float(rs.get("stall_min_distance", 0.15))
        self.route_static_steps = int(round(float(rs.get("static_after_s", 1.0)) * steps_per_s))
        self.route_scene_change = float(rs.get("scene_change", 0.01))
        self.route_max_plans = int(rs.get("max_plans_per_keyframe", 5))
        self.route_attempts = int(rs.get("attempts", 6))
        self.route_budget = float(rs.get("budget_s", 6.0))
        self.route_switch = float(rs.get("switch_distance", 0.2))
        self.route_wrist_weight = float(rs.get("wrist_weight", 1.0))
        self.route_staging_back = float(rs.get("staging_back", 0.1))
        self.route_staging_max = float(rs.get("staging_max", 0.4))
        self._route: Optional[dict] = None
        self._route_service: Optional[TransitPlannerService] = None
        self._route_requests = 0
        self._route_moved_at: Dict[int, int] = {}
        self._viz: Dict[str, Any] = {}
        print(f"[DREMA DYNAMIC SYSTEM] Route supervisor: {'on' if self.route_enabled else 'off'} (route through an obstacle "
              f"within {self.route_trigger * 100:.0f} cm or stall -> RRT-Connect in a separate process, clearances "
              f"{[round(c * 100, 1) for c in self.route_levels]} cm)")
        if self.global_ik is not None:
            print(f"[DREMA DYNAMIC SYSTEM] Keyframe posture choice: " + (
                f"joint motion + {self.posture_path_weight:.0f} rad/m x TCP path deviation, clear path"
                if self.posture_scoring else "closest posture"))
        print(f"[DREMA DYNAMIC SYSTEM] ✓ Object roles {'enabled' if self.role_registry else 'disabled'}.")

        # 3. Initialize Modular Perception Backend
        perc_mod_name = str(config.get_nested("perception.module", "vg_mapping_recurgs")).lower()
        if perc_mod_name == "vg_mapping_recurgs":
            self.perception: BasePerceptionModule = VGMappingPerceptionModule(config=self.config)
            print(f"[DREMA DYNAMIC SYSTEM] ✓ Submodule 1 (Perception: VG-Mapping + RecurGS) successfully initialized on {self.device.upper()}.")
        else:
            raise ValueError(f"[DREMA DYNAMIC SYSTEM] Unsupported perception module: '{perc_mod_name}' (expected 'vg_mapping_recurgs')")

        # 4. Visualization / recording hooks (MPC input dump, imagination overlay and recording): no-op unless configured
        self.visualization = build_visualization_hub(
            config, self.mpc_controller, self.digital_twin, viser_server=self.viser_server,
            scene_info=lambda: {'table_z': getattr(self.perception, 'z_table', None),
                                'table_bounds': getattr(self.perception, 'table_bounds', None)},
            roles_status=lambda: self.role_registry.status if self.role_registry is not None else [],
            annotations=self._viz_annotations
        )

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
        if self.role_registry is not None:
            self.role_registry.reset(self.digital_twin)
        self._yaw_choice.clear()
        self._keyframe_posture.clear()
        self._route = None
        self._route_moved_at.clear()
        self.mpc_controller.engine.guidance_q = None
        self._self_check_result = None
        self.digital_twin.reset()
        with self._mpc_lock:
            self.mpc_controller.reset()
            self._last_action = None
            self._last_action_time = 0.0
        if bool(self.config.get_nested("perception.cache.enabled", False)):
            self._try_startup_cache_restore()
        return True

    def _process_initial_scene_scan(
        self,
        semantic_labels: Optional[Dict[str, int]] = None,
        label_categories: Optional[Dict[int, str]] = None
    ) -> bool:
        """Processes initial 360 scene reconstruction or restores from pre-computed cache."""
        try:
            print("\n=======================================================")
            print(f"[DREMA DYNAMIC SYSTEM] Processing 360° Initial Scene Scan ({len(self.accumulated_scan_frames)} view point clouds)...")
            print("=======================================================")

            res: InitialScanResult = self.perception.process_initial_scan(
                scan_frames=self.accumulated_scan_frames,
                semantic_labels=semantic_labels or {},
                label_categories=label_categories or {},
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
                        'label_categories': label_categories or {},
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
                success = self._process_initial_scene_scan(semantic_labels=semantic_labels,
                                                           label_categories=unpack_label_categories(obs))
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
                            rot_str = ""
                            if 'raw_angular_velocity' in st:
                                rpy = np.degrees(Rotation.from_quat(st['orientation']).as_euler('xyz'))
                                rot_str = (f" | RPY: [{rpy[0]:+.1f}, {rpy[1]:+.1f}, {rpy[2]:+.1f}]deg"
                                           f" | |w|={float(np.linalg.norm(st['raw_angular_velocity'])):.3f}rad/s"
                                           f" ({'rotating' if st.get('is_rotating', False) else 'static'})")
                            print(f"  └─ [PREDICTOR] {name_o} (ID {oid}): Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | Vel: [{vel[0]:+.3f}, {vel[1]:+.3f}, {vel[2]:+.3f}]m/s (|v|={speed:.3f}m/s){rot_str}{pred_str}")

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

        if self.self_check_enabled and self._self_check_result is None:
            self._kinematic_self_check(robot_state)
        if self._self_check_result is False and self.self_check_stop:
            return drema_comm_pb2.ControlAction(
                timestamp=time.time(),
                timestep=robot_state.timestep,
                joint_velocities=[0.0] * 7,
                gripper_action=robot_state.gripper_open,
                safety_stop=True,
                status_message="ROBOT PROFILE SELF-CHECK FAILED: the kinematic model does not match the measured hand "
                               "pose (see the server log); fix robot.profile or set robot.self_check.on_failure: warn"
            )

        # Target orientation in the FrankaKinematics EE convention used by the MPC and by the hand footprint of the roles
        mpc_state = self._target_in_flange_frame(robot_state)

        # 0. Object roles: policies and held objects must be in place before the keyframe decisions, which check the
        # postures against them, and before the robot (and what it holds) is synced
        if self.role_registry is not None:
            self.role_registry.update(mpc_state, self.digital_twin, self.mpc_controller.kinematics)
        mpc_state = self._keyframe_decisions(robot_state, mpc_state)
        self.digital_twin.sync_gripper(robot_state.gripper_open)
        self._route_supervise(mpc_state)

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
        if mpc_state.target_available and len(mpc_state.target_pose) >= 3:
            target_goal = np.array(mpc_state.target_pose[:3], dtype=np.float32)

        if self.visualization.active:
            self.visualization.before_solve(mpc_state)

        # 4. Compute control action via MPC Controller (Digital Twin accessed internally)
        action = self.mpc_controller.compute_action(
            robot_state=mpc_state,
            target_goal=target_goal
        )
        if self.visualization.active:
            self.visualization.after_solve(mpc_state, action)

        if self.role_registry is not None:
            action.object_roles.extend(self.role_registry.status_protos(drema_comm_pb2))
            action.roles_revision_applied = self.role_registry.roles_revision
            action.status_message = f"{action.status_message} | {self.role_registry.summary()}"
        if self._route is not None and self._route['status']:
            action.status_message = f"{action.status_message} | route: {self._route['status']}"

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

    def _kinematic_self_check(self, robot_state: drema_comm_pb2.RobotState) -> None:
        """
        The hand pose predicted by the kinematic model from the measured joints against the hand pose the client
        measures: a robot profile that does not match the robot (joint offsets, TCP, base) shows up here, before the
        first motion, instead of as a missed grasp. Runs on the first state carrying both, after every reset.
        """
        if len(robot_state.ee_pose) < 7 or len(robot_state.joint_positions) != 7:
            return
        kin = self.mpc_controller.kinematics
        if kin.base_position is None and len(robot_state.robot_base_pos) >= 3:
            kin.base_position = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)
        p_fk, R_fk = kin.forward_kinematics_ee(np.asarray(robot_state.joint_positions, dtype=np.float64))
        position_error = float(np.linalg.norm(np.asarray(p_fk, dtype=np.float64) - np.asarray(robot_state.ee_pose[:3])))
        ok = position_error <= self.self_check_max_position
        rotation_note = "hand frame rotation not in the profile"
        if self.tcp_rotation_R is not None:
            R_hand = np.asarray(R_fk, dtype=np.float64).T @ Rotation.from_quat(robot_state.ee_pose[3:7]).as_matrix()
            rotation_error = float(np.degrees(np.linalg.norm(Rotation.from_matrix(self.tcp_rotation_R.T @ R_hand).as_rotvec())))
            ok = ok and rotation_error <= self.self_check_max_rotation
            rotation_note = f"hand frame rotation error {rotation_error:.2f} deg (max {self.self_check_max_rotation:.1f})"
        self._self_check_result = ok
        verdict = "OK" if ok else ("FAILED, control stopped" if self.self_check_stop else "FAILED, continuing (on_failure: warn)")
        print(f"[DREMA DYNAMIC SYSTEM] Robot profile self-check {verdict}: TCP position error {position_error * 1000:.1f} mm "
              f"(max {self.self_check_max_position * 1000:.1f}), {rotation_note}")

    def _target_in_flange_frame(self, robot_state: drema_comm_pb2.RobotState) -> drema_comm_pb2.RobotState:
        """
        Converts a Panda_tip target orientation (RLBench gripper_pose, PerAct keyposes) to the EE convention of
        FrankaKinematics. The tip->EE rotation is measured from the client's Panda_tip pose and the FK of the same
        joint configuration, so no hand-written constant (nominally 135 deg about z) can carry a sign error.
        """
        if robot_state.target_frame != drema_comm_pb2.FRAME_TIP or len(robot_state.target_pose) < 7:
            return robot_state
        if len(robot_state.ee_pose) >= 7 and len(robot_state.joint_positions) == 7:
            kin = self.mpc_controller.kinematics
            if kin.base_position is None and len(robot_state.robot_base_pos) >= 3:
                kin.base_position = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)
            _, R_fk = kin.forward_kinematics_ee(np.asarray(robot_state.joint_positions, dtype=np.float64))
            R_off = np.asarray(R_fk, dtype=np.float64).T @ Rotation.from_quat(robot_state.ee_pose[3:7]).as_matrix()
            if self._tip_offset_R is None:
                rv = Rotation.from_matrix(R_off).as_rotvec()
                print(f"[DREMA DYNAMIC SYSTEM] Panda_tip vs EE frame: {np.degrees(np.linalg.norm(rv)):.1f} deg about "
                      f"axis {np.round(rv / max(np.linalg.norm(rv), 1e-9), 3).tolist()} (EE frame)")
            self._tip_offset_R = R_off
        if self._tip_offset_R is None:
            return robot_state
        R_target_tip = Rotation.from_quat(robot_state.target_pose[3:7]).as_matrix()
        R_target_ee = R_target_tip @ self._tip_offset_R.T
        if self._yaw_symmetric(robot_state) and self._yaw_choice.get(int(robot_state.keyframe_index)):
            R_target_ee = R_target_ee @ Rotation.from_euler('z', np.pi).as_matrix()
        quat_ee = Rotation.from_matrix(R_target_ee).as_quat()
        converted = drema_comm_pb2.RobotState()
        converted.CopyFrom(robot_state)
        del converted.target_pose[:]
        converted.target_pose.extend([float(x) for x in robot_state.target_pose[:3]] + [float(x) for x in quat_ee])
        converted.target_frame = drema_comm_pb2.FRAME_FLANGE
        return converted

    def _yaw_symmetric(self, robot_state: drema_comm_pb2.RobotState) -> bool:
        return bool(self.yaw_symmetry and robot_state.HasField('yaw_symmetric') and robot_state.yaw_symmetric)

    def _keyframe_decisions(self, robot_state: drema_comm_pb2.RobotState,
                            mpc_state: drema_comm_pb2.RobotState) -> drema_comm_pb2.RobotState:
        """
        First cycle of a keyframe, with its roles in place: the gripper yaw of tip-frame targets symmetric about the
        gripper axis (the hand footprint the roles use is symmetric about it too), then the global IK posture.
        """
        key = int(robot_state.keyframe_index)
        if mpc_state is not robot_state and self._yaw_symmetric(robot_state) and key not in self._yaw_choice:
            R_given = Rotation.from_quat(mpc_state.target_pose[3:7]).as_matrix()
            self._reachable_gripper_yaw(robot_state, R_given)
            if self._yaw_choice[key]:
                quat = Rotation.from_matrix(R_given @ Rotation.from_euler('z', np.pi).as_matrix()).as_quat()
                turned = drema_comm_pb2.RobotState()
                turned.CopyFrom(mpc_state)
                del turned.target_pose[:]
                turned.target_pose.extend([float(x) for x in mpc_state.target_pose[:3]] + [float(x) for x in quat])
                mpc_state = turned
        self._seed_keyframe_posture(mpc_state)
        return mpc_state

    def _reachable_gripper_yaw(self, robot_state: drema_comm_pb2.RobotState, R_target: np.ndarray) -> np.ndarray:
        """
        A parallel gripper grasps the same way when turned by 180 deg about its axis: of the two equivalent targets,
        keeps the reachable one needing the smaller joint motion, unless it ends closer than YAW_MIN_LIMIT_MARGIN to a
        joint limit (joint 7 first of all). Decided once per keyframe, so the goal cannot flip while the arm moves.
        """
        key = int(robot_state.keyframe_index)
        if key not in self._yaw_choice:
            kin = self.mpc_controller.kinematics
            q_now = np.asarray(robot_state.joint_positions, dtype=np.float64)
            pos = np.asarray(robot_state.target_pose[:3], dtype=np.float64)
            scored, picks = [], {}
            for flip in (False, True):
                R = R_target @ Rotation.from_euler('z', np.pi).as_matrix() if flip else R_target
                if self.global_ik is not None:
                    pick = self._choose_posture(self.global_ik.solutions(pos, R), q_now)
                    picks[flip] = pick
                    ok, q_sol = pick is not None, (pick[0] if pick is not None else q_now)
                else:
                    q_sol, ok = kin.solve_dls_ik(q_now.astype(np.float32), pos, R)
                margin = float(np.min(np.minimum(q_sol - kin.Q_MIN, kin.Q_MAX - q_sol)))
                cost = picks[flip][2] if picks.get(flip) is not None else float(np.linalg.norm(q_sol - q_now))
                reachable = picks[flip][6] if picks.get(flip) is not None else bool(ok)
                scored.append((bool(ok), margin, cost, flip, q_sol, reachable))
            best = max(scored, key=lambda s: (s[0], s[5], s[1] >= self.YAW_MIN_LIMIT_MARGIN, -s[2]))
            choice, kept = best[3], ""
            if self.yaw_symmetry_held_limit > 0 and self._yaw_choice:
                previous = max(self._yaw_choice)
                inherited = self._yaw_choice[previous]
                # A gripper that is not open with nothing registered may still be closing on something whose turn
                # cannot be measured yet.
                pending = (robot_state.gripper_open < 0.9 and
                           not (self.role_registry is not None and self.role_registry.held))
                moved = self._held_turn_displacement() if choice != inherited else 0.0
                if choice != inherited and pending:
                    moved = float('inf')
                if moved > self.yaw_symmetry_held_limit:
                    reason = ("the gripper is closed on an object not registered yet" if pending else
                              f"the turn would move the held object by {moved * 100:.1f} cm")
                    # An unreachable kept yaw would stall the arm for good; a turned held object may still be placed.
                    if scored[1 if inherited else 0][0]:
                        choice = inherited
                        kept = f", kept from keyframe {previous}: {reason}"
                    else:
                        kept = f", turned although {reason}: the yaw of keyframe {previous} is unreachable here"
            self._yaw_choice[key] = choice
            if self.global_ik is not None:
                chosen = scored[1 if choice else 0]
                self._keyframe_posture[key] = chosen[4] if chosen[0] else None
                if chosen[0]:
                    self.mpc_controller.engine.last_ik_solution = chosen[4].astype(np.float32)
                kept += (", global IK posture " + self._posture_note(picks[choice])) if chosen[0] else ", no global IK solution"
            print(f"[DREMA DYNAMIC SYSTEM] Keyframe {key}: gripper yaw {'turned by 180 deg' if choice else 'as given'} "
                  f"(IK ok / joint-limit margin: as given {scored[0][0]}/{scored[0][1]:.2f} rad, "
                  f"turned {scored[1][0]}/{scored[1][1]:.2f} rad){kept}")
            self._route_start(key, pos, (R_target @ Rotation.from_euler('z', np.pi).as_matrix()) if choice else R_target,
                              f"gripper yaw {'turned by 180 deg' if choice else 'as given'}{kept}")
        if self._yaw_choice[key]:
            return R_target @ Rotation.from_euler('z', np.pi).as_matrix()
        return R_target

    def _choose_posture(self, solutions: List[np.ndarray], q_now: np.ndarray) -> Optional[Tuple]:
        """
        Of the global IK solutions at least YAW_MIN_LIMIT_MARGIN from the joint limits (all of them when none is) and
        not in collision with the current roles: the lowest joint motion plus path_weight times the largest distance of
        the TCP from the straight segment to its final position, along the joint-space line the guidance pulls the arm
        on; of the best few, the first whose line follows the route rule at the posture clearance. Without posture
        scoring, the closest solution. Returns (q, margin, cost, joint distance, TCP deviation, clear path, reachable).
        """
        if not solutions:
            return None
        kin = self.mpc_controller.kinematics
        q_now = np.asarray(q_now, dtype=np.float64)
        Q = np.asarray(solutions, dtype=np.float64)
        margins = np.min(np.minimum(Q - kin.Q_MIN, kin.Q_MAX - Q), axis=1)
        pool = np.flatnonzero(margins >= self.YAW_MIN_LIMIT_MARGIN)
        if len(pool) == 0:
            pool = np.arange(len(Q))
        distances = np.linalg.norm(Q - q_now, axis=1)
        if not self.posture_scoring:
            i = int(pool[np.argmin(distances[pool])])
            return Q[i], float(margins[i]), float(distances[i]), float(distances[i]), 0.0, True, True
        s = np.linspace(0.0, 1.0, 11)
        lines = q_now[None, None, :] + s[None, :, None] * (Q[pool] - q_now)[:, None, :]
        P = np.asarray(kin.batch_forward_kinematics_ee(lines.astype(np.float32)), dtype=np.float64)
        seg = P[:, -1:, :] - P[:, :1, :]
        t = np.clip(np.sum((P - P[:, :1, :]) * seg, axis=2) / np.maximum(np.sum(seg * seg, axis=2), 1e-12), 0.0, 1.0)
        deviations = np.max(np.linalg.norm(P - (P[:, :1, :] + t[..., None] * seg), axis=2), axis=1)
        costs = distances[pool] + self.posture_path_weight * deviations
        ranked = np.argsort(costs)
        free = ~np.any(self._clearances(Q[pool], self.posture_clearance)[0] <= 0.0, axis=1)
        usable = [int(r) for r in ranked if free[r]]
        for r in usable[:self.posture_candidates]:
            i = int(pool[r])
            if self._route_clear(q_now, Q[i], self.posture_clearance):
                return Q[i], float(margins[i]), float(costs[r]), float(distances[i]), float(deviations[r]), True, True
        r = usable[0] if usable else int(ranked[0])
        i = int(pool[r])
        return (Q[i], float(margins[i]), float(costs[r]) + 10.0, float(distances[i]), float(deviations[r]), False,
                bool(usable))

    def _clearances(self, Q: np.ndarray, cap: float) -> Tuple[np.ndarray, np.ndarray]:
        engine = self.mpc_controller.engine
        return clearance_matrix(self.digital_twin, engine.kin, Q, max(cap, 1e-3), sigma_1=engine.sigma_1,
                                kappa=engine.kappa, rho=engine.rho, dt=engine.dt)

    def _route_clear(self, q_from: np.ndarray, q_to: np.ndarray, clearance: float) -> bool:
        """The straight joint-space line follows the route rule at the given clearance."""
        D, _ = self._clearances(densify([q_from, q_to], self.ROUTE_RESOLUTION), clearance)
        return not route_violations(D, clearance).any()

    @staticmethod
    def _posture_note(pick: Tuple) -> str:
        path = "clear path" if pick[5] else ("NOT clear path" if pick[6] else "every posture of the target in collision")
        return (f"{pick[3]:.2f} rad from the current one, TCP off the straight line by {pick[4] * 100:.1f} cm, "
                f"joint-limit margin {pick[1]:.2f} rad, {path}")

    def _route_start(self, key: int, pos: np.ndarray, R: np.ndarray, note: str) -> None:
        """New keyframe: the route is the straight joint line to the chosen posture until a check says otherwise."""
        self.mpc_controller.engine.guidance_q = None
        q_goal = self._keyframe_posture.get(key)
        self._viz = {'key': key, 'posture': note}
        self._route = None
        if not self.route_enabled or self.global_ik is None or q_goal is None:
            return
        self._route = {'key': key, 'goal': np.asarray(q_goal, dtype=np.float64), 'pos': np.asarray(pos, dtype=np.float64),
                       'R': np.asarray(R, dtype=np.float64), 'state': 'straight', 'wps': None, 'i': 0, 'plans': 0,
                       'hist': [], 'last_check': None, 'notes': set(), 'status': ''}

    def _route_say(self, r: dict, text: str, once: Optional[str] = None) -> None:
        """Log line (once per keyframe when `once` is given), status for the client and the video banner."""
        if once is not None:
            if once in r['notes']:
                return
            r['notes'].add(once)
        r['status'] = text
        self._viz['transit'] = text
        print(f"[DREMA DYNAMIC SYSTEM] Keyframe {r['key']}: {text}")

    def _route_check(self, r: dict, q: np.ndarray, clearance: float) -> Dict[str, Any]:
        """
        Remaining route (current configuration, then the remaining waypoints or the chosen posture) checked with the
        route rule: where it first breaks it ('at', fraction of the joint length, None when it does not) and the
        objects there ('blockers'), the objects the arrival collides with ('occupied') or lies within the clearance
        of ('goal_band'), the TCP polyline ('tcp') and the samples breaking the rule ('bad').
        """
        kin = self.mpc_controller.kinematics
        nodes = [q] + ([np.asarray(w) for w in r['wps'][r['i']:]] if r['state'] == 'following' else [r['goal']])
        Q = densify(nodes, self.ROUTE_RESOLUTION)
        tcp = np.asarray(kin.batch_forward_kinematics_ee(Q[None].astype(np.float32)), dtype=np.float64)[0]
        D, ids = self._clearances(Q, clearance)
        bad = route_violations(D, clearance)
        rows = bad.any(axis=1)
        out = {'at': None, 'blockers': set(), 'occupied': {int(o) for o in ids[D[-1] <= 0.0]},
               'goal_band': {int(o) for o in ids[D[-1] < clearance]}, 'tcp': tcp, 'bad': rows}
        if rows.any():
            first = int(np.argmax(rows))
            out['at'] = float(np.sum(np.linalg.norm(np.diff(Q[:first + 1], axis=0), axis=1)) /
                              max(np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1)), 1e-9))
            out['blockers'] = {int(o) for o in ids[bad[first]]}
        return out

    def _object_static(self, oid: int, ts: int) -> bool:
        if oid < 0:
            return True
        return ts - self._route_moved_at.get(oid, -10 ** 9) >= self.route_static_steps

    def _route_track_motion(self, ts: int) -> None:
        tw = self.digital_twin
        predictor = getattr(tw, 'predictor', None)
        if predictor is None:
            return
        limit = self.role_registry.policy_contact_max_speed if self.role_registry is not None else 0.03
        with tw.state_lock:
            for oid in list(tw.tracked_objects):
                st = predictor.get_estimated_state(oid)
                if st is not None and 'velocity' in st and float(np.linalg.norm(st['velocity'])) > limit:
                    self._route_moved_at[int(oid)] = ts

    def _scene_signature(self) -> Dict[int, np.ndarray]:
        tw = self.digital_twin
        out = {}
        for oid in list(tw.tracked_objects):
            pose = tw.tracked_object_pose(oid)
            if pose is not None and oid not in tw.attachments:
                out[int(oid)] = np.asarray(pose[0], dtype=np.float64)
        return out

    def _names(self, ids: set) -> str:
        tw = self.digital_twin
        return ", ".join(sorted("table" if i < 0 else str(tw.tracked_objects.get(i, {}).get('name', i)) for i in ids)) or "?"

    def _route_request(self, r: dict, q: np.ndarray, start: np.ndarray, why: str) -> None:
        """Plans a route from `start` to the keyframe posture (any usable posture as fallback) in the background."""
        engine, kin = self.mpc_controller.engine, self.mpc_controller.kinematics
        if r['plans'] >= self.route_max_plans:
            self._route_block(r, q, f"{why}; planning attempts exhausted ({r['plans']})")
            return
        S = np.asarray(self.global_ik.solutions(r['pos'], r['R']), dtype=np.float64)
        usable = S[np.min(np.minimum(S - kin.Q_MIN, kin.Q_MAX - S), axis=1) >= self.YAW_MIN_LIMIT_MARGIN] if len(S) else S
        reachable = lambda G: ~np.any(self._clearances(G, self.route_levels[0])[0] <= 0.0, axis=1)
        goals = usable[reachable(usable)] if len(usable) else usable
        if not len(goals):
            self._route_block(r, q, f"{why}; every posture of the keyframe target is in collision")
            return
        if self._route_service is None:
            self._route_service = TransitPlannerService(os.path.dirname(os.path.abspath(__file__)))
        self._route_requests += 1
        preferred = r['goal'] if reachable(r['goal'][None])[0] else None
        req = dict(self._twin_snapshot(), key=r['key'], request_id=self._route_requests, start=start, goals=goals,
                   preferred_goal=preferred, clearance_levels=list(self.route_levels), base=np.asarray(kin.base_position, dtype=np.float64),
                   offsets=[float(x) for x in self.joint_angle_offsets], ee_offset=[float(x) for x in kin.EE_OFFSET],
                   joint_limits={'q_min': kin.Q_MIN.tolist(), 'q_max': kin.Q_MAX.tolist(), 'qd_max': kin.QD_MAX.tolist()},
                   urdf=self.robot_urdf_path, kappa=engine.kappa, rho=engine.rho, seed=1000 * r['key'] + r['plans'],
                   attempts=self.route_attempts, budget_s=self.route_budget, shortcut_tries=60,
                   wrist_weight=self.route_wrist_weight)
        self._route_service.submit(req)
        r.update(state='planning', request_id=self._route_requests, t_request=time.monotonic(), plans=r['plans'] + 1,
                 wps=None, i=0, hold=np.asarray(start, dtype=np.float64), hist=[])
        engine.guidance_q = np.asarray(start, dtype=np.float32)
        if np.linalg.norm(start - q) > 1e-6:
            self._viz['staging_skeleton'] = _link_skeletons(kin, np.asarray(start)[None])[0].astype(np.float32)
        self._viz.pop('path_tcp', None)
        self._viz.pop('wp_skeletons', None)
        self._route_say(r, f"{why}: planning a collision-free route in the background (attempt {r['plans']})"
                           + (", moving to a safe staging point" if np.linalg.norm(start - q) > 1e-6 else ", holding"))

    def _route_no_route(self, r: dict, q: np.ndarray, why: str) -> None:
        """A block only when the direct route collides with an obstacle; otherwise a narrow passage for the MPPI."""
        chk = self._route_check(dict(r, state='straight'), q, 0.0)
        if chk['at'] is not None:
            self._route_block(r, q, f"{why}; the direct route collides with {self._names(chk['blockers'])}")
            return
        r.update(state='narrow', wps=None, i=0, hist=[])
        self.mpc_controller.engine.guidance_q = None
        self._viz.pop('staging_skeleton', None)
        self._route_say(r, f"{why}, but the direct route does not collide: narrow passage, the MPPI continues")

    def _route_block(self, r: dict, q: np.ndarray, why: str) -> None:
        """No route: hold the current configuration and say why; resumes when the scene changes."""
        r.update(state='blocked', hold=np.asarray(q, dtype=np.float64), blocked_scene=self._scene_signature(), hist=[])
        self.mpc_controller.engine.guidance_q = np.asarray(q, dtype=np.float32)
        self._viz.pop('staging_skeleton', None)
        self._route_say(r, f"BLOCKED - {why}. Holding; resuming when the scene changes")

    def _route_poll(self, r: dict, q: np.ndarray) -> None:
        res = self._route_service.poll() if self._route_service is not None else None
        while res is not None:
            if r['state'] == 'planning' and res.get('key') == r['key'] and res.get('request_id') == r.get('request_id'):
                if res.get('path') is None:
                    self._route_no_route(r, q, f"no route with at least {self.route_levels[-1] * 100:.1f} cm clearance"
                                               + (f" ({res['error']})" if res.get('error') else ""))
                else:
                    info = res['info']
                    r.update(state='following', wps=[np.asarray(w, dtype=np.float64) for w in res['path']], i=0, hist=[])
                    self.mpc_controller.engine.guidance_q = r['wps'][0].astype(np.float32)
                    self._viz_route(r['wps'])
                    self._route_say(r, f"route ready after {time.monotonic() - r['t_request']:.1f} s to the {info['goal']}: "
                                       f"{len(r['wps']) - 2} intermediate waypoints, {info['joint_length']:.2f} rad, "
                                       f"clearance {info['clearance'] * 100:.1f} cm")
            res = self._route_service.poll()
        if r['state'] == 'planning' and time.monotonic() - r['t_request'] > self.route_budget + 15.0:
            self._route_block(r, q, "route planning timed out")

    def _route_advance(self, r: dict, q: np.ndarray) -> None:
        engine = self.mpc_controller.engine
        while np.linalg.norm(q - r['wps'][r['i']]) < self.route_switch:
            if r['i'] == len(r['wps']) - 1:
                r.update(state='done', hist=[], goal=r['wps'][-1])
                engine.guidance_q = None
                engine.last_ik_solution = r['wps'][-1].astype(np.float32)
                self._viz['wp_index'] = len(r['wps'])
                self._route_say(r, "planned route completed: the MPPI goes to the keyframe target as usual")
                return
            r['i'] += 1
            r['hist'] = []
            engine.guidance_q = r['wps'][r['i']].astype(np.float32)
            self._viz['wp_index'] = r['i']
            self._route_say(r, f"following the planned route: waypoint {r['i'] + 1} of {len(r['wps'])} "
                               f"(the green dashed arm is the current waypoint)")

    def _route_supervise(self, mpc_state: drema_comm_pb2.RobotState) -> None:
        """
        Every cycle, after the role update: collects planned routes, advances waypoints, checks the remaining route
        every check interval (static blockers -> plan, moving ones -> wait), watches for stalls and, after a declared
        block, replans only once the scene has changed.
        """
        r = self._route
        if r is None:
            return
        engine, kin = self.mpc_controller.engine, self.mpc_controller.kinematics
        if int(mpc_state.keyframe_index) != r['key']:
            self._route = None
            engine.guidance_q = None
            return
        ts = int(mpc_state.timestep)
        q = np.asarray(mpc_state.joint_positions, dtype=np.float64)
        self._route_track_motion(ts)
        self._route_poll(r, q)
        if r['state'] == 'following':
            self._route_advance(r, q)
        due = r['last_check'] is None or ts - r['last_check'] >= self.route_check_steps
        if r['state'] == 'blocked':
            if due:
                r['last_check'] = ts
                now = self._scene_signature()
                moved = max((float(np.linalg.norm(p - r['blocked_scene'][o])) for o, p in now.items()
                             if o in r['blocked_scene']), default=0.0)
                if set(now) != set(r['blocked_scene']) or moved >= self.route_scene_change:
                    r['blocked_scene'] = self._scene_signature()
                    chk = self._route_check(dict(r, state='straight'), q, self.route_trigger)
                    if chk['at'] is None:
                        r.update(state='straight', wps=None, i=0, hist=[])
                        engine.guidance_q = None
                        self._route_say(r, "the scene changed and the route is clear again: resuming")
                    elif all(self._object_static(o, ts) for o in chk['blockers']):
                        self._route_request(r, q, q, f"the scene changed, route still blocked by "
                                                     f"{self._names(chk['blockers'])}")
            return
        if r['state'] == 'planning':
            return
        if due and r['state'] != 'narrow':
            r['last_check'] = ts
            chk = self._route_check(r, q, self.route_trigger)
            blk, ids = chk['at'], chk['blockers']
            self._viz['line_tcp'], self._viz['line_blocked'] = chk['tcp'].astype(np.float32), chk['bad']
            occupying = {o for o in chk['occupied'] if not self._object_static(o, ts)}
            if occupying:
                if r['state'] != 'occupied':
                    r.update(state='occupied', hold=np.asarray(q, dtype=np.float64), hist=[])
                    engine.guidance_q = np.asarray(q, dtype=np.float32)
                self._route_say(r, f"waiting: the keyframe posture is occupied by moving {self._names(occupying)}; "
                                   f"holding", once=f"occupied:{sorted(occupying)}")
                return
            if r['state'] == 'occupied':
                r.update(state='straight', hist=[])
                engine.guidance_q = None
            if blk is None:
                if r['status'].startswith('waiting'):
                    self._route_say(r, "route clear again")
                elif not r['status']:
                    self._route_say(r, "route to the keyframe posture is clear: no planning needed")
            elif not all(self._object_static(o, ts) for o in ids):
                self._route_say(r, f"waiting: the route is crossed by moving {self._names(ids)} (the MPPI avoids it)",
                                once=f"wait:{sorted(ids)}")
            else:
                stop = max(0.0, blk - self.route_staging_back)
                start = q if (stop <= 0.0 or r['state'] == 'following') else q + min(stop, self.route_staging_max) * (r['goal'] - q)
                self._route_request(r, q, start, f"route through {self._names(ids)} (static) from {blk * 100:.0f}%")
                return
        if r['state'] == 'occupied':
            return
        hist = r['hist']
        guide = r['wps'][r['i']] if r['state'] == 'following' else (
            np.asarray(engine.last_ik_solution, dtype=np.float64) if engine.last_ik_solution is not None else r['goal'])
        d = float(np.linalg.norm(q - guide))
        hist.append((ts, d))
        while hist and ts - hist[0][0] > self.route_stall_steps:
            hist.pop(0)
        gripper_busy = mpc_state.HasField('gripper_moving') and mpc_state.gripper_moving
        near_target = (mpc_state.target_available and len(mpc_state.target_pose) >= 3 and
                       float(np.linalg.norm(np.asarray(kin.forward_kinematics_ee(q)[0]) - np.asarray(mpc_state.target_pose[:3]))) < 0.03)
        if (len(hist) < 2 or ts - hist[0][0] < self.route_stall_steps or gripper_busy or near_target
                or d < self.route_stall_min or hist[0][1] - d >= self.route_stall_progress):
            return
        r['hist'] = []
        if r['state'] == 'narrow':
            self._route_say(r, f"stall in the narrow passage ({d:.2f} rad to the guidance): left to the MPPI",
                            once="stall:narrow")
            return
        chk = self._route_check(r, q, float(engine.sigma_2))
        ids = chk['blockers']
        if chk['at'] is None and chk['goal_band']:
            self._route_say(r, f"stall at the arrival: the keyframe target lies within the collision band of "
                               f"{self._names(chk['goal_band'])} (a role matter, not a route one)",
                            once=f"goalband:{sorted(chk['goal_band'])}")
        elif chk['at'] is None:
            self._route_say(r, f"stall without any obstacle within the MPPI collision band ({d:.2f} rad to the guidance)",
                            once="stall:none")
        elif not all(self._object_static(o, ts) for o in ids):
            self._route_say(r, f"waiting: stalled next to moving {self._names(ids)}", once=f"wait:{sorted(ids)}")
        else:
            self._route_request(r, q, q, f"stall {d:.2f} rad from the guidance, {self._names(ids)} in the collision band")

    def _twin_snapshot(self) -> dict:
        """What the transit worker needs to rebuild the collision scene: objects, poses, roles, held objects, table."""
        tw = self.digital_twin
        with tw.state_lock:
            objects = []
            for oid, obj in tw.tracked_objects.items():
                pose = tw.tracked_object_pose(oid)
                if pose is None or obj.get('body_id', -1) < 0:
                    continue
                objects.append((int(oid), str(obj['mesh_path']), str(obj.get('name', oid)),
                                [float(x) for x in pose[0]], [float(x) for x in pose[1]], bool(obj.get('is_target', False))))
            snap = {'objects': objects, 'policies': dict(tw.object_policies),
                    'table_policy': getattr(tw, 'table_policy', None),
                    'held': {int(k): np.asarray(v, dtype=np.float64) for k, v in tw.attachments.items()},
                    'table_in_costs': bool(tw.table_in_collision_costs),
                    'vhacd': (bool(tw.obstacle_convex_decomposition), int(tw.vhacd_resolution), float(tw.vhacd_concavity))}
        bounds = getattr(self.perception, 'table_bounds', None)
        snap['table_z'] = float(getattr(self.perception, 'z_table', None) or tw.table_z)
        snap['table_bounds'] = None if bounds is None else [float(b) for b in bounds]
        return snap

    def _viz_route(self, wps: list) -> None:
        kin = self.mpc_controller.kinematics
        dense = [w0[None] + np.linspace(0.0, 1.0, max(2, int(np.ceil(np.linalg.norm(w1 - w0) / 0.05))))[:, None] * (w1 - w0)[None]
                 for w0, w1 in zip(wps[:-1], wps[1:])]
        self._viz['path_tcp'] = np.asarray(kin.batch_forward_kinematics_ee(np.concatenate(dense)[None].astype(np.float32)),
                                           dtype=np.float32)[0]
        self._viz['wp_skeletons'] = _link_skeletons(kin, np.asarray(wps, dtype=np.float64)).astype(np.float32)
        self._viz['wp_index'] = 0
        self._viz.pop('staging_skeleton', None)

    def _viz_annotations(self) -> Dict[str, Any]:
        """Per-action notes for the imagination video: keyframe decisions and route supervisor state."""
        v = self._viz
        if not v:
            return {}
        out = {'viz_keyframe': np.int64(v.get('key', -1)), 'viz_posture': np.array(str(v.get('posture', ''))),
               'viz_transit': np.array(str(v.get('transit', '')))}
        for name in ('line_tcp', 'line_blocked', 'path_tcp', 'wp_skeletons', 'staging_skeleton'):
            if name in v:
                out[f'viz_{name}'] = np.asarray(v[name])
        if 'wp_index' in v:
            out['viz_wp_index'] = np.int64(v['wp_index'])
        return out

    def _seed_keyframe_posture(self, mpc_state: drema_comm_pb2.RobotState) -> None:
        """
        Once per keyframe, for targets without the yaw choice: the global IK posture closest to the current one becomes
        the warm start of the per-cycle local IK, which otherwise stays in the joint-space family it started in.
        """
        if self.global_ik is None or not mpc_state.target_available or len(mpc_state.target_pose) < 7:
            return
        key = int(mpc_state.keyframe_index)
        if key in self._keyframe_posture:
            return
        kin = self.mpc_controller.kinematics
        q_now = np.asarray(mpc_state.joint_positions, dtype=np.float64)
        pick = self._choose_posture(self.global_ik.solutions(np.asarray(mpc_state.target_pose[:3], dtype=np.float64),
                                                            Rotation.from_quat(mpc_state.target_pose[3:7]).as_matrix()),
                                    q_now)
        self._keyframe_posture[key] = None if pick is None else pick[0]
        if pick is not None:
            self.mpc_controller.engine.last_ik_solution = pick[0].astype(np.float32)
        print(f"[DREMA DYNAMIC SYSTEM] Keyframe {key}: global IK posture " + (
            self._posture_note(pick) if pick is not None else "none: the target is out of reach"))
        self._route_start(key, np.asarray(mpc_state.target_pose[:3], dtype=np.float64),
                          Rotation.from_quat(mpc_state.target_pose[3:7]).as_matrix(),
                          "global IK posture " + (self._posture_note(pick) if pick is not None else "none: out of reach"))

    def _held_turn_displacement(self) -> float:
        """
        How far the held objects move (95th percentile over their points) when the gripper turns by 180 deg about its
        axis: about zero for objects symmetric about it (cubes, rings grasped at the centre), large for one held off-axis.
        """
        if self.role_registry is None or not self.role_registry.held:
            return 0.0
        turn = np.diag([-1.0, -1.0, 1.0])
        worst = 0.0
        for obj_id, tcp_T_obj in self.role_registry.held.items():
            local = self.digital_twin.object_body_points(obj_id)
            if local is None or not len(local):
                continue
            pts = np.asarray(local, dtype=np.float64) @ tcp_T_obj[:3, :3].T + tcp_T_obj[:3, 3]
            worst = max(worst, float(np.percentile(cKDTree(pts).query(pts @ turn.T)[0], 95)))
        return worst

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
                    categories = loaded.get('label_categories', {})
                    if 'robot_base_pos' in loaded and len(loaded['robot_base_pos']) >= 3:
                        self.robot_base_pos = np.array(loaded['robot_base_pos'], dtype=np.float32)
                    if 'robot_joint_positions' in loaded and len(loaded['robot_joint_positions']) > 0:
                        self.robot_joint_positions = list(loaded['robot_joint_positions'])
                    if 'reachability_radius' in loaded:
                        self.reachability_radius = float(loaded['reachability_radius'])
                elif isinstance(loaded, list):
                    self.accumulated_scan_frames = loaded
                    sem_labels, categories = {}, {}
                else:
                    return False

                # Temporarily disable perception cache loading so it re-runs reconstruction algorithm
                old_cache_enabled = self.perception.cache_enabled
                self.perception.cache_enabled = False
                success = self._process_initial_scene_scan(semantic_labels=sem_labels, label_categories=categories)
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
        if hasattr(self, 'visualization'):
            self.visualization.close()
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
