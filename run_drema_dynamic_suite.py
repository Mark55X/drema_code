#!/usr/bin/env python
"""
DREMA Dynamic Inference Suite (run_drema_dynamic_suite.py)

Central closed-loop inference server:
- Hosts gRPC server for multi-frequency communication with CoppeliaSim.
- Submodule 1: Dynamic VG-Mapping & RecurGS SE(3) pipeline for real-time 3D reconstruction and tracking.
- Submodule 2: PyBullet Digital Twin running synchronized physics simulation and proximity queries.
- Submodule 3: MPC Controller generating safe joint velocity actions for the Franka Panda arm.
"""

import os
import sys
import time
import queue
import argparse
import threading
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import mcubes
import trimesh
import viser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from drema.communication.grpc_server import DremaGrpcServer
from drema.communication.grpc_client import unpack_camera_frame
from drema.communication.proto import drema_comm_pb2
from drema.simulation.digital_twin import PyBulletDigitalTwin
from drema.controller.mpc_controller import MPCController

try:
    from drema.vg_mapping.closed_loop_pipeline import DREMAClosedLoopVGMappingPipeline, rotation_matrix_to_quaternion
except Exception as e:
    print("\n" + "=" * 75)
    print(f"[FATAL ERROR] Could not import DREMAClosedLoopVGMappingPipeline: {e}")
    print("DREMA requires VG-Mapping & RecurGS for real-time 3D reconstruction and tracking.")
    print("=" * 75 + "\n")
    sys.exit(1)


# Semantic Label Keyword Constants
ROBOT_KEYWORD_LABELS = (
    "panda", "link", "finger", "hand", "joint", "arm", "gripper", "wrist", "flange"
)
BACKGROUND_KEYWORD_LABELS = (
    "workspace", "table", "floor", "wall", "ceiling", "pillar",
    "sensor", "success", "camera", "head", "waypoint", "detector",
    "target", "goal", "marker", "dummy"
)
VIRTUAL_KEYWORD_LABELS = (
    "target", "goal", "marker", "dummy", "waypoint", "detector", "sensor", "success"
)


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


def extract_obstacle_mesh_from_tsdf(
    tsdf_map,
    z_min_cutoff: float,
    z_max_cutoff: Optional[float] = None,
    x_bounds: Optional[Tuple[float, float]] = None,
    y_bounds: Optional[Tuple[float, float]] = None,
    level: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts solid surface mesh above tabletop cutoff (z_min_cutoff) and within table bounds using Marching Cubes,
    isolating tabletop obstacles from the tabletop support plane without arbitrary robot distance thresholds.
    """
    F_np = tsdf_map.F.cpu().numpy().copy()
    W_np = tsdf_map.W.cpu().numpy().copy()

    # Mask unobserved voxels as free space
    F_np[W_np <= 0.5] = 1.0

    # Mask out everything below table cutoff to isolate tabletop support plane
    nx, ny, nz = F_np.shape
    origin_np = tsdf_map.origin.cpu().numpy()
    voxel_size = tsdf_map.voxel_size
    origin_z = float(origin_np[2])

    k_min = int(np.ceil((z_min_cutoff - origin_z) / voxel_size))
    k_min = max(0, min(nz, k_min))
    if k_min > 0:
        F_np[:, :, :k_min] = 1.0

    if z_max_cutoff is not None:
        k_max = int(np.floor((z_max_cutoff - origin_z) / voxel_size))
        k_max = max(0, min(nz, k_max))
        if k_max < nz:
            F_np[:, :, k_max:] = 1.0

    # Mask out regions outside tabletop horizontal bounds
    if x_bounds is not None or y_bounds is not None:
        xs = origin_np[0] + (np.arange(nx) + 0.5) * voxel_size
        ys = origin_np[1] + (np.arange(ny) + 0.5) * voxel_size
        xm, ym = np.meshgrid(xs, ys, indexing='ij')

        mask_free = np.zeros((nx, ny), dtype=bool)
        if x_bounds is not None:
            mask_free |= (xm < x_bounds[0]) | (xm > x_bounds[1])
        if y_bounds is not None:
            mask_free |= (ym < y_bounds[0]) | (ym > y_bounds[1])

        F_np[mask_free, :] = 1.0

    if F_np.min() <= level and F_np.max() >= level:
        vertices, triangles = mcubes.marching_cubes(F_np, level)
        vertices = origin_np + (vertices + 0.5) * voxel_size
        return vertices, triangles
    return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)


class DremaDynamicSuite:
    """
    Coordinator managing the 3 submodules and gRPC communication.
    """

    def __init__(
        self,
        port: int = 50051,
        visualize_pybullet: bool = False,
        visualize_viser: bool = True,
        viser_port: int = 8080,
        table_z_prior: float = 0.75,
        reachability_radius: float = 0.95,
        voxel_size: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.port = port
        self.device = device
        self.table_z_prior = table_z_prior
        self.reachability_radius = reachability_radius
        self.voxel_size = voxel_size
        self.visualize_viser = visualize_viser
        self.viser_port = viser_port
        self.viser_server = None
        self.viser_handles: Dict[str, Any] = {}

        # Robot base & dynamic active workspace (updated via initial scan & gRPC)
        self.robot_base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.robot_joint_positions: List[float] = []
        self.active_workspace_bounds: Optional[Dict[str, float]] = None
        self.workspace_bounds_t: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        print(f"==================================================")
        print(f"   DREMA Dynamic Inference Suite Starting...     ")
        print(f"   Device: {self.device} | Port: {self.port}       ")
        print(f"==================================================")

        # 0. Initialize Real-Time Viser 3D Web Visualizer if enabled
        if self.visualize_viser:
            try:
                self.viser_server = viser.ViserServer(host="0.0.0.0", port=self.viser_port)
                print(f"✓ [Viser 3D] Real-time Web Visualizer active at http://localhost:{self.viser_port}")
            except Exception as e:
                print(f"[Viser Notice] Could not start Viser on port {self.viser_port}: {e}")
                self.viser_server = None

        # 1. Initialize Submodule 2: PyBullet Digital Twin (Ground plane + Robot URDF)
        self.digital_twin = PyBulletDigitalTwin(
            visualize=visualize_pybullet,
            table_z=table_z_prior
        )
        print("✓ Submodule 2 (PyBullet Digital Twin) initialized.")

        # 2. Initialize Submodule 3: MPC Controller Stub
        self.mpc_controller = MPCController(
            num_joints=7,
            max_joint_velocity=0.15,
            safety_collision_distance=0.035
        )
        print("✓ Submodule 3 (MPC Controller) initialized.")

        # 3. Initialize Submodule 1: VG-Mapping & RecurGS SE(3) Pipeline (Fail-Fast Verification)
        try:
            self.vg_pipeline = DREMAClosedLoopVGMappingPipeline(
                pybullet_client=None,
                voxel_size=voxel_size,
                grid_dim=(128, 128, 128),
                origin=(-0.5, -0.5, 0.0),
                device=self.device
            )
            print(f"✓ Submodule 1 (VG-Mapping Closed-Loop Pipeline) successfully verified on {self.device.upper()}.")
        except Exception as e:
            print("\n" + "=" * 75)
            print(f"[FATAL ERROR] Failed to initialize DREMAClosedLoopVGMappingPipeline on {self.device}: {e}")
            print("=" * 75 + "\n")
            sys.exit(1)

        # Frame ingestion queue & worker thread
        self.frame_queue = queue.Queue(maxsize=3)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._vg_mapping_worker, daemon=True)
        self.worker_thread.start()

        # Initial 360-degree scan state & dynamic objects
        self.initial_scan_ready = False
        self.accumulated_scan_points = []
        self.accumulated_scan_frames = []
        self.semantic_labels = {}
        self.robot_ids = set()
        self.virtual_ids = set()
        self.dynamic_object_ids = set()
        self.tracked_objects = {}
        self.output_mesh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/scanned_meshes")
        os.makedirs(self.output_mesh_dir, exist_ok=True)

        self.scene_gaussians = {
            'xyz': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'rgb': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'scale': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'normal': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
            'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device)
        }

        # Telemetry
        self.total_frames_processed = 0
        self.total_actions_served = 0

        # 4. Initialize gRPC Server
        self.server = DremaGrpcServer(
            port=port,
            on_frame_callback=self.on_frame_received,
            on_action_callback=self.on_request_action,
            on_reset_callback=self.on_reset_episode
        )

    def _apply_semantic_robot_mask(
        self,
        depth_t: torch.Tensor,
        mask_np: Optional[np.ndarray]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Applies per-pixel CoppeliaSim entity handle segmentation mask to filter robot arm links
        and virtual simulation markers (target, goal, waypoint, marker) from TSDF and 3DGS.
        Returns:
            depth_masked: Depth tensor with robot and virtual marker pixels set to 0.0 (ignored by TSDF integration).
            mask_t: Integer tensor containing CoppeliaSim handle IDs per pixel (for VDC 3DGS filtering).
        """
        if mask_np is None:
            return depth_t, None
        mask_t = torch.from_numpy(mask_np.copy()).to(self.device)
        depth_masked = depth_t.clone()
        filter_ids = self.robot_ids | self.virtual_ids
        if len(filter_ids) > 0:
            f_ids = torch.tensor(list(filter_ids), device=self.device, dtype=mask_t.dtype)
            is_filtered = torch.isin(mask_t, f_ids)
            # 1-pixel morphological dilation to prevent boundary bleeding along link silhouettes
            filt_float = is_filtered.float().unsqueeze(0).unsqueeze(0)
            dilated = (torch.nn.functional.max_pool2d(filt_float, kernel_size=3, stride=1, padding=1).squeeze() > 0.5)
            depth_masked[0, dilated] = 0.0
        return depth_masked, mask_t

    def _add_viser_obstacle_mesh(self, obj_name: str, idx: int, comp):
        """Adds an extracted Marching Cubes obstacle surface mesh to the Viser 3D scene."""
        if self.viser_server is not None and comp is not None:
            try:
                self.viser_handles[f"mesh_{idx}"] = self.viser_server.scene.add_mesh_trimesh(
                    name=f"/marching_cubes/{obj_name}_{idx}",
                    mesh=comp
                )
            except Exception:
                pass

    def _init_viser_voxel_grid(
        self,
        grid_origin: Tuple[float, float, float],
        grid_dim: Tuple[int, int, int]
    ):
        """Initializes Voxel Grid wireframe, footprint grid, coordinate frame, and GUI toggles in Viser."""
        if self.viser_server is None:
            return

        gx, gy, gz = grid_origin
        nx_v, ny_v, nz_v = grid_dim
        s_v = self.voxel_size
        ext_x, ext_y, ext_z = nx_v * s_v, ny_v * s_v, nz_v * s_v
        cx = gx + ext_x / 2.0
        cy = gy + ext_y / 2.0
        cz = gz + ext_z / 2.0

        # 1. Emerald green wireframe bounding box
        self.viser_handles['vg_bbox'] = self.viser_server.scene.add_box(
            name="/voxel_grid/bbox",
            color=(0, 245, 100),
            dimensions=(ext_x, ext_y, ext_z),
            position=(cx, cy, cz),
            wireframe=True
        )
        # 2. Tabletop footprint grid aligned with bottom of voxel grid
        try:
            self.viser_handles['vg_base'] = self.viser_server.scene.add_grid(
                name="/voxel_grid/base_grid",
                width=ext_x,
                height=ext_y,
                plane="xy",
                position=(cx, cy, gz),
                cell_size=max(s_v * 5, 0.05),
                cell_color=(0, 180, 80),
                section_size=max(s_v * 10, 0.10),
                section_color=(0, 255, 128)
            )
        except Exception:
            pass

        # 3. 3D Coordinate frame at geometric center
        self.viser_handles['vg_center'] = self.viser_server.scene.add_frame(
            name="/voxel_grid/center",
            position=(cx, cy, cz),
            axes_length=0.15,
            axes_radius=0.004
        )
        # 4. Floating 3D label positioned right above the voxel grid
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
            f"### VG-Mapping Voxel Grid\n"
            f"- **Origin**: `[{gx:.3f}, {gy:.3f}, {gz:.3f}]` m\n"
            f"- **Dimensions**: `{nx_v} x {ny_v} x {nz_v}` voxels\n"
            f"- **Physical Size**: `{ext_x:.2f}m x {ext_y:.2f}m x {ext_z:.2f}m`\n"
            f"- **Resolution**: `{s_v * 100:.1f}` cm\n"
            f"- **Center**: `[{cx:.3f}, {cy:.3f}, {cz:.3f}]` m\n"
        )

        # 6. Interactive Visibility Checkboxes
        cb_bbox = self.viser_server.gui.add_checkbox("Show Voxel Grid BBox & Center", initial_value=True)
        @cb_bbox.on_update
        def _(_):
            is_vis = cb_bbox.value
            for k in ['vg_bbox', 'vg_base', 'vg_center', 'vg_label']:
                if k in self.viser_handles and self.viser_handles[k] is not None:
                    self.viser_handles[k].visible = is_vis

        cb_gaussians = self.viser_server.gui.add_checkbox("Show 3D Gaussians", initial_value=True)
        @cb_gaussians.on_update
        def _(_):
            if 'gaussians' in self.viser_handles and self.viser_handles['gaussians'] is not None:
                self.viser_handles['gaussians'].visible = cb_gaussians.value

        cb_voxels = self.viser_server.gui.add_checkbox("Show TSDF Surface Voxels", initial_value=True)
        @cb_voxels.on_update
        def _(_):
            if 'surface_voxels' in self.viser_handles and self.viser_handles['surface_voxels'] is not None:
                self.viser_handles['surface_voxels'].visible = cb_voxels.value

        cb_meshes = self.viser_server.gui.add_checkbox("Show Obstacle Meshes", initial_value=True)
        @cb_meshes.on_update
        def _(_):
            for k, handle in self.viser_handles.items():
                if k.startswith("mesh_") and handle is not None:
                    handle.visible = cb_meshes.value

    def _update_viser_surface_voxels(self):
        """Extracts and displays discrete TSDF surface voxels in Viser 3D Web Visualizer."""
        if self.viser_server is None:
            return
        try:
            with torch.no_grad():
                surf_mask = (self.vg_pipeline.tsdf_map.W > 0.5) & (self.vg_pipeline.tsdf_map.F.abs() < 0.12)
                if surf_mask.any():
                    surf_pts = self.vg_pipeline.tsdf_map.voxel_centers[surf_mask].detach().cpu().numpy()
                    surf_colors = np.zeros_like(surf_pts)
                    surf_colors[:, 0] = 0.05
                    surf_colors[:, 1] = 0.85
                    surf_colors[:, 2] = 0.95
                    if len(surf_pts) > 40000:
                        sub_idx = np.random.choice(len(surf_pts), 40000, replace=False)
                        surf_pts = surf_pts[sub_idx]
                        surf_colors = surf_colors[sub_idx]
                    self.viser_handles['surface_voxels'] = self.viser_server.scene.add_point_cloud(
                        name="/voxel_grid/surface_voxels",
                        points=surf_pts,
                        colors=surf_colors,
                        point_size=self.voxel_size * 0.7,
                        point_shape="square"
                    )
                    print(f"✓ [Viser 3D] Visualizing {len(surf_pts)} TSDF surface voxels at /voxel_grid/surface_voxels")
        except Exception as e:
            print(f"[Viser Notice] Could not add TSDF surface voxels: {e}")

    def _process_initial_scene_scan(self, semantic_labels: Optional[Dict[str, int]] = None) -> bool:
        """
        Processes the 200 orbital scanning views collected at t=0:
        1. Strict fail-fast check: aborts if 0 points were accumulated.
        2. Detects the true horizontal table surface and its spatial extent from the dense point cloud.
        3. Computes the Active Workspace by intersecting the table surface with the robot reachability cylinder.
        4. Dynamically initializes the TSDF voxel grid origin and dimensions to cover the Active Workspace.
        5. Spawns the physical table mesh in PyBullet Digital Twin.
        6. Discovers tabletop obstacle clusters strictly within the Active Workspace and spawns them.
        """
        print("\n=======================================================")
        print(f"[DREMA Suite] Processing 360° Initial Scene Scan ({len(self.accumulated_scan_points)} view point clouds)...")
        print("=======================================================")

        # Strict Fail-Fast Check: If no points collected, terminate immediately
        if len(self.accumulated_scan_points) == 0:
            print("\n" + "=" * 75)
            print("[FATAL ERROR] 360° initial scan collected 0 valid points!")
            print("No depth data was received from CoppeliaSim sensors. Cannot reconstruct table or scene.")
            print("Pipeline terminating immediately.")
            print("=" * 75 + "\n")
            self.stop()
            sys.exit(1)

        all_pts = np.vstack(self.accumulated_scan_points)
        print(f"[DREMA Suite Scan] Accumulated {len(all_pts)} dense 3D surface points across all views.")

        # =========================================================================
        # Table Detection & Workspace Geometry Configuration
        # =========================================================================
        MIN_TABLE_PLANE_POINTS = 400          # Minimum inlier points required to confirm table support
        TABLE_SURFACE_THICKNESS = 0.008        # Elevation tolerance (±8mm) for surface inliers
        ACTIVE_WORKSPACE_MARGIN = 0.05         # Reachability safety margin (meters) added to robot workspace
        TABLE_TSDF_LOWER_PADDING = 0.05        # Vertical volume padding below table surface for TSDF (meters)
        OBSTACLE_MIN_CLEARANCE_Z = 0.015       # Min clearance (meters) above table to isolate objects from surface noise
        ROBOT_BASE_RADIUS = 0.12               # Exclusion cylinder radius (meters) around robot base origin

        # 1. Robust Table Surface Detection via Dominant Horizontal Plane
        z_vals = all_pts[:, 2]
        z_min_scan = float(np.percentile(z_vals, 1))
        z_max_scan = float(np.percentile(z_vals, 99))
        if z_max_scan - z_min_scan < 0.10:
            z_min_scan -= 0.10
            z_max_scan += 0.10
        num_bins = max(30, int((z_max_scan - z_min_scan) / 0.005))
        hist, bin_edges = np.histogram(z_vals, bins=num_bins, range=(z_min_scan, z_max_scan))
        peak_indices = np.argsort(hist)[::-1]

        found_table = False
        z_table = 0.0
        table_pts = None

        # The tabletop is the dominant horizontal plane with substantial point support
        for p_idx in peak_indices[:10]:
            candidate_z = float(0.5 * (bin_edges[p_idx] + bin_edges[p_idx + 1]))
            cand_mask = np.abs(z_vals - candidate_z) <= TABLE_SURFACE_THICKNESS
            cand_pts = all_pts[cand_mask]

            if len(cand_pts) >= MIN_TABLE_PLANE_POINTS:
                z_table = candidate_z
                table_pts = cand_pts
                found_table = True
                break

        if not found_table or table_pts is None:
            print("\n" + "=" * 75)
            print("[FATAL ERROR] Could not detect a valid horizontal table surface in the 360° scan!")
            print(f"Analyzed {len(all_pts)} points between Z=[{z_min_scan:.2f}, {z_max_scan:.2f}]m.")
            print(f"No horizontal plane with at least {MIN_TABLE_PLANE_POINTS} points was found.")
            print("Pipeline terminating immediately.")
            print("=" * 75 + "\n")
            self.stop()
            sys.exit(1)

        # 2. Extract Full Physical Table Bounds directly from point distribution
        tab_x_min = float(np.percentile(table_pts[:, 0], 1))
        tab_x_max = float(np.percentile(table_pts[:, 0], 99))
        tab_y_min = float(np.percentile(table_pts[:, 1], 1))
        tab_y_max = float(np.percentile(table_pts[:, 1], 99))
        table_bounds = (tab_x_min, tab_x_max, tab_y_min, tab_y_max)
        print(f"✓ [DREMA Scan] Table surface detected at Z={z_table:.3f}m, full bounds: X[{tab_x_min:.3f}, {tab_x_max:.3f}], Y[{tab_y_min:.3f}, {tab_y_max:.3f}]")

        # 3. Post-Filtering: Compute Active Workspace (Intersection with Robot Reachability Cylinder)
        rb_x, rb_y, rb_z = float(self.robot_base_pos[0]), float(self.robot_base_pos[1]), float(self.robot_base_pos[2])
        r_reach = float(self.reachability_radius)

        reach_x_min = rb_x - (r_reach + ACTIVE_WORKSPACE_MARGIN)
        reach_x_max = rb_x + (r_reach + ACTIVE_WORKSPACE_MARGIN)
        reach_y_min = rb_y - (r_reach + ACTIVE_WORKSPACE_MARGIN)
        reach_y_max = rb_y + (r_reach + ACTIVE_WORKSPACE_MARGIN)

        act_x_min = max(tab_x_min, reach_x_min)
        act_x_max = min(tab_x_max, reach_x_max)
        act_y_min = max(tab_y_min, reach_y_min)
        act_y_max = min(tab_y_max, reach_y_max)
        act_z_min = z_table - TABLE_TSDF_LOWER_PADDING
        # Upper workspace bound: naturally bounded by robot reachability volume (no arbitrary height ceiling!)
        act_z_max = z_table + r_reach

        self.active_workspace_bounds = {
            'x_min': act_x_min, 'x_max': act_x_max,
            'y_min': act_y_min, 'y_max': act_y_max,
            'z_min': act_z_min, 'z_max': act_z_max,
            'z_table': z_table
        }
        print(f"✓ [DREMA Reachability] Robot Base: [{rb_x:.2f}, {rb_y:.2f}, {rb_z:.2f}], Reach Radius: {r_reach:.2f}m")
        print(f"✓ [DREMA Active Workspace] X[{act_x_min:.3f}, {act_x_max:.3f}], Y[{act_y_min:.3f}, {act_y_max:.3f}], Z[{act_z_min:.3f}, {act_z_max:.3f}]")

        # 4. Spawn Solid Table in PyBullet Digital Twin (Strictly active reachable workspace)
        active_table_bounds = (act_x_min, act_x_max, act_y_min, act_y_max)
        self.digital_twin.spawn_scanned_table(
            table_z=z_table,
            bounds=active_table_bounds
        )

        # 5. Dynamically Initialize TSDF Grid for Active Workspace in Submodule 1
        ext_x = act_x_max - act_x_min
        ext_y = act_y_max - act_y_min
        ext_z = act_z_max - act_z_min

        nx = max(32, int(np.ceil(ext_x / self.voxel_size)))
        ny = max(32, int(np.ceil(ext_y / self.voxel_size)))
        nz = max(32, int(np.ceil(ext_z / self.voxel_size)))

        nx = ((nx + 7) // 8) * 8
        ny = ((ny + 7) // 8) * 8
        nz = ((nz + 7) // 8) * 8

        grid_origin = (round(act_x_min, 4), round(act_y_min, 4), round(act_z_min, 4))
        grid_dim = (nx, ny, nz)

        print(f"✓ [VG-Mapping TSDF] Dynamic Grid configured: origin={grid_origin}, dim={grid_dim} ({nx*self.voxel_size:.2f}m x {ny*self.voxel_size:.2f}m x {nz*self.voxel_size:.2f}m)")
        self.vg_pipeline = DREMAClosedLoopVGMappingPipeline(
            pybullet_client=None,
            voxel_size=self.voxel_size,
            grid_dim=grid_dim,
            origin=grid_origin,
            device=self.device
        )

        # Render Voxel Grid Wireframe Bounding Box in PyBullet GUI
        self.digital_twin.draw_voxel_grid_bbox(
            origin=grid_origin,
            dim=grid_dim,
            voxel_size=self.voxel_size
        )

        # Update Viser Real-Time Web Visualizer with Voxel Grid
        self._init_viser_voxel_grid(grid_origin=grid_origin, grid_dim=grid_dim)

        # 6. Parse Semantic Labels using Keyword Constants
        self.semantic_labels = semantic_labels or {}
        id_to_name = {}
        self.robot_ids = set()
        self.virtual_ids = set()
        self.dynamic_object_ids = set()

        for name, num in self.semantic_labels.items():
            num = int(num)
            id_to_name[num] = name
            name_lower = name.lower()
            if any(kw in name_lower for kw in ROBOT_KEYWORD_LABELS):
                self.robot_ids.add(num)
            elif any(kw in name_lower for kw in VIRTUAL_KEYWORD_LABELS):
                self.virtual_ids.add(num)

            is_robot_or_bg = any(kw in name_lower for kw in (ROBOT_KEYWORD_LABELS + BACKGROUND_KEYWORD_LABELS))
            if not is_robot_or_bg:
                self.dynamic_object_ids.add(num)

        if len(self.semantic_labels) > 0:
            print(f"✓ [DREMA Scan] Parsed {len(self.semantic_labels)} semantic labels from CoppeliaSim:")
            print(f"   Robot Arm Link IDs ({len(self.robot_ids)}): {sorted(list(self.robot_ids))}")
            if len(self.virtual_ids) > 0:
                v_names = [f"'{id_to_name[v]}' (ID:{v})" for v in sorted(list(self.virtual_ids))]
                print(f"   Virtual/Target IDs to filter ({len(self.virtual_ids)}): {', '.join(v_names)}")
            if len(self.dynamic_object_ids) > 0:
                d_names = [f"'{id_to_name[d]}' (ID:{d})" for d in sorted(list(self.dynamic_object_ids))]
                print(f"   Scene Physical Object IDs ({len(self.dynamic_object_ids)}): {', '.join(d_names)}")
            else:
                print(f"   Scene Physical Object IDs: None detected from initial labels.")

        # 7. Ingest All 360° Scan Frames into VG-Mapping (TSDF Voxel Grid & 3D Gaussian Splats)
        num_views = len(self.accumulated_scan_frames)
        print(f"\n[VG-Mapping] Starting ingestion of {num_views} scan views into TSDF Voxel Grid & 3DGS...")

        workspace_bounds_t = (
            torch.tensor([act_x_min, act_y_min, act_z_min], dtype=torch.float32, device=self.device),
            torch.tensor([act_x_max, act_y_max, act_z_max], dtype=torch.float32, device=self.device)
        )
        self.workspace_bounds_t = workspace_bounds_t

        new_xyz_acc, new_rgb_acc, new_scale_acc, new_normal_acc, new_morton_acc, new_obj_id_acc = [], [], [], [], [], []
        raw_gaussians_count = 0

        for f_idx, frame_data in enumerate(self.accumulated_scan_frames):
            t_start_view = time.time()
            rgb_t = torch.from_numpy(frame_data['rgb'].copy()).permute(2, 0, 1).float().to(self.device) / 255.0
            depth_t = torch.from_numpy(frame_data['depth'].copy()).unsqueeze(0).to(self.device)
            k_t = torch.from_numpy(frame_data['intrinsics']).to(self.device)
            pose_t = torch.from_numpy(frame_data['extrinsics']).to(self.device)
            mask_np = frame_data.get('mask')

            # Filter out robot arm links & virtual targets via per-pixel semantic mask
            depth_tsdf, mask_t = self._apply_semantic_robot_mask(
                depth_t=depth_t,
                mask_np=mask_np
            )

            # Step 1: TSDF integration (robot & virtual targets masked to 0 so never carved into TSDF)
            self.vg_pipeline.step_1_ingest_frame(
                rgb=rgb_t,
                depth=depth_tsdf,
                intrinsic=k_t,
                camera_pose=pose_t
            )

            # Step 2: VDC Gaussian mapping (is_robot=True for both robot links and virtual targets)
            rendered_rgb = rgb_t.clone()
            rendered_depth = depth_tsdf.clone()

            new_g, prune_mask = self.vg_pipeline.step_2_online_mapping(
                rgb=rgb_t,
                depth=depth_tsdf,
                rendered_rgb=rendered_rgb,
                rendered_depth=rendered_depth,
                intrinsic=k_t,
                camera_pose=pose_t,
                current_morton_codes=self.scene_gaussians['morton'],
                mask=mask_t,
                workspace_bounds=workspace_bounds_t,
                is_initial_timestep=True,
                num_views=num_views,
                robot_ids=(self.robot_ids | self.virtual_ids),
                target_object_ids=self.dynamic_object_ids
            )

            n_new_g = len(new_g['xyz'])
            if n_new_g > 0 and len(self.robot_base_pos) >= 3:
                # Geometric exclusion of robot base mounting column (radius 0.13m, z >= table_z)
                rx, ry, rz = float(self.robot_base_pos[0]), float(self.robot_base_pos[1]), float(self.robot_base_pos[2])
                p_xy = new_g['xyz'][:, :2]
                dist_base = torch.sqrt((p_xy[:, 0] - rx) ** 2 + (p_xy[:, 1] - ry) ** 2)
                keep_geom = (dist_base > 0.13) | (new_g['xyz'][:, 2] < (rz - 0.05))
                if not torch.all(keep_geom):
                    for k in list(new_g.keys()):
                        if isinstance(new_g[k], torch.Tensor) and len(new_g[k]) == n_new_g:
                            new_g[k] = new_g[k][keep_geom]
                    n_new_g = len(new_g['xyz'])

            raw_gaussians_count += n_new_g
            t_view_ms = (time.time() - t_start_view) * 1000.0
            cam_name = frame_data.get('name', f'view_{f_idx}')
            print(f"   [Scan Ingest {f_idx+1:02d}/{num_views:02d}] View '{cam_name}': +{n_new_g:,} new Gaussians | Total Raw: {raw_gaussians_count:,} | {t_view_ms:.1f}ms")

            if n_new_g > 0:
                new_xyz_acc.append(new_g['xyz'])
                new_rgb_acc.append(new_g['rgb'])
                new_scale_acc.append(new_g['scale'])
                if 'normal' in new_g and len(new_g['normal']) == n_new_g:
                    new_normal_acc.append(new_g['normal'])
                else:
                    new_normal_acc.append(torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(n_new_g, 1))
                new_morton_acc.append(new_g['morton'])
                new_obj_id_acc.append(new_g.get('obj_id', torch.zeros(len(new_g['xyz']), dtype=torch.int32, device=self.device)))

        # Deduplication by Morton codes (1 Gaussian per 1cm voxel surface)
        if len(new_xyz_acc) > 0:
            added_xyz = torch.cat(new_xyz_acc, dim=0)
            added_rgb = torch.cat(new_rgb_acc, dim=0)
            added_scale = torch.cat(new_scale_acc, dim=0)
            added_normal = torch.cat(new_normal_acc, dim=0)
            added_morton = torch.cat(new_morton_acc, dim=0)
            added_obj_id = torch.cat(new_obj_id_acc, dim=0)

            if len(added_morton) > 0:
                perm = torch.argsort(added_morton)
                sorted_morton = added_morton[perm]
                uniq_mask = torch.ones_like(sorted_morton, dtype=torch.bool)
                uniq_mask[1:] = (sorted_morton[1:] != sorted_morton[:-1])
                keep_idx = perm[uniq_mask]

                self.scene_gaussians['xyz'] = added_xyz[keep_idx]
                self.scene_gaussians['rgb'] = added_rgb[keep_idx]
                self.scene_gaussians['scale'] = added_scale[keep_idx]
                self.scene_gaussians['normal'] = added_normal[keep_idx]
                self.scene_gaussians['morton'] = added_morton[keep_idx]
                self.scene_gaussians['obj_id'] = added_obj_id[keep_idx]

        retained_count = len(self.scene_gaussians['xyz'])
        reduction_pct = ((raw_gaussians_count - retained_count) / max(1, raw_gaussians_count)) * 100.0
        print(f"\n✓ [VG-Mapping] TSDF volumetric integration complete ({num_views} views).")
        print(f"✓ [VG-Mapping Deduplication] Morton surface pruning:")
        print(f"   Raw Gaussians accumulated: {raw_gaussians_count:,}")
        print(f"   Unique 1cm voxel surface Gaussians retained: {retained_count:,}")
        print(f"   Pruned redundant primitives: {raw_gaussians_count - retained_count:,} ({reduction_pct:.1f}% reduction).")

        # Update Viser with initial Gaussian Splatting scene
        self._update_viser_gaussians()

        # 8. Extract Tabletop Obstacle Meshes via TSDF Marching Cubes
        z_cutoff = z_table + OBSTACLE_MIN_CLEARANCE_Z
        verts, faces = extract_obstacle_mesh_from_tsdf(
            self.vg_pipeline.tsdf_map,
            z_min_cutoff=z_cutoff,
            z_max_cutoff=act_z_max,
            x_bounds=(tab_x_min + 0.02, tab_x_max - 0.02),
            y_bounds=(tab_y_min + 0.02, tab_y_max - 0.02),
            level=0.0
        )
        print(f"✓ [VG-Mapping Marching Cubes] Extracted raw obstacle surface mesh: {len(verts)} vertices, {len(faces)} faces.")

        spawned_obstacles = 0
        if len(verts) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            components = mesh.split(only_watertight=False)

            # If trimesh split yielded components, process each valid component
            valid_components = []
            for comp in components:
                ext = comp.bounds[1] - comp.bounds[0]
                center_c = (comp.bounds[0] + comp.bounds[1]) / 2.0
                # Must reside strictly within tabletop bounds and workspace ceiling
                in_table = (
                    (center_c[0] >= tab_x_min + 0.02) and (center_c[0] <= tab_x_max - 0.02) and
                    (center_c[1] >= tab_y_min + 0.02) and (center_c[1] <= tab_y_max - 0.02) and
                    (center_c[2] >= z_table + 0.01) and (center_c[2] <= act_z_max)
                )
                if in_table and np.max(ext) >= 0.02 and len(comp.vertices) >= 20:
                    valid_components.append(comp)

            # Fallback if no split components passed filter but raw mesh is valid
            if len(valid_components) == 0 and len(verts) >= 20:
                valid_components = [mesh]

            valid_components.sort(key=lambda c: len(c.vertices), reverse=True)
            print(f"✓ [VG-Mapping Marching Cubes] Discovered {len(valid_components)} distinct tabletop obstacle mesh(es).")

            # Semantic names for physical scene objects if available
            available_obj_names = [id_to_name[t] for t in self.dynamic_object_ids if t in id_to_name]

            for c_idx, comp in enumerate(valid_components):
                comp_verts = comp.vertices
                comp_faces = comp.faces
                min_b, max_b = comp.bounds[0], comp.bounds[1]
                center = (min_b + max_b) / 2.0
                extents = max_b - min_b

                if len(available_obj_names) > 0:
                    obj_name = available_obj_names.pop(0)
                else:
                    obj_name = f"obstacle_{spawned_obstacles}"

                # Center mesh vertices around center of bounding box
                verts_centered = comp_verts - center
                comp_centered = trimesh.Trimesh(vertices=verts_centered, faces=comp_faces)

                obs_mesh_file = os.path.join(self.output_mesh_dir, f"{obj_name}_{spawned_obstacles}.obj")
                comp_centered.export(obs_mesh_file)

                # Associate canonical Gaussian points from scene_gaussians
                xyz_g = self.scene_gaussians['xyz']
                rgb_g = self.scene_gaussians['rgb']
                in_box = (xyz_g[:, 0] >= min_b[0] - 0.02) & (xyz_g[:, 0] <= max_b[0] + 0.02) & \
                         (xyz_g[:, 1] >= min_b[1] - 0.02) & (xyz_g[:, 1] <= max_b[1] + 0.02) & \
                         (xyz_g[:, 2] >= min_b[2] - 0.01) & (xyz_g[:, 2] <= max_b[2] + 0.02)

                clean_xyz = xyz_g[in_box]
                clean_rgb = rgb_g[in_box]
                if len(clean_xyz) == 0:
                    clean_xyz = torch.from_numpy(comp_verts).float().to(self.device)
                    clean_rgb = torch.full((len(clean_xyz), 3), 0.5, dtype=torch.float32, device=self.device)

                # Real object color computed as mean RGB of its Gaussians
                if len(clean_rgb) > 0:
                    mean_rgb = clean_rgb.mean(dim=0).cpu().numpy()
                    obj_color = (
                        float(np.clip(mean_rgb[0], 0.0, 1.0)),
                        float(np.clip(mean_rgb[1], 0.0, 1.0)),
                        float(np.clip(mean_rgb[2], 0.0, 1.0)),
                        1.0
                    )
                else:
                    obj_color = (0.5, 0.5, 0.5, 1.0)

                body_id = self.digital_twin.spawn_mesh_object(
                    obj_id=spawned_obstacles,
                    mesh_file_path=obs_mesh_file,
                    initial_position=tuple(center.tolist()),
                    initial_orientation=(0, 0, 0, 1),
                    color=obj_color,
                    is_target=False
                )

                self.tracked_objects[spawned_obstacles] = {
                    'body_id': body_id,
                    'name': obj_name,
                    'initial_pos': tuple(center.tolist()),
                    'dims': extents.tolist(),
                    'canonical_points': {
                        'xyz': clean_xyz.clone(),
                        'rgb': clean_rgb.clone()
                    },
                    'last_pos': tuple(center.tolist()),
                    'last_quat': (0, 0, 0, 1),
                    'last_T': torch.eye(4, device=self.device)
                }
                # Add Marching Cubes mesh to Viser 3D Web Visualizer
                self._add_viser_obstacle_mesh(obj_name, spawned_obstacles, comp)

                print(f"   -> Object #{spawned_obstacles} ('{obj_name}'):")
                print(f"      Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] m | Extents: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}] m")
                print(f"      Associated Gaussians: {len(clean_xyz):,} | Real RGB Color: [{obj_color[0]:.2f}, {obj_color[1]:.2f}, {obj_color[2]:.2f}]")
                print(f"      Spawned in PyBullet Digital Twin (Body ID: {body_id})")
                spawned_obstacles += 1
        else:
            print("[DREMA Discovery] Marching Cubes yielded 0 tabletop obstacle vertices above table.")

        # 10. Load Franka Panda in PyBullet Digital Twin using REAL initial joint angles from CoppeliaSim
        if len(self.robot_joint_positions) > 0 and len(self.robot_base_pos) >= 3:
            self.digital_twin.load_robot(
                base_position=tuple(self.robot_base_pos.tolist()),
                joint_positions=list(self.robot_joint_positions)
            )

        self.initial_scan_ready = True
        print(f"\n✓ [DREMA Suite] Initial Scene Setup Complete! Active Workspace Ready.")
        print("=======================================================\n")
        return True

    def on_frame_received(self, obs: drema_comm_pb2.FrameObservation) -> Optional[drema_comm_pb2.StreamStatus]:
        """gRPC callback triggered when a camera frame arrives from CoppeliaSim."""
        # Update robot base, reachability radius, and joints if transmitted by client
        if len(obs.robot_base_pos) >= 3:
            self.robot_base_pos = np.array(obs.robot_base_pos[:3], dtype=np.float32)
        if obs.reachability_radius > 0:
            self.reachability_radius = float(obs.reachability_radius)
        if len(obs.joint_positions) > 0:
            self.robot_joint_positions = list(obs.joint_positions)
            if self.digital_twin.robot_id >= 0:
                self.digital_twin.sync_robot_state(self.robot_joint_positions)

        if obs.is_initial_scan:
            for f in obs.cameras:
                name, rgb, depth, extrinsics, intrinsics, near_clip, far_clip, mask = unpack_camera_frame(f)

                # Back-project depth points using camera projection
                pcd = pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics)
                # Valid points within the physical sensor clipping bounds
                valid = (depth > near_clip) & (depth < far_clip)
                pts = pcd[valid]

                # Accumulate all valid points across all scanning views (for table plane discovery)
                if len(pts) > 0:
                    self.accumulated_scan_points.append(pts)

                # Save the full observation frame for VG-Mapping TSDF & 3DGS ingestion
                self.accumulated_scan_frames.append({
                    'name': name,
                    'rgb': rgb,
                    'depth': depth,
                    'extrinsics': extrinsics,
                    'intrinsics': intrinsics,
                    'near_clip': near_clip,
                    'far_clip': far_clip,
                    'mask': mask
                })

            if obs.is_scan_finished:
                semantic_labels = dict(obs.semantic_labels) if obs.semantic_labels else None
                success = self._process_initial_scene_scan(semantic_labels=semantic_labels)
                return drema_comm_pb2.StreamStatus(
                    success=success,
                    message="Initial 360° scan processed and PyBullet Digital Twin populated" if success else "Scan processing failed",
                    received_timestep=0,
                    initial_scan_ready=self.initial_scan_ready
                )
            return drema_comm_pb2.StreamStatus(
                success=True,
                message="Initial scan frame ingested",
                received_timestep=0,
                initial_scan_ready=False
            )

        # Standard real-time streaming frame
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
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

    def _track_and_sync_objects(self):
        """Performs batched RecurGS SE(3) optimization and PyBullet physics synchronization."""
        if not (self.initial_scan_ready and len(self.tracked_objects) > 0 and len(self.scene_gaussians['xyz']) > 0 and self.active_workspace_bounds is not None):
            return

        try:
            objects_source = {}
            objects_target = {}
            initial_T_coarse_dict = {}

            xyz_curr = self.scene_gaussians['xyz']
            rgb_curr = self.scene_gaussians['rgb']
            ws = self.active_workspace_bounds
            z_tab = ws['z_table']

            # Foreground workspace candidate points above table surface
            table_clean_mask = (xyz_curr[:, 2] > (z_tab + 0.008)) & (xyz_curr[:, 2] <= ws['z_max']) & \
                               (xyz_curr[:, 0] >= ws['x_min']) & (xyz_curr[:, 0] <= ws['x_max']) & \
                               (xyz_curr[:, 1] >= ws['y_min']) & (xyz_curr[:, 1] <= ws['y_max'])
            cand_xyz = xyz_curr[table_clean_mask]
            cand_rgb = rgb_curr[table_clean_mask]

            for oid, obj_info in self.tracked_objects.items():
                src_xyz = obj_info['canonical_points']['xyz']
                src_rgb = obj_info['canonical_points']['rgb']
                if len(src_xyz) == 0 or len(cand_xyz) < 4:
                    continue

                # Subsample canonical points if large (up to 256 points)
                N_src = len(src_xyz)
                if N_src > 256:
                    sub_s = torch.randperm(N_src, device=self.device)[:256]
                    objects_source[oid] = {'xyz': src_xyz[sub_s], 'rgb': src_rgb[sub_s]}
                else:
                    objects_source[oid] = {'xyz': src_xyz, 'rgb': src_rgb}

                # Target points in proximity to last known position
                last_p = np.array(obj_info['last_pos'])
                dist_p = torch.norm(cand_xyz - torch.tensor(last_p, device=self.device, dtype=torch.float32), dim=1)
                near_mask = dist_p < 0.25
                if torch.any(near_mask) and near_mask.sum() >= 4:
                    tgt_xyz = cand_xyz[near_mask]
                    tgt_rgb = cand_rgb[near_mask]
                else:
                    tgt_xyz = cand_xyz
                    tgt_rgb = cand_rgb

                N_tgt = len(tgt_xyz)
                if N_tgt > 256:
                    sub_t = torch.randperm(N_tgt, device=self.device)[:256]
                    objects_target[oid] = {'xyz': tgt_xyz[sub_t], 'rgb': tgt_rgb[sub_t]}
                else:
                    objects_target[oid] = {'xyz': tgt_xyz, 'rgb': tgt_rgb}

                initial_T_coarse_dict[oid] = obj_info.get('last_T', torch.eye(4, device=self.device))

            # Batched RecurGS Lie algebra optimization
            if len(objects_source) > 0 and len(objects_target) > 0:
                T_fine_dict = self.vg_pipeline.step_3_estimate_multi_se3_motion(
                    objects_source=objects_source,
                    objects_target=objects_target,
                    initial_T_coarse_dict=initial_T_coarse_dict,
                    z_table=z_tab,
                    num_iterations=15,
                    icp_max_iters=12,
                    lr=3e-3,
                    tol=1e-4
                )

                for oid, T_fine in T_fine_dict.items():
                    self.tracked_objects[oid]['last_T'] = T_fine.detach()
                    c0 = self.tracked_objects[oid]['initial_pos']
                    half_h = self.tracked_objects[oid]['dims'][2] / 2.0

                    R_fine = T_fine[:3, :3]
                    t_fine = T_fine[:3, 3]
                    c0_t = torch.tensor(c0, dtype=torch.float32, device=self.device)
                    pos_w = R_fine @ c0_t + t_fine
                    pos_z = max(float(pos_w[2].item()), float(z_tab) + float(half_h))
                    new_pos = (float(pos_w[0].item()), float(pos_w[1].item()), pos_z)
                    quat = rotation_matrix_to_quaternion(R_fine)

                    self.digital_twin.sync_object_pose(oid, new_pos, quat)
                    self.tracked_objects[oid]['last_pos'] = new_pos
                    self.tracked_objects[oid]['last_quat'] = quat

                    # Synchronize Marching Cubes obstacle mesh in Viser Web Visualizer in real-time
                    mesh_handle = self.viser_handles.get(f"mesh_{oid}")
                    if mesh_handle is not None:
                        mesh_handle.position = new_pos
                        mesh_handle.wxyz = (quat[3], quat[0], quat[1], quat[2])

                    # Rotate obstacle surface normals with estimated SE(3) rotation
                    obj_mask = (self.scene_gaussians['obj_id'] == oid)
                    if torch.any(obj_mask) and 'normal' in self.scene_gaussians and len(self.scene_gaussians['normal']) == len(obj_mask):
                        self.scene_gaussians['normal'][obj_mask] = self.scene_gaussians['normal'][obj_mask] @ R_fine.T
        except Exception:
            pass

    def _vg_mapping_worker(self):
        """Background thread executing 3D reconstruction and SE(3) tracking."""
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

            # 1. Ingest streaming multi-camera frames into TSDF and VDC
            if self.vg_pipeline is not None and len(camera_views) > 0:
                workspace_bounds_t = self.workspace_bounds_t

                for cam_name, cam_data in camera_views.items():
                    try:
                        d_tensor = torch.from_numpy(cam_data['depth'].copy()).unsqueeze(0).to(self.device)
                        k_tensor = torch.from_numpy(cam_data['intrinsics'].copy()).to(self.device)
                        t_tensor = torch.from_numpy(cam_data['extrinsics'].copy()).to(self.device)
                        rgb_tensor = torch.from_numpy(cam_data['rgb'].copy()).permute(2, 0, 1).float().to(self.device) / 255.0
                        mask_np = cam_data.get('mask')

                        d_masked, mask_t = self._apply_semantic_robot_mask(
                            depth_t=d_tensor,
                            mask_np=mask_np
                        )

                        # Step 1: TSDF integration (robot depth masked to 0)
                        self.vg_pipeline.step_1_ingest_frame(
                            rgb=rgb_tensor,
                            depth=d_masked,
                            intrinsic=k_tensor,
                            camera_pose=t_tensor
                        )

                        # Step 2: VDC variation detection & raycast pruning
                        rendered_rgb = rgb_tensor.clone()
                        rendered_depth = d_masked.clone()

                        new_g, prune_mask = self.vg_pipeline.step_2_online_mapping(
                            rgb=rgb_tensor,
                            depth=d_masked,
                            rendered_rgb=rendered_rgb,
                            rendered_depth=rendered_depth,
                            intrinsic=k_tensor,
                            camera_pose=t_tensor,
                            current_morton_codes=self.scene_gaussians['morton'],
                            mask=mask_t,
                            workspace_bounds=workspace_bounds_t,
                            is_initial_timestep=False,
                            num_views=len(camera_views),
                            robot_ids=(self.robot_ids | self.virtual_ids),
                            target_object_ids=self.dynamic_object_ids
                        )

                        # Apply pruning
                        if len(prune_mask) > 0 and torch.any(prune_mask):
                            keep_mask = ~prune_mask
                            for k in ['xyz', 'rgb', 'scale', 'normal', 'morton', 'obj_id']:
                                if k in self.scene_gaussians and len(self.scene_gaussians[k]) == len(keep_mask):
                                    self.scene_gaussians[k] = self.scene_gaussians[k][keep_mask]

                        # Add new Gaussians with Morton deduplication & robot cylinder exclusion
                        if len(new_g['xyz']) > 0 and len(self.robot_base_pos) >= 3:
                            rx, ry, rz = float(self.robot_base_pos[0]), float(self.robot_base_pos[1]), float(self.robot_base_pos[2])
                            p_xy = new_g['xyz'][:, :2]
                            dist_base = torch.sqrt((p_xy[:, 0] - rx) ** 2 + (p_xy[:, 1] - ry) ** 2)
                            keep_geom = (dist_base > 0.13) | (new_g['xyz'][:, 2] < (rz - 0.05))
                            if not torch.all(keep_geom):
                                for k in list(new_g.keys()):
                                    if isinstance(new_g[k], torch.Tensor) and len(new_g[k]) == len(new_g['xyz']):
                                        new_g[k] = new_g[k][keep_geom]

                        if len(new_g['xyz']) > 0:
                            added_morton = new_g['morton']
                            if len(self.scene_gaussians['morton']) > 0:
                                occupied = torch.isin(added_morton, self.scene_gaussians['morton'])
                                non_dup = ~occupied
                            else:
                                non_dup = torch.ones_like(added_morton, dtype=torch.bool)

                            for k in ['xyz', 'rgb', 'scale', 'normal', 'morton', 'obj_id']:
                                if k in new_g:
                                    val = new_g[k]
                                elif k == 'normal':
                                    val = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(len(new_g['xyz']), 1)
                                else:
                                    val = torch.zeros(len(new_g['xyz']), dtype=torch.int32, device=self.device)
                                self.scene_gaussians[k] = torch.cat([self.scene_gaussians[k], val[non_dup]], dim=0)

                    except Exception as e:
                        pass

            # 2. RecurGS SE(3) Tracking & PyBullet Physics Synchronization
            self._track_and_sync_objects()

            self.total_frames_processed += 1
            elapsed = (time.time() - t0) * 1000.0

            # Periodically refresh Viser 3D web point cloud
            if self.total_frames_processed % 3 == 0:
                self._update_viser_gaussians()

            if self.total_frames_processed % 10 == 0:
                print(f"[Dynamic Inference #{timestep:04d}] Active Gaussians: {len(self.scene_gaussians['xyz']):,} | Loop Latency: {elapsed:.1f}ms | Tracked Objects: {len(self.tracked_objects)}")

            self.frame_queue.task_done()

    def _update_viser_gaussians(self):
        """Updates the live 3D Gaussian Splats in Viser Web Visualizer."""
        if self.viser_server is None or len(self.scene_gaussians['xyz']) == 0:
            return
        try:
            pts_np = self.scene_gaussians['xyz'].detach().cpu().numpy()
            rgb_np = self.scene_gaussians['rgb'].detach().cpu().numpy()
            scale_np = self.scene_gaussians['scale'].detach().cpu().numpy()
            rgb_np = np.clip(rgb_np, 0.0, 1.0)

            # Extract surface normals dynamically from scene_gaussians
            normals_np = self.scene_gaussians.get('normal', torch.empty(0)).detach().cpu().numpy() if 'normal' in self.scene_gaussians else None

            # Subsample if extremely large for responsive 60fps web streaming
            if len(pts_np) > 50000:
                sub = np.random.choice(len(pts_np), 50000, replace=False)
                pts_np = pts_np[sub]
                rgb_np = rgb_np[sub]
                scale_np = scale_np[sub]
                if normals_np is not None and len(normals_np) == len(self.scene_gaussians['xyz']):
                    normals_np = normals_np[sub]

            # Dynamically normalize surface normal vectors
            if normals_np is not None and len(normals_np) == len(pts_np):
                n_norms = np.linalg.norm(normals_np, axis=-1, keepdims=True)
                valid_n = (n_norms > 1e-4).squeeze(-1)
                normals_clean = np.zeros_like(normals_np)
                normals_clean[valid_n] = normals_np[valid_n] / n_norms[valid_n]
                normals_clean[~valid_n] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                normals_clean = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(pts_np), 1))

            # Dynamic scale: s_tan across local surface tangent plane, s_norm through surface thickness
            s_tan = np.maximum(scale_np[:, 0:1], 0.012)
            s_norm = np.maximum(scale_np[:, 1:2], 0.003)

            # Dynamic Anisotropic Covariance: Sigma = s_tan^2 * I + (s_norm^2 - s_tan^2) * (n n^T)
            # Aligns each disc-like 3D Gaussian flatly against the physical surface tangent plane (Eq. 16 VG-Mapping)
            nnT = normals_clean[:, :, None] @ normals_clean[:, None, :]
            eye3 = np.eye(3, dtype=np.float32)[None, :, :]
            covariances = (s_tan[:, :, None] ** 2) * eye3 + (s_norm[:, :, None] ** 2 - s_tan[:, :, None] ** 2) * nnT
            opacities = np.full((len(pts_np), 1), 0.95, dtype=np.float32)

            h = self.viser_server.scene.add_gaussian_splats(
                name="/scene/gaussians",
                centers=pts_np,
                covariances=covariances,
                rgbs=rgb_np,
                opacities=opacities,
                scale=1.0
            )
            self.viser_handles['gaussians'] = h
        except Exception:
            # Fallback to point cloud if add_gaussian_splats encounters an issue
            try:
                h = self.viser_server.scene.add_point_cloud(
                    name="/scene/gaussians",
                    points=pts_np,
                    colors=rgb_np,
                    point_size=0.008,
                    point_shape="circle"
                )
                self.viser_handles['gaussians'] = h
            except Exception:
                pass

    def on_request_action(self, robot_state: drema_comm_pb2.RobotState) -> drema_comm_pb2.ControlAction:
        """gRPC callback triggered when CoppeliaSim requests joint velocity command."""
        self.total_actions_served += 1

        # Update robot base and reachability if provided
        if len(robot_state.robot_base_pos) >= 3:
            self.robot_base_pos = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)
        if robot_state.reachability_radius > 0:
            self.reachability_radius = float(robot_state.reachability_radius)

        # 1. Update PyBullet Digital Twin robot configuration (load dynamically if not loaded yet)
        if self.digital_twin.robot_id < 0 and len(self.robot_base_pos) >= 3:
            self.digital_twin.load_robot(
                base_position=tuple(self.robot_base_pos.tolist()),
                joint_positions=list(robot_state.joint_positions) if len(robot_state.joint_positions) > 0 else None
            )
        elif len(robot_state.joint_positions) > 0:
            self.digital_twin.sync_robot_state(robot_state.joint_positions)

        # 2. Step PyBullet physics forward
        self.digital_twin.step()

        # 3. Extract target goal (if provided by CoppeliaSim oracle or PerAct policy)
        target_goal = None
        if robot_state.target_available and len(robot_state.target_pose) >= 3:
            target_goal = np.array(robot_state.target_pose[:3], dtype=np.float32)

        # 4. Compute control action via MPC Controller
        action = self.mpc_controller.compute_action(
            robot_state=robot_state,
            digital_twin=self.digital_twin,
            target_goal=target_goal
        )

        if self.total_actions_served % 100 == 0:
            mode = "ACTIVE" if robot_state.task_active else "IDLE"
            print(f"[MPC Controller] Action #{self.total_actions_served} ({mode} | {action.status_message})")

        return action

    def on_reset_episode(self, reset_req: drema_comm_pb2.ResetRequest) -> bool:
        """gRPC callback triggered when an episode resets."""
        print(f"[DREMA Suite] Resetting episode {reset_req.episode_index} for task {reset_req.task_name}...")
        self.initial_scan_ready = False
        self.accumulated_scan_points = []
        self.accumulated_scan_frames = []
        self.semantic_labels = {}
        self.robot_ids = set()
        self.virtual_ids = set()
        self.dynamic_object_ids = set()
        self.tracked_objects = {}
        self.active_workspace_bounds = None
        self.workspace_bounds_t = None
        self.robot_joint_positions = []
        self.scene_gaussians = {
            'xyz': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'rgb': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'scale': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'normal': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
            'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device)
        }
        self.digital_twin.reset()
        self.mpc_controller.reset()
        return True

    def start(self):
        self.server.start()
        print(f"✓ DREMA Dynamic Inference Suite fully ready and listening for CoppeliaSim!")

    def spin(self):
        """Keeps main thread alive until interrupted."""
        try:
            while not self.stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down DREMA Dynamic Inference Suite...")
        finally:
            self.stop()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        self.server.stop(grace=0.5)
        self.digital_twin.close()
        if self.viser_server is not None:
            try:
                self.viser_server.stop()
            except Exception:
                pass
        print("✓ DREMA Suite cleanly stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DREMA Dynamic Inference Suite")
    parser.add_argument("--port", type=int, default=50051, help="gRPC Server port (default: 50051)")
    parser.add_argument("--visualize_pybullet", action="store_true", help="Open PyBullet GUI window for real-time visualization")
    parser.add_argument("--visualize_viser", action="store_true", default=True, help="Launch real-time 3D Viser Web Visualizer (default: True)")
    parser.add_argument("--no_viser", dest="visualize_viser", action="store_false", help="Disable Viser Web Visualizer")
    parser.add_argument("--viser_port", type=int, default=8080, help="Viser Web Visualizer port (default: 8080)")
    parser.add_argument("--table_z_prior", type=float, default=0.75, help="Preliminary prior for table Z search in meters (default: 0.75)")
    parser.add_argument("--reachability_radius", type=float, default=0.95, help="Robot maximum reachable radius in meters (default: 0.95)")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="TSDF voxel grid resolution (default: 0.01m)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device (cuda/cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suite = DremaDynamicSuite(
        port=args.port,
        visualize_pybullet=args.visualize_pybullet,
        visualize_viser=args.visualize_viser,
        viser_port=args.viser_port,
        table_z_prior=args.table_z_prior,
        reachability_radius=args.reachability_radius,
        voxel_size=args.voxel_size,
        device=args.device
    )
    suite.start()
    suite.spin()
