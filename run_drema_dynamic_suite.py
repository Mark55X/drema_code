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
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse import csgraph

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


class DremaDynamicSuite:
    """
    Coordinator managing the 3 submodules and gRPC communication.
    """

    def __init__(
        self,
        port: int = 50051,
        visualize_pybullet: bool = False,
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

        # Robot base & dynamic active workspace (updated via initial scan & gRPC)
        self.robot_base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.active_workspace_bounds: Optional[Dict[str, float]] = None

        print(f"==================================================")
        print(f"   DREMA Dynamic Inference Suite Starting...     ")
        print(f"   Device: {self.device} | Port: {self.port}       ")
        print(f"==================================================")

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
        self.tracked_objects = {}
        self.output_mesh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/scanned_meshes")
        os.makedirs(self.output_mesh_dir, exist_ok=True)

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

        # 4. Spawn Solid Table in PyBullet Digital Twin (Full geometry)
        self.digital_twin.spawn_scanned_table(
            table_z=z_table,
            bounds=table_bounds
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

        # 6. Discover Tabletop Obstacles (Strictly within Active Workspace, no arbitrary height ceiling)
        dist_to_robot = np.sqrt((all_pts[:, 0] - rb_x)**2 + (all_pts[:, 1] - rb_y)**2)
        obj_mask = (all_pts[:, 2] > (z_table + OBSTACLE_MIN_CLEARANCE_Z)) & (all_pts[:, 2] <= act_z_max) & \
                   (all_pts[:, 0] >= act_x_min) & (all_pts[:, 0] <= act_x_max) & \
                   (all_pts[:, 1] >= act_y_min) & (all_pts[:, 1] <= act_y_max) & \
                   (dist_to_robot <= r_reach) & (dist_to_robot >= ROBOT_BASE_RADIUS)
        obj_pts = all_pts[obj_mask]
        print(f"[DREMA Discovery] Tabletop obstacle points detected in active workspace: {len(obj_pts)}")

        if len(obj_pts) >= 30:
            voxel_idx = np.floor(obj_pts / self.voxel_size).astype(int)
            _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
            ds_pts = obj_pts[unique_idx]

            tree = cKDTree(ds_pts)
            adj = tree.sparse_distance_matrix(tree, max_distance=0.04)
            n_comp, labels = csgraph.connected_components(adj)

            cluster_list = []
            for lbl in range(n_comp):
                c_pts = ds_pts[labels == lbl]
                if len(c_pts) >= 15:
                    cluster_list.append(c_pts)

            cluster_list.sort(key=lambda c: len(c), reverse=True)
            print(f"[DREMA Discovery] Found {len(cluster_list)} distinct object cluster(s) in active workspace.")

            spawned_obstacles = 0
            for c_idx, c_pts in enumerate(cluster_list):
                center = np.mean(c_pts, axis=0)
                extents = np.max(c_pts, axis=0) - np.min(c_pts, axis=0)
                max_dim = float(np.max(extents))

                is_tunnel_obstacle = (max_dim > 0.10)
                obj_name = "tunnel_obstacle" if is_tunnel_obstacle else f"object_{c_idx}"

                print(f"  Cluster #{c_idx} ({obj_name}): {len(c_pts)} points, center=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}], size=[{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")

                pts_centered = c_pts - center
                obs_mesh_file = os.path.join(self.output_mesh_dir, f"scanned_obstacle_{spawned_obstacles}.obj")

                try:
                    hull_o = ConvexHull(pts_centered)
                    with open(obs_mesh_file, "w") as f:
                        for v in pts_centered:
                            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        for s in hull_o.simplices:
                            f.write(f"f {s[0]+1} {s[1]+1} {s[2]+1}\n")
                except Exception:
                    hx, hy, hz = [max(0.02, float(d) / 2.0) for d in extents]
                    verts = [[-hx,-hy,-hz],[hx,-hy,-hz],[hx,hy,-hz],[-hx,hy,-hz],
                             [-hx,-hy,hz],[hx,-hy,hz],[hx,hy,hz],[-hx,hy,hz]]
                    faces = [(1,2,3),(1,3,4),(5,7,6),(5,8,7),(1,6,2),(1,5,6),
                             (2,7,3),(2,6,7),(3,8,4),(3,7,8),(4,5,1),(4,8,5)]
                    with open(obs_mesh_file, "w") as f:
                        for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                        for s in faces: f.write(f"f {s[0]} {s[1]} {s[2]}\n")

                body_id = self.digital_twin.spawn_mesh_object(
                    obj_id=spawned_obstacles,
                    mesh_file_path=obs_mesh_file,
                    initial_position=tuple(center.tolist()),
                    initial_orientation=(0, 0, 0, 1),
                    color=(0.2, 0.45, 0.85, 1.0) if is_tunnel_obstacle else (0.85, 0.2, 0.2, 1.0),
                    is_target=not is_tunnel_obstacle
                )

                self.tracked_objects[spawned_obstacles] = {
                    'body_id': body_id,
                    'name': obj_name,
                    'initial_pos': tuple(center.tolist()),
                    'canonical_pts': pts_centered,
                    'last_pos': tuple(center.tolist()),
                    'last_quat': (0, 0, 0, 1),
                    'dims': extents.tolist()
                }
                print(f"✓ Spawned '{obj_name}' in PyBullet Digital Twin (Body ID: {body_id}) from 3D scan mesh")
                spawned_obstacles += 1

        self.initial_scan_ready = True
        print(f"\n✓ [DREMA Suite] Initial Scene Setup Complete! Active Workspace Ready.")
        print("=======================================================\n")
        return True

    def on_frame_received(self, obs: drema_comm_pb2.FrameObservation) -> Optional[drema_comm_pb2.StreamStatus]:
        """gRPC callback triggered when a camera frame arrives from CoppeliaSim."""
        # Update robot base and reachability radius if transmitted by client
        if len(obs.robot_base_pos) >= 3:
            self.robot_base_pos = np.array(obs.robot_base_pos[:3], dtype=np.float32)
        if obs.reachability_radius > 0:
            self.reachability_radius = float(obs.reachability_radius)

        if obs.is_initial_scan:
            for f in obs.cameras:
                name, rgb, depth, extrinsics, intrinsics, near_clip, far_clip = unpack_camera_frame(f)

                # Back-project depth points using camera projection
                pcd = pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics)
                # Valid points within the physical sensor clipping bounds
                valid = (depth > near_clip) & (depth < far_clip)
                pts = pcd[valid]

                # Accumulate all valid points across all scanning views (no premature arbitrary box cropping)
                if len(pts) > 0:
                    self.accumulated_scan_points.append(pts)

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
                name, rgb, depth, extrinsics, intrinsics, near_clip, far_clip = unpack_camera_frame(f)
                camera_views[name] = {
                    'rgb': rgb,
                    'depth': depth,
                    'extrinsics': extrinsics,
                    'intrinsics': intrinsics,
                    'near_clipping': near_clip,
                    'far_clipping': far_clip
                }

            # If VG-Mapping pipeline is available, integrate depth frames
            if self.vg_pipeline is not None and len(camera_views) > 0:
                try:
                    first_cam = list(camera_views.values())[0]
                    d_tensor = torch.from_numpy(first_cam['depth'].copy()).to(self.device)
                    k_tensor = torch.from_numpy(first_cam['intrinsics'].copy()).to(self.device)
                    t_tensor = torch.from_numpy(first_cam['extrinsics'].copy()).to(self.device)
                    rgb_tensor = torch.from_numpy(first_cam['rgb'].copy()).float().to(self.device) / 255.0

                    self.vg_pipeline.step_1_ingest_frame(
                        rgb=rgb_tensor,
                        depth=d_tensor,
                        intrinsic=k_tensor,
                        camera_pose=t_tensor
                    )
                except Exception as e:
                    print(f"[VG-Mapping Worker] Exception during frame {timestep} integration: {e}")

            # If obstacle 0 is tracked, update its position via real-time centroid tracking in Active Workspace
            if self.initial_scan_ready and 0 in self.tracked_objects and len(camera_views) > 0 and self.active_workspace_bounds is not None:
                try:
                    first_cam = list(camera_views.values())[0]
                    depth_rt = first_cam['depth']
                    ext_rt = first_cam['extrinsics']
                    int_rt = first_cam['intrinsics']
                    near_c = first_cam['near_clipping']
                    far_c = first_cam['far_clipping']

                    H, W = depth_rt.shape
                    u_g, v_g = np.meshgrid(np.arange(0, W, 4), np.arange(0, H, 4))
                    d_vals = depth_rt[v_g, u_g]
                    valid = (d_vals > near_c) & (d_vals < far_c)

                    u_v, v_v, d_v = u_g[valid], v_g[valid], d_vals[valid]
                    fx, fy = int_rt[0, 0], int_rt[1, 1]
                    cx, cy = int_rt[0, 2], int_rt[1, 2]
                    x_c = (u_v - cx) * d_v / fx
                    y_c = (v_v - cy) * d_v / fy
                    p_c = np.stack([x_c, y_c, d_v, np.ones_like(d_v)], axis=-1)
                    p_w = (ext_rt @ p_c.T).T[:, :3]

                    ws = self.active_workspace_bounds
                    z_tab = ws['z_table']
                    rb_x, rb_y = float(self.robot_base_pos[0]), float(self.robot_base_pos[1])
                    dist_rt = np.sqrt((p_w[:, 0] - rb_x)**2 + (p_w[:, 1] - rb_y)**2)

                    obs_m = (p_w[:, 2] > (z_tab + 0.015)) & (p_w[:, 2] <= ws['z_max']) & \
                            (p_w[:, 0] >= ws['x_min']) & (p_w[:, 0] <= ws['x_max']) & \
                            (p_w[:, 1] >= ws['y_min']) & (p_w[:, 1] <= ws['y_max']) & \
                            (dist_rt <= self.reachability_radius) & (dist_rt >= 0.12)
                    pts_obs_curr = p_w[obs_m]

                    if len(pts_obs_curr) >= 20:
                        curr_center = np.mean(pts_obs_curr, axis=0)
                        self.digital_twin.sync_object_pose(0, tuple(curr_center.tolist()), (0, 0, 0, 1))
                        self.tracked_objects[0]['last_pos'] = tuple(curr_center.tolist())
                except Exception:
                    pass

            self.total_frames_processed += 1
            elapsed = (time.time() - t0) * 1000.0

            if self.total_frames_processed % 20 == 0:
                print(f"[VG-Mapping] Ingested Frame #{timestep} ({len(camera_views)} views) in {elapsed:.1f}ms")

            self.frame_queue.task_done()

    def on_request_action(self, robot_state: drema_comm_pb2.RobotState) -> drema_comm_pb2.ControlAction:
        """gRPC callback triggered when CoppeliaSim requests joint velocity command."""
        self.total_actions_served += 1

        # Update robot base and reachability if provided
        if len(robot_state.robot_base_pos) >= 3:
            self.robot_base_pos = np.array(robot_state.robot_base_pos[:3], dtype=np.float32)
        if robot_state.reachability_radius > 0:
            self.reachability_radius = float(robot_state.reachability_radius)

        # 1. Update PyBullet Digital Twin robot joint configuration
        if len(robot_state.joint_positions) > 0:
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
        self.tracked_objects = {}
        self.active_workspace_bounds = None
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
        print("✓ DREMA Suite cleanly stopped.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DREMA Dynamic Inference Suite")
    parser.add_argument("--port", type=int, default=50051, help="gRPC Server port (default: 50051)")
    parser.add_argument("--visualize_pybullet", action="store_true", help="Open PyBullet GUI window for real-time visualization")
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
        table_z_prior=args.table_z_prior,
        reachability_radius=args.reachability_radius,
        voxel_size=args.voxel_size,
        device=args.device
    )
    suite.start()
    suite.spin()
