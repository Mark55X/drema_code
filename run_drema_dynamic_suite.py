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

# Ensure local packages and master-thesis workspace packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from drema.communication.grpc_server import DremaGrpcServer
from drema.communication.grpc_client import unpack_camera_frame
from drema.communication.proto import drema_comm_pb2
from drema.simulation.digital_twin import PyBulletDigitalTwin
from drema.controller.mpc_controller import MPCController

try:
    from drema.vg_mapping.closed_loop_pipeline import DREMAClosedLoopVGMappingPipeline, rotation_matrix_to_quaternion
    HAS_VG_MAPPING = True
except Exception as e:
    print(f"[Warning] Could not import DREMAClosedLoopVGMappingPipeline: {e}")
    HAS_VG_MAPPING = False


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
        table_z: float = 0.75,
        voxel_size: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.port = port
        self.device = device
        self.table_z = table_z
        self.voxel_size = voxel_size

        print(f"==================================================")
        print(f"   DREMA Dynamic Inference Suite Starting...     ")
        print(f"   Device: {self.device} | Port: {self.port}       ")
        print(f"==================================================")

        # 1. Initialize Submodule 2: PyBullet Digital Twin (Baseline: robot + table)
        self.digital_twin = PyBulletDigitalTwin(
            visualize=visualize_pybullet,
            table_z=table_z
        )
        print("✓ Submodule 2 (PyBullet Digital Twin) initialized.")

        # 2. Initialize Submodule 3: MPC Controller Stub
        self.mpc_controller = MPCController(
            num_joints=7,
            max_joint_velocity=0.15,
            safety_collision_distance=0.035
        )
        print("✓ Submodule 3 (MPC Controller) initialized.")

        # 3. Initialize Submodule 1: VG-Mapping & RecurGS SE(3) Pipeline
        self.vg_pipeline = None
        if HAS_VG_MAPPING:
            try:
                self.vg_pipeline = DREMAClosedLoopVGMappingPipeline(
                    pybullet_client=None,  # We manage PyBullet directly via digital_twin
                    voxel_size=voxel_size,
                    grid_dim=(128, 128, 128),
                    origin=(-0.26, -0.64, table_z - 0.20),
                    device=self.device
                )
                print("✓ Submodule 1 (VG-Mapping Closed-Loop Pipeline) initialized.")
            except Exception as e:
                print(f"[Notice] VG-Mapping pipeline initialization deferred/mock: {e}")

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

    def _process_initial_scene_scan(self, semantic_labels: Optional[Dict[str, int]] = None):
        """
        Processes the 200 orbital scanning views collected at t=0:
        1. Segments the dominant horizontal plane to determine exact table elevation z_table and 2D bounds.
        2. Generates a solid 3D table surface mesh (.obj) via 3D Convex Hull and spawns it in PyBullet.
        3. Discovers foreground obstacle clusters above the table (e.g. tunnel), builds a 3D mesh,
           and spawns them in PyBullet with spawn_mesh_object.
        4. Saves canonical points for online RecurGS SE(3) tracking.
        """
        print("\n=======================================================")
        print(f"[DREMA Suite] Processing 360° Initial Scene Scan ({len(self.accumulated_scan_points)} view point clouds)...")
        print("=======================================================")

        if len(self.accumulated_scan_points) == 0:
            print("[DREMA Suite Warning] No scanning points accumulated. Using default bounds.")
            self.digital_twin.spawn_scanned_table(table_z=self.table_z, bounds=(-0.50, 1.10, -0.55, 0.55))
            self.initial_scan_ready = True
            return

        all_pts = np.vstack(self.accumulated_scan_points)
        print(f"[DREMA Suite Scan] Accumulated {len(all_pts)} dense 3D surface points across all views.")

        # 1. Table Detection via Z-elevation histogram
        hist, bin_edges = np.histogram(all_pts[:, 2], bins=50, range=(0.70, 0.80))
        peak_bin = np.argmax(hist)
        z_table = float(0.5 * (bin_edges[peak_bin] + bin_edges[peak_bin + 1]))

        table_pts_mask = np.abs(all_pts[:, 2] - z_table) <= 0.008
        table_pts = all_pts[table_pts_mask]

        if len(table_pts) > 100:
            x_min = float(np.percentile(table_pts[:, 0], 1))
            x_max = float(np.percentile(table_pts[:, 0], 99))
            y_min = float(np.percentile(table_pts[:, 1], 1))
            y_max = float(np.percentile(table_pts[:, 1], 99))
        else:
            x_min, x_max = -0.50, 1.10
            y_min, y_max = -0.55, 0.55

        table_bounds = (x_min, x_max, y_min, y_max)
        print(f"[DREMA Discovery] Table surface detected at Z={z_table:.3f}m, bounds: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}]")

        # Spawn solid table in PyBullet Digital Twin (extends from ground Z=0 to Z=z_table)
        self.digital_twin.spawn_scanned_table(
            table_z=z_table,
            bounds=table_bounds
        )

        # 2. Fast Semantic Clustering for Tabletop Obstacles & Target
        # Select points above table surface in workspace (X > 0.05 avoids robot arm at X <= -0.05)
        obj_mask = (all_pts[:, 2] > (z_table + 0.015)) & (all_pts[:, 2] < (z_table + 0.35)) & \
                   (all_pts[:, 0] > 0.05) & (all_pts[:, 0] < 0.70) & \
                   (all_pts[:, 1] > -0.50) & (all_pts[:, 1] < 0.50)
        obj_pts = all_pts[obj_mask]
        print(f"[DREMA Discovery] Tabletop object points detected: {len(obj_pts)}")

        if len(obj_pts) >= 30:
            # Fast voxel downsampling (1cm grid)
            voxel_size = 0.01
            voxel_idx = np.floor(obj_pts / voxel_size).astype(int)
            _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
            ds_pts = obj_pts[unique_idx]

            # Fast Euclidean clustering via KDTree connected components (< 2ms)
            tree = cKDTree(ds_pts)
            adj = tree.sparse_distance_matrix(tree, max_distance=0.04)
            n_comp, labels = csgraph.connected_components(adj)

            cluster_list = []
            for lbl in range(n_comp):
                c_pts = ds_pts[labels == lbl]
                if len(c_pts) >= 15:
                    cluster_list.append(c_pts)

            cluster_list.sort(key=lambda c: len(c), reverse=True)
            print(f"[DREMA Discovery] Found {len(cluster_list)} distinct object cluster(s) above the table.")

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
        print(f"\n✓ [DREMA Suite] Initial Scene Setup Complete! Ready for dynamic execution.")
        print("=======================================================\n")

    def on_frame_received(self, obs: drema_comm_pb2.FrameObservation) -> Optional[drema_comm_pb2.StreamStatus]:
        """gRPC callback triggered when a camera frame arrives from CoppeliaSim."""
        if obs.is_initial_scan:
            for f in obs.cameras:
                name, rgb, depth, extrinsics, intrinsics = unpack_camera_frame(f)
                if self.vg_pipeline is not None:
                    try:
                        d_tensor = torch.from_numpy(depth.copy()).to(self.device)
                        k_tensor = torch.from_numpy(intrinsics.copy()).to(self.device)
                        t_tensor = torch.from_numpy(extrinsics.copy()).to(self.device)
                        rgb_tensor = torch.from_numpy(rgb.copy()).float().to(self.device) / 255.0
                        self.vg_pipeline.step_1_ingest_frame(
                            rgb=rgb_tensor,
                            depth=d_tensor,
                            intrinsic=k_tensor,
                            camera_pose=t_tensor
                        )
                    except Exception:
                        pass

                # Back-project depth points using exact CoppeliaSim camera projection
                pcd = pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics)
                valid = (depth > 0.1) & (depth < 3.5)
                pts = pcd[valid]

                # Filter to table workspace: X in [-0.55, 1.15], Y in [-0.65, 0.65], Z in [0.65, 1.50]
                ws = (pts[:, 0] >= -0.55) & (pts[:, 0] <= 1.15) & \
                     (pts[:, 1] >= -0.65) & (pts[:, 1] <= 0.65) & \
                     (pts[:, 2] >= 0.65) & (pts[:, 2] <= 1.50)
                if np.any(ws):
                    self.accumulated_scan_points.append(pts[ws])

            if obs.is_scan_finished:
                semantic_labels = dict(obs.semantic_labels) if obs.semantic_labels else None
                self._process_initial_scene_scan(semantic_labels=semantic_labels)
                return drema_comm_pb2.StreamStatus(
                    success=True,
                    message="Initial 360° scan processed and PyBullet Digital Twin populated",
                    received_timestep=0,
                    initial_scan_ready=True
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
                name, rgb, depth, extrinsics, intrinsics = unpack_camera_frame(f)
                camera_views[name] = {
                    'rgb': rgb,
                    'depth': depth,
                    'extrinsics': extrinsics,
                    'intrinsics': intrinsics
                }

            # If VG-Mapping pipeline is available, integrate depth frames
            if self.vg_pipeline is not None and len(camera_views) > 0:
                try:
                    # Ingest first available camera (or all views)
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

            # If obstacle 0 is tracked, update its position via real-time centroid tracking
            if self.initial_scan_ready and 0 in self.tracked_objects and len(camera_views) > 0:
                try:
                    first_cam = list(camera_views.values())[0]
                    depth_rt = first_cam['depth']
                    ext_rt = first_cam['extrinsics']
                    int_rt = first_cam['intrinsics']

                    H, W = depth_rt.shape
                    u_g, v_g = np.meshgrid(np.arange(0, W, 4), np.arange(0, H, 4))
                    d_vals = depth_rt[v_g, u_g]
                    valid = (d_vals > 0.1) & (d_vals < 2.5)

                    u_v, v_v, d_v = u_g[valid], v_g[valid], d_vals[valid]
                    fx, fy = int_rt[0, 0], int_rt[1, 1]
                    cx, cy = int_rt[0, 2], int_rt[1, 2]
                    x_c = (u_v - cx) * d_v / fx
                    y_c = (v_v - cy) * d_v / fy
                    p_c = np.stack([x_c, y_c, d_v, np.ones_like(d_v)], axis=-1)
                    p_w = (ext_rt @ p_c.T).T[:, :3]

                    z_tab = self.digital_twin.table_z
                    obs_m = (p_w[:, 2] > (z_tab + 0.008)) & \
                            (p_w[:, 0] >= 0.05) & (p_w[:, 0] <= 0.65) & \
                            (p_w[:, 1] >= -0.45) & (p_w[:, 1] <= 0.45)
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
    parser.add_argument("--table_z", type=float, default=0.75, help="Table surface Z elevation in meters (default: 0.75)")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="TSDF voxel grid resolution (default: 0.01m)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device (cuda/cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    suite = DremaDynamicSuite(
        port=args.port,
        visualize_pybullet=args.visualize_pybullet,
        table_z=args.table_z,
        voxel_size=args.voxel_size,
        device=args.device
    )
    suite.start()
    suite.spin()
