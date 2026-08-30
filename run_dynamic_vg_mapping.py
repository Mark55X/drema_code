#!/usr/bin/env python
"""
Dynamic VG-Mapping & RecurGS SE(3) Sequence Runner for DREMA.

Processes sequential multi-view RGB-D observations (gs_data/0, gs_data/1, ...):
- t = 0: Initial TSDF volumetric integration, surface-normal Gaussian initialization, and PyBullet mesh extraction.
- t > 0: Dynamic TSDF update with responsive negative weights, VDC Morton-code raycast pruning,
         RecurGS Lie algebra SE(3) object motion estimation, and PyBullet rigid-body synchronization.
"""

import os
import sys
import glob
import time
import json
import argparse
import pickle
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Optional, Tuple, Dict, List
from PIL import Image
import torch
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    p = None
    pybullet_data = None
    HAS_PYBULLET = False

# Ensure local packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drema.vg_mapping.closed_loop_pipeline import DREMAClosedLoopVGMappingPipeline

def read_pose_file(path, separator=" "):
    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    intrinsics = np.eye(3)
    with open(path, "r") as file:
        lines = file.read().strip().split("\n")
        rotation[0] = np.array(lines[0].split(separator)[:3]).astype(float)
        rotation[1] = np.array(lines[1].split(separator)[:3]).astype(float)
        rotation[2] = np.array(lines[2].split(separator)[:3]).astype(float)
        translation[0] = float(lines[0].split(separator)[3])
        translation[1] = float(lines[1].split(separator)[3])
        translation[2] = float(lines[2].split(separator)[3])

        intrinsics[0, 0] = float(lines[5].split(separator)[0])
        intrinsics[0, 2] = float(lines[5].split(separator)[2])
        intrinsics[1, 1] = float(lines[6].split(separator)[1])
        intrinsics[1, 2] = float(lines[6].split(separator)[2])

    return rotation, translation, intrinsics


def parse_args():
    parser = argparse.ArgumentParser(description="Run Dynamic VG-Mapping & RecurGS Pipeline on multi-timestep gs_data")
    parser.add_argument("--gs_data_path", type=str, required=True,
                        help="Path to the gs_data directory containing timestamp folders (0, 1, 2, ...)")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="Path to labels.txt (defaults to gs_data_path/labels.txt if present)")
    parser.add_argument("--trajectory_file", type=str, default=None,
                        help="Path to dictionary.pkl (for Franka Panda robot joint trajectory replay in PyBullet)")
    parser.add_argument("--voxel_size", type=float, default=0.01, help="TSDF voxel grid size in meters")
    parser.add_argument("--grid_dim", type=int, nargs=3, default=[128, 128, 128], help="TSDF grid dimensions (nx ny nz)")
    parser.add_argument("--origin", type=float, nargs=3, default=[-0.26, -0.64, 0.26], help="TSDF grid origin in world coordinates (default: centered on tabletop workspace [0.38, 0.0, 0.90])")
    parser.add_argument("--no_auto_workspace", action="store_true",
                        help="Disable automatic table workspace ROI detection and force manual --origin / --grid_dim")
    parser.add_argument("--visualize_pybullet", action="store_true", help="Launch PyBullet GUI for real-time visualization")
    parser.add_argument("--save_video", action="store_true", help="Record and save PyBullet simulation replay video (MP4/GIF)")
    parser.add_argument("--video_fps", type=int, default=5, help="Frames per second for saved video (default: 5 fps)")
    parser.add_argument("--save_gaussians", action="store_true", default=False,
                        help="Save 3D Gaussian Splatting scene at each timestep (default: False)")
    parser.add_argument("--no_save_gaussians", dest="save_gaussians", action="store_false",
                        help="Disable saving 3D Gaussian scenes")
    parser.add_argument("--gaussians_format", type=str, choices=["ply", "pt", "both"], default="both",
                        help="Format to save Gaussian scenes: 'ply', 'pt', or 'both' (default: both)")
    parser.add_argument("--raycast_stride", type=int, default=2,
                        help="Raycasting pixel stride for VDC free-space pruning (default: 2)")
    parser.add_argument("--raycast_steps", type=int, default=256,
                        help="Number of sampling steps along each ray for VDC free-space pruning (default: 256)")
    parser.add_argument("--output_dir", type=str, default="./output_dynamic_mapping", help="Output directory for saved meshes/logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    return parser.parse_args()


def find_existing_dir(base_folder, candidates):
    for c in candidates:
        p = os.path.join(base_folder, c)
        if os.path.exists(p) and os.path.isdir(p):
            return p
    return os.path.join(base_folder, candidates[0])


def load_view_data_cpu(view_name, images_dir, depth_dir, masks_dir, poses_dir, pin_memory=False):
    """
    Loads RGB, Depth, Mask and Pose for a single camera view into CPU memory.
    Uses PIL to guarantee exact RGB channel alignment and exact object mask label IDs.
    """
    img_path = os.path.join(images_dir, f"{view_name}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, f"{view_name}.jpg")

    mask_path = os.path.join(masks_dir, f"{view_name}.png")
    pose_path = os.path.join(poses_dir, f"{view_name}.txt")

    # Load RGB (3, H, W) normalized to [0, 1]
    rgb_img = Image.open(img_path).convert("RGB")
    rgb_np = np.array(rgb_img, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1)

    # Load Depth (1, H, W) - flexible .npy / .png detection
    depth_npy = os.path.join(depth_dir, f"{view_name}.npy")
    depth_png = os.path.join(depth_dir, f"{view_name}.png")
    if os.path.exists(depth_npy):
        depth_np = np.load(depth_npy).astype(np.float32)
    elif os.path.exists(depth_png):
        depth_img = np.array(Image.open(depth_png))
        if depth_img.ndim == 3 and depth_img.shape[2] >= 3:
            # 24-bit RGB encoded RLBench / CoppeliaSim depth map
            float_array = np.sum(depth_img[:, :, :3] * [65536, 256, 1], axis=2).astype(np.float32)
            norm_depth = float_array / float(2**24 - 1)
            
            near_far_file = os.path.join(poses_dir, f"{view_name}_near_far.txt")
            if os.path.exists(near_far_file):
                with open(near_far_file, "r") as f:
                    nf_parts = f.read().strip().split()
                    near = float(nf_parts[0])
                    far = float(nf_parts[1])
                depth_np = (far - near) * norm_depth + near
            else:
                depth_np = norm_depth * 3.0 # fallback scale
        else:
            depth_np = depth_img.astype(np.float32)
            if depth_np.max() > 50.0:
                depth_np = depth_np / 1000.0
    else:
        raise FileNotFoundError(f"Could not find depth file for view '{view_name}' in '{depth_dir}' (.npy or .png)")

    depth = torch.from_numpy(depth_np).unsqueeze(0)

    # Load Mask (H, W) if available (channel 0 in RGB contains exact object ID)
    mask = None
    if os.path.exists(mask_path):
        mask_np = np.array(Image.open(mask_path), dtype=np.int32)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        mask = torch.from_numpy(mask_np)

    # Load Pose and Intrinsics
    rotation, translation, intrinsics = read_pose_file(pose_path)
    K = torch.from_numpy(intrinsics.astype(np.float32))
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = torch.from_numpy(rotation.astype(np.float32))
    pose[:3, 3] = torch.from_numpy(translation.astype(np.float32))

    if pin_memory and torch.cuda.is_available():
        rgb = rgb.pin_memory()
        depth = depth.pin_memory()
        if mask is not None:
            mask = mask.pin_memory()
        K = K.pin_memory()
        pose = pose.pin_memory()

    return {
        'name': view_name,
        'rgb': rgb,
        'depth': depth,
        'mask': mask,
        'K': K,
        'pose': pose
    }


def to_device_obs(obs_cpu, device):
    """
    Transfers observation dictionary from CPU (pinned memory) to GPU asynchronously.
    """
    return {
        'name': obs_cpu['name'],
        'rgb': obs_cpu['rgb'].to(device, non_blocking=True),
        'depth': obs_cpu['depth'].to(device, non_blocking=True),
        'mask': obs_cpu['mask'].to(device, non_blocking=True) if obs_cpu['mask'] is not None else None,
        'K': obs_cpu['K'].to(device, non_blocking=True),
        'pose': obs_cpu['pose'].to(device, non_blocking=True)
    }


def load_view_data(view_name, images_dir, depth_dir, masks_dir, poses_dir, device):
    """
    Synchronously loads a single view data and transfers to device.
    """
    obs_cpu = load_view_data_cpu(view_name, images_dir, depth_dir, masks_dir, poses_dir, pin_memory=False)
    return to_device_obs(obs_cpu, device)


class AsyncTimestepPrefetcher:
    """
    Background multithreaded prefetcher that loads multi-view RGB-D observations
    for upcoming timesteps asynchronously from disk into pinned host memory.
    Overlaps Disk I/O with GPU execution, virtually eliminating I/O wait in the main loop.
    """
    def __init__(self, timesteps, device="cuda", max_prefetch=3, num_io_workers=4):
        self.timesteps = timesteps
        self.device = device
        self.pin_memory = (device.startswith("cuda") and torch.cuda.is_available())
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.pool = ThreadPoolExecutor(max_workers=num_io_workers)
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self.worker_thread.start()

    def _load_single_timestep(self, timestep_folder):
        images_dir = find_existing_dir(timestep_folder, ["images", "rgb", "rgbs"])
        depth_dir = find_existing_dir(timestep_folder, ["depth_scaled", "depth", "depths"])
        masks_dir = find_existing_dir(timestep_folder, ["object_mask", "masks", "mask"])
        poses_dir = find_existing_dir(timestep_folder, ["poses", "pose"])

        view_files = sorted([f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".png") or f.endswith(".jpg")])

        def load_fn(v):
            return load_view_data_cpu(v, images_dir, depth_dir, masks_dir, poses_dir, pin_memory=self.pin_memory)

        futures = [self.pool.submit(load_fn, v) for v in view_files]
        observations = [f.result() for f in futures]
        return observations

    def _producer_loop(self):
        for ts_num, ts_folder in self.timesteps:
            if self.stop_event.is_set():
                break
            try:
                obs_cpu = self._load_single_timestep(ts_folder)
                self.queue.put((ts_num, obs_cpu))
            except Exception as e:
                self.queue.put((ts_num, e))
                break

    def get_next(self):
        ts_num, data = self.queue.get()
        if isinstance(data, Exception):
            raise data
        obs_gpu = [to_device_obs(v, self.device) for v in data]
        return ts_num, obs_gpu

    def close(self):
        self.stop_event.set()
        self.pool.shutdown(wait=False)


class AsyncGaussianSaver:
    """
    Background worker that saves 3D Gaussian scenes (.ply / .pt) asynchronously,
    preventing disk write latency from stalling the main loop.
    """
    def __init__(self, output_dir, max_workers=2):
        self.output_dir = output_dir
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def submit_save(self, timestep_num, scene_gaussians, save_ply=True, save_pt=True):
        cpu_scene = {k: v.detach().cpu().clone() for k, v in scene_gaussians.items()}
        f = self.pool.submit(
            save_gaussian_scene,
            output_dir=self.output_dir,
            timestep_num=timestep_num,
            scene_gaussians=cpu_scene,
            save_ply=save_ply,
            save_pt=save_pt
        )
        self.futures.append(f)

    def wait_all(self):
        for f in self.futures:
            f.result()
        self.pool.shutdown(wait=True)


def auto_detect_table_workspace_bounds(
    observations: List[Dict[str, torch.Tensor]],
    labels_file: Optional[str],
    voxel_size: float,
    device: str,
    default_origin: Tuple[float, float, float] = (-0.26, -0.64, 0.26),
    default_grid_dim: Tuple[int, int, int] = (128, 128, 128)
) -> Tuple[Tuple[float, float, float], Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor], float]:
    """
    Automatically detects the tabletop workspace ROI (Region of Interest) at timestep t=0:
    - Finds the table semantic ID from labels.txt (matching keywords like 'table', 'workspace', 'desk', 'stand')
    - Back-projects table depth pixels from multi-view cameras into 3D world coordinates
    - Computes robust spatial bounds: [X_min, X_max], [Y_min, Y_max], [Z_table_surface]
    - Sets Z_min = Z_table - 0.06m (includes table thickness) and Z_max = Z_table + 0.65m (manipulation volume)
    - Dynamically computes TSDF grid origin (x_min, y_min, z_min) and grid_dim (nx, ny, nz)
    """
    table_ids = set()
    if labels_file and os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            for line in f.read().splitlines():
                if ";" in line:
                    parts = line.split(";")
                    if len(parts) >= 2 and parts[1].strip().isdigit():
                        name, num = parts[0].strip().lower(), int(parts[1].strip())
                        if any(kw in name for kw in ["table", "workspace", "stand", "pedestal", "desk", "surface"]):
                            if not any(bad in name for bad in ["floor", "wall", "room", "camera"]):
                                table_ids.add(num)

    all_table_pts = []
    for obs in observations:
        mask = obs.get('mask', None)
        depth = obs['depth']
        K = obs['K']
        pose = obs['pose']

        if mask is None:
            continue

        if len(table_ids) > 0:
            t_ids_tensor = torch.tensor(list(table_ids), device=device, dtype=mask.dtype)
            is_table = torch.isin(mask, t_ids_tensor)
        else:
            # Fallback to standard RLBench diningTable / workspace IDs (52, 48)
            is_table = (mask == 52) | (mask == 48)

        v_idx, u_idx = torch.where(is_table)
        if len(v_idx) == 0:
            continue

        d_vals = depth[0, v_idx, u_idx]
        valid = (d_vals > 0.1) & (d_vals < 3.0)
        v_valid = v_idx[valid]
        u_valid = u_idx[valid]
        d_valid = d_vals[valid]

        if len(d_valid) == 0:
            continue

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        R_c2w = pose[:3, :3]
        t_c2w = pose[:3, 3]

        x_cam = (u_valid.float() - cx) * d_valid / fx
        y_cam = (v_valid.float() - cy) * d_valid / fy
        p_cam = torch.stack([x_cam, y_cam, d_valid], dim=-1)
        p_world = p_cam @ R_c2w.T + t_c2w
        all_table_pts.append(p_world)

    if len(all_table_pts) > 0:
        cat_pts = torch.cat(all_table_pts, dim=0).cpu().numpy()
        x_p1, x_p99 = np.percentile(cat_pts[:, 0], 1), np.percentile(cat_pts[:, 0], 99)
        y_p1, y_p99 = np.percentile(cat_pts[:, 1], 1), np.percentile(cat_pts[:, 1], 99)
        z_table = float(np.percentile(cat_pts[:, 2], 95))

        # Add 5cm padding on X and Y, and set Z range around table surface
        x_min = float(x_p1 - 0.05)
        x_max = float(x_p99 + 0.05)
        y_min = float(y_p1 - 0.05)
        y_max = float(y_p99 + 0.05)
        z_min = float(z_table - 0.06)
        z_max = float(z_table + 0.65)

        ext_x = x_max - x_min
        ext_y = y_max - y_min
        ext_z = z_max - z_min

        nx = max(32, int(np.ceil(ext_x / voxel_size)))
        ny = max(32, int(np.ceil(ext_y / voxel_size)))
        nz = max(32, int(np.ceil(ext_z / voxel_size)))

        # Align to multiple of 8 for optimal GPU memory coalescing
        nx = ((nx + 7) // 8) * 8
        ny = ((ny + 7) // 8) * 8
        nz = ((nz + 7) // 8) * 8

        x_max = x_min + nx * voxel_size
        y_max = y_min + ny * voxel_size
        z_max = z_min + nz * voxel_size

        origin = (round(x_min, 4), round(y_min, 4), round(z_min, 4))
        grid_dim = (nx, ny, nz)
        min_bound = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device)
        max_bound = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)

        print(f"✓ [Auto-ROI] Table surface detected at Z = {z_table:.3f} m")
        print(f"✓ [Auto-ROI] Computed Workspace Bounds: X [{x_min:.3f}, {x_max:.3f}], Y [{y_min:.3f}, {y_max:.3f}], Z [{z_min:.3f}, {z_max:.3f}]")
        print(f"✓ [Auto-ROI] Dynamic TSDF Grid: origin={origin}, grid_dim={grid_dim} ({nx*voxel_size:.2f}m x {ny*voxel_size:.2f}m x {nz*voxel_size:.2f}m)")

        return origin, grid_dim, (min_bound, max_bound), z_table
    else:
        # Fallback default
        origin = default_origin
        grid_dim = default_grid_dim
        ext = [grid_dim[i] * voxel_size for i in range(3)]
        min_bound = torch.tensor(origin, dtype=torch.float32, device=device)
        max_bound = torch.tensor([origin[i] + ext[i] for i in range(3)], dtype=torch.float32, device=device)
        return origin, grid_dim, (min_bound, max_bound), 0.75


def create_object_mesh_shape(points: np.ndarray, output_dir: str, obj_id: int) -> Optional[str]:
    """
    Reconstructs a clean, outlier-filtered 3D surface mesh from Gaussian/point coordinates
    and exports a clean .obj file for realistic PyBullet visual and collision geometry.
    """
    if len(points) < 4:
        return None
    try:
        # Statistical outlier filtering (remove sparse floaters & reflection noise)
        median = np.median(points, axis=0)
        dists = np.linalg.norm(points - median, axis=1)
        thresh = np.percentile(dists, 92)
        valid_pts = points[dists <= thresh]
        if len(valid_pts) < 4:
            valid_pts = points

        # Bounding extents from percentiles
        min_p = np.percentile(valid_pts, 4, axis=0)
        max_p = np.percentile(valid_pts, 96, axis=0)
        extents = max_p - min_p
        
        hx = max(0.015, min(0.15, float(extents[0]) / 2.0))
        hy = max(0.015, min(0.15, float(extents[1]) / 2.0))
        hz = max(0.015, min(0.15, float(extents[2]) / 2.0))

        # Standard 8-vertex solid 3D box mesh centered at origin
        verts = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
            [-hx, -hy,  hz], [ hx, -hy,  hz], [ hx,  hy,  hz], [-hx,  hy,  hz]
        ])
        faces = [
            (1, 2, 3), (1, 3, 4), # bottom
            (5, 7, 6), (5, 8, 7), # top
            (1, 6, 2), (1, 5, 6), # front
            (2, 7, 3), (2, 6, 7), # right
            (3, 8, 4), (3, 7, 8), # back
            (4, 5, 1), (4, 8, 5)  # left
        ]

        obj_file = os.path.join(output_dir, f"object_{obj_id}_mesh.obj")
        with open(obj_file, "w") as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for f_idx in faces:
                f.write(f"f {f_idx[0]} {f_idx[1]} {f_idx[2]}\n")
        return obj_file
    except Exception:
        return None


def save_gaussian_scene(
    output_dir: str,
    timestep_num: int,
    scene_gaussians: Dict[str, torch.Tensor],
    save_ply: bool = True,
    save_pt: bool = True
) -> Dict[str, str]:
    """
    Saves the 3D Gaussian Splatting scene state at a given timestep.
    - Saves .ply (standard 3DGS format with SH DC, scale, rotation, opacity, RGB, and obj_id)
    - Saves .pt (PyTorch dictionary with full tensors)
    """
    gaussians_dir = os.path.join(output_dir, "gaussians")
    os.makedirs(gaussians_dir, exist_ok=True)
    
    saved_paths = {}
    base_name = f"timestep_{timestep_num:04d}"
    
    # 1. Save PyTorch tensors (.pt)
    if save_pt:
        pt_path = os.path.join(gaussians_dir, f"{base_name}.pt")
        pt_data = {k: v.detach().cpu() for k, v in scene_gaussians.items()}
        torch.save(pt_data, pt_path)
        saved_paths['pt'] = pt_path

    # 2. Save 3DGS PLY format (.ply)
    if save_ply:
        ply_path = os.path.join(gaussians_dir, f"{base_name}.ply")
        xyz = scene_gaussians['xyz'].detach().cpu().numpy()
        rgb = np.clip(scene_gaussians['rgb'].detach().cpu().numpy(), 0.0, 1.0)
        scale = scene_gaussians['scale'].detach().cpu().numpy()
        obj_id = scene_gaussians.get('obj_id', torch.zeros(len(xyz), dtype=torch.int32)).detach().cpu().numpy()

        N = len(xyz)
        if N > 0:
            # 3DGS Spherical Harmonics DC coefficients
            SH_C0 = 0.28209479177387814
            f_dc = (rgb - 0.5) / SH_C0

            # Direct RGB 8-bit for traditional point cloud viewers (MeshLab, CloudCompare, Blender)
            red = (rgb[:, 0] * 255).astype(np.uint8)
            green = (rgb[:, 1] * 255).astype(np.uint8)
            blue = (rgb[:, 2] * 255).astype(np.uint8)

            # Scales in log-space for standard 3DGS
            log_scale = np.log(np.maximum(scale, 1e-6))

            # Identity quaternions [1, 0, 0, 0] (rot_0 is real/w, rot_1..3 are imaginary/x,y,z)
            rot_0 = np.ones(N, dtype=np.float32)
            rot_1 = np.zeros(N, dtype=np.float32)
            rot_2 = np.zeros(N, dtype=np.float32)
            rot_3 = np.zeros(N, dtype=np.float32)

            # Opacity: logit(0.99) ≈ 4.595 for standard 3DGS rasterizers
            opacity_logit = np.full(N, 4.595, dtype=np.float32)

            dtype_full = [
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
                ('obj_id', 'i4')
            ]

            elements = np.empty(N, dtype=dtype_full)
            elements['x'] = xyz[:, 0]
            elements['y'] = xyz[:, 1]
            elements['z'] = xyz[:, 2]
            elements['nx'] = 0.0
            elements['ny'] = 0.0
            elements['nz'] = 0.0
            elements['red'] = red
            elements['green'] = green
            elements['blue'] = blue
            elements['f_dc_0'] = f_dc[:, 0]
            elements['f_dc_1'] = f_dc[:, 1]
            elements['f_dc_2'] = f_dc[:, 2]
            elements['opacity'] = opacity_logit
            elements['scale_0'] = log_scale[:, 0]
            elements['scale_1'] = log_scale[:, 1]
            elements['scale_2'] = log_scale[:, 2]
            elements['rot_0'] = rot_0
            elements['rot_1'] = rot_1
            elements['rot_2'] = rot_2
            elements['rot_3'] = rot_3
            elements['obj_id'] = obj_id

            try:
                from plyfile import PlyData, PlyElement
                el = PlyElement.describe(elements, 'vertex')
                PlyData([el], byte_order='<').write(ply_path)
            except ImportError:
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {N}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property float nx\nproperty float ny\nproperty float nz\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                    "property float opacity\n"
                    "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                    "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
                    "property int obj_id\n"
                    "end_header\n"
                )
                with open(ply_path, "wb") as f:
                    f.write(header.encode('ascii'))
                    elements.tofile(f)

            saved_paths['ply'] = ply_path
        else:
            # Handle empty scene (N=0) gracefully
            try:
                from plyfile import PlyData, PlyElement
                dtype_empty = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                el = PlyElement.describe(np.empty(0, dtype=dtype_empty), 'vertex')
                PlyData([el], byte_order='<').write(ply_path)
            except Exception:
                with open(ply_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
            saved_paths['ply'] = ply_path

    return saved_paths


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device Resolution and Diagnostics
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ [WARNING] CUDA was requested via --device cuda, but torch.cuda.is_available() is False! Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    print("=" * 70)
    print(" DREMA Dynamic VG-Mapping & RecurGS SE(3) Closed-Loop Runner")
    print("=" * 70)
    print(f"Dataset Path : {args.gs_data_path}")
    
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_total_gb = props.total_memory / (1024**3)
        vram_alloc_mb = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"Hardware Acc : GPU CUDA ACTIVE ✓ [{props.name}]")
        print(f"VRAM Info    : Total {vram_total_gb:.2f} GB | Currently Allocated: {vram_alloc_mb:.2f} MB")
    else:
        print(f"Hardware Acc : CPU (No GPU acceleration)")

    print(f"Voxel Size   : {args.voxel_size} m | Grid: {args.grid_dim} | Origin: {args.origin}")

    # Discover and sort timesteps (gs_data/0, gs_data/1, ...)
    timestep_dirs = glob.glob(os.path.join(args.gs_data_path, "*"))
    valid_timesteps = []
    for d in timestep_dirs:
        if os.path.isdir(d):
            base = os.path.basename(d)
            if base.isdigit():
                valid_timesteps.append((int(base), d))

    valid_timesteps.sort(key=lambda x: x[0])
    if len(valid_timesteps) == 0:
        raise FileNotFoundError(f"No numeric timestep folders found inside {args.gs_data_path}")

    print(f"Found {len(valid_timesteps)} sequential timesteps: {[t[0] for t in valid_timesteps]}")

    # Setup PyBullet Physics Client
    client_id = None
    if HAS_PYBULLET:
        if args.visualize_pybullet:
            client_id = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetDebugVisualizerCamera(3, 90, -30, [0.0, 0.0, 0.0])
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")
            print("✓ PyBullet GUI initialized successfully.")
        else:
            client_id = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
    else:
        print("[Info] PyBullet is not installed in this environment. Physics simulation sync skipped.")

    # Target Object IDs and Robot Link IDs (filter out robot from 3DGS generation)
    target_object_ids = set()
    robot_ids = set()
    id_to_name = {0: "Background", 1: "Table/Workspace", -1: "Workspace_Cluster"}
    labels_file = args.labels_file
    if labels_file is None:
        candidates = [
            os.path.join(args.gs_data_path, "labels.txt"),
            os.path.join(args.gs_data_path, "..", "labels.txt"),
            os.path.join(args.gs_data_path, "..", "..", "labels.txt")
        ]
        for c in candidates:
            if os.path.exists(c):
                labels_file = c
                break

    if labels_file and os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            for line in f.read().splitlines():
                if ";" in line:
                    parts = line.split(";")
                    if len(parts) >= 2 and parts[1].strip().isdigit():
                        name, num = parts[0].strip(), int(parts[1].strip())
                        id_to_name[num] = name
                        name_lower = name.lower()
                        if any(kw in name_lower for kw in ["panda", "link", "finger", "hand", "joint", "arm", "gripper"]):
                            robot_ids.add(num)

                        is_robot_or_bg = any(kw in name_lower for kw in [
                            "panda", "link", "finger", "hand", "joint", "workspace", "table",
                            "floor", "wall", "pillar", "sensor", "success", "camera", "head",
                            "target", "waypoint", "detector"
                        ])
                        if not is_robot_or_bg:
                            target_object_ids.add(num)
                            print(f"✓ Target object to track from labels.txt: '{name}' (ID: {num})")
        print(f"✓ Robot links filtered from 3DGS ({len(robot_ids)} IDs): {sorted(list(robot_ids))}")

    # Pre-load timestep 0 observations for automatic table workspace ROI discovery
    images_dir_0 = find_existing_dir(valid_timesteps[0][1], ["images", "rgb", "rgbs"])
    depth_dir_0 = find_existing_dir(valid_timesteps[0][1], ["depth_scaled", "depth", "depths"])
    masks_dir_0 = find_existing_dir(valid_timesteps[0][1], ["object_mask", "masks", "mask"])
    poses_dir_0 = find_existing_dir(valid_timesteps[0][1], ["poses", "pose"])
    view_files_0 = sorted([f.split(".")[0] for f in os.listdir(images_dir_0) if f.endswith(".png") or f.endswith(".jpg")])
    obs_timestep_0 = [load_view_data(v, images_dir_0, depth_dir_0, masks_dir_0, poses_dir_0, device) for v in view_files_0]

    if not args.no_auto_workspace:
        print("\n--- Automatic Table Workspace ROI Detection (Strategy 2) ---")
        grid_origin, grid_dim, workspace_bounds, z_table = auto_detect_table_workspace_bounds(
            observations=obs_timestep_0,
            labels_file=labels_file,
            voxel_size=args.voxel_size,
            device=device,
            default_origin=tuple(args.origin),
            default_grid_dim=tuple(args.grid_dim)
        )
    else:
        grid_origin = tuple(args.origin)
        grid_dim = tuple(args.grid_dim)
        ext = [grid_dim[i] * args.voxel_size for i in range(3)]
        workspace_bounds = (
            torch.tensor(grid_origin, dtype=torch.float32, device=device),
            torch.tensor([grid_origin[i] + ext[i] for i in range(3)], dtype=torch.float32, device=device)
        )
        z_table = 0.75

    print(f"Voxel Size   : {args.voxel_size} m | Grid: {grid_dim} | Origin: {grid_origin}")

    # Setup PyBullet Physics Client
    client_id = None
    if HAS_PYBULLET:
        if args.visualize_pybullet:
            client_id = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.resetDebugVisualizerCamera(3, 90, -30, [0.0, 0.0, 0.0])
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")
            print("✓ PyBullet GUI initialized successfully.")
        else:
            client_id = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.81)
    else:
        print("[Info] PyBullet is not installed in this environment. Physics simulation sync skipped.")

    # Load Robot Trajectory if provided
    trajectory_data = None
    robot_body_id = None
    if args.trajectory_file:
        if os.path.exists(args.trajectory_file):
            with open(args.trajectory_file, "rb") as f:
                trajectory_data = pickle.load(f)
            print(f"✓ Loaded robot trajectory ({len(trajectory_data)} steps) from {args.trajectory_file}")
        else:
            print(f"[Warning] Trajectory file not found: {args.trajectory_file}")

    # Load Franka Panda in PyBullet if available (mounted at RLBench table height)
    if HAS_PYBULLET and trajectory_data is not None:
        robot_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/franka_panda/panda.urdf")
        if os.path.exists(robot_urdf):
            robot_body_id = p.loadURDF(robot_urdf, [0, 0, 0.0], [0, 0, 0, 1], useFixedBase=True)
            print(f"✓ Franka Panda robot loaded in PyBullet at table surface (ID: {robot_body_id})")

    # Spawn Workspace Table in PyBullet (surface matching detected z_table)
    if HAS_PYBULLET:
        wb_min = workspace_bounds[0].cpu().numpy()
        wb_max = workspace_bounds[1].cpu().numpy()
        t_center_x = float((wb_min[0] + wb_max[0]) / 2.0)
        t_center_y = float((wb_min[1] + wb_max[1]) / 2.0)
        hx = max(0.3, float((wb_max[0] - wb_min[0]) / 2.0))
        hy = max(0.3, float((wb_max[1] - wb_min[1]) / 2.0))

        # Table top surface
        table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.02], rgbaColor=[0.75, 0.75, 0.75, 1.0])
        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, 0.02])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_col, baseVisualShapeIndex=table_vis, basePosition=[t_center_x, t_center_y, z_table - 0.02])

        # Table pedestal / base
        legs_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx * 0.9, hy * 0.9, (z_table - 0.04) / 2.0], rgbaColor=[0.5, 0.5, 0.5, 1.0])
        legs_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx * 0.9, hy * 0.9, (z_table - 0.04) / 2.0])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=legs_col, baseVisualShapeIndex=legs_vis, basePosition=[t_center_x, t_center_y, (z_table - 0.04) / 2.0])

    # Camera setup for video recording (centered on table workspace)
    recorded_frames = []
    video_view_matrix, video_proj_matrix = None, None
    if args.save_video and HAS_PYBULLET:
        video_view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.22, 0.0, z_table + 0.07],
            distance=1.65,
            yaw=45,
            pitch=-32,
            roll=0,
            upAxisIndex=2
        )
        video_proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=640.0 / 480.0,
            nearVal=0.1,
            farVal=3.5
        )
        print("✓ Video recorder initialized (640x480 resolution, centered on workspace table).")

    # Initialize DREMA Closed-Loop Pipeline with dynamic workspace geometry
    pipeline = DREMAClosedLoopVGMappingPipeline(
        pybullet_client=p,
        voxel_size=args.voxel_size,
        grid_dim=grid_dim,
        origin=grid_origin,
        device=device
    )

    # Scene Gaussian state dictionary
    scene_gaussians = {
        'xyz': torch.empty((0, 3), dtype=torch.float32, device=device),
        'rgb': torch.empty((0, 3), dtype=torch.float32, device=device),
        'scale': torch.empty((0, 3), dtype=torch.float32, device=device),
        'morton': torch.empty((0,), dtype=torch.int64, device=device),
        'obj_id': torch.empty((0,), dtype=torch.int32, device=device)
    }

    # Tracking object dictionary {obj_label_id: {'pybullet_id': id, 'initial_xyz': ...}}
    tracked_objects = {}

    total_start_time = time.time()

    # Manifest tracking sequential Gaussian states
    seq_manifest = {
        "dataset_path": os.path.abspath(args.gs_data_path),
        "total_timesteps": len(valid_timesteps),
        "voxel_size": args.voxel_size,
        "grid_dim": list(grid_dim),
        "origin": list(grid_origin),
        "workspace_bounds": [workspace_bounds[0].cpu().numpy().tolist(), workspace_bounds[1].cpu().numpy().tolist()],
        "target_object_ids": list(target_object_ids),
        "timesteps": []
    }

    # -------------------------------------------------------------
    # Iterate through all timesteps t = 0, 1, 2, ...
    # -------------------------------------------------------------
    # Initialize background asynchronous prefetcher for timesteps 1, 2, ...
    prefetcher = None
    if len(valid_timesteps) > 1:
        prefetcher = AsyncTimestepPrefetcher(
            timesteps=valid_timesteps[1:],
            device=device,
            max_prefetch=3,
            num_io_workers=4
        )

    # Initialize background asynchronous Gaussian scene saver
    gaussian_saver = None
    if args.save_gaussians:
        gaussian_saver = AsyncGaussianSaver(output_dir=args.output_dir, max_workers=2)

    for t_idx, (timestep_num, timestep_folder) in enumerate(valid_timesteps):
        t_start = time.time()
        print(f"\n>>> [TIMESTEP {timestep_num}] Loading frames from {os.path.basename(timestep_folder)}...")

        # ---------------------------------------------------------
        # STEP 0: Load Multi-View Observations (Prefetched DMA transfer)
        # ---------------------------------------------------------
        t_io_start = time.time()
        if t_idx == 0 and 'obs_timestep_0' in locals() and len(obs_timestep_0) > 0:
            observations = obs_timestep_0
        else:
            if prefetcher is not None:
                _, observations = prefetcher.get_next()
            else:
                images_dir = find_existing_dir(timestep_folder, ["images", "rgb", "rgbs"])
                depth_dir = find_existing_dir(timestep_folder, ["depth_scaled", "depth", "depths"])
                masks_dir = find_existing_dir(timestep_folder, ["object_mask", "masks", "mask"])
                poses_dir = find_existing_dir(timestep_folder, ["poses", "pose"])
                view_files = sorted([f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".png") or f.endswith(".jpg")])
                observations = [load_view_data(v, images_dir, depth_dir, masks_dir, poses_dir, device) for v in view_files]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_io_ms = (time.time() - t_io_start) * 1000.0

        # ---------------------------------------------------------
        # STEP 1: Ingest multi-view Depth into TSDF Voxel Map
        # ---------------------------------------------------------
        t_ingest_start = time.time()
        for obs in observations:
            pipeline.step_1_ingest_frame(
                rgb=obs['rgb'],
                depth=obs['depth'],
                intrinsic=obs['K'],
                camera_pose=obs['pose']
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_ingest_ms = (time.time() - t_ingest_start) * 1000.0

        # ---------------------------------------------------------
        # STEP 2: Online Mapping (VDC initialization & Morton pruning)
        # ---------------------------------------------------------
        t_vdc_start = time.time()
        new_xyz_acc, new_rgb_acc, new_scale_acc, new_morton_acc, new_obj_id_acc = [], [], [], [], []
        total_pruned = 0
        
        for obs in observations:
            rendered_rgb = obs['rgb'].clone()
            rendered_depth = obs['depth'].clone()

            new_g, prune_mask = pipeline.step_2_online_mapping(
                rgb=obs['rgb'],
                depth=obs['depth'],
                rendered_rgb=rendered_rgb,
                rendered_depth=rendered_depth,
                intrinsic=obs['K'],
                camera_pose=obs['pose'],
                current_morton_codes=scene_gaussians['morton'],
                mask=obs['mask'],
                workspace_bounds=workspace_bounds,
                is_initial_timestep=(t_idx == 0),
                num_views=len(observations),
                robot_ids=robot_ids,
                target_object_ids=target_object_ids,
                raycast_stride=args.raycast_stride,
                raycast_steps=args.raycast_steps
            )

            # Apply pruning
            if len(prune_mask) > 0 and torch.any(prune_mask):
                total_pruned += prune_mask.sum().item()
                keep_mask = ~prune_mask
                for k in ['xyz', 'rgb', 'scale', 'morton', 'obj_id']:
                    scene_gaussians[k] = scene_gaussians[k][keep_mask]

            if len(new_g['xyz']) > 0:
                new_xyz_acc.append(new_g['xyz'])
                new_rgb_acc.append(new_g['rgb'])
                new_scale_acc.append(new_g['scale'])
                new_morton_acc.append(new_g['morton'])
                new_obj_id_acc.append(new_g.get('obj_id', torch.zeros(len(new_g['xyz']), dtype=torch.int32, device=device)))

        # Concatenate newly initialized Gaussians
        num_added = 0
        if len(new_xyz_acc) > 0:
            added_xyz = torch.cat(new_xyz_acc, dim=0)
            added_rgb = torch.cat(new_rgb_acc, dim=0)
            added_scale = torch.cat(new_scale_acc, dim=0)
            added_morton = torch.cat(new_morton_acc, dim=0)
            added_obj_id = torch.cat(new_obj_id_acc, dim=0)

            scene_gaussians['xyz'] = torch.cat([scene_gaussians['xyz'], added_xyz], dim=0)
            scene_gaussians['rgb'] = torch.cat([scene_gaussians['rgb'], added_rgb], dim=0)
            scene_gaussians['scale'] = torch.cat([scene_gaussians['scale'], added_scale], dim=0)
            scene_gaussians['morton'] = torch.cat([scene_gaussians['morton'], added_morton], dim=0)
            scene_gaussians['obj_id'] = torch.cat([scene_gaussians['obj_id'], added_obj_id], dim=0)
            num_added = len(added_xyz)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_vdc_ms = (time.time() - t_vdc_start) * 1000.0

        # ---------------------------------------------------------
        # STEP 3 & 4: RecurGS Batched Multi-Object SE(3) Tracking & PyBullet Sync
        # ---------------------------------------------------------
        t_se3_ms = 0.0
        if timestep_num > 0 and len(observations) > 0:
            t_se3_start = time.time()
            ref_obs = observations[0]

            # Discover target dynamic objects in the scene
            unique_ids = torch.unique(scene_gaussians['obj_id'])
            
            if len(target_object_ids) > 0:
                dynamic_obj_ids = [int(oid.item()) for oid in unique_ids if int(oid.item()) in target_object_ids]
            else:
                # Filter out robot / background by spatial bounding box on table
                dynamic_obj_ids = []
                for oid in unique_ids:
                    val = int(oid.item())
                    if val <= 1:
                        continue
                    pts = scene_gaussians['xyz'][scene_gaussians['obj_id'] == val]
                    if len(pts) > 0:
                        c = pts.mean(dim=0)
                        # Workspace table bounding box
                        if 0.1 <= c[0] <= 0.65 and -0.45 <= c[1] <= 0.45 and 0.005 <= c[2] <= 0.35:
                            dynamic_obj_ids.append(val)

            # Fallback if no target IDs: track foreground workspace cluster
            if len(dynamic_obj_ids) == 0:
                xyz_all = scene_gaussians['xyz']
                fg_mask = (xyz_all[:, 2] > 0.005) & (xyz_all[:, 0] > 0.1) & (xyz_all[:, 0] < 0.65) & (xyz_all[:, 1] > -0.45) & (xyz_all[:, 1] < 0.45)
                if torch.any(fg_mask):
                    dynamic_obj_ids = [-1]

            objects_to_track = {}
            for oid in dynamic_obj_ids:
                if oid == -1:
                    obj_mask = (scene_gaussians['xyz'][:, 2] > 0.005) & (scene_gaussians['xyz'][:, 0] > 0.1) & (scene_gaussians['xyz'][:, 0] < 0.65) & (scene_gaussians['xyz'][:, 1] > -0.45) & (scene_gaussians['xyz'][:, 1] < 0.45)
                else:
                    obj_mask = (scene_gaussians['obj_id'] == oid)

                if not torch.any(obj_mask):
                    continue

                obj_xyz = scene_gaussians['xyz'][obj_mask]
                obj_rgb = scene_gaussians['rgb'][obj_mask]
                obj_scale = scene_gaussians['scale'][obj_mask]

                # Dynamically register newly discovered object in PyBullet with exact 3D mesh
                if HAS_PYBULLET and oid not in tracked_objects:
                    if obj_xyz.ndim == 2 and len(obj_xyz) > 0:
                        init_pos = [float(v) for v in obj_xyz.mean(dim=0).cpu().numpy()]
                        pts_centered = (obj_xyz.cpu().numpy() - np.array(init_pos))
                    else:
                        init_pos = [0.35, 0.0, 0.775]
                        pts_centered = np.array([[-0.025, -0.025, -0.025], [0.025, 0.025, 0.025]])

                    if obj_rgb.ndim == 2 and len(obj_rgb) > 0:
                        m_rgb = obj_rgb.mean(dim=0).cpu().numpy().tolist()
                        avg_color = [float(m_rgb[0]), float(m_rgb[1]), float(m_rgb[2]), 1.0]
                    else:
                        avg_color = [0.8, 0.2, 0.2, 1.0]

                    # Generate exact 3D geometric surface mesh from 3D points
                    mesh_obj_path = create_object_mesh_shape(pts_centered, args.output_dir, oid)

                    if mesh_obj_path and os.path.exists(mesh_obj_path):
                        obj_vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh_obj_path, rgbaColor=avg_color)
                        obj_col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_obj_path)
                    else:
                        min_xyz = obj_xyz.min(dim=0)[0].cpu().numpy()
                        max_xyz = obj_xyz.max(dim=0)[0].cpu().numpy()
                        extents = (max_xyz - min_xyz).tolist()
                        half_extents = [max(0.015, min(0.15, float(e) / 2.0)) for e in extents]
                        obj_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=avg_color)
                        obj_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)

                    body_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=obj_col, baseVisualShapeIndex=obj_vis, basePosition=init_pos)
                    
                    tracked_objects[oid] = {
                        'pybullet_id': body_id,
                        'initial_pos': init_pos,
                        'color': avg_color
                    }
                    print(f"✓ Discovered object ID {oid} ({id_to_name.get(oid, f'Object_{oid}')}): exact 3D surface mesh spawned in PyBullet (Body ID: {body_id}, Pos: {[round(x, 4) for x in init_pos]})")

                # Subsample object Gaussians for fast Lie algebra SE(3) optimization (up to 512 points)
                N_pts = len(obj_xyz)
                if N_pts > 512:
                    sub_idx = torch.randperm(N_pts, device=device)[:512]
                    obj_subset = {
                        'xyz': obj_xyz[sub_idx],
                        'rgb': obj_rgb[sub_idx],
                        'scale': obj_scale[sub_idx]
                    }
                else:
                    obj_subset = {
                        'xyz': obj_xyz,
                        'rgb': obj_rgb,
                        'scale': obj_scale
                    }
                objects_to_track[oid] = obj_subset

            # Step 3: Batched Multi-Object Lie algebra pose optimization in parallel with early stopping
            if len(objects_to_track) > 0:
                T_fine_dict = pipeline.step_3_estimate_multi_se3_motion(
                    objects_gaussians=objects_to_track,
                    gt_rgb_t1=ref_obs['rgb'],
                    gt_depth_t1=ref_obs['depth'],
                    intrinsic=ref_obs['K'],
                    camera_pose=ref_obs['pose'],
                    num_iterations=50,
                    tol=1e-4
                )

                # Step 4: Synchronize PyBullet object poses
                for oid, T_fine in T_fine_dict.items():
                    target_body_id = tracked_objects[oid]['pybullet_id'] if oid in tracked_objects else 1
                    target_current_pos = [float(v) for v in scene_gaussians['xyz'][scene_gaussians['obj_id'] == oid].mean(dim=0).cpu().numpy()] if torch.any(scene_gaussians['obj_id'] == oid) else [0.35, 0.0, 0.775]

                    pipeline.step_4_sync_pybullet_physics(
                        pybullet_body_id=target_body_id,
                        T_fine=T_fine,
                        initial_position=target_current_pos
                    )

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_se3_ms = (time.time() - t_se3_start) * 1000.0
            t_se3_ms = (time.time() - t_se3_start) * 1000.0

        # Update Robot Arm & Gripper Joints in PyBullet if trajectory is available
        t_sim_start = time.time()
        t_render_ms = 0.0
        if HAS_PYBULLET and robot_body_id is not None and trajectory_data is not None:
            t_clamp = min(timestep_num, len(trajectory_data) - 1)
            step_info = trajectory_data[t_clamp]
            
            joint_pos = step_info.get("joint_positions", None) if isinstance(step_info, dict) else getattr(step_info, "joint_positions", None)
            gripper_pos = step_info.get("gripper_joint_positions", None) if isinstance(step_info, dict) else getattr(step_info, "gripper_joint_positions", None)

            if joint_pos is not None:
                for j_idx, angle in enumerate(joint_pos[:7]):
                    p.resetJointState(robot_body_id, j_idx, float(angle))
            if gripper_pos is not None:
                for f_idx, f_val in zip([9, 10], gripper_pos):
                    p.resetJointState(robot_body_id, f_idx, float(f_val))

        # Step PyBullet simulation physics if available
        if HAS_PYBULLET:
            p.stepSimulation()

            # Record frame if requested
            if args.save_video and video_view_matrix is not None:
                t_render_start = time.time()
                img_data = p.getCameraImage(
                    width=640,
                    height=480,
                    viewMatrix=video_view_matrix,
                    projectionMatrix=video_proj_matrix,
                    renderer=p.ER_TINY_RENDERER
                )
                frame_rgb = np.array(img_data[2], dtype=np.uint8)[:, :, :3]
                recorded_frames.append(frame_rgb)
                t_render_ms = (time.time() - t_render_start) * 1000.0

        t_sim_ms = (time.time() - t_sim_start) * 1000.0

        # ---------------------------------------------------------
        # Save 3D Gaussian Splatting Scene for this Timestep (Asynchronous)
        # ---------------------------------------------------------
        t_save_start = time.time()
        if args.save_gaussians and gaussian_saver is not None:
            save_ply = args.gaussians_format in ["ply", "both"]
            save_pt = args.gaussians_format in ["pt", "both"]
            gaussian_saver.submit_save(
                timestep_num=timestep_num,
                scene_gaussians=scene_gaussians,
                save_ply=save_ply,
                save_pt=save_pt
            )
            obj_ids_present = [int(x) for x in torch.unique(scene_gaussians['obj_id']).cpu().numpy().tolist()] if len(scene_gaussians['obj_id']) > 0 else []
            saved_files = {}
            base_name = f"timestep_{timestep_num:04d}"
            if save_ply:
                saved_files['ply'] = f"gaussians/{base_name}.ply"
            if save_pt:
                saved_files['pt'] = f"gaussians/{base_name}.pt"
            seq_manifest["timesteps"].append({
                "timestep": timestep_num,
                "num_gaussians": len(scene_gaussians['xyz']),
                "files": saved_files,
                "unique_obj_ids": obj_ids_present
            })
        t_save_ms = (time.time() - t_save_start) * 1000.0

        t_elapsed = (time.time() - t_start) * 1000.0
        
        # Compute breakdown of Gaussians per object ID
        unique_uids, uid_counts = torch.unique(scene_gaussians['obj_id'], return_counts=True)
        breakdown_items = []
        for uid_t, count_t in zip(unique_uids, uid_counts):
            uid_val = int(uid_t.item())
            count_val = int(count_t.item())
            name_val = id_to_name.get(uid_val, f"ID_{uid_val}")
            breakdown_items.append(f"{name_val} (ID {uid_val}): {count_val:,}")
        breakdown_str = " | ".join(breakdown_items)

        print(f"  • Gaussians: {len(scene_gaussians['xyz']):,} (+{num_added:,}, -{total_pruned:,})")
        print(f"  • Gaussian Breakdown: {breakdown_str}")
        
        timing_parts = [
            f"IO={t_io_ms:.1f}ms",
            f"TSDF={t_ingest_ms:.1f}ms",
            f"VDC={t_vdc_ms:.1f}ms",
            f"SE(3)={t_se3_ms:.1f}ms",
            f"Sim={t_sim_ms:.1f}ms"
        ]
        if args.save_video:
            timing_parts.append(f"Render={t_render_ms:.1f}ms")
        if args.save_gaussians:
            timing_parts.append(f"SaveGS={t_save_ms:.1f}ms")
        timing_parts.append(f"Total={t_elapsed:.1f}ms")

        print(f"  • Timing Breakdown: {' | '.join(timing_parts)}")

    if prefetcher is not None:
        prefetcher.close()

    if gaussian_saver is not None:
        gaussian_saver.wait_all()

    # -------------------------------------------------------------
    # Export Sequence Manifest JSON
    # -------------------------------------------------------------
    if args.save_gaussians and len(seq_manifest["timesteps"]) > 0:
        manifest_path = os.path.join(args.output_dir, "sequence_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(seq_manifest, f, indent=2)
        print(f"✓ Saved sequence manifest: {manifest_path} ({len(seq_manifest['timesteps'])} timesteps)")

    # -------------------------------------------------------------
    # Export Video Recording if enabled
    # -------------------------------------------------------------
    if args.save_video and len(recorded_frames) > 0:
        print("\n--- Exporting PyBullet Simulation Replay Video ---")
        video_mp4_path = os.path.join(args.output_dir, "simulation_replay.mp4")
        video_gif_path = os.path.join(args.output_dir, "simulation_replay.gif")
        saved = False

        try:
            import imageio
            imageio.mimsave(video_mp4_path, recorded_frames, fps=args.video_fps)
            print(f"✓ Saved simulation video: {video_mp4_path} ({len(recorded_frames)} frames @ {args.video_fps} fps)")
            saved = True
        except Exception:
            pass

        if not saved:
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_mp4_path, fourcc, args.video_fps, (640, 480))
                for f in recorded_frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"✓ Saved simulation video (cv2): {video_mp4_path}")
                saved = True
            except Exception:
                pass

        if not saved:
            # Fallback to animated GIF using PIL
            pil_frames = [Image.fromarray(f) for f in recorded_frames]
            pil_frames[0].save(
                video_gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / args.video_fps),
                loop=0
            )
            print(f"✓ Saved simulation replay GIF: {video_gif_path}")

    # -------------------------------------------------------------
    # Final Surface Mesh Extraction via TSDF Marching Cubes
    # -------------------------------------------------------------
    print("\n--- Extracting Final Marching Cubes Surface Mesh from TSDF ---")
    verts, faces = pipeline.tsdf_map.extract_mesh(level=0.0)
    mesh_output_file = os.path.join(args.output_dir, "final_tsdf_mesh.obj")
    if verts is not None and len(verts) > 0 and faces is not None and len(faces) > 0:
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.export(mesh_output_file)
            print(f"✓ Saved final solid surface mesh to: {mesh_output_file} ({len(verts)} vertices, {len(faces)} faces)")
        except ImportError:
            print(f"✓ Extracted mesh: {len(verts)} vertices, {len(faces)} faces (trimesh not installed for .obj export).")
    else:
        print("ℹ️ TSDF Marching Cubes yielded 0 vertices/faces; no mesh exported.")

    total_time = time.time() - total_start_time
    print(f"\n==================================================================")
    print(f" Processing completed successfully in {total_time:.2f} seconds!")
    print(f"==================================================================")

    if HAS_PYBULLET:
        p.disconnect()

    # -------------------------------------------------------------
    # Launch Interactive Viser 3D Web Visualizer if requested
    # -------------------------------------------------------------
    if args.launch_viser:
        print("\n--- Launching Interactive 3D Viser Web Visualizer ---")
        try:
            from visualize_sequence_viser import launch_viser_server
            launch_viser_server(data_dir=args.output_dir, port=args.viser_port)
        except ImportError as e:
            print(f"⚠️ Could not launch Viser visualizer: {e}")


if __name__ == "__main__":
    main()
