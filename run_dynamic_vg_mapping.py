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
import argparse
import pickle
import numpy as np
from PIL import Image
import torch
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
    parser.add_argument("--origin", type=float, nargs=3, default=[-0.64, -0.64, -0.64], help="TSDF grid origin in world coordinates")
    parser.add_argument("--visualize_pybullet", action="store_true", help="Launch PyBullet GUI for real-time visualization")
    parser.add_argument("--output_dir", type=str, default="./output_dynamic_mapping", help="Output directory for saved meshes/logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    return parser.parse_args()


def find_existing_dir(base_folder, candidates):
    for c in candidates:
        p = os.path.join(base_folder, c)
        if os.path.exists(p) and os.path.isdir(p):
            return p
    return os.path.join(base_folder, candidates[0])


def load_view_data(view_name, images_dir, depth_dir, masks_dir, poses_dir, device):
    """
    Loads RGB, Depth, Mask and Pose for a single camera view.
    Supports .png, .jpg for images, and .npy / .png for depth.
    """
    img_path = os.path.join(images_dir, f"{view_name}.png")
    if not os.path.exists(img_path):
        img_path = os.path.join(images_dir, f"{view_name}.jpg")

    mask_path = os.path.join(masks_dir, f"{view_name}.png")
    pose_path = os.path.join(poses_dir, f"{view_name}.txt")

    # Load RGB (3, H, W) normalized to [0, 1]
    rgb_img = Image.open(img_path).convert("RGB")
    rgb_np = np.array(rgb_img, dtype=np.float32) / 255.0
    rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device).permute(2, 0, 1)

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

    depth = torch.tensor(depth_np, dtype=torch.float32, device=device).unsqueeze(0)

    # Load Mask (H, W) if available
    mask = None
    if os.path.exists(mask_path):
        mask_np = np.array(Image.open(mask_path), dtype=np.int32)
        mask = torch.tensor(mask_np, dtype=torch.int32, device=device)

    # Load Pose and Intrinsics
    rotation, translation, intrinsics = read_pose_file(pose_path)
    
    K = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
    pose[:3, 3] = torch.tensor(translation, dtype=torch.float32, device=device)

    return {
        'name': view_name,
        'rgb': rgb,
        'depth': depth,
        'mask': mask,
        'K': K,
        'pose': pose
    }


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
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        print(f"Hardware Acc : GPU CUDA ACTIVE ✓ [{gpu_name}]")
        print(f"VRAM Info    : Total {vram_total:.2f} GB | Currently Allocated: {vram_alloc:.2f} MB")
    else:
        print(f"Hardware Acc : CPU (No GPU acceleration)")

    print(f"Voxel Size   : {args.voxel_size} m | Grid: {args.grid_dim} | Origin: {args.origin}")

    # Discover and sort timesteps
    timestep_dirs = [d for d in glob.glob(os.path.join(args.gs_data_path, "*")) if os.path.isdir(d)]
    
    valid_timesteps = []
    for d in timestep_dirs:
        base = os.path.basename(d)
        if base.isdigit():
            valid_timesteps.append((int(base), d))
    
    valid_timesteps.sort(key=lambda x: x[0])
    
    if len(valid_timesteps) == 0:
        raise ValueError(f"No numeric timestep folders (0, 1, 2, ...) found in {args.gs_data_path}")

    print(f"Found {len(valid_timesteps)} sequential timesteps: {[t[0] for t in valid_timesteps]}")

    # Setup PyBullet client if available
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

    # Load Franka Panda in PyBullet if available
    if HAS_PYBULLET and trajectory_data is not None:
        robot_urdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/franka_panda/panda.urdf")
        if os.path.exists(robot_urdf):
            robot_body_id = p.loadURDF(robot_urdf, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
            print(f"✓ Franka Panda robot loaded in PyBullet (ID: {robot_body_id})")

    # Initialize DREMA Closed-Loop Pipeline
    pipeline = DREMAClosedLoopVGMappingPipeline(
        pybullet_client=p,
        voxel_size=args.voxel_size,
        grid_dim=tuple(args.grid_dim),
        origin=tuple(args.origin),
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

    # -------------------------------------------------------------
    # Iterate through all timesteps t = 0, 1, 2, ...
    # -------------------------------------------------------------
    for t_idx, (timestep_num, timestep_folder) in enumerate(valid_timesteps):
        t_start = time.time()
        print(f"\n>>> [TIMESTEP {timestep_num}] Loading frames from {os.path.basename(timestep_folder)}...")

        images_dir = find_existing_dir(timestep_folder, ["images", "rgb", "rgbs"])
        depth_dir = find_existing_dir(timestep_folder, ["depth_scaled", "depth", "depths"])
        masks_dir = find_existing_dir(timestep_folder, ["object_mask", "masks", "mask"])
        poses_dir = find_existing_dir(timestep_folder, ["poses", "pose"])

        view_files = sorted([f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".png") or f.endswith(".jpg")])
        
        # Load all camera observations for this timestep
        observations = []
        for v in view_files:
            obs = load_view_data(v, images_dir, depth_dir, masks_dir, poses_dir, device)
            observations.append(obs)

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
        t_ingest_ms = (time.time() - t_ingest_start) * 1000.0

        # ---------------------------------------------------------
        # STEP 2: Online Mapping (VDC initialization & Morton pruning)
        # ---------------------------------------------------------
        t_vdc_start = time.time()
        new_xyz_acc, new_rgb_acc, new_scale_acc, new_morton_acc = [], [], [], []
        total_pruned = 0
        
        for obs in observations:
            # Simple rendered placeholder or point-splatted render
            rendered_rgb = obs['rgb'].clone()
            rendered_depth = obs['depth'].clone()

            new_g, prune_mask = pipeline.step_2_online_mapping(
                rgb=obs['rgb'],
                depth=obs['depth'],
                rendered_rgb=rendered_rgb,
                rendered_depth=rendered_depth,
                intrinsic=obs['K'],
                camera_pose=obs['pose'],
                current_morton_codes=scene_gaussians['morton']
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

        # Concatenate newly initialized Gaussians
        num_added = 0
        if len(new_xyz_acc) > 0:
            added_xyz = torch.cat(new_xyz_acc, dim=0)
            added_rgb = torch.cat(new_rgb_acc, dim=0)
            added_scale = torch.cat(new_scale_acc, dim=0)
            added_morton = torch.cat(new_morton_acc, dim=0)
            added_obj_id = torch.zeros(len(added_xyz), dtype=torch.int32, device=device)

            scene_gaussians['xyz'] = torch.cat([scene_gaussians['xyz'], added_xyz], dim=0)
            scene_gaussians['rgb'] = torch.cat([scene_gaussians['rgb'], added_rgb], dim=0)
            scene_gaussians['scale'] = torch.cat([scene_gaussians['scale'], added_scale], dim=0)
            scene_gaussians['morton'] = torch.cat([scene_gaussians['morton'], added_morton], dim=0)
            scene_gaussians['obj_id'] = torch.cat([scene_gaussians['obj_id'], added_obj_id], dim=0)
            num_added = len(added_xyz)

        t_vdc_ms = (time.time() - t_vdc_start) * 1000.0

        # ---------------------------------------------------------
        # STEP 3 & 4: RecurGS SE(3) Motion Estimation & PyBullet Sync
        # ---------------------------------------------------------
        t_se3_ms = 0.0
        if timestep_num > 0 and len(observations) > 0:
            t_se3_start = time.time()
            # Primary reference camera view for pose refinement
            ref_obs = observations[0]

            if len(scene_gaussians['xyz']) > 0:
                # Subsample points for fast, robust SE(3) optimization (max 2048 points)
                N_pts = len(scene_gaussians['xyz'])
                if N_pts > 2048:
                    sub_idx = torch.randperm(N_pts, device=device)[:2048]
                    obj_subset = {
                        'xyz': scene_gaussians['xyz'][sub_idx],
                        'rgb': scene_gaussians['rgb'][sub_idx],
                        'scale': scene_gaussians['scale'][sub_idx]
                    }
                else:
                    obj_subset = {
                        'xyz': scene_gaussians['xyz'],
                        'rgb': scene_gaussians['rgb'],
                        'scale': scene_gaussians['scale']
                    }

                T_fine = pipeline.step_3_estimate_se3_motion(
                    object_gaussians_t0=obj_subset,
                    gt_rgb_t1=ref_obs['rgb'],
                    gt_depth_t1=ref_obs['depth'],
                    intrinsic=ref_obs['K'],
                    camera_pose=ref_obs['pose'],
                    num_iterations=50
                )

                # Sync with PyBullet
                pipeline.step_4_sync_pybullet_physics(
                    pybullet_body_id=1,
                    T_fine=T_fine
                )
            t_se3_ms = (time.time() - t_se3_start) * 1000.0

        # Update Robot Arm & Gripper Joints in PyBullet if trajectory is available
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

        t_elapsed = (time.time() - t_start) * 1000.0
        print(f"  • Gaussians: {len(scene_gaussians['xyz'])} (+{num_added}, -{total_pruned})")
        print(f"  • Timing Breakdown: TSDF={t_ingest_ms:.1f}ms | VDC={t_vdc_ms:.1f}ms | SE(3)={t_se3_ms:.1f}ms | Total={t_elapsed:.1f}ms")

    # -------------------------------------------------------------
    # Final Surface Mesh Extraction via TSDF Marching Cubes
    # -------------------------------------------------------------
    print("\n--- Extracting Final Marching Cubes Surface Mesh from TSDF ---")
    verts, faces = pipeline.tsdf_map.extract_mesh(level=0.0)
    mesh_output_file = os.path.join(args.output_dir, "final_tsdf_mesh.obj")
    
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(mesh_output_file)
        print(f"✓ Saved final solid surface mesh to: {mesh_output_file} ({len(verts)} vertices, {len(faces)} faces)")
    except ImportError:
        print(f"✓ Extracted mesh: {len(verts)} vertices, {len(faces)} faces (trimesh not installed for .obj export).")

    total_time = time.time() - total_start_time
    print(f"\n==================================================================")
    print(f" Processing completed successfully in {total_time:.2f} seconds!")
    print(f"==================================================================")

    if HAS_PYBULLET:
        p.disconnect()


if __name__ == "__main__":
    main()
