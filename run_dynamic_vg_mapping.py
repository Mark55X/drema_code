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
    parser.add_argument("--voxel_size", type=float, default=0.01, help="TSDF voxel grid size in meters")
    parser.add_argument("--grid_dim", type=int, nargs=3, default=[128, 128, 128], help="TSDF grid dimensions (nx ny nz)")
    parser.add_argument("--origin", type=float, nargs=3, default=[-0.64, -0.64, -0.64], help="TSDF grid origin in world coordinates")
    parser.add_argument("--visualize_pybullet", action="store_true", help="Launch PyBullet GUI for real-time visualization")
    parser.add_argument("--output_dir", type=str, default="./output_dynamic_mapping", help="Output directory for saved meshes/logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    return parser.parse_args()


def load_view_data(view_name, images_dir, depth_dir, masks_dir, poses_dir, device):
    """
    Loads RGB, Depth, Mask and Pose for a single camera view.
    """
    img_path = os.path.join(images_dir, f"{view_name}.png")
    depth_path = os.path.join(depth_dir, f"{view_name}.npy")
    mask_path = os.path.join(masks_dir, f"{view_name}.png")
    pose_path = os.path.join(poses_dir, f"{view_name}.txt")

    # Load RGB (3, H, W) normalized to [0, 1]
    rgb_img = Image.open(img_path).convert("RGB")
    rgb_np = np.array(rgb_img, dtype=np.float32) / 255.0
    rgb = torch.tensor(rgb_np, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Load Depth (1, H, W)
    depth_np = np.load(depth_path).astype(np.float32)
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
    device = args.device

    print("=" * 70)
    print(" DREMA Dynamic VG-Mapping & RecurGS SE(3) Closed-Loop Runner")
    print("=" * 70)
    print(f"Dataset Path : {args.gs_data_path}")
    print(f"Device       : {device}")
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

        images_dir = os.path.join(timestep_folder, "images")
        depth_dir = os.path.join(timestep_folder, "depth_scaled")
        masks_dir = os.path.join(timestep_folder, "object_mask")
        poses_dir = os.path.join(timestep_folder, "poses")

        view_files = sorted([f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".png")])
        
        # Load all camera observations for this timestep
        observations = []
        for v in view_files:
            obs = load_view_data(v, images_dir, depth_dir, masks_dir, poses_dir, device)
            observations.append(obs)

        # ---------------------------------------------------------
        # STEP 1: Ingest multi-view Depth into TSDF Voxel Map
        # ---------------------------------------------------------
        for obs in observations:
            pipeline.step_1_ingest_frame(
                rgb=obs['rgb'],
                depth=obs['depth'],
                intrinsic=obs['K'],
                camera_pose=obs['pose']
            )

        # ---------------------------------------------------------
        # STEP 2: Online Mapping (VDC initialization & Morton pruning)
        # ---------------------------------------------------------
        new_xyz_acc, new_rgb_acc, new_scale_acc, new_morton_acc = [], [], [], []
        
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
                keep_mask = ~prune_mask
                for k in ['xyz', 'rgb', 'scale', 'morton', 'obj_id']:
                    scene_gaussians[k] = scene_gaussians[k][keep_mask]

            if len(new_g['xyz']) > 0:
                new_xyz_acc.append(new_g['xyz'])
                new_rgb_acc.append(new_g['rgb'])
                new_scale_acc.append(new_g['scale'])
                new_morton_acc.append(new_g['morton'])

        # Concatenate newly initialized Gaussians
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

        print(f"  • Total scene Gaussians: {len(scene_gaussians['xyz'])}")

        # ---------------------------------------------------------
        # STEP 3 & 4: RecurGS SE(3) Motion Estimation & PyBullet Sync
        # ---------------------------------------------------------
        if timestep_num > 0 and len(observations) > 0:
            # Primary reference camera view for pose refinement
            ref_obs = observations[0]

            # If objects are segmented, perform SE(3) alignment per object
            if len(scene_gaussians['xyz']) > 0:
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
                    camera_pose=ref_obs['pose']
                )

                # Sync with PyBullet
                pipeline.step_4_sync_pybullet_physics(
                    pybullet_body_id=1,
                    T_fine=T_fine
                )

        # Step PyBullet simulation physics if available
        if HAS_PYBULLET:
            p.stepSimulation()

        t_elapsed = (time.time() - t_start) * 1000.0
        print(f"  ✓ Timestep {timestep_num} completed in {t_elapsed:.1f} ms")

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
