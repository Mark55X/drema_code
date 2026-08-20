import os
import shutil
import numpy as np
from PIL import Image
import torch
import subprocess
import sys

def create_synthetic_gs_data(base_path):
    os.makedirs(base_path, exist_ok=True)
    
    for t in [0, 1]:
        t_dir = os.path.join(base_path, str(t))
        images_dir = os.path.join(t_dir, "images")
        depth_dir = os.path.join(t_dir, "depth_scaled")
        masks_dir = os.path.join(t_dir, "object_mask")
        poses_dir = os.path.join(t_dir, "poses")
        
        for d in [images_dir, depth_dir, masks_dir, poses_dir]:
            os.makedirs(d, exist_ok=True)
            
        for cam_id in ["0001", "0002"]:
            # Image 64x64
            img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(images_dir, f"{cam_id}.png"))
            
            # Depth 64x64
            depth = np.ones((64, 64), dtype=np.float32) * 0.5
            np.save(os.path.join(depth_dir, f"{cam_id}.npy"), depth)
            
            # Mask 64x64
            mask = np.ones((64, 64), dtype=np.uint8)
            Image.fromarray(mask).save(os.path.join(masks_dir, f"{cam_id}.png"))
            
            # Pose .txt
            pose_content = """1.0 0.0 0.0 0.0
0.0 1.0 0.0 0.0
0.0 0.0 1.0 -0.5
0.0 0.0 0.0 1.0

100.0 0.0 32.0
0.0 100.0 32.0
0.0 0.0 1.0"""
            with open(os.path.join(poses_dir, f"{cam_id}.txt"), "w") as f:
                f.write(pose_content)

def run_test():
    test_dir = "/tmp/test_synthetic_gs_data"
    output_dir = "/tmp/test_output_mapping"
    shutil.rmtree(output_dir, ignore_errors=True)
    create_synthetic_gs_data(test_dir)
    print("Synthetic dataset created at", test_dir)
    
    python_bin = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_dynamic_vg_mapping.py")
    cmd = [
        python_bin,
        script_path,
        "--gs_data_path", test_dir,
        "--grid_dim", "32", "32", "32",
        "--voxel_size", "0.02",
        "--origin", "-0.32", "-0.32", "-0.32",
        "--output_dir", output_dir,
        "--save_gaussians",
        "--gaussians_format", "both"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.returncode != 0:
        print("STDERR:\n", res.stderr)
        assert False, f"Runner test failed with code {res.returncode}"
    
    # Check that Gaussians and manifest were saved
    manifest_path = os.path.join(output_dir, "sequence_manifest.json")
    assert os.path.exists(manifest_path), f"Missing {manifest_path}"
    
    ply_0 = os.path.join(output_dir, "gaussians", "timestep_0000.ply")
    pt_0 = os.path.join(output_dir, "gaussians", "timestep_0000.pt")
    assert os.path.exists(ply_0), f"Missing {ply_0}"
    assert os.path.exists(pt_0), f"Missing {pt_0}"
    
    print(f"✓ Verified sequence manifest: {manifest_path}")
    print(f"✓ Verified Gaussian PLY: {ply_0} ({os.path.getsize(ply_0)} bytes)")
    print(f"✓ Verified Gaussian PT: {pt_0} ({os.path.getsize(pt_0)} bytes)")
    print("✓ Dynamic Sequence Runner End-to-End Test PASSED!")
    shutil.rmtree(test_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)

if __name__ == "__main__":
    run_test()
