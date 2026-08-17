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
    create_synthetic_gs_data(test_dir)
    print("Synthetic dataset created at", test_dir)
    
    python_bin = sys.executable
    cmd = [
        python_bin,
        "drema_code/run_dynamic_vg_mapping.py",
        "--gs_data_path", test_dir,
        "--grid_dim", "32", "32", "32",
        "--voxel_size", "0.02",
        "--origin", "-0.32", "-0.32", "-0.32",
        "--output_dir", "/tmp/test_output_mapping"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.returncode != 0:
        print("STDERR:\n", res.stderr)
        assert False, f"Runner test failed with code {res.returncode}"
    
    print("✓ Dynamic Sequence Runner End-to-End Test PASSED!")
    shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    run_test()
