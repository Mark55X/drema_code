import torch
import numpy as np
import sys
import os

# Ensure package paths are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_drema_closed_loop_pipeline():
    print("--- DREMA Integration Test: Closed-Loop Pipeline ---")
    from drema.vg_mapping.closed_loop_pipeline import DREMAClosedLoopVGMappingPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = DREMAClosedLoopVGMappingPipeline(device=device)

    # Create synthetic RGB-D observations
    rgb = torch.rand((3, 64, 64), device=device)
    depth = torch.ones((1, 64, 64), device=device) * 0.5
    rendered_rgb = torch.rand((3, 64, 64), device=device)
    rendered_depth = torch.ones((1, 64, 64), device=device) * 0.5

    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], device=device)
    pose = torch.eye(4, device=device)

    # Step 1: Ingest frame
    pipeline.step_1_ingest_frame(rgb, depth, K, pose)
    print("✓ Step 1: Ingest frame PASSED.")

    # Step 2: Online mapping
    current_morton = torch.tensor([10, 20, 30], dtype=torch.int64, device=device)
    new_gaussians, prune_mask = pipeline.step_2_online_mapping(
        rgb, depth, rendered_rgb, rendered_depth, K, pose, current_morton
    )
    print(f"✓ Step 2: Online mapping PASSED. New Gaussians initialized: {len(new_gaussians['xyz'])}")

    # Step 3: SE(3) Motion estimation
    obj_gaussians = {
        'xyz': torch.randn((50, 3), device=device) * 0.1,
        'rgb': torch.rand((50, 3), device=device),
        'scale': torch.ones((50, 3), device=device) * 0.01
    }
    T_fine = pipeline.step_3_estimate_se3_motion(obj_gaussians, rgb, depth, K, pose)
    print(f"✓ Step 3: SE(3) Lie algebra motion estimation PASSED. T_fine shape: {T_fine.shape}")

    # Step 4: Sync PyBullet physics
    pipeline.step_4_sync_pybullet_physics(pybullet_body_id=1, T_fine=T_fine)
    print("✓ Step 4: PyBullet physics sync handler PASSED.")

if __name__ == "__main__":
    test_drema_closed_loop_pipeline()
    print("\n✓ DREMA VG-MAPPING INTEGRATION TEST PASSED SUCCESSFULLY!")
