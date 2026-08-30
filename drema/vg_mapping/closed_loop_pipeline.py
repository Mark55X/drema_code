import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Set

from vgmapping_drema.tsdf import TSDFVoxelMap
from vgmapping_drema.vdc import VariationAwareDensityController
from vgmapping_drema.recurgs_se3 import RecurGSLieAlgebraAligner, exp_se3, icp_coarse_alignment

def rotation_matrix_to_quaternion(R: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    Converts 3x3 PyTorch rotation matrix to PyBullet quaternion (x, y, z, w).
    """
    R_np = R.detach().cpu().numpy()
    tr = np.trace(R_np)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R_np[2, 1] - R_np[1, 2]) / S
        qy = (R_np[0, 2] - R_np[2, 0]) / S
        qz = (R_np[1, 0] - R_np[0, 1]) / S
    elif (R_np[0, 0] > R_np[1, 1]) and (R_np[0, 0] > R_np[2, 2]):
        S = np.sqrt(1.0 + R_np[0, 0] - R_np[1, 1] - R_np[2, 2]) * 2
        qw = (R_np[2, 1] - R_np[1, 2]) / S
        qx = 0.25 * S
        qy = (R_np[0, 1] + R_np[1, 0]) / S
        qz = (R_np[0, 2] + R_np[2, 0]) / S
    elif R_np[1, 1] > R_np[2, 2]:
        S = np.sqrt(1.0 + R_np[1, 1] - R_np[0, 0] - R_np[2, 2]) * 2
        qw = (R_np[0, 2] - R_np[2, 0]) / S
        qx = (R_np[0, 1] + R_np[1, 0]) / S
        qy = 0.25 * S
        qz = (R_np[1, 2] + R_np[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R_np[2, 2] - R_np[0, 0] - R_np[1, 1]) * 2
        qw = (R_np[1, 0] - R_np[0, 1]) / S
        qx = (R_np[0, 2] + R_np[2, 0]) / S
        qy = (R_np[1, 2] + R_np[2, 1]) / S
        qz = 0.25 * S

    return (float(qx), float(qy), float(qz), float(qw))


class DREMAClosedLoopVGMappingPipeline:
    """
    Closed-Loop Visual-Physics Pipeline for DREMA and PyBullet simulation.
    
    4-Step Architecture:
    - Step 1: RGB-D ingest + segmentation mask ingest + TSDF voxel integration
    - Step 2: Online variation detection (AVD/GVD) & Morton-code raycast pruning
    - Step 3: RecurGS SE(3) Lie algebra parameter estimation xi in se(3)
    - Step 4: PyBullet rigid object sync via resetBasePositionAndOrientation
    """
    def __init__(
        self,
        pybullet_client=None,
        voxel_size: float = 0.01,
        grid_dim: Tuple[int, int, int] = (128, 128, 128),
        origin: Tuple[float, float, float] = (-0.64, -0.64, -0.64),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.p = pybullet_client
        self.device = device
        self.tsdf_map = TSDFVoxelMap(voxel_size=voxel_size, grid_dim=grid_dim, origin=origin, device=device)
        self.vdc = VariationAwareDensityController(device=device)
        self.se3_aligner = RecurGSLieAlgebraAligner(device=device)

        self.tracked_objects: Dict[int, Dict[str, torch.Tensor]] = {}

    def step_1_ingest_frame(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor
    ):
        """
        Step 1: Ingest dual/single RGB-D observation and integrate into TSDF.
        """
        self.tsdf_map.integrate_depth_frame(depth, intrinsic, camera_pose)

    def step_2_online_mapping(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        rendered_rgb: torch.Tensor,
        rendered_depth: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor,
        current_morton_codes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        workspace_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_initial_timestep: bool = False,
        num_views: int = 1,
        robot_ids: Optional[Set[int]] = None,
        target_object_ids: Optional[Set[int]] = None,
        raycast_stride: int = 2,
        raycast_steps: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Step 2: Variation detection & Morton code raycast pruning.
        Returns newly initialized Gaussians and pruning boolean mask.
        """
        prune_mask = self.vdc.prune_gaussians_via_morton(
            depth_obs=depth,
            intrinsic=intrinsic,
            pose=camera_pose,
            tsdf_map=self.tsdf_map,
            gaussian_morton_codes=current_morton_codes,
            stride=raycast_stride,
            num_steps=raycast_steps
        )

        new_gaussians = self.vdc.detect_and_initialize_gaussians(
            rgb_obs=rgb,
            depth_obs=depth,
            rendered_rgb=rendered_rgb,
            rendered_depth=rendered_depth,
            intrinsic=intrinsic,
            pose=camera_pose,
            tsdf_map=self.tsdf_map,
            mask_obs=mask,
            workspace_bounds=workspace_bounds,
            is_initial_timestep=is_initial_timestep,
            num_views=num_views,
            robot_ids=robot_ids,
            target_object_ids=target_object_ids
        )

        return new_gaussians, prune_mask

    def step_3_estimate_se3_motion(
        self,
        object_gaussians_t0: Dict[str, torch.Tensor],
        gt_rgb_t1: torch.Tensor,
        gt_depth_t1: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor,
        num_iterations: int = 50
    ) -> torch.Tensor:
        """
        Step 3: Estimate rigid transformation T_fine in SE(3) using RecurGS Lie algebra optimization.
        """
        T_fine = self.se3_aligner.optimize_se3_pose(
            object_gaussians=object_gaussians_t0,
            gt_rgb=gt_rgb_t1,
            gt_depth=gt_depth_t1,
            intrinsic=intrinsic,
            camera_pose=camera_pose,
            num_iterations=num_iterations
        )
        return T_fine

    def step_3_estimate_multi_se3_motion(
        self,
        objects_gaussians: Dict[int, Dict[str, torch.Tensor]],
        gt_rgb_t1: torch.Tensor,
        gt_depth_t1: torch.Tensor,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor,
        num_iterations: int = 50,
        tol: float = 1e-4
    ) -> Dict[int, torch.Tensor]:
        """
        Step 3 (Multi-Object): Simultaneously estimate independent rigid transformations T_fine_k in SE(3)
        for all K dynamic objects in parallel on GPU.
        """
        return self.se3_aligner.optimize_multi_object_se3_pose(
            objects_gaussians=objects_gaussians,
            gt_rgb=gt_rgb_t1,
            gt_depth=gt_depth_t1,
            intrinsic=intrinsic,
            camera_pose=camera_pose,
            num_iterations=num_iterations,
            tol=tol
        )

    def step_4_sync_pybullet_physics(
        self,
        pybullet_body_id: int,
        T_fine: torch.Tensor,
        initial_position: Optional[Tuple[float, float, float]] = None
    ):
        """
        Step 4: Update PyBullet rigid body pose by applying estimated delta T_fine to initial world pose.
        """
        if self.p is None:
            print(f"[Warning] PyBullet client not attached. Computed T_fine:\n{T_fine}")
            return

        R_fine = T_fine[:3, :3]
        t_fine = T_fine[:3, 3]

        if initial_position is not None:
            init_pos_tensor = torch.tensor(initial_position, dtype=torch.float32, device=T_fine.device)
            pos_world = init_pos_tensor + t_fine
            quat = rotation_matrix_to_quaternion(R_fine)
            pos = (float(pos_world[0]), float(pos_world[1]), float(pos_world[2]))
        else:
            quat = rotation_matrix_to_quaternion(R_fine)
            pos = (float(t_fine[0]), float(t_fine[1]), float(t_fine[2]))

        self.p.resetBasePositionAndOrientation(pybullet_body_id, pos, quat)
        print(f"✓ PyBullet body ID {pybullet_body_id} synced to pos: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), quat: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
