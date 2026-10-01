#!/usr/bin/env python
"""
VG-Mapping & RecurGS Perception Backend for DREMA Suite.
Implements BasePerceptionModule using TSDF volumetric mapping, Variation-aware Density Control (VDC),
and Lie algebra SE(3) rigid body motion tracking.
"""

import os
import sys
import time
import json
from typing import Optional, Tuple, List, Dict, Any, Set

import numpy as np
import torch
import trimesh

# Ensure parent master-thesis directory is in sys.path for vgmapping_drema
thesis_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if thesis_root not in sys.path:
    sys.path.insert(0, thesis_root)

import mcubes
from .base_perception import BasePerceptionModule, InitialScanResult, StreamingUpdateResult, DiscoveredObstacle
from ..vg_mapping.closed_loop_pipeline import DREMAClosedLoopVGMappingPipeline, rotation_matrix_to_quaternion
from ..config import ConfigDict


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


class VGMappingPerceptionModule(BasePerceptionModule):
    """
    Modular Perception Backend implementing VG-Mapping (TSDF + 3DGS) and RecurGS tracking.
    """

    def __init__(self, config: Optional[ConfigDict] = None):
        if config is None:
            config = ConfigDict()

        self.config = config
        raw_device = str(config.get_nested("system.device", "auto")).lower()
        if raw_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif raw_device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = raw_device
        self.voxel_size = float(config.get_nested("perception.voxel_size", 0.01))

        # Workspace priors
        ws_cfg = config.get_nested("perception.workspace", {})
        self.table_z_prior = float(ws_cfg.get("table_z_prior", 0.75))
        self.reachability_radius = float(ws_cfg.get("reachability_radius", 0.95))
        self.robot_base_radius = float(ws_cfg.get("robot_base_radius", 0.13))
        self.obstacle_min_clearance_z = float(ws_cfg.get("obstacle_min_clearance_z", 0.015))

        # Gaussians configuration
        gs_cfg = config.get_nested("perception.gaussians", {})
        self.scale_tangent_min = float(gs_cfg.get("scale_tangent_min", 0.012))
        self.scale_normal_min = float(gs_cfg.get("scale_normal_min", 0.003))
        self.opacity_init = float(gs_cfg.get("opacity_init", 0.95))

        # Tracking configuration
        tr_cfg = config.get_nested("perception.tracking", {})
        self.se3_iterations = int(tr_cfg.get("se3_iterations", 15))
        self.se3_icp_iterations = int(tr_cfg.get("se3_icp_iterations", 12))
        self.se3_subsample = int(tr_cfg.get("subsample", 256))
        self.se3_lr = float(tr_cfg.get("learning_rate", 0.003))
        self.se3_tol = float(tr_cfg.get("tolerance", 0.0001))
        self.proximity_radius = float(tr_cfg.get("proximity_search_radius", 0.25))

        # Mapping & Raycast Pruning configuration
        map_cfg = config.get_nested("perception.mapping", {})
        self.closed_loop_avd = bool(map_cfg.get("closed_loop_avd", True))
        self.raycast_stride = int(map_cfg.get("raycast_stride", 1))
        self.raycast_steps = map_cfg.get("raycast_steps", None)
        self.tau_s = float(map_cfg.get("tau_s", 0.6))
        self.tau_p = float(map_cfg.get("tau_p", 0.2))
        self.tau_floater = float(map_cfg.get("tau_floater", 0.95))
        self.max_weight = float(map_cfg.get("max_weight", 3.0))
        self.safety_margin_factor = float(map_cfg.get("safety_margin_factor", 1.0))

        # Diagnostics & Timing Breakdown
        diag_cfg = config.get_nested("perception.diagnostics", {})
        self.log_interval_frames = int(diag_cfg.get("log_interval_frames", 10))
        self.enable_timing_breakdown = bool(diag_cfg.get("enable_timing_breakdown", True))

        # Photometric SGD Optimization (VG-Mapping Paper Sec. III-B.3, Eq. 10)
        sgd_cfg = config.get_nested("perception.sgd", {})
        self.enable_sgd = bool(sgd_cfg.get("enabled", False))
        self.sgd_steps = int(sgd_cfg.get("steps", 5 if self.enable_sgd else 0))
        self.sgd_lr_color = float(sgd_cfg.get("lr_color", 0.01))
        self.sgd_lr_opacity = float(sgd_cfg.get("lr_opacity", 0.05))
        self.sgd_lambda_ssim = float(sgd_cfg.get("lambda_ssim", 0.2))
        self.sgd_prune_opacity_threshold = float(sgd_cfg.get("prune_opacity_threshold", 0.05))

        # Caching configuration
        cache_cfg = config.get_nested("perception.cache", {})
        self.cache_enabled = bool(cache_cfg.get("enabled", False))
        self.cache_save_on_scan = bool(cache_cfg.get("save_on_scan", False))
        self.cache_dir = str(cache_cfg.get("cache_dir", "cache/scene_init"))

        # Internal pipeline state
        self.vg_pipeline: Optional[DREMAClosedLoopVGMappingPipeline] = None
        self.scene_gaussians: Dict[str, torch.Tensor] = {
            'xyz': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'rgb': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'scale': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'normal': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
            'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device),
            'opacity': torch.empty((0, 1), dtype=torch.float32, device=self.device)
        }

        self.tracked_objects: Dict[int, Dict[str, Any]] = {}
        self.discovered_obstacles: List[DiscoveredObstacle] = []
        self.semantic_labels: Dict[str, int] = {}
        self.robot_ids: Set[int] = set()
        self.virtual_ids: Set[int] = set()
        self.dynamic_object_ids: Set[int] = set()

        self.robot_base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.z_table: float = self.table_z_prior
        self.table_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self.active_workspace_bounds: Dict[str, float] = {}
        self.grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.grid_dim: Tuple[int, int, int] = (64, 64, 64)
        self.workspace_bounds_t: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _apply_semantic_robot_mask(
        self,
        depth_t: torch.Tensor,
        mask_np: Optional[np.ndarray]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Applies per-pixel semantic segmentation mask with morphological dilation to filter robot and virtual markers."""
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

    def process_initial_scan(
        self,
        scan_frames: List[Dict[str, Any]],
        semantic_labels: Dict[str, int],
        robot_base_pos: np.ndarray,
        digital_twin: Optional[Any] = None,
        reachability_radius: Optional[float] = None
    ) -> InitialScanResult:
        """Processes initial 360 scene reconstruction or restores from pre-computed cache."""
        self.semantic_labels = semantic_labels or {}
        if len(robot_base_pos) >= 3:
            self.robot_base_pos = np.array(robot_base_pos[:3], dtype=np.float32)
        if reachability_radius is not None and reachability_radius > 0:
            self.reachability_radius = float(reachability_radius)

        # 1. Attempt Cache Restoration if enabled
        if self.cache_enabled:
            cached_res = self.load_cache(self.cache_dir, digital_twin=digital_twin)
            if cached_res is not None:
                return cached_res

        # 2. Parse Semantic Labels
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

        # 3. Detect Table Surface from Accumulated Point Clouds
        all_pts_list = [f['point_cloud'] for f in scan_frames if 'point_cloud' in f and len(f['point_cloud']) > 0]
        if len(all_pts_list) == 0:
            raise RuntimeError("Initial scan collected 0 valid point clouds.")

        all_pts = np.vstack(all_pts_list)
        z_vals = all_pts[:, 2]
        z_min_scan = float(np.percentile(z_vals, 1))
        z_max_scan = float(np.percentile(z_vals, 99))
        num_bins = max(30, int((z_max_scan - z_min_scan) / 0.005))
        hist, bin_edges = np.histogram(z_vals, bins=num_bins, range=(z_min_scan, z_max_scan))
        peak_indices = np.argsort(hist)[::-1]

        found_table = False
        z_table = self.table_z_prior
        table_pts = None
        for p_idx in peak_indices[:10]:
            candidate_z = float(0.5 * (bin_edges[p_idx] + bin_edges[p_idx + 1]))
            cand_mask = np.abs(z_vals - candidate_z) <= 0.008
            cand_pts = all_pts[cand_mask]
            if len(cand_pts) >= 400:
                z_table = candidate_z
                table_pts = cand_pts
                found_table = True
                break

        if not found_table or table_pts is None:
            z_table = self.table_z_prior
            tab_x_min, tab_x_max = -0.50, 1.10
            tab_y_min, tab_y_max = -0.55, 0.55
        else:
            tab_x_min = float(np.percentile(table_pts[:, 0], 0.5))
            tab_x_max = float(np.percentile(table_pts[:, 0], 99.5))
            tab_y_min = float(np.percentile(table_pts[:, 1], 0.5))
            tab_y_max = float(np.percentile(table_pts[:, 1], 99.5))

        self.z_table = z_table
        rb_x, rb_y, rb_z = float(self.robot_base_pos[0]), float(self.robot_base_pos[1]), float(self.robot_base_pos[2])
        r_reach = self.reachability_radius
        margin = 0.05

        act_x_min = max(tab_x_min, rb_x - r_reach - margin)
        act_x_max = min(tab_x_max, rb_x + r_reach + margin)
        act_y_min = max(tab_y_min, rb_y - r_reach - margin)
        act_y_max = min(tab_y_max, rb_y + r_reach + margin)
        act_z_min = z_table - 0.05
        act_z_max = z_table + 0.95

        self.table_bounds = (act_x_min, act_x_max, act_y_min, act_y_max)
        self.active_workspace_bounds = {
            'x_min': act_x_min, 'x_max': act_x_max,
            'y_min': act_y_min, 'y_max': act_y_max,
            'z_min': act_z_min, 'z_max': act_z_max,
            'z_table': z_table
        }

        print(f"✓ [DREMA Scan] Table surface detected at Z={z_table:.3f}m, bounds: X[{tab_x_min:.3f}, {tab_x_max:.3f}], Y[{tab_y_min:.3f}, {tab_y_max:.3f}]")
        print(f"✓ [DREMA Reachability] Robot Base: [{rb_x:.2f}, {rb_y:.2f}, {rb_z:.2f}], Reach Radius: {r_reach:.2f}m")
        print(f"✓ [DREMA Active Workspace] X[{act_x_min:.3f}, {act_x_max:.3f}], Y[{act_y_min:.3f}, {act_y_max:.3f}], Z[{act_z_min:.3f}, {act_z_max:.3f}]")

        # 4. Spawn Solid Table in Digital Twin
        if digital_twin is not None:
            digital_twin.spawn_scanned_table(table_z=z_table, bounds=self.table_bounds)

        # 5. Dynamically Initialize TSDF Grid for Active Workspace
        ext_x = act_x_max - act_x_min
        ext_y = act_y_max - act_y_min
        ext_z = act_z_max - act_z_min

        nx = max(32, int(np.ceil(ext_x / self.voxel_size)))
        ny = max(32, int(np.ceil(ext_y / self.voxel_size)))
        nz = max(32, int(np.ceil(ext_z / self.voxel_size)))

        nx = ((nx + 7) // 8) * 8
        ny = ((ny + 7) // 8) * 8
        nz = ((nz + 7) // 8) * 8

        self.grid_origin = (round(act_x_min, 4), round(act_y_min, 4), round(act_z_min, 4))
        self.grid_dim = (nx, ny, nz)

        print(f"✓ [VG-Mapping TSDF] Dynamic Grid configured: origin={self.grid_origin}, dim={self.grid_dim} ({nx*self.voxel_size:.2f}m x {ny*self.voxel_size:.2f}m x {nz*self.voxel_size:.2f}m)")
        self.vg_pipeline = DREMAClosedLoopVGMappingPipeline(
            pybullet_client=None,
            voxel_size=self.voxel_size,
            grid_dim=self.grid_dim,
            origin=self.grid_origin,
            max_weight=self.max_weight,
            tau_s=self.tau_s,
            tau_p=self.tau_p,
            tau_floater=self.tau_floater,
            safety_margin_factor=self.safety_margin_factor,
            device=self.device
        )


        # 6. Ingest All Scan Frames into VG-Mapping (TSDF + 3DGS)
        num_views = len(scan_frames)
        print(f"\n[VG-Mapping] Starting ingestion of {num_views} scan views into TSDF Voxel Grid & 3DGS...")

        workspace_bounds_t = (
            torch.tensor([act_x_min, act_y_min, act_z_min], dtype=torch.float32, device=self.device),
            torch.tensor([act_x_max, act_y_max, act_z_max], dtype=torch.float32, device=self.device)
        )
        self.workspace_bounds_t = workspace_bounds_t

        new_xyz_acc, new_rgb_acc, new_scale_acc, new_normal_acc, new_morton_acc, new_obj_id_acc = [], [], [], [], [], []
        raw_gaussians_count = 0

        for f_idx, frame_data in enumerate(scan_frames):
            t_start_view = time.time()
            rgb_t = torch.from_numpy(frame_data['rgb'].copy()).permute(2, 0, 1).float().to(self.device) / 255.0
            depth_t = torch.from_numpy(frame_data['depth'].copy()).unsqueeze(0).to(self.device)
            k_t = torch.from_numpy(frame_data['intrinsics']).to(self.device)
            pose_t = torch.from_numpy(frame_data['extrinsics']).to(self.device)
            mask_np = frame_data.get('mask')

            depth_tsdf, mask_t = self._apply_semantic_robot_mask(depth_t=depth_t, mask_np=mask_np)

            # Step 1: TSDF integration (robot masked to 0)
            self.vg_pipeline.step_1_ingest_frame(rgb=rgb_t, depth=depth_tsdf, intrinsic=k_t, camera_pose=pose_t)

            # Step 2: VDC Gaussian mapping (depth_tsdf passed to avoid phantom robot primitives)
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
                # Geometric exclusion of robot base mounting column
                rx, ry, rz = float(self.robot_base_pos[0]), float(self.robot_base_pos[1]), float(self.robot_base_pos[2])
                p_xy = new_g['xyz'][:, :2]
                dist_base = torch.sqrt((p_xy[:, 0] - rx) ** 2 + (p_xy[:, 1] - ry) ** 2)
                keep_geom = (dist_base > self.robot_base_radius) | (new_g['xyz'][:, 2] < (rz - 0.05))
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

        # Deduplication by Morton codes (1 Gaussian per voxel)
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
        self.scene_gaussians['opacity'] = torch.full((retained_count, 1), self.opacity_init, device=self.device, dtype=torch.float32)

        # Optional Paper-Compliant Photometric SGD on Initial Scan (Sec. III-B.3)
        if self.enable_sgd and retained_count > 0 and len(scan_frames) > 0:
            n_scan_iters = max(10, self.sgd_steps * 2)
            print(f"\n[VG-Mapping Initial Scan] Executing Photometric SGD Optimization ({n_scan_iters} iterations across {len(scan_frames)} scan views)...")
            t_sgd_scan_start = time.perf_counter()
            try:
                from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
                from ..gaussian_splatting_utils.loss_utils import l1_loss, ssim
                from ..vg_mapping.closed_loop_pipeline import getProjectionMatrix
                import math

                op_clamped = torch.clamp(self.scene_gaussians['opacity'], 1e-4, 1.0 - 1e-4)
                opacity_logit = torch.logit(op_clamped).detach().requires_grad_(True)
                rgb_param = self.scene_gaussians['rgb'].clone().detach().requires_grad_(True)

                sgd_opt = torch.optim.Adam([
                    {'params': [opacity_logit], 'lr': self.sgd_lr_opacity},
                    {'params': [rgb_param], 'lr': self.sgd_lr_color}
                ])

                means3D = self.scene_gaussians['xyz']
                screenspace_pts = torch.zeros_like(means3D)
                scales = self.scene_gaussians['scale']
                rotations = torch.zeros((retained_count, 4), device=self.device, dtype=torch.float32)
                rotations[:, 0] = 1.0

                prepped_scan_views = []
                for f_data in scan_frames:
                    c2w_t = torch.from_numpy(f_data['extrinsics'].copy()).to(self.device, dtype=torch.float32)
                    k_t = torch.from_numpy(f_data['intrinsics'].copy()).to(self.device, dtype=torch.float32)
                    raw_rgb = f_data['rgb']
                    if isinstance(raw_rgb, np.ndarray):
                        if raw_rgb.ndim == 3 and raw_rgb.shape[2] == 3:
                            gt_rgb = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).float().to(self.device)
                        else:
                            gt_rgb = torch.from_numpy(raw_rgb.copy()).float().to(self.device)
                    else:
                        gt_rgb = raw_rgb.clone().to(self.device, dtype=torch.float32)
                        if gt_rgb.ndim == 3 and gt_rgb.shape[2] == 3:
                            gt_rgb = gt_rgb.permute(2, 0, 1)

                    if gt_rgb.max() > 1.0:
                        gt_rgb = gt_rgb / 255.0
                    c_h, c_w = int(gt_rgb.shape[1]), int(gt_rgb.shape[2])

                    c_fx, c_fy = float(k_t[0, 0]), float(k_t[1, 1])
                    c_fovx = 2.0 * math.atan(c_w / (2.0 * c_fx))
                    c_fovy = 2.0 * math.atan(c_h / (2.0 * c_fy))
                    c_tanfovx = math.tan(c_fovx * 0.5)
                    c_tanfovy = math.tan(c_fovy * 0.5)

                    c_cam_center = c2w_t[:3, 3]
                    c_w2c = torch.inverse(c2w_t)
                    c_view_transform = c_w2c.transpose(0, 1).contiguous()
                    c_projmatrix = getProjectionMatrix(znear=0.01, zfar=20.0, fovX=c_fovx, fovY=c_fovy).transpose(0, 1).to(self.device)
                    c_full_proj = (c_view_transform.unsqueeze(0).bmm(c_projmatrix.unsqueeze(0))).squeeze(0).contiguous()

                    c_settings = GaussianRasterizationSettings(
                        image_height=int(c_h),
                        image_width=int(c_w),
                        tanfovx=c_tanfovx,
                        tanfovy=c_tanfovy,
                        bg=torch.zeros(3, device=self.device, dtype=torch.float32),
                        scale_modifier=1.0,
                        viewmatrix=c_view_transform,
                        projmatrix=c_full_proj,
                        sh_degree=0,
                        campos=c_cam_center,
                        prefiltered=False,
                        debug=False
                    )
                    prepped_scan_views.append({
                        'rasterizer': GaussianRasterizer(raster_settings=c_settings),
                        'gt_rgb': gt_rgb,
                        'gt_rgb_4d': gt_rgb.unsqueeze(0)
                    })

                for it in range(n_scan_iters):
                    sgd_opt.zero_grad()
                    step_loss = torch.tensor(0.0, device=self.device)
                    curr_alpha = torch.sigmoid(opacity_logit)
                    curr_colors = torch.clamp(rgb_param, 0.0, 1.0)

                    for pv in prepped_scan_views:
                        rendered, _ = pv['rasterizer'](
                            means3D=means3D,
                            means2D=screenspace_pts,
                            shs=None,
                            colors_precomp=curr_colors,
                            opacities=curr_alpha,
                            scales=scales,
                            rotations=rotations,
                            cov3D_precomp=None
                        )
                        ll1 = l1_loss(rendered, pv['gt_rgb'])
                        ssim_val = ssim(rendered.unsqueeze(0), pv['gt_rgb_4d'])
                        view_loss = (1.0 - self.sgd_lambda_ssim) * ll1 + self.sgd_lambda_ssim * (1.0 - ssim_val)
                        step_loss = step_loss + view_loss

                    step_loss.backward()
                    sgd_opt.step()

                with torch.no_grad():
                    self.scene_gaussians['opacity'] = torch.sigmoid(opacity_logit).detach()
                    self.scene_gaussians['rgb'] = torch.clamp(rgb_param, 0.0, 1.0).detach()

                    if self.sgd_prune_opacity_threshold > 0.0:
                        keep_op = (self.scene_gaussians['opacity'].squeeze(-1) >= self.sgd_prune_opacity_threshold)
                        if not torch.all(keep_op):
                            n_before = len(self.scene_gaussians['xyz'])
                            for k in list(self.scene_gaussians.keys()):
                                if isinstance(self.scene_gaussians[k], torch.Tensor) and len(self.scene_gaussians[k]) == len(keep_op):
                                    self.scene_gaussians[k] = self.scene_gaussians[k][keep_op]
                            print(f"✓ [SGD Initial Pruning] Pruned {n_before - len(self.scene_gaussians['xyz'])} low-opacity primitives.")

                t_sgd_scan = (time.perf_counter() - t_sgd_scan_start) * 1000.0
                print(f"✓ [VG-Mapping Initial Scan] SGD Optimization completed in {t_sgd_scan:.1f}ms! Final primitives: {len(self.scene_gaussians['xyz']):,}")

            except Exception as e_sgd:
                print(f"[VG MAPPING PERCEPTION Warning] Error during initial scan SGD optimization: {e_sgd}")

        # 7. Extract Tabletop Obstacle Meshes via Marching Cubes
        z_cutoff = z_table + self.obstacle_min_clearance_z
        verts, faces = extract_obstacle_mesh_from_tsdf(
            self.vg_pipeline.tsdf_map,
            z_min_cutoff=z_cutoff,
            z_max_cutoff=act_z_max,
            x_bounds=(tab_x_min + 0.02, tab_x_max - 0.02),
            y_bounds=(tab_y_min + 0.02, tab_y_max - 0.02),
            level=0.0
        )
        print(f"✓ [VG-Mapping Marching Cubes] Extracted raw obstacle surface mesh: {len(verts)} vertices, {len(faces)} faces.")

        self.discovered_obstacles.clear()
        if len(verts) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            components = mesh.split(only_watertight=False)

            valid_components = []
            for comp in components:
                ext = comp.bounds[1] - comp.bounds[0]
                center_c = (comp.bounds[0] + comp.bounds[1]) / 2.0
                in_table = (
                    (center_c[0] >= tab_x_min + 0.02) and (center_c[0] <= tab_x_max - 0.02) and
                    (center_c[1] >= tab_y_min + 0.02) and (center_c[1] <= tab_y_max - 0.02) and
                    (center_c[2] >= z_table + 0.01) and (center_c[2] <= act_z_max)
                )
                if in_table and np.max(ext) >= 0.02 and len(comp.vertices) >= 20:
                    valid_components.append(comp)

            if len(valid_components) == 0 and len(verts) >= 20:
                valid_components = [mesh]

            valid_components.sort(key=lambda c: len(c.vertices), reverse=True)
            print(f"✓ [VG-Mapping Marching Cubes] Discovered {len(valid_components)} distinct tabletop obstacle mesh(es).")

            available_obj_names = [id_to_name[t] for t in self.dynamic_object_ids if t in id_to_name]
            os.makedirs("assets/scanned_meshes", exist_ok=True)

            for c_idx, comp in enumerate(valid_components):
                comp_verts = comp.vertices
                comp_faces = comp.faces
                min_b, max_b = comp.bounds[0], comp.bounds[1]
                centroid = list((min_b + max_b) / 2.0)
                extents = list(max_b - min_b)

                if len(available_obj_names) > 0:
                    obj_name = available_obj_names.pop(0)
                else:
                    obj_name = f"obstacle_{c_idx}"

                mesh_path = f"assets/scanned_meshes/{obj_name}_{c_idx}.obj"
                verts_centered = comp_verts - np.array(centroid)
                comp_centered = trimesh.Trimesh(vertices=verts_centered, faces=comp_faces)
                comp_centered.export(mesh_path)

                # Discover real RGB color from associated surface Gaussians
                g_xyz = self.scene_gaussians['xyz']
                in_box = (
                    (g_xyz[:, 0] >= min_b[0] - 0.02) & (g_xyz[:, 0] <= max_b[0] + 0.02) &
                    (g_xyz[:, 1] >= min_b[1] - 0.02) & (g_xyz[:, 1] <= max_b[1] + 0.02) &
                    (g_xyz[:, 2] >= min_b[2] - 0.01) & (g_xyz[:, 2] <= max_b[2] + 0.02)
                )
                if torch.any(in_box):
                    mean_color = self.scene_gaussians['rgb'][in_box].mean(dim=0).cpu().numpy().tolist()
                else:
                    mean_color = [0.15, 0.45, 0.85]

                pb_body_id = -1
                if digital_twin is not None:
                    pb_body_id = digital_twin.spawn_scanned_mesh_obstacle(
                        mesh_path=mesh_path,
                        initial_pos=centroid,
                        initial_quat=(0, 0, 0, 1),
                        name=f"{obj_name}_{c_idx}",
                        is_target=False,
                        mass=0.0,
                        color=mean_color + [1.0],
                        obj_id=c_idx
                    )

                # Initialize canonical tracking point cloud
                if torch.any(in_box):
                    c_xyz = self.scene_gaussians['xyz'][in_box]
                    c_rgb = self.scene_gaussians['rgb'][in_box]
                else:
                    v_t = torch.tensor(comp_verts, dtype=torch.float32, device=self.device)
                    c_xyz = v_t
                    c_rgb = torch.tensor(mean_color, dtype=torch.float32, device=self.device).repeat(len(v_t), 1)

                self.tracked_objects[c_idx] = {
                    'name': obj_name,
                    'mesh_path': mesh_path,
                    'initial_pos': centroid,
                    'initial_quat': (0.0, 0.0, 0.0, 1.0),
                    'last_pos': centroid,
                    'last_quat': (0.0, 0.0, 0.0, 1.0),
                    'dims': extents,
                    'color': mean_color,
                    'pybullet_id': pb_body_id,
                    'is_target': False,
                    'canonical_points': {
                        'xyz': c_xyz,
                        'rgb': c_rgb
                    }
                }

                self.discovered_obstacles.append(DiscoveredObstacle(
                    oid=c_idx,
                    name=obj_name,
                    mesh_path=mesh_path,
                    initial_pos=tuple(centroid),
                    initial_quat=(0.0, 0.0, 0.0, 1.0),
                    dims=tuple(extents),
                    rgb=mean_color,
                    is_target=False,
                    pybullet_body_id=pb_body_id
                ))

                print(f"✓ Digital Twin: Spawned mesh object ID {c_idx} (PyBullet Body ID: {pb_body_id}, Target=False)")
                print(f"   -> Object #{c_idx} ('{obj_name}'):")
                print(f"      Center: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}] m | Extents: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}] m")
                print(f"      Associated Gaussians: {in_box.sum().item():,} | Real RGB Color: [{mean_color[0]:.2f}, {mean_color[1]:.2f}, {mean_color[2]:.2f}]")

        # 8. Save to cache if enabled
        if self.cache_save_on_scan:
            self.save_cache(self.cache_dir)

        return InitialScanResult(
            table_z=self.z_table,
            table_bounds=self.table_bounds,
            active_workspace_bounds=self.active_workspace_bounds,
            grid_origin=self.grid_origin,
            grid_dim=self.grid_dim,
            voxel_size=self.voxel_size,
            discovered_obstacles=self.discovered_obstacles,
            raw_gaussians_count=raw_gaussians_count,
            retained_gaussians_count=retained_count,
            restored_from_cache=False
        )

    def render_scene_view(
        self,
        intrinsic: torch.Tensor,
        camera_pose: torch.Tensor,
        width: int,
        height: int,
        bg_color: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Closed-Loop 3DGS Forward Rasterization for Appearance-based Variation Detection (AVD).
        Renders the active Gaussian map from the viewpoint of the camera (paper Section III-B.1).
        """
        N = len(self.scene_gaussians['xyz'])
        if N == 0:
            if bg_color is None:
                return torch.zeros((3, height, width), device=self.device, dtype=torch.float32)
            return bg_color.view(3, 1, 1).repeat(1, height, width)

        try:
            import math
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
            from drema.gaussian_splatting_utils.graphics_utils import getProjectionMatrix

            fx = float(intrinsic[0, 0].item())
            fy = float(intrinsic[1, 1].item())
            tanfovx = float(width / (2.0 * fx))
            tanfovy = float(height / (2.0 * fy))
            fovx = float(2.0 * math.atan(tanfovx))
            fovy = float(2.0 * math.atan(tanfovy))

            c2w = camera_pose.to(self.device)
            cam_center = c2w[:3, 3]
            w2c = torch.inverse(c2w)
            world_view_transform = w2c.transpose(0, 1).contiguous()
            projmatrix = getProjectionMatrix(znear=0.01, zfar=20.0, fovX=fovx, fovY=fovy).transpose(0, 1).to(self.device)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0).contiguous()

            if bg_color is None:
                bg = torch.zeros(3, device=self.device, dtype=torch.float32)
            else:
                bg = bg_color.to(self.device, dtype=torch.float32)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(height),
                image_width=int(width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg,
                scale_modifier=1.0,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=0,
                campos=cam_center,
                prefiltered=False,
                debug=False
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means3D = self.scene_gaussians['xyz'].contiguous()
            screenspace_points = torch.zeros_like(means3D, requires_grad=False)
            colors_precomp = self.scene_gaussians['rgb'].contiguous()
            scales = self.scene_gaussians['scale'].contiguous()

            # Paper Eq. 15-16: Rotation is identity matrix (quaternion w=1, x=0, y=0, z=0)
            rotations = torch.zeros((N, 4), device=self.device, dtype=torch.float32)
            rotations[:, 0] = 1.0

            opacities = self.scene_gaussians.get('opacity', torch.full((N, 1), self.opacity_init, device=self.device, dtype=torch.float32)).contiguous()

            with torch.no_grad():
                rendered_img, _ = rasterizer(
                    means3D=means3D,
                    means2D=screenspace_points,
                    shs=None,
                    colors_precomp=colors_precomp,
                    opacities=opacities,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=None
                )
            if not getattr(self, '_render_logged', False):
                print(f"✓ [VG MAPPING PERCEPTION] Closed-Loop 3DGS Forward Rasterizer active (rasterizing {N:,} Gaussians at {width}x{height} via diff-gaussian-rasterization)")
                self._render_logged = True
            return torch.clamp(rendered_img, 0.0, 1.0)
        except Exception as e:
            if not getattr(self, '_render_warned', False):
                print(f"[VG MAPPING PERCEPTION Warning] 3DGS forward rasterization failed: {e}")
                self._render_warned = True
            return None

    def update_streaming_frame(
        self,
        timestep: int,
        camera_views: Dict[str, Dict[str, Any]],
        robot_state: Optional[Dict[str, Any]] = None,
        digital_twin: Optional[Any] = None
    ) -> StreamingUpdateResult:
        """Processes incoming multi-camera streaming frames and updates dynamic 3DGS & tracking."""
        t0 = time.perf_counter()
        t_tsdf_total = 0.0
        t_prune_total = 0.0
        t_render_total = 0.0
        t_detect_total = 0.0
        t_sgd_total = 0.0
        t_vdc_total = 0.0
        t_se3_total = 0.0
        total_pruned_in_frame = 0
        total_added_in_frame = 0
        tracked_poses = {}
        tracked_deltas = {}
        all_active_surface_mortons = None

        if self.vg_pipeline is not None and len(camera_views) > 0:
            workspace_bounds_t = self.workspace_bounds_t

            # Multi-view confirmed surface fusion:
            # Pre-gather 3D surface voxels actively confirmed present across ALL camera views in this timestep
            # to guarantee that a grazing ray from one view never prunes an active surface visible to another view.
            active_surface_morton_list = []
            for c_name, c_data in camera_views.items():
                try:
                    c_d = torch.from_numpy(c_data['depth'].copy()).unsqueeze(0).to(self.device)
                    c_k = torch.from_numpy(c_data['intrinsics'].copy()).to(self.device)
                    c_p = torch.from_numpy(c_data['extrinsics'].copy()).to(self.device)
                    c_d_masked, _ = self._apply_semantic_robot_mask(depth_t=c_d, mask_np=c_data.get('mask'))
                    H_c, W_c = c_d_masked.shape[1], c_d_masked.shape[2]
                    v_g, u_g = torch.meshgrid(torch.arange(0, H_c, 2, device=self.device), torch.arange(0, W_c, 2, device=self.device), indexing='ij')
                    u_f, v_f = u_g.flatten(), v_g.flatten()
                    d_f = c_d_masked[0, v_f, u_f]
                    valid_d = (d_f > 0.1) & (d_f < 3.5)
                    if torch.any(valid_d):
                        u_v, v_v, d_v = u_f[valid_d], v_f[valid_d], d_f[valid_d]
                        x_c = (u_v.float() - c_k[0, 2]) * d_v / c_k[0, 0]
                        y_c = (v_v.float() - c_k[1, 2]) * d_v / c_k[1, 1]
                        p_cam_v = torch.stack([x_c, y_c, d_v], dim=-1)
                        p_w_v = p_cam_v @ c_p[:3, :3].T + c_p[:3, 3]
                        m_v = self.vg_pipeline.tsdf_map.point_to_morton(p_w_v)
                        active_surface_morton_list.append(m_v[m_v >= 0])
                except Exception:
                    pass

            if len(active_surface_morton_list) > 0:
                all_active_surface_mortons = torch.cat(active_surface_morton_list).unique()
            else:
                all_active_surface_mortons = None

            for cam_name, cam_data in camera_views.items():
                try:
                    d_tensor = torch.from_numpy(cam_data['depth'].copy()).unsqueeze(0).to(self.device)
                    k_tensor = torch.from_numpy(cam_data['intrinsics'].copy()).to(self.device)
                    t_tensor = torch.from_numpy(cam_data['extrinsics'].copy()).to(self.device)
                    rgb_tensor = torch.from_numpy(cam_data['rgb'].copy()).permute(2, 0, 1).float().to(self.device) / 255.0
                    mask_np = cam_data.get('mask')

                    d_masked, mask_t = self._apply_semantic_robot_mask(depth_t=d_tensor, mask_np=mask_np)

                    # Step 1: Raycast Pruning of deleted objects against prior TSDF map (Eq. 17)
                    # + Direct Floater Pruning for confirmed free space where F > tau_floater (Paper Sec. III-B.2)
                    t_prune_start = time.perf_counter()
                    prune_mask = self.vg_pipeline.vdc.prune_gaussians_via_morton(
                        depth_obs=d_masked,
                        intrinsic=k_tensor,
                        pose=t_tensor,
                        tsdf_map=self.vg_pipeline.tsdf_map,
                        gaussian_morton_codes=self.scene_gaussians['morton'],
                        stride=self.raycast_stride,
                        num_steps=self.raycast_steps,
                        confirmed_surface_mortons=all_active_surface_mortons
                    )

                    floater_mask = self.vg_pipeline.vdc.prune_floaters_via_tsdf(
                        gaussian_xyz=self.scene_gaussians['xyz'],
                        tsdf_map=self.vg_pipeline.tsdf_map,
                        tau_floater=self.tau_floater
                    )
                    total_prune_mask = prune_mask | floater_mask

                    n_pruned = 0
                    if len(total_prune_mask) > 0 and torch.any(total_prune_mask):
                        n_pruned = int(total_prune_mask.sum().item())
                        total_pruned_in_frame += n_pruned
                        keep_mask = ~total_prune_mask
                        for k in ['xyz', 'rgb', 'scale', 'normal', 'morton', 'obj_id', 'opacity']:
                            if k in self.scene_gaussians and len(self.scene_gaussians[k]) == len(keep_mask):
                                self.scene_gaussians[k] = self.scene_gaussians[k][keep_mask]
                    t_prune_total += (time.perf_counter() - t_prune_start) * 1000.0

                    # Step 2: TSDF integration of current observation (updates F and W)
                    t_s1 = time.perf_counter()
                    self.vg_pipeline.step_1_ingest_frame(
                        rgb=rgb_tensor,
                        depth=d_masked,
                        intrinsic=k_tensor,
                        camera_pose=t_tensor
                    )
                    t_tsdf_total += (time.perf_counter() - t_s1) * 1000.0

                    # Step 3: Closed-Loop 3DGS Forward Rasterization (render active Gaussian map)
                    t_render_start = time.perf_counter()
                    if self.closed_loop_avd:
                        rendered_rgb = self.render_scene_view(
                            intrinsic=k_tensor,
                            camera_pose=t_tensor,
                            width=rgb_tensor.shape[2],
                            height=rgb_tensor.shape[1]
                        )
                        if rendered_rgb is None:
                            rendered_rgb = rgb_tensor.clone()
                    else:
                        rendered_rgb = rgb_tensor.clone()
                    rendered_depth = d_masked.clone()
                    if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                        torch.cuda.synchronize()
                    t_render_total += (time.perf_counter() - t_render_start) * 1000.0

                    # Step 4: VDC variation detection & initialization on newly observed surfaces
                    t_det_start = time.perf_counter()
                    new_g = self.vg_pipeline.vdc.detect_and_initialize_gaussians(
                        rgb_obs=rgb_tensor,
                        depth_obs=d_masked,
                        rendered_rgb=rendered_rgb,
                        rendered_depth=rendered_depth,
                        intrinsic=k_tensor,
                        pose=t_tensor,
                        tsdf_map=self.vg_pipeline.tsdf_map,
                        mask_obs=mask_t,
                        workspace_bounds=workspace_bounds_t,
                        is_initial_timestep=False,
                        num_views=len(camera_views),
                        robot_ids=(self.robot_ids | self.virtual_ids),
                        target_object_ids=self.dynamic_object_ids
                    )

                    # 2. Add new Gaussians with Morton deduplication & robot cylinder exclusion
                    if len(new_g['xyz']) > 0 and len(self.robot_base_pos) >= 3:
                        rx, ry, rz = float(self.robot_base_pos[0]), float(self.robot_base_pos[1]), float(self.robot_base_pos[2])
                        p_xy = new_g['xyz'][:, :2]
                        dist_base = torch.sqrt((p_xy[:, 0] - rx) ** 2 + (p_xy[:, 1] - ry) ** 2)
                        keep_geom = (dist_base > self.robot_base_radius) | (new_g['xyz'][:, 2] < (rz - 0.05))
                        if not torch.all(keep_geom):
                            n_g_orig = len(new_g['xyz'])
                            for k in list(new_g.keys()):
                                if isinstance(new_g[k], torch.Tensor) and len(new_g[k]) == n_g_orig:
                                    new_g[k] = new_g[k][keep_geom]

                    if len(new_g['xyz']) > 0:
                        # 0. Filter out invalid sentinel morton codes if any (< 0)
                        valid_m = (new_g['morton'] >= 0)
                        if not torch.all(valid_m):
                            for k in list(new_g.keys()):
                                if isinstance(new_g[k], torch.Tensor) and len(new_g[k]) == len(valid_m):
                                    new_g[k] = new_g[k][valid_m]

                        # 1. Deduplicate within the newly initialized batch (keep 1 Gaussian per voxel)
                        added_m = new_g['morton']
                        if len(added_m) > 1:
                            perm = torch.argsort(added_m)
                            sorted_m = added_m[perm]
                            uniq_mask = torch.ones_like(sorted_m, dtype=torch.bool)
                            uniq_mask[1:] = (sorted_m[1:] != sorted_m[:-1])
                            keep_idx = perm[uniq_mask]
                            for k in list(new_g.keys()):
                                if isinstance(new_g[k], torch.Tensor) and len(new_g[k]) == len(added_m):
                                    new_g[k] = new_g[k][keep_idx]

                        # 2. Filter out additions to voxels that are already densely occupied
                        if len(self.scene_gaussians['morton']) > 0 and len(new_g['morton']) > 0:
                            unoccupied = ~torch.isin(new_g['morton'], self.scene_gaussians['morton'])
                            if not torch.all(unoccupied):
                                for k in list(new_g.keys()):
                                    if isinstance(new_g[k], torch.Tensor) and len(new_g[k]) == len(unoccupied):
                                        new_g[k] = new_g[k][unoccupied]

                        # 3. Commit newly initialized surface Gaussians
                        n_added = len(new_g['xyz'])
                        total_added_in_frame += n_added

                        for k in ['xyz', 'rgb', 'scale', 'normal', 'morton', 'obj_id', 'opacity']:
                            if k in new_g:
                                val = new_g[k]
                            elif k == 'normal':
                                val = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(len(new_g['xyz']), 1)
                            elif k == 'opacity':
                                val = torch.full((len(new_g['xyz']), 1), self.opacity_init, device=self.device, dtype=torch.float32)
                            else:
                                val = torch.zeros(len(new_g['xyz']), dtype=torch.int32, device=self.device)
                            self.scene_gaussians[k] = torch.cat([self.scene_gaussians[k], val], dim=0)

                    t_detect_total += (time.perf_counter() - t_det_start) * 1000.0

                except Exception as e:
                    print(f"[VG MAPPING PERCEPTION Warning] Error processing camera view '{cam_name}': {e}")

            # Step 2.b: Paper-Compliant Photometric SGD Optimization (Paper Sec. III-B.3, Eq. 10)
            t_sgd_total = 0.0
            if self.enable_sgd and self.sgd_steps > 0 and len(self.scene_gaussians.get('xyz', [])) > 0:
                t_sgd_start = time.perf_counter()
                try:
                    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
                    from ..gaussian_splatting_utils.loss_utils import l1_loss, ssim
                    from ..vg_mapping.closed_loop_pipeline import getProjectionMatrix
                    import math

                    N_g = len(self.scene_gaussians['xyz'])
                    if 'opacity' not in self.scene_gaussians or len(self.scene_gaussians['opacity']) != N_g:
                        self.scene_gaussians['opacity'] = torch.full((N_g, 1), self.opacity_init, device=self.device, dtype=torch.float32)

                    op_clamped = torch.clamp(self.scene_gaussians['opacity'], 1e-4, 1.0 - 1e-4)
                    opacity_logit = torch.logit(op_clamped).detach().requires_grad_(True)
                    rgb_param = self.scene_gaussians['rgb'].clone().detach().requires_grad_(True)

                    sgd_opt = torch.optim.Adam([
                        {'params': [opacity_logit], 'lr': self.sgd_lr_opacity},
                        {'params': [rgb_param], 'lr': self.sgd_lr_color}
                    ])

                    means3D = self.scene_gaussians['xyz']
                    screenspace_pts = torch.zeros_like(means3D)
                    scales = self.scene_gaussians['scale']
                    rotations = torch.zeros((N_g, 4), device=self.device, dtype=torch.float32)
                    rotations[:, 0] = 1.0

                    prepped_views = []
                    for c_name, c_data in camera_views.items():
                        c2w_t = torch.from_numpy(c_data['extrinsics'].copy()).to(self.device, dtype=torch.float32)
                        k_t = torch.from_numpy(c_data['intrinsics'].copy()).to(self.device, dtype=torch.float32)
                        raw_rgb = c_data['rgb']
                        if isinstance(raw_rgb, np.ndarray):
                            if raw_rgb.ndim == 3 and raw_rgb.shape[2] == 3:
                                gt_rgb = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).float().to(self.device)
                            else:
                                gt_rgb = torch.from_numpy(raw_rgb.copy()).float().to(self.device)
                        else:
                            gt_rgb = raw_rgb.clone().to(self.device, dtype=torch.float32)
                            if gt_rgb.ndim == 3 and gt_rgb.shape[2] == 3:
                                gt_rgb = gt_rgb.permute(2, 0, 1)

                        if gt_rgb.max() > 1.0:
                            gt_rgb = gt_rgb / 255.0
                        c_h, c_w = int(gt_rgb.shape[1]), int(gt_rgb.shape[2])

                        c_fx, c_fy = float(k_t[0, 0]), float(k_t[1, 1])
                        c_fovx = 2.0 * math.atan(c_w / (2.0 * c_fx))
                        c_fovy = 2.0 * math.atan(c_h / (2.0 * c_fy))
                        c_tanfovx = math.tan(c_fovx * 0.5)
                        c_tanfovy = math.tan(c_fovy * 0.5)

                        c_cam_center = c2w_t[:3, 3]
                        c_w2c = torch.inverse(c2w_t)
                        c_view_transform = c_w2c.transpose(0, 1).contiguous()
                        c_projmatrix = getProjectionMatrix(znear=0.01, zfar=20.0, fovX=c_fovx, fovY=c_fovy).transpose(0, 1).to(self.device)
                        c_full_proj = (c_view_transform.unsqueeze(0).bmm(c_projmatrix.unsqueeze(0))).squeeze(0).contiguous()

                        c_settings = GaussianRasterizationSettings(
                            image_height=int(c_h),
                            image_width=int(c_w),
                            tanfovx=c_tanfovx,
                            tanfovy=c_tanfovy,
                            bg=torch.zeros(3, device=self.device, dtype=torch.float32),
                            scale_modifier=1.0,
                            viewmatrix=c_view_transform,
                            projmatrix=c_full_proj,
                            sh_degree=0,
                            campos=c_cam_center,
                            prefiltered=False,
                            debug=False
                        )
                        prepped_views.append({
                            'rasterizer': GaussianRasterizer(raster_settings=c_settings),
                            'gt_rgb': gt_rgb,
                            'gt_rgb_4d': gt_rgb.unsqueeze(0)
                        })

                    for step_i in range(self.sgd_steps):
                        sgd_opt.zero_grad()
                        step_loss = torch.tensor(0.0, device=self.device)
                        curr_alpha = torch.sigmoid(opacity_logit)
                        curr_colors = torch.clamp(rgb_param, 0.0, 1.0)

                        for pv in prepped_views:
                            rendered, _ = pv['rasterizer'](
                                means3D=means3D,
                                means2D=screenspace_pts,
                                shs=None,
                                colors_precomp=curr_colors,
                                opacities=curr_alpha,
                                scales=scales,
                                rotations=rotations,
                                cov3D_precomp=None
                            )
                            ll1 = l1_loss(rendered, pv['gt_rgb'])
                            ssim_val = ssim(rendered.unsqueeze(0), pv['gt_rgb_4d'])
                            view_loss = (1.0 - self.sgd_lambda_ssim) * ll1 + self.sgd_lambda_ssim * (1.0 - ssim_val)
                            step_loss = step_loss + view_loss

                        step_loss.backward()
                        sgd_opt.step()

                    with torch.no_grad():
                        self.scene_gaussians['opacity'] = torch.sigmoid(opacity_logit).detach()
                        self.scene_gaussians['rgb'] = torch.clamp(rgb_param, 0.0, 1.0).detach()

                        # Paper floater & low-opacity pruning (Sec. III-B.3)
                        if self.sgd_prune_opacity_threshold > 0.0:
                            keep_op = (self.scene_gaussians['opacity'].squeeze(-1) >= self.sgd_prune_opacity_threshold)
                            if not torch.all(keep_op):
                                for k in list(self.scene_gaussians.keys()):
                                    if isinstance(self.scene_gaussians[k], torch.Tensor) and len(self.scene_gaussians[k]) == len(keep_op):
                                        self.scene_gaussians[k] = self.scene_gaussians[k][keep_op]

                except Exception as e_sgd:
                    print(f"[VG MAPPING PERCEPTION Warning] Error during photometric SGD optimization: {e_sgd}")
                t_sgd_total = (time.perf_counter() - t_sgd_start) * 1000.0

            t_vdc_total = t_prune_total + t_render_total + t_detect_total + t_sgd_total

        # Step 3: RecurGS SE(3) Tracking & Digital Twin Synchronization
        if len(self.tracked_objects) > 0 and len(self.scene_gaussians['xyz']) > 0:
            t_s3 = time.perf_counter()
            objects_source = {}
            objects_target = {}
            initial_T_coarse_dict = {}

            xyz_curr = self.scene_gaussians['xyz']
            rgb_curr = self.scene_gaussians['rgb']
            ws = self.active_workspace_bounds
            z_tab = self.z_table

            table_clean_mask = (
                (xyz_curr[:, 2] > (z_tab + 0.008)) & (xyz_curr[:, 2] <= ws['z_max']) &
                (xyz_curr[:, 0] >= ws['x_min']) & (xyz_curr[:, 0] <= ws['x_max']) &
                (xyz_curr[:, 1] >= ws['y_min']) & (xyz_curr[:, 1] <= ws['y_max'])
            )
            cand_xyz = xyz_curr[table_clean_mask]
            cand_rgb = rgb_curr[table_clean_mask]

            for oid, obj_info in self.tracked_objects.items():
                src_xyz = obj_info['canonical_points']['xyz']
                src_rgb = obj_info['canonical_points']['rgb']
                if len(src_xyz) == 0 or len(cand_xyz) < 4:
                    continue

                # Configurable subsample
                N_src = len(src_xyz)
                if N_src > self.se3_subsample:
                    sub_s = torch.randperm(N_src, device=self.device)[:self.se3_subsample]
                    objects_source[oid] = {'xyz': src_xyz[sub_s], 'rgb': src_rgb[sub_s]}
                else:
                    objects_source[oid] = {'xyz': src_xyz, 'rgb': src_rgb}

                # Proximity search around last known position
                last_p = np.array(obj_info['last_pos'])
                dist_p = torch.norm(cand_xyz - torch.tensor(last_p, device=self.device, dtype=torch.float32), dim=1)
                near_mask = dist_p < self.proximity_radius
                if torch.any(near_mask) and near_mask.sum() >= 4:
                    tgt_xyz = cand_xyz[near_mask]
                    tgt_rgb = cand_rgb[near_mask]
                else:
                    tgt_xyz = cand_xyz
                    tgt_rgb = cand_rgb

                N_tgt = len(tgt_xyz)
                if N_tgt > self.se3_subsample:
                    sub_t = torch.randperm(N_tgt, device=self.device)[:self.se3_subsample]
                    objects_target[oid] = {'xyz': tgt_xyz[sub_t], 'rgb': tgt_rgb[sub_t]}
                else:
                    objects_target[oid] = {'xyz': tgt_xyz, 'rgb': tgt_rgb}

                initial_T_coarse_dict[oid] = obj_info.get('last_T', torch.eye(4, device=self.device))

            if len(objects_source) > 0 and len(objects_target) > 0:
                T_fine_dict = self.vg_pipeline.step_3_estimate_multi_se3_motion(
                    objects_source=objects_source,
                    objects_target=objects_target,
                    initial_T_coarse_dict=initial_T_coarse_dict,
                    z_table=z_tab,
                    num_iterations=self.se3_iterations,
                    icp_max_iters=self.se3_icp_iterations,
                    lr=self.se3_lr,
                    tol=self.se3_tol
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

                    # Compute displacement delta relative to previous position
                    old_pos = np.array(self.tracked_objects[oid]['last_pos'])
                    delta_p = np.array(new_pos) - old_pos
                    tracked_deltas[oid] = delta_p

                    if digital_twin is not None:
                        digital_twin.sync_object_pose(oid, new_pos, quat)

                    self.tracked_objects[oid]['last_pos'] = new_pos
                    self.tracked_objects[oid]['last_quat'] = quat
                    tracked_poses[oid] = (new_pos, quat)

                    # Pure VG-Mapping: Gaussians have surface normals assigned directly from TSDF
                    # gradients upon initialization; no artificial rototranslation of normals.

                    # Dynamic Object Gaussian Management (DREMA RecurGS Tracking):
                    # Clean up stale Gaussians belonging specifically to this object (obj_id == oid)
                    # that fall outside the tracked bounding hull as the object moves.
                    # The static table/background (obj_id != oid) is completely untouched.
                    if len(self.scene_gaussians.get('xyz', [])) > 0:
                        g_obj_id = self.scene_gaussians.get('obj_id', None)
                        if g_obj_id is not None and torch.any(g_obj_id == oid):
                            dims = self.tracked_objects[oid].get('dims', [0.1, 0.1, 0.1])
                            r_bbox = max(float(dims[0]), float(dims[1])) * 0.75 + 0.05
                            pos_tensor = torch.tensor(new_pos[:2], device=self.device, dtype=torch.float32)
                            d_xy = torch.norm(self.scene_gaussians['xyz'][:, :2] - pos_tensor, dim=1)

                            stale_obj_mask = (g_obj_id == oid) & (d_xy > r_bbox)
                            if torch.any(stale_obj_mask):
                                keep_mask = ~stale_obj_mask
                                for k in list(self.scene_gaussians.keys()):
                                    if isinstance(self.scene_gaussians[k], torch.Tensor) and len(self.scene_gaussians[k]) == len(keep_mask):
                                        self.scene_gaussians[k] = self.scene_gaussians[k][keep_mask]

            t_se3_total = (time.perf_counter() - t_s3) * 1000.0

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Periodic structured diagnostic report
        if self.enable_timing_breakdown and (timestep % self.log_interval_frames == 0 or timestep <= 2):
            tsdf_active_voxels = 0
            w_max = 0.0
            w_mean = 0.0
            if self.vg_pipeline is not None and self.vg_pipeline.tsdf_map is not None:
                with torch.no_grad():
                    tsdf_m = self.vg_pipeline.tsdf_map
                    tsdf_active_voxels = int(((tsdf_m.W > 0.5) & (tsdf_m.F.abs() < 0.12)).sum().item())
                    w_max = float(tsdf_m.W.max().item())
                    w_pos = tsdf_m.W[tsdf_m.W > 0]
                    w_mean = float(w_pos.mean().item()) if len(w_pos) > 0 else 0.0

            fps = 1000.0 / max(1.0, total_elapsed_ms)
            num_cams = len(camera_views)
            active_g = len(self.scene_gaussians.get('xyz', []))

            print(f"\n[VG MAPPING PERCEPTION #{timestep:04d}]")
            print(f"  ├─ Step 1 (TSDF Ingest): {t_tsdf_total:.1f}ms ({num_cams} views)")
            sgd_log = f" | SGD ({self.sgd_steps} iters): {t_sgd_total:.1f}ms" if (self.enable_sgd and self.sgd_steps > 0) else ""
            print(f"  ├─ Step 2 (VDC Mapping): {t_vdc_total:.1f}ms [Prune: {t_prune_total:.1f}ms | 3DGS Render: {t_render_total:.1f}ms | Init: {t_detect_total:.1f}ms{sgd_log}] | Pruned: {total_pruned_in_frame} | Added: {total_added_in_frame} | Active 3DGS: {active_g:,}")
            if len(tracked_deltas) > 0:
                for oid, d_p in tracked_deltas.items():
                    name_o = self.tracked_objects[oid]['name']
                    pos_o = self.tracked_objects[oid]['last_pos']
                    print(f"  ├─ Step 3 (RecurGS Tracking): {t_se3_total:.1f}ms | Obj #{oid} ('{name_o}'): pos=[{pos_o[0]:.3f}, {pos_o[1]:.3f}, {pos_o[2]:.3f}] | delta=[{d_p[0]:+.4f}, {d_p[1]:+.4f}, {d_p[2]:+.4f}]m")

                    # Generic tracking verification: dynamic object surface retention vs orphan stains
                    if len(self.scene_gaussians.get('xyz', [])) > 0:
                        with torch.no_grad():
                            g_xyz = self.scene_gaussians['xyz']
                            g_rgb = self.scene_gaussians['rgb']
                            g_obj_id = self.scene_gaussians.get('obj_id', None)

                            # Generic match by semantic ID or canonical color similarity
                            col_raw = self.tracked_objects[oid].get('color', [0.2, 0.4, 0.8])
                            obj_col = torch.tensor(col_raw[:3], device=self.device, dtype=torch.float32)
                            col_dist = torch.norm(g_rgb - obj_col, dim=1)
                            if g_obj_id is not None and torch.any(g_obj_id == oid):
                                is_this_obj = (g_obj_id == oid)
                            else:
                                is_this_obj = (col_dist < 0.25)

                            dims = self.tracked_objects[oid].get('dims', [0.1, 0.1, 0.1])
                            r_bbox = max(float(dims[0]), float(dims[1])) * 0.75 + 0.05
                            pos_tensor = torch.tensor(pos_o[:2], device=self.device, dtype=torch.float32)
                            d_xy = torch.norm(g_xyz[:, :2] - pos_tensor, dim=1)

                            active_on_obj = int((is_this_obj & (d_xy <= r_bbox)).sum().item())
                            orphans = is_this_obj & (d_xy > r_bbox)
                            n_orph = int(orphans.sum().item())
                            if n_orph > 0 and self.vg_pipeline is not None and self.vg_pipeline.tsdf_map is not None:
                                orph_xyz = g_xyz[orphans]
                                orph_m = self.scene_gaussians['morton'][orphans]
                                orph_f, orph_w = self.vg_pipeline.tsdf_map.query_tsdf_and_weight(orph_xyz)
                                z_min, z_max = float(orph_xyz[:, 2].min().item()), float(orph_xyz[:, 2].max().item())
                                f_mean = float(orph_f.mean().item())
                                w_mean_orph = float(orph_w.mean().item())
                                in_surf = 0
                                if all_active_surface_mortons is not None and len(all_active_surface_mortons) > 0:
                                    in_surf = int(torch.isin(orph_m, all_active_surface_mortons).sum().item())
                                print(f"  │    └─ Obj #{oid} ('{name_o}') Tracking State: {active_on_obj:,} on body | {n_orph} orphan stains (Z=[{z_min:.3f}, {z_max:.3f}], F_mean={f_mean:.2f}, W_mean={w_mean_orph:.1f}, {in_surf}/{n_orph} on camera surface)")
                            else:
                                print(f"  │    └─ Obj #{oid} ('{name_o}') Tracking State: {active_on_obj:,} on body | 0 orphan stains (CLEAN)")
            else:
                print(f"  ├─ Step 3 (RecurGS Tracking): {t_se3_total:.1f}ms | No objects actively tracked")
            print(f"  ├─ TSDF Voxel Grid: {tsdf_active_voxels:,} surface voxels | W_max: {w_max:.1f} | W_mean: {w_mean:.1f}")
            print(f"  └─ Total Step Latency: {total_elapsed_ms:.1f}ms ({fps:.1f} Hz)\n")

        return StreamingUpdateResult(
            timestep=timestep,
            active_gaussians_count=len(self.scene_gaussians.get('xyz', [])),
            tracked_object_poses=tracked_poses,
            latency_ms=total_elapsed_ms
        )

    def get_viser_splats_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Returns centers, covariances, rgbs, and opacities formatted for Viser 3D Web Visualizer."""
        if len(self.scene_gaussians['xyz']) == 0:
            return None

        pts_np = self.scene_gaussians['xyz'].detach().cpu().numpy()
        rgb_np = np.clip(self.scene_gaussians['rgb'].detach().cpu().numpy(), 0.0, 1.0)
        scale_np = self.scene_gaussians['scale'].detach().cpu().numpy()

        normals_np = None
        if 'normal' in self.scene_gaussians and len(self.scene_gaussians['normal']) == len(pts_np):
            normals_np = self.scene_gaussians['normal'].detach().cpu().numpy()

        # Max splats cap for browser 60fps responsiveness
        max_splats = int(self.config.get_nested("system.viser.max_splats", 50000))
        if len(pts_np) > max_splats:
            sub = np.random.choice(len(pts_np), max_splats, replace=False)
            pts_np = pts_np[sub]
            rgb_np = rgb_np[sub]
            scale_np = scale_np[sub]
            if normals_np is not None:
                normals_np = normals_np[sub]

        # Normalize surface normal vectors
        if normals_np is not None:
            n_norms = np.linalg.norm(normals_np, axis=-1, keepdims=True)
            valid_n = (n_norms > 1e-4).squeeze(-1)
            normals_clean = np.zeros_like(normals_np)
            normals_clean[valid_n] = normals_np[valid_n] / n_norms[valid_n]
            normals_clean[~valid_n] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            normals_clean = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(pts_np), 1))

        # Dynamic scale: s_tan across local surface tangent plane, s_norm through thickness
        s_tan = np.maximum(scale_np[:, 0:1], self.scale_tangent_min)
        s_norm = np.maximum(scale_np[:, 1:2], self.scale_normal_min)

        # Dynamic Anisotropic Covariance: Sigma = s_tan^2 * I + (s_norm^2 - s_tan^2) * (n n^T)
        nnT = normals_clean[:, :, None] @ normals_clean[:, None, :]
        eye3 = np.eye(3, dtype=np.float32)[None, :, :]
        covariances = (s_tan[:, :, None] ** 2) * eye3 + (s_norm[:, :, None] ** 2 - s_tan[:, :, None] ** 2) * nnT
        opacities = np.full((len(pts_np), 1), self.opacity_init, dtype=np.float32)

        return {
            'centers': pts_np,
            'covariances': covariances,
            'rgbs': rgb_np,
            'opacities': opacities
        }

    def save_cache(self, cache_dir: str) -> bool:
        """Saves reconstructed initial scene state to disk."""
        try:
            os.makedirs(cache_dir, exist_ok=True)
            t_start = time.time()

            # 1. Save tensors on CPU for portability
            g_cpu = {k: v.detach().cpu() for k, v in self.scene_gaussians.items()}
            torch.save(g_cpu, os.path.join(cache_dir, "scene_gaussians.pt"))

            if self.vg_pipeline is not None and self.vg_pipeline.tsdf_map is not None:
                tsdf_data = {
                    'F': self.vg_pipeline.tsdf_map.F.detach().cpu(),
                    'W': self.vg_pipeline.tsdf_map.W.detach().cpu()
                }
                torch.save(tsdf_data, os.path.join(cache_dir, "tsdf_map.pt"))

            # 2. Save tracked objects tensors
            t_obj_cpu = {}
            for oid, obj in self.tracked_objects.items():
                obj_c = dict(obj)
                obj_c['canonical_points'] = {
                    'xyz': obj['canonical_points']['xyz'].detach().cpu(),
                    'rgb': obj['canonical_points']['rgb'].detach().cpu()
                }
                if 'last_T' in obj_c and isinstance(obj_c['last_T'], torch.Tensor):
                    obj_c['last_T'] = obj_c['last_T'].detach().cpu()
                t_obj_cpu[oid] = obj_c
            torch.save(t_obj_cpu, os.path.join(cache_dir, "tracked_objects.pt"))

            # 3. Save JSON metadata
            meta = {
                'z_table': float(self.z_table),
                'table_bounds': [float(x) for x in self.table_bounds],
                'active_workspace_bounds': {k: float(v) for k, v in self.active_workspace_bounds.items()},
                'grid_origin': [float(x) for x in self.grid_origin],
                'grid_dim': [int(x) for x in self.grid_dim],
                'voxel_size': float(self.voxel_size),
                'semantic_labels': {k: int(v) for k, v in self.semantic_labels.items()},
                'robot_ids': sorted(list(self.robot_ids)),
                'virtual_ids': sorted(list(self.virtual_ids)),
                'dynamic_object_ids': sorted(list(self.dynamic_object_ids)),
                'discovered_obstacles': [
                    {
                        'oid': obs.oid,
                        'name': obs.name,
                        'mesh_path': obs.mesh_path,
                        'initial_pos': list(obs.initial_pos),
                        'initial_quat': list(obs.initial_quat),
                        'dims': list(obs.dims),
                        'rgb': list(obs.rgb),
                        'is_target': obs.is_target,
                        'pybullet_body_id': obs.pybullet_body_id
                    } for obs in self.discovered_obstacles
                ]
            }
            with open(os.path.join(cache_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            t_elapsed = time.time() - t_start
            print(f"✓ [Perception Cache] Successfully saved initial scene state to '{cache_dir}' in {t_elapsed:.2f}s!")
            return True
        except Exception as e:
            print(f"[Perception Cache Warning] Failed to save scene cache: {e}")
            return False

    def load_cache(
        self,
        cache_dir: str,
        digital_twin: Optional[Any] = None
    ) -> Optional[InitialScanResult]:
        """Loads reconstructed initial scene state from disk."""
        meta_path = os.path.join(cache_dir, "metadata.json")
        g_path = os.path.join(cache_dir, "scene_gaussians.pt")
        tsdf_path = os.path.join(cache_dir, "tsdf_map.pt")
        t_obj_path = os.path.join(cache_dir, "tracked_objects.pt")

        if not (os.path.exists(meta_path) and os.path.exists(g_path) and os.path.exists(tsdf_path)):
            print(f"[Perception Cache] Cache files not found in '{cache_dir}'. Proceeding with full 360° scan.")
            return None

        try:
            t_start = time.time()
            print(f"\n=======================================================")
            print(f"⚡ [Perception Cache] Restoring initial scene state from '{cache_dir}'...")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.z_table = float(meta['z_table'])
            self.table_bounds = tuple(meta['table_bounds'])
            self.active_workspace_bounds = meta['active_workspace_bounds']
            self.grid_origin = tuple(meta['grid_origin'])
            self.grid_dim = tuple(meta['grid_dim'])
            self.voxel_size = float(meta.get('voxel_size', self.voxel_size))
            self.semantic_labels = meta.get('semantic_labels', {})
            self.robot_ids = set(meta.get('robot_ids', []))
            self.virtual_ids = set(meta.get('virtual_ids', []))
            self.dynamic_object_ids = set(meta.get('dynamic_object_ids', []))

            act_x_min, act_x_max = self.active_workspace_bounds['x_min'], self.active_workspace_bounds['x_max']
            act_y_min, act_y_max = self.active_workspace_bounds['y_min'], self.active_workspace_bounds['y_max']
            act_z_min, act_z_max = self.active_workspace_bounds['z_min'], self.active_workspace_bounds['z_max']
            self.workspace_bounds_t = (
                torch.tensor([act_x_min, act_y_min, act_z_min], dtype=torch.float32, device=self.device),
                torch.tensor([act_x_max, act_y_max, act_z_max], dtype=torch.float32, device=self.device)
            )

            # Load 3D Gaussians
            g_loaded = torch.load(g_path, map_location=self.device, weights_only=False)
            for k in ['xyz', 'rgb', 'scale', 'normal', 'morton', 'obj_id', 'opacity']:
                if k in g_loaded:
                    self.scene_gaussians[k] = g_loaded[k].to(self.device)
            if 'opacity' not in self.scene_gaussians or len(self.scene_gaussians['opacity']) != len(self.scene_gaussians['xyz']):
                self.scene_gaussians['opacity'] = torch.full((len(self.scene_gaussians['xyz']), 1), self.opacity_init, device=self.device, dtype=torch.float32)

            # Re-initialize TSDF Voxel Pipeline and restore F, W tensors (with weight clamping)
            self.vg_pipeline = DREMAClosedLoopVGMappingPipeline(
                pybullet_client=None,
                voxel_size=self.voxel_size,
                grid_dim=self.grid_dim,
                origin=self.grid_origin,
                max_weight=self.max_weight,
                tau_s=self.tau_s,
                tau_p=self.tau_p,
                tau_floater=self.tau_floater,
                safety_margin_factor=self.safety_margin_factor,
                device=self.device
            )
            tsdf_loaded = torch.load(tsdf_path, map_location=self.device, weights_only=False)
            self.vg_pipeline.tsdf_map.F.copy_(tsdf_loaded['F'].to(self.device))
            self.vg_pipeline.tsdf_map.W.copy_(torch.clamp(tsdf_loaded['W'].to(self.device), min=0.0, max=self.max_weight))

            # Load tracked objects
            if os.path.exists(t_obj_path):
                t_obj_loaded = torch.load(t_obj_path, map_location=self.device, weights_only=False)
                self.tracked_objects = {}
                for oid, obj in t_obj_loaded.items():
                    obj_dev = dict(obj)
                    obj_dev['canonical_points'] = {
                        'xyz': obj['canonical_points']['xyz'].to(self.device),
                        'rgb': obj['canonical_points']['rgb'].to(self.device)
                    }
                    if 'last_T' in obj_dev and isinstance(obj_dev['last_T'], torch.Tensor):
                        obj_dev['last_T'] = obj_dev['last_T'].to(self.device)
                    self.tracked_objects[oid] = obj_dev

            # Restore Discovered Obstacles
            self.discovered_obstacles.clear()
            for obs_d in meta.get('discovered_obstacles', []):
                self.discovered_obstacles.append(DiscoveredObstacle(
                    oid=obs_d['oid'],
                    name=obs_d['name'],
                    mesh_path=obs_d['mesh_path'],
                    initial_pos=tuple(obs_d['initial_pos']),
                    initial_quat=tuple(obs_d['initial_quat']),
                    dims=tuple(obs_d['dims']),
                    rgb=obs_d['rgb'],
                    is_target=obs_d.get('is_target', False),
                    pybullet_body_id=obs_d.get('pybullet_body_id', -1)
                ))

            # Spawn into Digital Twin
            if digital_twin is not None:
                digital_twin.spawn_scanned_table(table_z=self.z_table, bounds=self.table_bounds)
                for obs in self.discovered_obstacles:
                    if os.path.exists(obs.mesh_path):
                        pb_id = digital_twin.spawn_scanned_mesh_obstacle(
                            mesh_path=obs.mesh_path,
                            initial_pos=obs.initial_pos,
                            initial_quat=obs.initial_quat,
                            name=obs.name,
                            is_target=obs.is_target,
                            mass=0.0,
                            color=obs.rgb + [1.0],
                            obj_id=obs.oid
                        )
                        obs.pybullet_body_id = pb_id
                        if obs.oid in self.tracked_objects:
                            self.tracked_objects[obs.oid]['pybullet_id'] = pb_id

            t_elapsed = time.time() - t_start
            retained_count = len(self.scene_gaussians['xyz'])
            print(f"✓ [Perception Cache] Restored {retained_count:,} Gaussians, TSDF grid {self.grid_dim}, and {len(self.discovered_obstacles)} obstacles in {t_elapsed:.3f}s!")
            print(f"=======================================================\n")

            return InitialScanResult(
                table_z=self.z_table,
                table_bounds=self.table_bounds,
                active_workspace_bounds=self.active_workspace_bounds,
                grid_origin=self.grid_origin,
                grid_dim=self.grid_dim,
                voxel_size=self.voxel_size,
                discovered_obstacles=self.discovered_obstacles,
                raw_gaussians_count=retained_count,
                retained_gaussians_count=retained_count,
                restored_from_cache=True
            )

        except Exception as e:
            print(f"[Perception Cache Error] Failed to restore cache from '{cache_dir}': {e}. Falling back to live scan.")
            return None

    def get_viser_surface_voxels(self) -> Optional[Dict[str, np.ndarray]]:
        """Extracts discrete TSDF surface voxels for Viser visualization."""
        if self.vg_pipeline is None or self.vg_pipeline.tsdf_map is None:
            return None
        try:
            with torch.no_grad():
                tsdf_map = self.vg_pipeline.tsdf_map
                surf_mask = (tsdf_map.W > 0.5) & (tsdf_map.F.abs() < 0.12)
                if surf_mask.any():
                    surf_pts = tsdf_map.voxel_centers[surf_mask].detach().cpu().numpy()
                    surf_colors = np.zeros_like(surf_pts)
                    surf_colors[:, 0] = 0.05
                    surf_colors[:, 1] = 0.85
                    surf_colors[:, 2] = 0.95
                    if len(surf_pts) > 40000:
                        sub_idx = np.random.choice(len(surf_pts), 40000, replace=False)
                        surf_pts = surf_pts[sub_idx]
                        surf_colors = surf_colors[sub_idx]
                    return {
                        'points': surf_pts,
                        'colors': surf_colors,
                        'point_size': self.voxel_size * 0.7
                    }
        except Exception:
            pass
        return None

    def reset(self) -> None:
        """Resets dynamic perception state for a new episode."""
        self.scene_gaussians = {
            'xyz': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'rgb': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'scale': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'normal': torch.empty((0, 3), dtype=torch.float32, device=self.device),
            'morton': torch.empty((0,), dtype=torch.int64, device=self.device),
            'obj_id': torch.empty((0,), dtype=torch.int32, device=self.device)
        }
        self.tracked_objects.clear()
        self.discovered_obstacles.clear()

    def shutdown(self) -> None:
        """Cleans up perception tensors."""
        self.scene_gaussians.clear()
        self.tracked_objects.clear()
        self.discovered_obstacles.clear()
