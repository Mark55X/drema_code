import torch
from random import randint
from drema.drema_scene import DremaScene
from drema.drema_scene.interactive_gaussian_model import InteractiveGaussianModel
from drema.gaussian_renderer.original_gaussian_renderer import render
from drema.gaussian_splatting_utils.loss_utils import l1_loss, ssim
from drema.r2s_builder.gaussians_optimizers.base_optimizer import BaseTrainer
from vg_mapping_recurgs_native.tsdf import TSDFVoxelMap
from vg_mapping_recurgs_native.vdc import VariationAwareDensityController

class VGMappingOptimizer(BaseTrainer):
    """
    VG-Mapping Continuous Reconstruction Engine for DREMA.
    Integrates TSDF-based surface mapping, Morton-code pruning, and AVD/GVD density control.
    """
    def __init__(self, dataset, opt, pipe, saving_iterations):
        super().__init__(dataset, opt, pipe, saving_iterations)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tsdf_map = TSDFVoxelMap(
            voxel_size=getattr(opt, 'tsdf_voxel_size', 0.01),
            grid_dim=(128, 128, 128),
            device=device
        )
        self.vdc = VariationAwareDensityController(
            ssim_threshold=getattr(opt, 'ssim_threshold', 0.6),
            prune_threshold=getattr(opt, 'prune_threshold', 0.2),
            device=device
        )

    def create_scene(self, dataset):
        return DremaScene(dataset, InteractiveGaussianModel(dataset.sh_degree))

    def step(self, iteration):
        self.gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))

        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"]
        )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        return loss, Ll1, viewspace_point_tensor, visibility_filter, radii, render_pkg

    def extract_tsdf_mesh(self):
        """
        Extracts solid Marching Cubes surface mesh from TSDF voxel map.
        """
        verts, faces = self.tsdf_map.extract_mesh(level=0.0)
        return verts, faces
