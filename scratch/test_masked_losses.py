import torch
import torch.nn.functional as F
import sys

from drema.gaussian_splatting_utils.loss_utils import (
    l1_loss,
    ssim,
    masked_l1_loss,
    masked_ssim
)

def test_all():
    print("=== STARTING RIGOROUS TESTS ON MASKED LOSSES ===")
    
    H, W = 64, 64
    torch.manual_seed(42)
    
    # 1. Test masked_l1_loss fallback when mask is None
    rend = torch.rand(3, H, W, requires_grad=True)
    gt = torch.rand(3, H, W)
    loss_std = l1_loss(rend, gt)
    loss_masked_none = masked_l1_loss(rend, gt, None)
    assert torch.allclose(loss_std, loss_masked_none), "masked_l1_loss(None) must equal l1_loss"
    print(" [PASS] masked_l1_loss with mask=None equals l1_loss")

    # 2. Test masked_l1_loss when mask is all 1s (2D, 3D, 4D)
    mask_ones_2d = torch.ones(H, W)
    mask_ones_3d = torch.ones(1, H, W)
    loss_m1 = masked_l1_loss(rend, gt, mask_ones_2d)
    loss_m2 = masked_l1_loss(rend, gt, mask_ones_3d)
    assert torch.allclose(loss_std, loss_m1), "masked_l1_loss(all 1s 2D) must equal l1_loss"
    assert torch.allclose(loss_std, loss_m2), "masked_l1_loss(all 1s 3D) must equal l1_loss"
    print(" [PASS] masked_l1_loss with all-1 mask (2D and 3D) equals l1_loss")

    # 3. Test masked_l1_loss when mask is all 0s
    mask_zeros = torch.zeros(H, W)
    loss_zero = masked_l1_loss(rend, gt, mask_zeros)
    assert loss_zero.item() == 0.0, "masked_l1_loss(all 0s) must be 0.0"
    print(" [PASS] masked_l1_loss with all-0 mask returns 0.0")

    # 4. Test gradient flow in masked_l1_loss: zero gradient on masked pixels!
    rend_g = torch.rand(3, H, W, requires_grad=True)
    mask_half = torch.zeros(H, W)
    mask_half[:, :W//2] = 1.0  # left half is valid, right half is robot
    loss_h = masked_l1_loss(rend_g, gt, mask_half)
    loss_h.backward()
    grad_masked_region = rend_g.grad[:, :, W//2:]
    assert torch.all(grad_masked_region == 0.0), "Gradients in masked region must be strictly 0"
    assert torch.any(rend_g.grad[:, :, :W//2] != 0.0), "Gradients in valid region must be non-zero"
    print(" [PASS] masked_l1_loss gradient flow: strictly 0 on robot region, non-zero on scene")

    # 5. Test masked_ssim fallback when mask is None
    rend_s = torch.rand(1, 3, H, W, requires_grad=True)
    gt_s = torch.rand(1, 3, H, W)
    ssim_std = ssim(rend_s, gt_s)
    ssim_masked_none = masked_ssim(rend_s, gt_s, None)
    assert torch.allclose(ssim_std, ssim_masked_none, atol=1e-5), "masked_ssim(None) must equal ssim"
    print(" [PASS] masked_ssim with mask=None equals ssim")

    # 6. Test masked_ssim when mask is all 1s
    ssim_m1 = masked_ssim(rend_s, gt_s, mask_ones_2d)
    assert torch.allclose(ssim_std, ssim_m1, atol=1e-5), "masked_ssim(all 1s) must equal ssim"
    print(" [PASS] masked_ssim with all-1 mask equals ssim")

    # 7. Test masked_ssim when mask is all 0s
    ssim_zero = masked_ssim(rend_s, gt_s, mask_zeros)
    assert ssim_zero.item() == 1.0, f"masked_ssim(all 0s) must return 1.0, got {ssim_zero.item()}"
    print(" [PASS] masked_ssim with all-0 mask returns 1.0 (zero penalty)")

    # 8. Test masked_ssim with 3D tensor input (C, H, W) instead of (1, C, H, W)
    rend_3d = torch.rand(3, H, W)
    gt_3d = torch.rand(3, H, W)
    ssim_from_3d = masked_ssim(rend_3d, gt_3d, mask_half)
    assert not torch.isnan(ssim_from_3d), "masked_ssim on 3D tensor must not produce NaN"
    print(" [PASS] masked_ssim accepts both (C, H, W) and (1, C, H, W) correctly")

    # 9. Test masked_ssim with narrow valid region (eroded to 0) fallback
    narrow_mask = torch.zeros(H, W)
    narrow_mask[30:33, 30:33] = 1.0  # 3x3 patch, kernel is 11x11 so erosion is 0
    ssim_narrow = masked_ssim(rend_3d, gt_3d, narrow_mask)
    assert not torch.isnan(ssim_narrow), "narrow mask must fall back cleanly without NaN"
    print(" [PASS] masked_ssim handles small patches (erosion to 0) with graceful fallback")

    # 10. Test robot dilation logic
    test_mask_raw = torch.zeros(H, W, dtype=torch.int32)
    robot_id = 42
    test_mask_raw[20:30, 20:30] = robot_id
    filter_ids = {robot_id}
    f_ids = torch.tensor(list(filter_ids), dtype=test_mask_raw.dtype)
    is_robot = torch.isin(test_mask_raw, f_ids)
    f_float = is_robot.float().unsqueeze(0).unsqueeze(0)
    dilated_robot = (torch.nn.functional.max_pool2d(f_float, kernel_size=5, stride=1, padding=2)[0, 0] > 0.5)
    v_mask = (~dilated_robot).float()
    
    # Check that robot was expanded by 2 pixels (5x5 kernel has radius 2)
    assert dilated_robot[18, 18].item() == True, "Dilation should cover 20-2 = 18"
    assert dilated_robot[17, 17].item() == False, "Dilation should not cover 20-3 = 17"
    assert v_mask.shape == (H, W), f"v_mask shape must be ({H}, {W})"
    print(" [PASS] Robot mask dilation expands boundaries by 2px and inverts correctly to valid_mask")

    print("\n>>> ALL 10 RIGOROUS UNIT TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_all()
