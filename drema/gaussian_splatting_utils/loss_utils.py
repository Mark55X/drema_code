#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import Optional
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def masked_l1_loss(network_output: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes L1 loss strictly averaged over valid pixels specified by mask (mask > 0.5).
    If mask is None, falls back to standard unmasked l1_loss.
    """
    if mask is None:
        return torch.abs(network_output - gt).mean()

    diff = torch.abs(network_output - gt)
    m = (mask > 0.5).to(device=diff.device, dtype=diff.dtype)
    if m.dim() == 2:
        m = m.unsqueeze(0).expand_as(diff)
    elif m.dim() == 3 and m.shape[0] == 1:
        m = m.expand_as(diff)
    elif m.dim() == 4 and m.shape[1] == 1:
        m = m.expand_as(diff)

    denom = m.sum().clamp(min=1.0)
    return (diff * m).sum() / denom


def masked_ssim(img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None, window_size: int = 11) -> torch.Tensor:
    """
    Computes SSIM over valid pixels without edge artifacts.
    Evaluates ssim_map over the full continuous image, and evaluates spatial average
    over the eroded valid mask (radius = window_size // 2) so only patches 100% inside
    the valid domain are counted in the loss.
    If mask is None, falls back to standard unmasked ssim.
    """
    if mask is None:
        return ssim(img1, img2, window_size=window_size)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1_val = 0.01 ** 2
    C2_val = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1_val) * (2 * sigma12 + C2_val)) / ((mu1_sq + mu2_sq + C1_val) * (sigma1_sq + sigma2_sq + C2_val))

    m_cast = (mask > 0.5).to(device=img1.device, dtype=img1.dtype)
    if m_cast.dim() == 2:
        m_2d = m_cast.unsqueeze(0).unsqueeze(0)
    elif m_cast.dim() == 3:
        m_2d = m_cast.unsqueeze(0)
    else:
        m_2d = m_cast

    r = window_size // 2
    # Erode valid mask so boundary patches containing masked pixels are not averaged
    eroded_mask = 1.0 - F.max_pool2d(1.0 - m_2d, kernel_size=2 * r + 1, stride=1, padding=r)
    eroded_mask = (eroded_mask > 0.5).type_as(img1)

    if eroded_mask.sum() == 0:
        if m_2d.sum() == 0:
            return torch.tensor(1.0, device=img1.device, dtype=img1.dtype)
        eval_mask = m_2d.expand_as(ssim_map) if ssim_map.dim() == 4 else m_2d
    else:
        eval_mask = eroded_mask.expand_as(ssim_map) if ssim_map.dim() == 4 else eroded_mask

    denom = eval_mask.sum().clamp(min=1.0)
    return (ssim_map * eval_mask).sum() / denom