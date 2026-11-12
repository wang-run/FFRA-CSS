"""Losses and evaluation metrics (Sections 3.2.3 and 4.1)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Training losses
# ----------------------------------------------------------------------------
class MRAELoss(nn.Module):
    """MRAE over all pixel-band elements (Eq. 15), used for training and validation."""

    def forward(self, output, target):
        return torch.mean(torch.abs(output - target) / (target + 1e-6))


class SAMLoss(nn.Module):
    """SAM loss 1 - cos(theta) (Eq. 16), computed in fp32."""

    def forward(self, output, target):
        cos = F.cosine_similarity(output.float(), target.float(), dim=1, eps=1e-7)
        return torch.mean(1.0 - torch.clamp(cos, -1.0, 1.0))


def sam_weight(iteration, warmup_iters, lambda_max):
    """Cosine warm-up of the SAM loss weight (Eq. 17)."""
    if iteration < warmup_iters:
        return lambda_max * 0.5 * (1 - math.cos(math.pi * iteration / warmup_iters))
    return lambda_max


# ----------------------------------------------------------------------------
# Evaluation metrics (outputs are clipped to [0, 1] before evaluation)
# ----------------------------------------------------------------------------
def masked_mrae(output, target, threshold=0.01):
    """Test-time MRAE over elements whose ground truth exceeds ``threshold`` (Eq. 26)."""
    output = output.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mask = target > threshold
    if not torch.any(mask):
        return torch.zeros((), device=output.device)
    return torch.mean(torch.abs(output[mask] - target[mask]) / target[mask])


def rmse(output, target):
    """Eq. (27)."""
    return torch.sqrt(torch.mean((output - target) ** 2) + 1e-6)


def psnr(output, target, data_range=255):
    """Eq. (28); images are scaled to [0, 255], which is equivalent to a peak value of 1."""
    n, c, h, w = target.shape
    t = (target.clamp(0.0, 1.0) * data_range).reshape(n, c * h * w)
    o = (output.clamp(0.0, 1.0) * data_range).reshape(n, c * h * w)
    mse = ((t - o) ** 2).sum(dim=1) / (c * h * w)
    return torch.mean(10.0 * torch.log((data_range ** 2) / (mse + 1e-6)) / np.log(10.0))


def _gaussian_window(size=11, sigma=1.5):
    g = torch.tensor([math.exp(-(x - size // 2) ** 2 / (2 * sigma ** 2)) for x in range(size)])
    g = (g / g.sum()).unsqueeze(1)
    return g.mm(g.t()).float().unsqueeze(0).unsqueeze(0)


def ssim(output, target, window_size=11):
    """Band-wise SSIM with an 11x11 Gaussian window (sigma 1.5), averaged over bands and pixels."""
    c = output.shape[1]
    window = _gaussian_window(window_size).expand(c, 1, window_size, window_size).to(output.device)
    pad = window_size // 2
    mu1 = F.conv2d(output, window, padding=pad, groups=c)
    mu2 = F.conv2d(target, window, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = F.conv2d(output * output, window, padding=pad, groups=c) - mu1_sq
    s2 = F.conv2d(target * target, window, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(output * target, window, padding=pad, groups=c) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return ssim_map.mean()


def sam(output, target):
    """Mean spectral angle in radians over all pixels (Eq. 29)."""
    cos = F.cosine_similarity(output, target, dim=1)
    cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.mean(torch.acos(cos))


def roi_cc(output, target, roi=50):
    """Pearson correlation of the ROI-averaged spectra (Eq. 30).

    The ROI is a centered square with side min(50, H // 2, W // 2).
    Inputs are single images [31, H, W] as numpy arrays.
    """
    _, h, w = target.shape
    s = min(roi, h // 2, w // 2)
    x0, y0 = max(0, (w - s) // 2), max(0, (h - s) // 2)
    gt = target[:, y0:y0 + s, x0:x0 + s].mean(axis=(1, 2))
    pr = output[:, y0:y0 + s, x0:x0 + s].mean(axis=(1, 2))
    if gt.size == 0:
        return 0.0
    return float(np.corrcoef(gt, pr)[0, 1])


def evaluate(output, target, mrae_threshold=0.01):
    """All six metrics for one image; output/target are [1, 31, H, W] tensors."""
    output = output.float().clamp(0.0, 1.0)
    target = target.float()
    return {
        "MRAE": masked_mrae(output, target, mrae_threshold).item(),
        "RMSE": rmse(output, target).item(),
        "PSNR": psnr(output, target).item(),
        "SSIM": ssim(output, target).item(),
        "SAM": sam(output, target).item(),
        "CC": roi_cc(output[0].cpu().numpy(), target[0].cpu().numpy()),
    }


def validation_metrics(output, target):
    """Metrics used for checkpoint selection during training (unmasked MRAE, as in Eq. 15)."""
    return {
        "MRAE": MRAELoss()(output, target).item(),
        "RMSE": rmse(output, target).item(),
        "PSNR": psnr(output, target).item(),
        "SSIM": ssim(output, target).item(),
        "SAM": sam(output, target).item(),
    }


class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
