"""Optional ground-truth boundary weighting for four-phase cross entropy.

Class order matches metrics.py: Al=0, Primary_Si=1, Al3Ni=2, Eutectic_Si=3.
The default multiplier of one is ordinary CE. A multiplier of two is the
proposed experimental setting, not an established improvement. No metadata,
predicted boundaries, label modifications, or trainable parameters are used.
"""

import math
from numbers import Real

import torch
from torch.nn import functional as F


def _thin_boundary_mask(target, radius):
    """Four-neighbor label transitions, dilated, then restricted to GT 2/3.

    Both pixels adjoining a transition are marked. Square max pooling dilates
    this initial edge set by Chebyshev distance <= radius; radius=2 therefore
    uses a 5x5 kernel. The outside of an image is not an extra label/boundary.
    This definition differs by up to one pixel from some distance-to-interface
    conventions, so the discrete construction is specified explicitly.
    """
    edge = torch.zeros_like(target, dtype=torch.bool)
    vertical = target[:, 1:, :] != target[:, :-1, :]
    horizontal = target[:, :, 1:] != target[:, :, :-1]
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal
    if radius:
        edge = F.max_pool2d(
            edge[:, None].float(), kernel_size=2 * radius + 1,
            stride=1, padding=radius,
        )[:, 0].bool()
    return edge & ((target == 2) | (target == 3))


def phase_cross_entropy(logits, target, thin_weight=1., radius=2):
    """Mean of independently normalized, optionally weighted image CE losses.

    Args:
        logits: Raw [B, 4, H, W] phase logits; slice auxiliary channels outside
            this function. Float16/bfloat16 inputs are promoted to float32.
        target: Unchanged [B, H, W] long labels in {0, 1, 2, 3}; no ignore label.
        thin_weight: Positive finite CE multiplier on the selected GT pixels.
            One disables weighting, and two is the proposed thin-phase trial.
        radius: Nonnegative integer radius for dilation of the discrete edges.

    Let M be the selected GT-2/3 band and w = 1+(thin_weight-1)*M. The result is
    mean_b(sum_hw(w*CE) / sum_hw(w)), not a batch-global weighted average. This
    keeps images with more selected boundary pixels from obtaining a larger
    image weight. It is a crop loss, not the original-image competition mIoU.
    """
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError('logits must have shape [B, 4, H, W]')
    if target.ndim != 3 or target.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError('target must match logits as [B, H, W]')
    if min(target.shape) < 1:
        raise ValueError('Batch and spatial dimensions must be positive')
    if target.dtype != torch.long:
        raise ValueError('target must contain torch.long class indices')
    if not logits.is_floating_point():
        raise ValueError('logits must be floating point')
    if (not isinstance(thin_weight, Real) or isinstance(thin_weight, bool)
            or not math.isfinite(thin_weight) or thin_weight <= 0):
        raise ValueError('thin_weight must be a positive finite number')
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 0:
        raise ValueError('radius must be a nonnegative integer')
    # A full-resolution sum in fp16 can overflow even for ordinary CE values.
    # The cast preserves autograd and does not reduce float64 check precision.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    if thin_weight == 1:
        return F.cross_entropy(logits, target)
    per_pixel = F.cross_entropy(logits, target, reduction='none')
    selected = _thin_boundary_mask(target, radius)
    weights = 1 + (float(thin_weight) - 1) * selected.to(per_pixel.dtype)
    numerator = (per_pixel * weights).sum(dim=(1, 2))
    denominator = weights.sum(dim=(1, 2))
    return (numerator / denominator).mean()
