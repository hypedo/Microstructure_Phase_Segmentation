"""Eutectic-specific, per-image soft-clDice auxiliary objective.

Based on Shit et al., CVPR 2021 (not a new topology loss):
https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html
The finite soft-skeleton approximation is applied only to phase 3. This does
not guarantee topology preservation or force disconnected GT regions to join.
The ordinary four-class segmentation objectives remain active.
"""
import torch
from torch.nn import functional as F


def soft_erode(image):
    vertical = -F.max_pool2d(-image, (3, 1), 1, (1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), 1, (0, 1))
    return torch.minimum(vertical, horizontal)


def soft_skeleton(image, iterations=5):
    if iterations < 0:
        raise ValueError('iterations must be nonnegative')
    image = image.float()
    opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = F.max_pool2d(soft_erode(image), 3, 1, 1)
        difference = F.relu(image - opened)
        skeleton = skeleton + F.relu(difference - skeleton * difference)
    return skeleton


def eutectic_cldice(probabilities, target, iterations=5):
    """Equal image weighting; empty/unskeletonized GT uses the main loss only.

    probabilities: four-class softmax [B,4,H,W]. target: index mask [B,H,W].
    Five erosion iterations focus this auxiliary objective on narrow features;
    broad components remain supervised by CE, Dice and Lovasz as before.
    Computation stays in FP32 even during mixed-precision CNN training.
    """
    if probabilities.ndim != 4 or probabilities.shape[1] != 4:
        raise ValueError('Expected four-class probabilities')
    if target.shape != (probabilities.shape[0], *probabilities.shape[2:]):
        raise ValueError('Target must match batch and spatial dimensions')
    pred = probabilities[:, 3:4].float()
    truth = (target == 3).unsqueeze(1).float()
    predicted_skeleton = soft_skeleton(pred, iterations)
    with torch.no_grad():
        target_skeleton = soft_skeleton(truth, iterations)
    dimensions = (1, 2, 3)
    predicted_mass = predicted_skeleton.sum(dimensions)
    target_mass = target_skeleton.sum(dimensions)
    precision = ((predicted_skeleton * truth).sum(dimensions) + 1.) / (predicted_mass + 1.)
    recall = ((target_skeleton * pred).sum(dimensions) + 1.) / (target_mass + 1.)
    loss = 1. - 2. * precision * recall / (precision + recall).clamp_min(1e-7)
    active = (target_mass > 0).float()
    return (loss * active).sum() / active.sum().clamp_min(1.)
