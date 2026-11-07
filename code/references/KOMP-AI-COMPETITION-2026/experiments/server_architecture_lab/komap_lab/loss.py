"""R022 loss recipe: CE + weighted Dice + per-image present-class Lovasz."""
import torch
from torch import nn
from torch.nn import functional as F


def lovasz_gradient(sorted_ground_truth):
    total = sorted_ground_truth.sum()
    intersection = total - sorted_ground_truth.float().cumsum(0)
    union = total + (1 - sorted_ground_truth).float().cumsum(0)
    gradient = 1. - intersection / union.clamp_min(1.)
    if len(sorted_ground_truth) > 1:
        gradient[1:] -= gradient[:-1].clone()
    return gradient


def present_class_lovasz(probabilities, target, class_weights):
    """One image at a time; absent GT classes do not enter its weighted mean."""
    losses = []
    for image_prob, image_target in zip(probabilities, target):
        flat_prob = image_prob.permute(1, 2, 0).reshape(-1, image_prob.shape[0])
        flat_target = image_target.reshape(-1)
        classes = torch.unique(flat_target)
        class_losses, weights = [], []
        for cls in classes.tolist():
            fg = (flat_target == cls).to(flat_prob.dtype)
            if not fg.any():
                continue
            errors = (fg - flat_prob[:, cls]).abs()
            errors_sorted, order = torch.sort(errors, descending=True)
            fg_sorted = fg[order]
            class_losses.append(torch.dot(errors_sorted, lovasz_gradient(fg_sorted)))
            weights.append(class_weights[cls])
        if class_losses:
            weights = torch.stack(weights)
            losses.append((torch.stack(class_losses) * weights).sum() / weights.sum())
    return torch.stack(losses).mean()


class R022Loss(nn.Module):
    """0.5 unweighted CE + 0.25 weighted Dice + 0.25 weighted Lovasz."""
    def __init__(self, ce_weight=.5, dice_weight=.25, lovasz_weight=.25,
                 class_weights=(1., 1., 2., 1.)):
        super().__init__()
        if (ce_weight, dice_weight, lovasz_weight) != (.5, .25, .25):
            raise ValueError('R022 loss weights are fixed at .5/.25/.25')
        if len(class_weights) != 4 or any(v <= 0 for v in class_weights):
            raise ValueError('Four positive class weights required')
        self.ce_weight, self.dice_weight, self.lovasz_weight = ce_weight, dice_weight, lovasz_weight
        self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, target):
        if logits.shape[-2:] != target.shape[-2:]:
            raise ValueError(f'Logits {tuple(logits.shape)} must match center target {tuple(target.shape)}')
        ce = F.cross_entropy(logits, target)
        probabilities = logits.softmax(1)
        one_hot = F.one_hot(target, 4).permute(0, 3, 1, 2).to(probabilities.dtype)
        intersection = (probabilities * one_hot).sum((0, 2, 3))
        denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
        dice_by_class = 1. - (2. * intersection + 1e-5) / (denominator + 1e-5)
        weights = self.class_weights / self.class_weights.sum()
        dice = (dice_by_class * weights).sum()
        lovasz = present_class_lovasz(probabilities, target, self.class_weights)
        return self.ce_weight * ce + self.dice_weight * dice + self.lovasz_weight * lovasz
