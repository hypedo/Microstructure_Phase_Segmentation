"""Zero-initialized full-resolution residual detail refinement.

Experimental motivation: preserve grayscale and short-range boundary cues
that a stride-two encoder may lose. This is not a research novelty claim.
There is no dataset/label/filename access and no phase-specific hard rule.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(width: int) -> int:
    return next(groups for groups in range(min(8, width), 0, -1) if width % groups == 0)


class ResidualDetailRefiner(nn.Module):
    """Wrap a segmentation backbone returning N x classes x H x W logits.

    The first four channels are mutually exclusive phase logits. If
    ``classes=5``, channel five is an independent boundary logit, matching
    the existing morphology loss. All channels receive learned residuals.

    ``x`` is the same normalized RGB tensor the backbone normally receives.
    Features are normalized RGB, four phase probabilities, the optional
    boundary probability, a 3x3 local grayscale high-pass, and optionally
    two fixed Sobel derivatives. No features are detached: training remains
    end-to-end. The zero projection initially preserves backbone logits.

    To initialize from an unwrapped checkpoint, first load that state into
    the backbone (or ``wrapper.backbone``), then save future checkpoints
    using the wrapper's full state dict. Wrapped keys carry ``backbone.``.
    """

    def __init__(self, backbone: nn.Module, classes: int = 4, width: int = 24,
                 include_sobel: bool = True):
        super().__init__()
        if classes not in (4, 5):
            raise ValueError('classes must be 4 phases or 4 phases plus 1 boundary channel')
        if not isinstance(width, int) or width < 1:
            raise ValueError('width must be a positive integer')
        self.backbone = backbone
        self.classes = classes
        self.include_sobel = bool(include_sobel)
        # RGB + phase probabilities + optional boundary + high-pass + Sobel.
        feature_channels = 3 + 4 + int(classes == 5) + 1 + 2 * self.include_sobel
        groups = _group_count(width)
        self.detail_features = nn.Sequential(
            nn.Conv2d(feature_channels, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Conv2d(width, classes, kernel_size=1)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        sobel = torch.tensor([
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        ]).unsqueeze(1) / 8.
        self.register_buffer('sobel_kernels', sobel, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f'Expected normalized NCHW RGB, got {tuple(x.shape)}')
        if min(x.shape[-2:]) < 2:
            raise ValueError('Spatial dimensions must be at least 2 pixels')
        coarse = self.backbone(x)
        if not isinstance(coarse, torch.Tensor):
            raise TypeError('Backbone must return a logits tensor')
        expected = (x.shape[0], self.classes, *x.shape[-2:])
        if tuple(coarse.shape) != expected:
            raise ValueError(f'Backbone logits must have shape {expected}; got {tuple(coarse.shape)}')
        gray = x.mean(dim=1, keepdim=True)
        high_pass = gray - F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1,
                                      count_include_pad=False)
        features = [x, coarse[:, :4].softmax(dim=1)]
        if self.classes == 5:
            features.append(coarse[:, 4:5].sigmoid())
        features.append(high_pass)
        if self.include_sobel:
            derivatives = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'),
                                   self.sobel_kernels.to(dtype=gray.dtype))
            features.append(derivatives)
        detail = self.detail_features(torch.cat(features, dim=1))
        return coarse + self.projection(detail)


def smoke_test() -> None:
    """CPU-only shape, identity, gradient and checkpoint round-trip checks.

    This uses synthetic tensors only; no data or model download is needed.
    Run ``python src/detail_refinement.py`` in an environment with PyTorch.
    """
    torch.manual_seed(7)
    for classes in (4, 5):
        for sobel in (False, True):
            base = nn.Conv2d(3, classes, kernel_size=1)
            model = ResidualDetailRefiner(base, classes=classes, width=24,
                                          include_sobel=sobel)
            x = torch.randn(2, 3, 37, 53)
            coarse = base(x)
            result = model(x)
            assert result.shape == (2, classes, 37, 53)
            assert torch.equal(result, coarse), 'Zero initialization must preserve logits exactly'
            result.square().mean().backward()
            assert torch.isfinite(model.projection.weight.grad).all()
            assert model.projection.weight.grad.abs().sum() > 0
            assert base.weight.grad.abs().sum() > 0
            restored = ResidualDetailRefiner(nn.Conv2d(3, classes, 1), classes=classes,
                                             width=24, include_sobel=sobel)
            restored.load_state_dict(model.state_dict(), strict=True)
            assert torch.equal(restored(x), result)
    print('Detail-refiner CPU smoke checks passed: 4/5 channels, Sobel on/off, identity, gradients, state round trip.')


if __name__ == '__main__':
    smoke_test()
