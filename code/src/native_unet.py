# 도메인과 U-Net 연결: 광학현미경의 Eutectic Si는 가늘고 경계 비중이 높다.
# 첫 학습 특징을 원본 해상도에서 만들고 마지막 decoder까지 skip으로 전달하여,
# 초기 stride로 미세한 상을 소실시키는 위험을 줄이는 설계를 비교했다.
# U-Net 원천 구조: https://arxiv.org/abs/1505.04597
# Residual/GroupNorm/SiLU는 구현 선택이다. 이러한 블록 자체의 원천 발명을 주장하지 않는다.
# base=64의 채널은 64,128,256,512,768이며 네 번의 pooling과 네 번의 upsampling을 쓴다.
# 보조 경계 logit을 갖는 checkpoint에서는 상 분할 추론 시 첫 네 logit만 사용한다.

"""A residual U-Net control with a learned skip at the original resolution.

Inputs are externally normalized RGB tensors [B, 3, H, W]. Outputs are raw
[B, classes, H, W] logits. Four classes represent the phases; a fifth logit
can be used by the training pipeline as an auxiliary boundary prediction.
No external weights, model downloads, normalization, or softmax are used here.

The full-resolution first block and encoder-to-decoder concatenations follow
the established U-Net design. Residual blocks, GroupNorm, SiLU and bilinear
upsampling are implementation choices, not a claim of a new architecture.
This control tests whether retaining learned features before the first
downsampling helps thin phases. Improvement requires a measured ablation.

Primary references (this is not an exact reproduction of either paper):
Ronneberger et al., U-Net (2015):
    https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/
Zhang et al., Road Extraction by Deep Residual U-Net (2017):
    https://arxiv.org/abs/1711.10684
"""

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels):
    """At least two channels per group, including a batch-one 1x1 map."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    raise ValueError('GroupNorm requires at least two channels')


class ResidualBlock(nn.Module):
    """Two stride-one 3x3 convolutions, with a projection only if needed."""

    def __init__(self, in_channels, out_channels, normalization='group'):
        super().__init__()
        if normalization not in ('group','batch'):
            raise ValueError('normalization must be group or batch')
        def norm():
            return (nn.GroupNorm(_groups(out_channels), out_channels) if normalization=='group'
                    else nn.BatchNorm2d(out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = norm()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = norm()
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1, bias=False))
        self.activation = nn.SiLU(inplace=False)

    def forward(self, x):
        identity = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + identity)


class UpBlock(nn.Module):
    """Upsample, concatenate the matching encoder feature, then refine."""

    def __init__(self, in_channels, skip_channels, out_channels, normalization='group'):
        super().__init__()
        self.refine = ResidualBlock(in_channels + skip_channels, out_channels, normalization)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class NativeResolutionUNet(nn.Module):
    """Five scales, four pooling stages, and a full-resolution final skip.

    For base_channels=32 the encoder widths are (32, 64, 128, 256, 384).
    All learned encoder convolutions have stride one. Max pooling performs
    the four reductions. The final decoder block receives the first learned
    full-resolution encoder feature, not only the raw RGB input.

    Images are padded on the right/bottom to a multiple of 16 and logits are
    cropped to the exact input size. Reflection padding is used when valid;
    replication handles tiny images for which reflection is undefined.
    """

    def __init__(self, classes=4, base_channels=32, normalization='group', context='none'):
        super().__init__()
        if classes not in (4, 5):
            raise ValueError('classes must be 4 phases or 4 phases plus boundary')
        if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 2:
            raise ValueError('base_channels must be an integer >= 2')
        self.classes = classes
        self.base_channels = base_channels
        self.normalization = normalization
        self.channels = tuple(base_channels * factor for factor in (1, 2, 4, 8, 12))
        self.stem = ResidualBlock(3, self.channels[0], normalization)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.down_blocks = nn.ModuleList(
            ResidualBlock(cin, cout, normalization) for cin, cout in zip(self.channels[:-1], self.channels[1:])
        )
        self.up_blocks = nn.ModuleList(
            UpBlock(self.channels[level + 1], self.channels[level], self.channels[level], normalization)
            for level in reversed(range(4))
        )
        self.head = nn.Conv2d(self.channels[0], classes, 1)
        if context not in ('none','nonlocal'):
            raise ValueError('context must be none or nonlocal')
        self.context = nn.Identity()
        if context=='nonlocal':
            from nonlocal_context import PooledNonLocalContext
            # Extra module initialization must not alter data-loader RNG seeds
            # or the shared backbone's initialized parameters in this ablation.
            with torch.random.fork_rng(devices=[]):
                self.context=PooledNonLocalContext(self.channels[-1])

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('Expected [B, 3, H, W] normalized RGB tensor')
        if x.shape[0] < 1 or min(x.shape[-2:]) < 1:
            raise ValueError('Batch and spatial dimensions must be positive')
        height, width = x.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        if pad_h or pad_w:
            mode = 'reflect' if pad_h < height and pad_w < width else 'replicate'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        features = [self.stem(x)]
        for block in self.down_blocks:
            features.append(block(self.pool(features[-1])))
        x = self.context(features[-1])
        for block, skip in zip(self.up_blocks, reversed(features[:-1])):
            x = block(x, skip)
        return self.head(x)[..., :height, :width]
