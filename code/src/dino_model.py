"""DINOv2 intermediate features with a trainable high-resolution phase decoder.

Input: externally ImageNet-normalized RGB, [B, 3, H, W]. Grayscale replication
and normalization belong to the data pipeline, not to this module. Output:
unnormalized [B, classes, H, W] logits, where classes is 4 or 5. When classes=5,
the fifth channel is available to train.py's auxiliary boundary loss.

This is a task-specific experimental adapter, not the published DPT model and
not a claim of architectural novelty. Four same-resolution ViT feature maps
are reassembled into a feature pyramid. Actual high-resolution information
comes from the independent full/half/quarter-resolution convolutional skips.

Public API source checked at timm v1.0.15 and v1.0.30:
https://huggingface.co/docs/timm/en/feature_extraction
Pretraining: https://arxiv.org/abs/2304.07193
Decoder inspiration: https://arxiv.org/abs/2103.13413

Example integration (train.py is deliberately not modified here):
    model = DinoPhaseSegmenter(classes=5 if config.get('morphology') else 4,
                               pretrained=pretrained)

Smoke checks to run on the training host before launching an experiment:
    model = DinoPhaseSegmenter(classes=5, pretrained=False).eval()
    output = model(torch.randn(1, 3, 129, 173))
    assert output.shape == (1, 5, 129, 173)
    assert torch.isfinite(output).all()
"""
from pathlib import Path
import inspect

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import timm


DEFAULT_ENCODER = 'vit_small_patch14_dinov2.lvd142m'
DEFAULT_WEIGHT_REVISION = '4610ca143709d58a633b6397a74412c2c3842454'
DEFAULT_INDICES = (2, 5, 8, 11)


def _groups(channels):
    for groups in (8, 4, 2, 1):
        # Leave at least two channels per group so even a 1x1 deepest map with
        # batch size one has more than one value for GroupNorm's statistics.
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualRefine(nn.Module):
    """Two local convolutions; GroupNorm also works with a small GPU batch."""
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.body(x))


def _resize(x, shape):
    if x.shape[-2:] == shape:
        return x
    return F.interpolate(x, size=shape, mode='bilinear', align_corners=False)


class DinoPhaseSegmenter(nn.Module):
    """Four DINOv2 block features + pyramid fusion + full-resolution CNN skips.

    Args:
        classes: 4 phase logits, or 4 phase logits plus one boundary logit.
        pretrained: If False, never loads external/local encoder weights. This
            is the path for restoring the complete trained model state_dict.
        encoder_weights: Optional path to a timm-format encoder .safetensors or
            .pth state_dict. With pretrained=True and no path, use public timm
            weights. The literal 'none' disables pretraining explicitly.
        encoder_name: Built-in timm model identifier, not a hub URI. The default
            is ViT-S/14 DINOv2; a compatible DINOv2 backbone can be supplied.
        decoder_channels: Width of the reassembled transformer pyramid.
        stem_channels: Width of the native-resolution CNN skip. Half/quarter
            skip widths are twice/four times this value.
        freeze_encoder: Keep the backbone in eval mode with no encoder grads.
        gradient_checkpointing: Recompute backbone features in backward using
            torch non-reentrant checkpointing; independent of timm's older
            forward_intermediates implementations that omit this feature.

    All newly added modules use standard PyTorch initialization. The pretrained
    encoder is never reinitialized while creating the decoder.
    """
    def __init__(self, classes=4, pretrained=True, encoder_weights=None,
                 encoder_name=DEFAULT_ENCODER, decoder_channels=64,
                 stem_channels=24, indices=DEFAULT_INDICES,
                 freeze_encoder=False, gradient_checkpointing=False):
        super().__init__()
        if classes not in (4, 5):
            raise ValueError('KoMaP expects 4 class logits or 4 plus one boundary logit')
        if decoder_channels < 8 or stem_channels < 8:
            raise ValueError('Decoder/stem channel counts must be >=8')
        if len(indices) != 4 or tuple(sorted(set(indices))) != tuple(indices):
            raise ValueError('Provide exactly four distinct, increasing block indices')
        if any(not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError('Intermediate block indices must be nonnegative integers')
        if ':' in encoder_name or '/' in encoder_name or 'dinov2' not in encoder_name:
            raise ValueError('Use a built-in timm DINOv2 model name, not a remote hub URI')

        self.classes = classes
        self.encoder_name = encoder_name
        self.indices = tuple(indices)
        self.decoder_channels = decoder_channels
        self.stem_channels = stem_channels
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.timm_version = timm.__version__
        self._encoder_frozen = False
        weights_disabled = encoder_weights is not None and str(encoder_weights).lower() == 'none'
        load_remote = bool(pretrained and encoder_weights is None)
        create_options = dict(num_classes=0, global_pool='', img_size=518,
                              dynamic_img_size=True, dynamic_img_pad=False,
                              patch_drop_rate=0.0)
        if load_remote and encoder_name == DEFAULT_ENCODER:
            # A built-in identifier keeps architecture configuration local; this
            # overlay pins only the public weights to an inspected HF revision.
            create_options['pretrained_cfg_overlay'] = {
                'hf_hub_id': f'timm/{DEFAULT_ENCODER}@{DEFAULT_WEIGHT_REVISION}',
            }
        self.encoder = timm.create_model(encoder_name, pretrained=load_remote, **create_options)
        if not hasattr(self.encoder, 'forward_intermediates'):
            raise RuntimeError('DINO adapter needs timm with forward_intermediates (source checked in 1.0.15 and 1.0.30)')
        required_arguments = {'indices', 'norm', 'output_fmt', 'intermediates_only'}
        actual_arguments = set(inspect.signature(self.encoder.forward_intermediates).parameters)
        if not required_arguments.issubset(actual_arguments):
            raise RuntimeError(f'Unsupported timm {timm.__version__} intermediate-feature API')
        self.patch_size = tuple(self.encoder.patch_embed.patch_size)
        if self.patch_size != (14, 14):
            raise ValueError(f'This adapter is configured for DINOv2 patch size 14, got {self.patch_size}')
        if self.indices[-1] >= len(self.encoder.blocks):
            raise ValueError('Requested intermediate index exceeds encoder depth')

        if pretrained and encoder_weights is not None and not weights_disabled:
            path = Path(encoder_weights)
            if path.suffix == '.safetensors':
                from safetensors.torch import load_file
                state = load_file(str(path), device='cpu')
            else:
                state = torch.load(path, map_location='cpu', weights_only=True)
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
            # Local files must use timm encoder names, not raw Meta or whole
            # segmenter checkpoint names. Fail rather than silently skip keys.
            self.encoder.load_state_dict(state, strict=True)

        width, local = decoder_channels, stem_channels
        self.stem_full = nn.Sequential(ConvNormAct(3, local), ResidualRefine(local))
        self.stem_half = nn.Sequential(ConvNormAct(local, local * 2, stride=2), ResidualRefine(local * 2))
        self.stem_quarter = nn.Sequential(ConvNormAct(local * 2, local * 4, stride=2), ResidualRefine(local * 4))
        channels = self.encoder.num_features
        self.projections = nn.ModuleList([ConvNormAct(channels, width, kernel_size=1) for _ in range(4)])
        self.pyramid_refine = nn.ModuleList([ResidualRefine(width) for _ in range(4)])
        self.fuse_quarter = nn.Sequential(ConvNormAct(width + local * 4, width), ResidualRefine(width))
        self.fuse_half = nn.Sequential(ConvNormAct(width + local * 2, local * 2), ResidualRefine(local * 2))
        self.fuse_full = nn.Sequential(ConvNormAct(local * 3, local), ResidualRefine(local))
        self.head = nn.Conv2d(local, classes, kernel_size=1)
        self.set_encoder_trainable(not freeze_encoder)

    def set_encoder_trainable(self, trainable=True):
        self._encoder_frozen = not bool(trainable)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(bool(trainable))
        self.encoder.train(self.training and bool(trainable))

    def train(self, mode=True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def _encoder_features(self, x):
        # This public API returns four NCHW tensors and removes prefix tokens.
        return tuple(self.encoder.forward_intermediates(
            x, indices=list(self.indices), norm=True, output_fmt='NCHW',
            intermediates_only=True,
        ))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point():
            raise ValueError('Expected an externally normalized floating-point [B,3,H,W] tensor')
        height, width = x.shape[-2:]
        if height < 1 or width < 1:
            raise ValueError('Input height and width must be positive')
        pad_h, pad_w = (-height) % 14, (-width) % 14
        if pad_h or pad_w:
            mode = 'reflect' if pad_h < height and pad_w < width else 'replicate'
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

        if self._encoder_frozen:
            with torch.no_grad():
                features = self._encoder_features(x)
        elif self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            features = checkpoint(self._encoder_features, x, use_reentrant=False)
        else:
            features = self._encoder_features(x)
        if len(features) != 4:
            raise RuntimeError('Expected four intermediate DINOv2 feature tensors')

        full = self.stem_full(x)
        half = self.stem_half(full)
        quarter = self.stem_quarter(half)
        qh, qw = quarter.shape[-2:]
        shapes = [(max(1, (qh + factor - 1) // factor), max(1, (qw + factor - 1) // factor))
                  for factor in (1, 2, 4, 8)]
        pyramid = [_resize(project(feature), shape)
                   for feature, project, shape in zip(features, self.projections, shapes)]
        merged = self.pyramid_refine[3](pyramid[3])
        for index in (2, 1, 0):
            merged = self.pyramid_refine[index](_resize(merged, shapes[index]) + pyramid[index])
        merged = self.fuse_quarter(torch.cat([merged, quarter], dim=1))
        merged = self.fuse_half(torch.cat([_resize(merged, half.shape[-2:]), half], dim=1))
        merged = self.fuse_full(torch.cat([_resize(merged, full.shape[-2:]), full], dim=1))
        return self.head(merged)[..., :height, :width]

    def parameter_groups(self, learning_rate, encoder_lr_multiplier=.1, weight_decay=.01):
        """Optional optimizer groups; schedulers must preserve each lr_scale.

        This method is not used automatically. A scheduler that replaces every
        group lr with the same value would erase the intended encoder/head ratio.
        """
        if learning_rate <= 0 or not 0 < encoder_lr_multiplier <= 1:
            raise ValueError('learning_rate > 0 and 0 < encoder_lr_multiplier <= 1 required')
        encoder_parameters = [p for p in self.encoder.parameters() if p.requires_grad]
        other_parameters = [p for name, p in self.named_parameters()
                            if not name.startswith('encoder.') and p.requires_grad]
        groups = []
        if encoder_parameters:
            groups.append({'params': encoder_parameters, 'lr': learning_rate * encoder_lr_multiplier,
                           'lr_scale': encoder_lr_multiplier, 'weight_decay': weight_decay, 'name': 'encoder'})
        groups.append({'params': other_parameters, 'lr': learning_rate,
                       'lr_scale': 1., 'weight_decay': weight_decay, 'name': 'decoder_and_rgb_stem'})
        return groups
