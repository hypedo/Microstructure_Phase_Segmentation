"""Bounded residual correction of frozen out-of-fold phase probabilities.

This module never runs or trains the base model. Gray input is [B,1,H,W] in
[0,1], and base probabilities are [B,4,H,W], in metrics.py class order.
Outputs are four raw logits. No filename metadata or component attributes
are used. Zero output initialization preserves the base argmax.
"""
import torch
from torch import nn
from torch.nn import functional as F


class OOFPhaseRefiner(nn.Module):
    """Probability calibration or grayscale-plus-probability spatial context.

    logits = log(normalize(clamp(P))) + 2*tanh(delta).
    Clipping makes finite logs while preserving class order, including ties.
    Frozen teacher tensors are detached at the interface by design.
    """
    def __init__(self, mode='spatial', width=32, epsilon=1e-7):
        super().__init__()
        if mode not in ('probability', 'spatial', 'spatial_multiscale'):
            raise ValueError('mode must be probability, spatial, or spatial_multiscale')
        if width != 32:
            raise ValueError('The fixed comparison uses width=32')
        if not 0 < epsilon < .25:
            raise ValueError('epsilon must be between 0 and .25')
        self.mode, self.width, self.epsilon = mode, width, float(epsilon)
        if mode == 'probability':
            self.features = nn.Identity()
            self.output = nn.Conv2d(4, 4, 1)
        else:
            layers = []
            for channels, dilation in zip((8, width, width), (1, 2, 4)):
                layers.extend((nn.Conv2d(channels, width, 3, padding=dilation,
                                        dilation=dilation, bias=False),
                               nn.GroupNorm(8, width), nn.SiLU(inplace=False)))
            self.features = nn.Sequential(*layers)
            self.output = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('sobel', torch.tensor(
            [[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
             [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]
        )[:, None] / 8.)

        if mode == 'spatial_multiscale':
            # Construct the common 8-channel trunk first using the exact same
            # RNG draws as spatial, then extend it without changing shared
            # weights. The six new inputs start at zero influence, but their
            # weights become trainable once the output projection learns.
            common = self.features[0]
            expanded = nn.Conv2d(14, width, 3, padding=1, bias=False)
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :8].copy_(common.weight)
            self.features[0] = expanded

    def forward(self, gray, probabilities, extra_features=None):
        if probabilities.ndim != 4 or probabilities.shape[1] != 4:
            raise ValueError('Expected probabilities [B,4,H,W]')
        if gray.shape != (probabilities.shape[0], 1, *probabilities.shape[-2:]):
            raise ValueError('Expected matching gray [B,1,H,W]')
        if min(gray.shape) < 1 or not gray.is_floating_point() or not probabilities.is_floating_point():
            raise ValueError('Inputs must be nonempty floating tensors')
        # Keep log probabilities in FP32 under autocast and stop teacher grads.
        p = probabilities.detach().float().clamp_min(self.epsilon)
        p = p / p.sum(1, keepdim=True)
        logp = p.log()
        if self.mode == 'probability':
            features = logp
        else:
            gray = gray.detach().float()
            pad = F.pad(gray, (1, 1, 1, 1), mode='replicate')
            highpass = gray - F.avg_pool2d(pad, 3, stride=1)
            gradient = F.conv2d(pad, self.sobel.float())
            features = torch.cat((gray, p, highpass, gradient), dim=1)
        if self.mode == 'spatial_multiscale':
            if extra_features is None or extra_features.shape != (gray.shape[0], 6, *gray.shape[-2:]):
                raise ValueError('spatial_multiscale requires 6 full-image input-only maps cropped jointly; never compute them on a crop')
            features = torch.cat((features, extra_features.detach().float()), dim=1)
        elif extra_features is not None:
            raise ValueError('Extra feature maps belong only to spatial_multiscale')
        delta = self.output(self.features(features)).float()
        return logp + 2. * delta.tanh()


MULTISCALE_NAMES = ['highpass_sigma2','highpass_sigma8','highpass_sigma16',
                    'localstd_sigma2','localstd_sigma8','localstd_sigma16']


def multiscale_input_features(gray):
    """Six scalar maps from the complete native grayscale image, never GT.

    gray is numpy[H,W] in[0,1]. At sigma2/8/16 pixels, return gray-G(gray)
    and sqrt(max(G(gray**2)-G(gray)**2,0)). Gaussian support is explicitly
    +/-ceil(4*sigma), BORDER_REFLECT_101, with float64 moment calculation to
    avoid spurious variance from float32 cancellation. Outputs are float32.
    Compute once BEFORE crop/D4/reflection padding, and transform all maps
    together. These scalar filters commute with D4 up to numeric tolerance.

    Established multiscale image-feature motivation (not an exact ilastik
    feature implementation or a novelty claim):
    https://www.ilastik.org/documentation/pixelclassification/pixelclassification
    Gaussian implementation API:
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
    """
    import cv2
    import numpy as np
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2 or min(gray.shape) < 1 or not np.isfinite(gray).all() or gray.min() < 0 or gray.max() > 1:
        raise ValueError('Expected finite full native grayscale[H,W] in[0,1]')
    highpass, localstd = [], []
    for sigma in (2., 8., 16.):
        kernel = 2 * int(np.ceil(4*sigma)) + 1
        smooth = cv2.GaussianBlur(gray, (kernel,kernel), sigmaX=sigma,
                                  sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        second = cv2.GaussianBlur(gray*gray, (kernel,kernel), sigmaX=sigma,
                                  sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        highpass.append(gray-smooth)
        localstd.append(np.sqrt(np.maximum(second-smooth*smooth, 0.)))
    return np.stack(highpass+localstd).astype(np.float32)


@torch.inference_mode()
def refined_probabilities(model, gray, probabilities, device, tile=512, overlap=128, extra_features=None):
    """Native-resolution tiled inference, B1/noTTA, on already computed P.

    gray: numpy [H,W] float32 in [0,1]. P: numpy [4,H,W].
    The fixed 512/128 inference geometry is shared by both learned modes.
    Refiner evaluation uses FP32 without autocast. Base cache numeric
    provenance remains separate and unchanged.
    Validation targets are neither accepted nor accessed. No silent fallback.
    """
    import numpy as np
    if tile != 512 or overlap != 128:
        raise ValueError('The fixed comparison uses tile512 and overlap128')
    gray = np.asarray(gray, dtype=np.float32)
    p = np.asarray(probabilities, dtype=np.float32)
    if gray.ndim != 2 or p.shape != (4, *gray.shape) or min(gray.shape) < 1:
        raise ValueError('Mismatched gray/probability native shapes')
    if not np.isfinite(gray).all() or not np.isfinite(p).all() or gray.min() < 0 or gray.max() > 1 or p.min() < 0:
        raise ValueError('Invalid input values')
    if not np.allclose(p.sum(0), 1, rtol=0, atol=.002):
        raise ValueError('Input probability sums are not near one')
    if model.mode == 'spatial_multiscale':
        if extra_features is None:
            extra_features = multiscale_input_features(gray)
        extra_features = np.asarray(extra_features, dtype=np.float32)
        if extra_features.shape != (6,*gray.shape) or not np.isfinite(extra_features).all():
            raise ValueError('Invalid full-image multiscale features')
    elif extra_features is not None:
        raise ValueError('Extra features supplied for a basic mode')
    height, width = gray.shape
    ph, pw = max(tile, height), max(tile, width)
    # Match native patch training's reflected bottom/right padding. Numpy's
    # reflect also supports tiny dimensions without fabricating a new label.
    gray = np.pad(gray, ((0, ph-height), (0, pw-width)), mode='reflect')
    p = np.pad(p, ((0, 0), (0, ph-height), (0, pw-width)), mode='reflect')
    if extra_features is not None:
        extra_features = np.pad(extra_features, ((0,0),(0,ph-height),(0,pw-width)), mode='reflect')
    def starts(length):
        return sorted(set(range(0, max(1, length-tile+1), tile-overlap)) | {length-tile})
    win = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), .02).astype(np.float32)
    sums = np.zeros((4, ph, pw), np.float32)
    count = np.zeros((ph, pw), np.float32)
    model.eval()
    device = torch.device(device)
    for y in starts(ph):
        for x in starts(pw):
            g = torch.from_numpy(np.ascontiguousarray(gray[y:y+tile, x:x+tile]))[None, None].to(device)
            q = torch.from_numpy(np.ascontiguousarray(p[:, y:y+tile, x:x+tile]))[None].to(device)
            extras = None
            if extra_features is not None:
                extras = torch.from_numpy(np.ascontiguousarray(extra_features[:,y:y+tile,x:x+tile]))[None].to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=False):
                pred = model(g, q, extras).softmax(1)[0].float().cpu().numpy()
            sums[:, y:y+tile, x:x+tile] += pred * win
            count[y:y+tile, x:x+tile] += win
    return (sums/count[None])[:, :height, :width]


def self_test():
    """Small CPU-only checks; no models, data, downloads or GPU are needed."""
    torch.set_num_threads(2)
    torch.manual_seed(314)
    results = {}
    for mode in ('probability', 'spatial'):
        model = OOFPhaseRefiner(mode)
        p = torch.rand(2, 4, 23, 31)
        p = (p/p.sum(1, keepdim=True)).requires_grad_()
        gray = torch.rand(2, 1, 23, 31, requires_grad=True)
        out = model(gray, p)
        assert out.shape == p.shape and torch.isfinite(out).all()
        assert torch.equal(out.argmax(1), p.argmax(1))
        assert torch.allclose(out.softmax(1), p, atol=1e-6, rtol=0)
        target = torch.randint(0, 4, (2, 23, 31))
        loss = F.cross_entropy(out, target)
        loss.backward()
        assert torch.isfinite(model.output.weight.grad).all()
        assert model.output.weight.grad.abs().sum() > 0
        assert gray.grad is None and p.grad is None
        # Zero last layer correctly gives no feature gradient on step one;
        # after one update the spatial trunk must receive a usable gradient.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step(); optimizer.zero_grad()
        F.cross_entropy(model(gray, p), target).backward()
        if mode == 'spatial':
            assert model.features[0].weight.grad.abs().sum() > 0
        clone = OOFPhaseRefiner(mode)
        clone.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(clone(gray, p), model(gray, p))
        assert (model(gray, p)-(p.detach().clamp_min(model.epsilon)/p.detach().clamp_min(model.epsilon).sum(1, keepdim=True)).log()).abs().max() <= 2
        # Exact zeros and tied maxima preserve the original argmax too.
        ties = torch.tensor([.5, .5, 0., 0.]).view(1,4,1,1)
        fresh = OOFPhaseRefiner(mode)
        assert torch.equal(fresh(torch.zeros(1,1,1,1), ties).argmax(1), ties.argmax(1))
        results[mode] = {'shape':list(out.shape), 'identity_argmax':True,
                         'frozen_teacher':True, 'gradient':True,
                         'strict_state_roundtrip':True,
                         'parameters':sum(v.numel() for v in model.parameters())}
    return results


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), indent=2))
