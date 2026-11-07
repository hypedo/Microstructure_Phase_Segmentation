"""Full-image phase supervision and prediction with explicit padding/cropping.

``forward_padded`` keeps autograd and returns logits at the original size, so
training can use the original mask without fabricating padded target labels.
``full_image_probabilities`` handles one uint8 RGB image, normalization and
optional D4 inference. It deliberately has no tiled/OOM fallback: changing the
context after an allocation failure would silently change the experiment.
"""

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def forward_padded(model, normalized_tensor, multiple=32):
    """Differentiably pad normalized RGB, run the model, and crop raw logits.

    The input is [B, 3, H, W]. Only right/bottom padding is added, using
    reflection where valid and replication for tiny dimensions. The model
    must return four phase logits, optionally followed by a boundary logit,
    at the same padded spatial resolution. Model train/eval state is unchanged.
    """
    if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple < 1:
        raise ValueError('multiple must be a positive integer')
    if normalized_tensor.ndim != 4 or normalized_tensor.shape[1] != 3:
        raise ValueError('Expected normalized RGB tensor [B, 3, H, W]')
    if min(normalized_tensor.shape) < 1 or not normalized_tensor.is_floating_point():
        raise ValueError('Expected positive dimensions and floating-point RGB')
    height, width = normalized_tensor.shape[-2:]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    x = normalized_tensor
    if pad_h or pad_w:
        mode = 'reflect' if pad_h < height and pad_w < width else 'replicate'
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    logits = model(x)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError('Model must return a [B, 4 or 5, H, W] logits tensor')
    if (logits.shape[0] != x.shape[0] or logits.shape[1] not in (4, 5)
            or logits.shape[-2:] != x.shape[-2:]):
        raise ValueError('Model logits must match padded input size and have 4 or 5 channels')
    return logits[..., :height, :width]


@torch.inference_mode()
def full_image_probabilities(model, imageRGBnumpy, device, tta=False, canonical_width=0):
    """Return float32 [4, H, W] probabilities at the original image resolution.

    Input is one uint8 RGB array. Canonical resizing and ImageNet normalization
    match train.py. CUDA uses float16 autocast; softmax and averaging use float32.
    The caller should select model.eval(), as with the existing tiled helper.
    Each of the eight D4 views is run separately with batch size one. On a
    rectangle a 90-degree rotation swaps H/W; crop before applying its inverse.
    Probability accumulation is on CPU to avoid retaining eight GPU outputs.
    Exceptions, including insufficient GPU memory, propagate to the caller.
    """
    if (not isinstance(imageRGBnumpy, np.ndarray) or imageRGBnumpy.ndim != 3
            or imageRGBnumpy.shape[2] != 3 or min(imageRGBnumpy.shape[:2]) < 1):
        raise ValueError('Expected a nonempty H x W x 3 RGB numpy array')
    if imageRGBnumpy.dtype != np.uint8:
        raise ValueError('RGB image must be uint8 in the 0..255 range')
    if (not isinstance(canonical_width, int) or isinstance(canonical_width, bool)
            or canonical_width < 0):
        raise ValueError('canonical_width must be a nonnegative integer')
    device = torch.device(device)
    image = imageRGBnumpy
    original_height, original_width = image.shape[:2]
    if canonical_width and original_width != canonical_width:
        size = (canonical_width, round(original_height * canonical_width / original_width))
        if size[1] < 1:
            raise ValueError('canonical_width would produce an empty image height')
        image = cv2.resize(
            image, size,
            interpolation=cv2.INTER_AREA if size[0] < original_width else cv2.INTER_LINEAR,
        )
    height, width = image.shape[:2]
    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
    tensor = tensor.to(device=device, dtype=torch.float32) / 255
    # Identical constants and arithmetic to train.normalize.
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    tensor = (tensor - mean) / std
    transforms = ([(k, flip) for flip in (False, True) for k in range(4)]
                  if tta else [(0, False)])
    probabilities = np.zeros((4, height, width), dtype=np.float32)
    for rotation, flip in transforms:
        transformed = torch.rot90(tensor, rotation, (-2, -1))
        if flip:
            transformed = transformed.flip(-1)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = forward_padded(model, transformed)
        probability = logits[:, :4].float().softmax(dim=1)
        if flip:
            probability = probability.flip(-1)
        probability = torch.rot90(probability, -rotation, (-2, -1))
        probabilities += probability[0].cpu().numpy()
        del transformed, logits, probability
    probabilities /= len(transforms)
    if (height, width) != (original_height, original_width):
        probabilities = np.stack([
            cv2.resize(channel, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            for channel in probabilities
        ])
    return probabilities
