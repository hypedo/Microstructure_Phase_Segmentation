import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ['Primary Si', 'Al3Ni', 'Eutectic Si', 'Al']
PALETTE = np.array([[76, 178, 76], [25, 76, 153], [204, 204, 204], [153, 127, 76]], dtype=np.uint8)


def decode_mask(rgb):
    rgb = np.asarray(rgb)[..., :3]
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for index, color in enumerate(PALETTE):
        labels[np.all(rgb == color, axis=-1)] = index
    if np.any(labels == 255):
        unknown = np.unique(rgb[labels == 255], axis=0)
        raise ValueError(f'Unknown mask colors: {unknown[:10].tolist()}; use original RGB masks, not JPEG')
    return labels


def raw_gray(path):
    with Image.open(path) as image:
        return np.array(image.convert('L'), dtype=np.uint8)


def patch_origins(height, width, patch, stride):
    if patch <= 0 or not 0 < stride <= patch:
        raise ValueError('Require patch > 0 and 0 < stride <= patch')
    if min(height, width) < patch:
        raise ValueError(f'Image {height}x{width} smaller than patch {patch}; no implicit resize/padding')
    axes = []
    for length in (height, width):
        positions = list(range(0, length - patch + 1, stride))
        if positions[-1] != length - patch:
            positions.append(length - patch)
        axes.append(positions)
    return [(y, x) for y in axes[0] for x in axes[1]]


def extract_context(padded_image, y, x, context=448):
    """Input is padded by context//4 on each edge; (y,x) names its center tile."""
    return padded_image[y:y + context, x:x + context]


def image_paths(root, split):
    paths = sorted((Path(root) / split / 'images').glob('*.png'))
    if not paths:
        raise ValueError(f'No PNG files: {Path(root) / split / "images"}')
    return paths


def read_pair(root, split, path):
    image = raw_gray(path)
    mask_path = Path(root) / split / 'masks' / path.name.replace('_image', '_mask')
    with Image.open(mask_path) as mask:
        labels = decode_mask(np.asarray(mask.convert('RGB')))
    if image.shape != labels.shape:
        raise ValueError(f'Image/mask size mismatch: {path}')
    return image, labels


def manifest(root, include_valid=True, patch=224):
    """Content identity without absolute paths. SSL mode never reads Valid/Test."""
    root = Path(root)
    result, seen = {}, {}
    for split in ('train', 'valid') if include_valid else ('train',):
        result[split] = []
        for path in image_paths(root, split):
            image = raw_gray(path)
            if min(image.shape) < patch:
                raise ValueError(f'Image smaller than patch: {path}')
            pixel_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if pixel_hash in seen and seen[pixel_hash][0] != split:
                raise ValueError(f'Identical Train/Valid image: {seen[pixel_hash][1]} and {path.name}')
            seen[pixel_hash] = (split, path.name)
            entry = {'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'shape': list(image.shape)}
            if include_valid:
                read_pair(root, split, path)  # validates palette, pair existence, shape
                mask = root / split / 'masks' / path.name.replace('_image', '_mask')
                entry['mask_sha256'] = hashlib.sha256(mask.read_bytes()).hexdigest()
            result[split].append(entry)
    return result


class NativePatchDataset(Dataset):
    """Context448 input, center224 target, rare classes 1/2 balanced by source."""
    def __init__(self, root, context=448, target=224, stride=112, rare_fraction=.10,
                 enriched_probability=.5, rare_classes=(1, 2), samples_per_epoch=420, photometric=True):
        if context != 2 * target or not 0 < stride <= target:
            raise ValueError('Require Context448=2*target224 and stride<=target')
        self.paths = image_paths(root, 'train')
        self.context, self.target = context, target
        self.margin, self.samples_per_epoch = (context - target) // 2, samples_per_epoch
        self.images, self.padded, self.labels, self.origins = [], [], [], []
        self.enriched = {int(cls): {} for cls in rare_classes}
        self.enriched_probability, self.photometric = enriched_probability, photometric
        for index, path in enumerate(self.paths):
            image, labels = read_pair(root, 'train', path)
            origins = patch_origins(*image.shape, target, stride)
            self.images.append(image)
            self.padded.append(np.pad(image, self.margin, mode='reflect'))
            self.labels.append(labels)
            self.origins.append(origins)
            for cls in rare_classes:
                integral = np.pad((labels == cls).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                positives = [(y, x) for y, x in origins
                             if integral[y+target, x+target] - integral[y, x+target] - integral[y+target, x]
                             + integral[y, x] >= rare_fraction * target * target]
                if positives:
                    self.enriched[int(cls)][index] = positives
        if enriched_probability > 0:
            missing = [cls for cls, sources in self.enriched.items() if not sources]
            if missing:
                raise ValueError(f'No rare-enriched tile for class(es) {missing}')

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if torch.rand(()) < self.enriched_probability:
            # R022: first choose Al3Ni or Eutectic equally, then source uniformly.
            cls = self.rare_classes[int(torch.randint(len(self.rare_classes), ()))]
            sources = tuple(self.enriched[cls])
            source = sources[int(torch.randint(len(sources), ()))]
            tiles = self.enriched[cls][source]
        else:
            source = int(torch.randint(len(self.paths), ()))
            tiles = self.origins[source]
        y, x = tiles[int(torch.randint(len(tiles), ()))]
        image = torch.from_numpy(self.padded[source][y:y+self.context, x:x+self.context].copy()).float()[None] / 255
        target = torch.from_numpy(self.labels[source][y:y+self.target, x:x+self.target].astype(np.int64, copy=True))
        image = self._photometric(image)
        k = int(torch.randint(4, ()))
        image, target = image.rot90(k, (-2, -1)), target.rot90(k, (-2, -1))
        if bool(torch.randint(2, ())):
            image, target = image.flip(-1), target.flip(-1)
        return image, target, source

    @property
    def rare_classes(self):
        return tuple(self.enriched)

    def _photometric(self, image):
        if not self.photometric:
            return image
        # R022: contrast+brightness p=.5, gamma p=.3, Gaussian noise p=.15.
        if torch.rand(()) < .5:
            contrast = torch.empty(()).uniform_(.9, 1.1)
            brightness = torch.empty(()).uniform_(-.08, .08)
            mean = image.mean((-2, -1), keepdim=True)
            image = (image - mean) * contrast + mean + brightness
        if torch.rand(()) < .3:
            image = image.clamp(0, 1).pow(torch.empty(()).uniform_(.9, 1.1))
        if torch.rand(()) < .15:
            image = image + torch.randn_like(image) * .01
        return image.clamp(0, 1)


class FullResolutionDataset:
    def __init__(self, root, split):
        if split not in ('valid', 'test'):
            raise ValueError('Evaluation split must be valid or test')
        self.paths = image_paths(root, split)
        self.images, self.labels = [], []
        for path in self.paths:
            if split == 'valid':
                image, label = read_pair(root, split, path)
                self.labels.append(label)
            else:
                image = raw_gray(path)
            self.images.append(image)
