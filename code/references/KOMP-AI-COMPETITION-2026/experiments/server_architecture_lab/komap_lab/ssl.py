"""Train-only multistage SSL. Transfers encoder only, never gates/projection."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as TF

from .data import image_paths, raw_gray, patch_origins
from .models import GrayEncoder


class MultiStageSSL(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.encoder = GrayEncoder(config['model']['backbone'], pretrained, **config['normalization'])
        widths = self.encoder.channels[1:]
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in widths])
        s = config['ssl']
        self.projection = nn.Sequential(nn.Linear(sum(widths), s['projection_width'], bias=False),
                                        nn.BatchNorm1d(s['projection_width']), nn.ReLU(),
                                        nn.Linear(s['projection_width'], s['embedding_width']))

    def forward(self, x):
        features = self.encoder(x)[1:]
        return self.projection(torch.cat([f.mean((-2, -1)) * g.sigmoid() for f, g in zip(features, self.gates)], 1))


def nt_xent(z1, z2, temperature=.07):
    if z1.shape != z2.shape or z1.ndim != 2 or len(z1) < 2 or temperature <= 0:
        raise ValueError('Contrastive loss needs matching batches >=2 and positive temperature')
    n = len(z1)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    similarity = z @ z.T / temperature
    similarity = similarity.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float('-inf'))
    return F.cross_entropy(similarity, (torch.arange(2 * n, device=z.device) + n) % (2 * n))


class TrainViews:
    def __init__(self, root, patch):
        self.paths = image_paths(Path(root), 'train')
        self.images = [raw_gray(p) for p in self.paths]
        self.origins = [patch_origins(*image.shape, patch, patch // 2) for image in self.images]
        self.patch = patch
        self.crop = RandomResizedCrop(patch, scale=(.5, 1.), ratio=(.9, 1.1),
                                     interpolation=InterpolationMode.BILINEAR, antialias=True)
        self.jitter = ColorJitter(brightness=.1, contrast=.1)

    def view(self, anchor):
        image = self.crop(anchor)
        image = TF.rotate(image, 90 * int(torch.randint(4, ())))
        if bool(torch.randint(2, ())):
            image = TF.hflip(image)
        return TF.to_tensor(self.jitter(image))

    def batch(self, size):
        if not 2 <= size <= len(self.images):
            raise ValueError(f'SSL batch must contain 2..{len(self.images)} distinct Train sources')
        sources = torch.randperm(len(self.images))[:size]
        first, second = [], []
        for source in sources.tolist():
            tiles = self.origins[source]
            y, x = tiles[int(torch.randint(len(tiles), ()))]
            anchor = TF.to_pil_image(self.images[source][y:y+self.patch, x:x+self.patch])
            first.append(self.view(anchor))
            second.append(self.view(anchor))
        return torch.stack(first), torch.stack(second)
