"""Proposed combinations, NOT exact reproductions of the referenced papers."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def resize(x, reference):
    size = reference.shape[-2:] if isinstance(reference, torch.Tensor) else reference
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class SCSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(1, channels // 16), 1),
                                     nn.ReLU(), nn.Conv2d(max(1, channels // 16), channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel(x) + x * self.spatial(x)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, scse=True):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                                   nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
        self.attention = SCSE(cout) if scse else nn.Identity()

    def forward(self, x):
        return self.attention(self.block(x))


class SharedWeightRRCU(nn.Module):
    """Two recurrent steps with shared Conv-BN weights and an outer residual."""
    def __init__(self, cin, cout, iterations=2):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
        self.recurrent_conv = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.recurrent_norm = nn.BatchNorm2d(cout)
        self.iterations = iterations
        self.activation = nn.ReLU()
        self.attention = SCSE(cout)

    def forward(self, x):
        residual = self.project(x)
        state = torch.zeros_like(residual)
        for _ in range(self.iterations):
            state = self.activation(self.recurrent_norm(self.recurrent_conv(residual + state)))
        return self.attention(residual + state)


def sample_pixels(x, coords):
    # Explicit border-padded bilinear gathering, also supports MPS backward.
    n, c, h, w = x.shape
    oh, ow = coords.shape[1:3]
    xx, yy = coords[..., 0].clamp(0, w - 1), coords[..., 1].clamp(0, h - 1)
    x0, y0 = xx.floor().long(), yy.floor().long()
    x1, y1 = (x0 + 1).clamp(max=w - 1), (y0 + 1).clamp(max=h - 1)
    wx, wy = (xx - x0)[:, None], (yy - y0)[:, None]
    def at(y, z):
        indices = (y * w + z).reshape(n, 1, -1).expand(-1, c, -1)
        return x.flatten(2).gather(2, indices).reshape(n, c, oh, ow)
    return ((1 - wx) * (1 - wy) * at(y0, x0) + wx * (1 - wy) * at(y0, x1)
            + (1 - wx) * wy * at(y1, x0) + wx * wy * at(y1, x1))


class DySampleLP(nn.Module):
    """Final 2x LP upsampling, 4 groups; based on DySample's sampling rule."""
    def __init__(self, channels=64, groups=4):
        super().__init__()
        if channels % groups:
            raise ValueError('DySample channels must be divisible by groups')
        self.groups = groups
        self.offset = nn.Conv2d(channels, 8 * groups, 1)
        nn.init.normal_(self.offset.weight, std=.001)
        nn.init.zeros_(self.offset.bias)
        yy, xx = torch.meshgrid(torch.tensor([-.25, .25]), torch.tensor([-.25, .25]), indexing='ij')
        self.register_buffer('positions', torch.stack([xx, yy]).reshape(2, 1, 4, 1, 1)
                             .repeat(1, groups, 1, 1, 1).reshape(1, 8 * groups, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        offsets = (.25 * self.offset(x) + self.positions).reshape(b, 2, 4 * self.groups, h, w)
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                               torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
        points = offsets + torch.stack([xx, yy])[None, :, None]
        coords = F.pixel_shuffle(points.reshape(b, -1, h, w), 2).reshape(b, 2, self.groups, 2 * h, 2 * w)
        coords = coords.permute(0, 2, 3, 4, 1).reshape(b * self.groups, 2 * h, 2 * w, 2)
        return sample_pixels(x.reshape(b * self.groups, c // self.groups, h, w), coords).reshape(b, c, 2 * h, 2 * w)


class GrayEncoder(nn.Module):
    def __init__(self, backbone='resnet101', pretrained=False,
                 mean=(.485, .456, .406), std=(.229, .224, .225)):
        super().__init__()
        weights = {'resnet34': models.ResNet34_Weights, 'resnet101': models.ResNet101_Weights}
        if backbone not in weights:
            raise ValueError('Supported backbones: resnet101, resnet34 (debug/control)')
        base = getattr(models, backbone)(weights=weights[backbone].DEFAULT if pretrained else None)
        # R022 repeats grayscale into RGB and keeps the original V2 weights.
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.channels = [64, 64, 128, 256, 512] if backbone == 'resnet34' else [64, 256, 512, 1024, 2048]
        self.register_buffer('input_mean', torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[1] != 3:
            raise ValueError('Expected grayscale (1 channel) or repeated RGB input')
        stem = self.relu(self.bn1(self.conv1((x - self.input_mean) / self.input_std)))
        e1 = self.layer1(self.maxpool(stem))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return [stem, e1, e2, e3, e4]


class HighFrequency(nn.Module):
    """Our skip residual: 3x3 mean lowpass, learned depthwise/pointwise projection."""
    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                     nn.Conv2d(channels, channels, 1, bias=False))
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        low = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        return x + self.gain * self.project(x - low)


class StageCrossAttention(nn.Module):
    def __init__(self, shallow, deep, mode='attention', width=128, heads=4):
        super().__init__()
        self.mode = mode
        self.query, self.key_value = nn.Conv2d(shallow, width, 1), nn.Conv2d(deep, width, 1)
        self.norm_q, self.norm_kv = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True) if mode == 'attention' else None
        self.concat = nn.Conv2d(width * 2, width, 1) if mode == 'concat' else None
        self.output = nn.Conv2d(width, shallow, 1)
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, shallow, deep):
        q, kv = self.query(shallow), self.key_value(deep)
        if self.mode == 'attention':
            qt = self.norm_q(q.flatten(2).transpose(1, 2))
            kt = self.norm_kv(kv.flatten(2).transpose(1, 2))
            # Q and K/V retain their native stage3/stage4 sizes.
            fused = self.attention(qt, kt, kt, need_weights=False)[0]
            fused = fused.transpose(1, 2).reshape_as(q)
        else:
            fused = self.concat(torch.cat([q, resize(kv, q)], 1))
        return shallow + self.gain * self.output(fused)


class MultiDilation(nn.Module):
    def __init__(self, channels=128, adaptive=True):
        super().__init__()
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                                                              groups=channels, bias=False),
                                                    nn.Conv2d(channels, channels, 1, bias=False),
                                                    nn.BatchNorm2d(channels), nn.ReLU()) for d in (1, 2, 3)])
        self.selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 3, 1)) if adaptive else None
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        weights = self.selector(x).softmax(1) if self.selector is not None else x.new_full((len(x), 3, 1, 1), 1 / 3)
        combined = sum(branch(x) * weights[:, i:i + 1] for i, branch in enumerate(self.branches))
        return x + self.gain * combined


class WeightedSkip(nn.Module):
    def __init__(self, deep, shallow, width):
        super().__init__()
        self.deep, self.shallow = nn.Conv2d(deep, width, 1), nn.Conv2d(shallow, width, 1)
        self.weights = nn.Parameter(torch.zeros(2))

    def forward(self, deep, shallow, hf_residual=None):
        weights = self.weights.softmax(0)
        fused = weights[0] * self.deep(resize(deep, shallow)) + weights[1] * self.shallow(shallow)
        # HF bypass remains separate from the shallow/deep scalar weighting.
        if hf_residual is not None:
            fused = fused + F.conv2d(hf_residual, self.shallow.weight, bias=None)
        return fused


class UNetDecoder(nn.Module):
    def __init__(self, channels, upsampling='dysample', dilation='none', weighted=False,
                 rrcu_deepest=False, rrcu_seed=1747):
        super().__init__()
        incoming, widths = [channels[-1], 256, 128, 64], [256, 128, 64, 64]
        skips = list(reversed(channels[:-1]))
        self.blocks = nn.ModuleList([ConvBlock(cin + skip, cout) for cin, skip, cout in zip(incoming, skips, widths)])
        if rrcu_deepest:
            original_attention = self.blocks[0].attention
            # Keep every other decoder weight and the subsequent sampling RNG
            # identical to B. Seed only the CPU generator used for allocation.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(rrcu_seed)
                replacement = SharedWeightRRCU(incoming[0] + skips[0], widths[0], iterations=2)
            replacement.attention = original_attention
            self.blocks[0] = replacement
        self.weighted = weighted
        self.fusions = nn.ModuleDict()
        if weighted:
            # Replace two shallow concat junctions; match their output width.
            for i in (2, 3):
                self.fusions[str(i)] = WeightedSkip(incoming[i], skips[i], widths[i])
                self.blocks[i] = ConvBlock(widths[i], widths[i])
        if dilation != 'none':
            self.blocks[1].attention = MultiDilation(128, adaptive=dilation == 'adaptive')
        self.upsampling = upsampling
        self.upsampler = DySampleLP(64) if upsampling == 'dysample' else nn.Identity()
        self.final = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 4, 1)

    def forward(self, features, size, hf_residuals=None):
        x = features[-1]
        for i, (skip, block) in enumerate(zip(reversed(features[:-1]), self.blocks)):
            residual = (hf_residuals or {}).get(3 - i)
            merged = self.fusions[str(i)](x, skip, residual) if str(i) in self.fusions else torch.cat([resize(x, skip), skip], 1)
            x = block(merged)
        x = self.upsampler(x) if self.upsampling == 'dysample' else resize(x, size)
        if x.shape[-2:] != size:
            x = resize(x, size)
        return self.head(self.final(x))


class UPerDecoder(nn.Module):
    """PPM + FPN; DySample only at the final 1/2 -> 1 resolution step."""
    def __init__(self, channels, upsampling='dysample', width=128):
        super().__init__()
        self.ppm = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(channels[-1], width, 1, bias=False),
                                               nn.GroupNorm(8, width), nn.ReLU()) for s in (1, 2, 3, 6)])
        self.top = ConvBlock(channels[-1] + 4 * width, width, scse=False)
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels[1:-1]])
        self.smooth = nn.ModuleList([ConvBlock(width, width, scse=False) for _ in range(3)])
        self.fuse = ConvBlock(4 * width, 64, scse=False)
        self.half_stage = ConvBlock(64, 64, scse=False)
        self.upsampling = upsampling
        self.up = DySampleLP(64) if upsampling == 'dysample' else nn.Identity()
        self.final, self.head = ConvBlock(64, 32, scse=False), nn.Conv2d(32, 4, 1)

    def forward(self, features, size):
        deep = features[-1]
        top = self.top(torch.cat([deep] + [resize(branch(deep), deep) for branch in self.ppm], 1))
        pyramid = [top]
        for i in (2, 1, 0):
            top = self.smooth[i](self.lateral[i](features[i + 1]) + resize(top, features[i + 1]))
            pyramid.append(top)
        x = self.fuse(torch.cat([resize(p, features[1]) for p in pyramid], 1))
        x = self.half_stage(resize(x, features[0]))
        x = self.up(x) if self.upsampling == 'dysample' else resize(x, size)
        return self.head(self.final(resize(x, size) if x.shape[-2:] != size else x))


class NestedDecoder(nn.Module):
    """Dense nested skips on five encoder levels, one final head, no deep supervision."""
    def __init__(self, channels, upsampling='dysample'):
        super().__init__()
        widths = [64, 64, 128, 256, 512]
        self.project = nn.ModuleList([nn.Conv2d(c, w, 1) for c, w in zip(channels, widths)])
        self.nodes = nn.ModuleDict({f'{i}_{j}': ConvBlock(j * widths[i] + widths[i + 1], widths[i])
                                   for j in range(1, 5) for i in range(5 - j)})
        self.upsampling = upsampling
        self.up = DySampleLP(64) if upsampling == 'dysample' else nn.Identity()
        self.final, self.head = ConvBlock(64, 32), nn.Conv2d(32, 4, 1)

    def forward(self, features, size):
        nodes = {(i, 0): p(x) for i, (p, x) in enumerate(zip(self.project, features))}
        for j in range(1, 5):
            for i in range(5 - j):
                values = [nodes[i, k] for k in range(j)] + [resize(nodes[i + 1, j - 1], nodes[i, 0])]
                nodes[i, j] = self.nodes[f'{i}_{j}'](torch.cat(values, 1))
        x = self.up(nodes[0, 4]) if self.upsampling == 'dysample' else resize(nodes[0, 4], size)
        return self.head(self.final(resize(x, size) if x.shape[-2:] != size else x))


class ParallelSwin(nn.Module):
    def __init__(self, channels, pretrained=False, mode='gated', size=224):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError('C09 requires: pip install -r requirements-swin.txt') from exc
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained, features_only=True,
                                     out_indices=(0, 1, 2, 3), img_size=size)
        self.swin_channels = self.swin.feature_info.channels()
        self.projections = nn.ModuleList([nn.Conv2d(cin, cout, 1) for cin, cout in zip(self.swin_channels, channels[1:])])
        self.mode = mode
        self.fusions = nn.ModuleList([nn.Conv2d(2 * c, c, 1) for c in channels[1:]])
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def forward(self, image, features):
        transformer = self.swin((image.repeat(1, 3, 1, 1) - self.rgb_mean) / self.rgb_std)
        result = [features[0]]
        for i, (cnn, swin) in enumerate(zip(features[1:], transformer)):
            # timm Swin returns NHWC; inspect output_fmt rather than guessing a dimension.
            if str(getattr(self.swin, 'output_fmt', 'NHWC')).upper().endswith('NHWC'):
                swin = swin.permute(0, 3, 1, 2).contiguous()
            if swin.shape[1] != self.swin_channels[i]:
                raise RuntimeError(f'Unexpected timm Swin feature layout: {tuple(swin.shape)}')
            projected = resize(self.projections[i](swin), cnn)
            fused = self.fusions[i](torch.cat([cnn, projected], 1))
            result.append(fused.sigmoid() * cnn + (1 - fused.sigmoid()) * projected if self.mode == 'gated' else fused)
        return result


class FourDirectionMamba(nn.Module):
    """CUDA Mamba1 over row/column forward/reverse sequences. Derived VSS-like block.

    Uses genuine mamba-ssm kernels, NOT the original VMamba SS2D implementation.
    One shared Mamba per block across four directions; internal width 128.
    """
    def __init__(self, channels, width=128):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            raise RuntimeError('C10 requires a compatible Linux CUDA mamba-ssm installation; see README') from exc
        self.reduce, self.expand = nn.Conv2d(channels, width, 1), nn.Conv2d(width, channels, 1)
        self.norm = nn.LayerNorm(width)
        self.mamba = Mamba(d_model=width, d_state=16, d_conv=4, expand=2)
        self.gain = nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        if x.device.type != 'cuda':
            raise RuntimeError('C10 Mamba requires CUDA; no substitute block is used on CPU/MPS')
        z = self.reduce(x)
        b, c, h, w = z.shape
        row = z.flatten(2).transpose(1, 2)
        col = z.transpose(2, 3).flatten(2).transpose(1, 2)
        sequences = [row, row.flip(1), col, col.flip(1)]
        outputs = [self.mamba(self.norm(s)) for s in sequences]
        row_out = (outputs[0] + outputs[1].flip(1)).transpose(1, 2).reshape(b, c, h, w)
        col_out = (outputs[2] + outputs[3].flip(1)).transpose(1, 2).reshape(b, c, w, h).transpose(2, 3)
        return x + self.gain * self.expand((row_out + col_out) / 4)


class SegmentationModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        architecture = config['model']
        self.target_size = config['train']['target']
        self.encoder = GrayEncoder(architecture['backbone'], pretrained, **config['normalization'])
        channels = self.encoder.channels
        decoder_kind = 'unet' if architecture['decoder'] == 'unet_rrcu' else architecture['decoder']
        decoder = {'unet': UNetDecoder, 'upernet': UPerDecoder, 'unetpp': NestedDecoder}[decoder_kind]
        options = {'upsampling': architecture['upsampling']}
        if architecture['decoder'] in ('unet', 'unet_rrcu'):
            options.update(dilation=architecture['dilation'], weighted=architecture['weighted_skip'],
                           rrcu_deepest=architecture['decoder'] == 'unet_rrcu',
                           rrcu_seed=config['train']['seed'] + 1705)
        self.decoder = decoder(channels, **options)
        self.weighted_skip = architecture['weighted_skip']
        # Allocate each add-on with an independent RNG; the 2x2 comparison shares
        # baseline, HF and cross-attention initialization wherever applicable.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['train']['seed'] + 1701)
            self.hf = nn.ModuleDict({str(i): HighFrequency(channels[i]) for i in (0, 1)}) if architecture['hf'] else nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['train']['seed'] + 1702)
            self.cross = StageCrossAttention(channels[3], channels[4], architecture['cross']) if architecture['cross'] != 'none' else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['train']['seed'] + 1703)
            self.hybrid = ParallelSwin(channels, pretrained, architecture['hybrid'], config['train']['patch']) if architecture['hybrid'] != 'none' else None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['train']['seed'] + 1704)
            self.mamba = nn.Sequential(*[FourDirectionMamba(channels[-1]) for _ in range(architecture['mamba_blocks'])])

    def forward(self, x):
        features = self.encoder(x)
        if self.hybrid is not None:
            features = self.hybrid(x, features)
        features[-1] = self.mamba(features[-1])
        if self.cross is not None:
            features[3] = self.cross(features[3], features[4])
        hf_residuals = {}
        for key, block in self.hf.items():
            index = int(key)
            enhanced = block(features[index])
            if self.weighted_skip:
                hf_residuals[index] = enhanced - features[index]
            else:
                features[index] = enhanced
        logits = (self.decoder(features, x.shape[-2:], hf_residuals) if self.weighted_skip
                  else self.decoder(features, x.shape[-2:]))
        # The R022 training API supervises the center 224 only. Evaluation gets
        # full Context448 logits so halo inference can inverse-transform then crop.
        if self.training:
            if logits.shape[-2:] == x.shape[-2:]:
                margin = (x.shape[-1] - self.target_size) // 2
                logits = logits[..., margin:margin + self.target_size, margin:margin + self.target_size]
        return logits


def load_encoder(encoder, path, kind):
    """Strict RGB ResNet encoder transfer; accepts torchvision or package SSL state."""
    from pathlib import Path
    import hashlib
    state = torch.load(path, map_location='cpu', weights_only=True)
    metadata = state.get('metadata', {}) if isinstance(state, dict) else {}
    if kind == 'ssl' and metadata.get('pretraining') != 'multi-stage-ssl':
        raise ValueError('C12 needs an encoder.pt exported by this package\'s ssl command')
    for key in ('encoder', 'state_dict', 'model'):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise ValueError('Expected a tensor state dictionary')
    cleaned = {}
    for key, value in state.items():
        for prefix in ('module.', 'encoder.', 'backbone.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key.startswith(('fc.', 'classifier.')):
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'Unexpected non-tensor checkpoint entry {key}')
        if key in cleaned:
            raise ValueError(f'Duplicate encoder key {key}')
        cleaned[key] = value
    expected = encoder.state_dict()
    if kind == 'micronet':
        # Explicit grayscale adaptation for RGB classification checkpoints.
        if (cleaned.get('conv1.weight') is not None
                and cleaned['conv1.weight'].shape != expected['conv1.weight'].shape):
            raise ValueError('MicroNet conv1 must match RGB-repeated ResNet101; channel summation is not the R022 input recipe')
        for key in ('input_mean', 'input_std'):
            cleaned.setdefault(key, expected[key])
    if set(cleaned) != set(expected):
        raise ValueError(f'Encoder key mismatch: missing={sorted(set(expected)-set(cleaned))[:12]}, '
                         f'extra={sorted(set(cleaned)-set(expected))[:12]}; no partial loading allowed')
    if any(cleaned[k].shape != expected[k].shape for k in expected):
        raise ValueError('Encoder tensor shapes do not match the selected backbone')
    if any(not torch.equal(cleaned[k], expected[k]) for k in ('input_mean', 'input_std')):
        raise ValueError('Checkpoint normalization differs from config; set normalization explicitly')
    encoder.load_state_dict(cleaned, strict=True)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {'kind': kind, 'file': Path(path).name, 'sha256': digest, 'metadata': metadata}
