# Implementation and source notes

This package adapts the project's existing KoMaP sampling, loss, metrics and
DySample-style upsampling code. The combination modules are derived designs,
not full official reproductions or claims of new scientific novelty.

External dependencies and public pretrained weights retain their upstream
licenses. Their source code/weights are not bundled here.

- DySample sampling principle: https://github.com/tiny-smart/dysample
- scSE: https://arxiv.org/abs/1808.08127
- UPerNet PPM/FPN: https://arxiv.org/abs/1807.10221
- UNet++ dense nested skips: https://arxiv.org/abs/1807.10165
- timm Swin feature extractor: https://github.com/huggingface/pytorch-image-models
- Mamba library: https://github.com/state-spaces/mamba
- MicroNet weights: https://github.com/nasa/pretrained-microscopy-models
- MatSSL inspiration: https://arxiv.org/abs/2507.18184

Additional metallography references and implementation differences are in
docs/METALLOGRAPHIC_NOVELTY_BREAKDOWN_KO.md and the main README.
