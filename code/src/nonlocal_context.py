"""Pooled non-local residual context for microscopy phase segmentation.

Established non-local attention, adapted to the bottleneck of Native U-Net.
Wang et al., CVPR 2018: https://arxiv.org/abs/1711.07971
Queries retain the complete bottleneck grid. Keys and values use at most a
16x16 area-averaged grid, keeping attention cost bounded for larger images.
No positions, filenames, source groups, labels, or external weights are used.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PooledNonLocalContext(nn.Module):
    def __init__(self, channels, hidden=128, max_grid=16):
        super().__init__()
        self.hidden=min(hidden,channels)
        self.max_grid=max_grid
        self.norm=nn.GroupNorm(8 if channels%8==0 else 1,channels)
        self.query=nn.Conv2d(channels,self.hidden,1,bias=False)
        self.key=nn.Conv2d(channels,self.hidden,1,bias=False)
        self.value=nn.Conv2d(channels,self.hidden,1,bias=False)
        self.output=nn.Conv2d(self.hidden,channels,1,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,x):
        batch,channels,height,width=x.shape
        normalized=self.norm(x)
        pooled=F.adaptive_avg_pool2d(normalized,(min(height,self.max_grid),min(width,self.max_grid)))
        query=self.query(normalized).flatten(2).transpose(1,2)
        key=self.key(pooled).flatten(2)
        value=self.value(pooled).flatten(2).transpose(1,2)
        # Explicit FP32 affinities avoid overflow with AMP feature magnitudes.
        affinity=torch.bmm(query.float(),key.float())/math.sqrt(self.hidden)
        context=torch.bmm(affinity.softmax(-1),value.float())
        context=context.transpose(1,2).reshape(batch,self.hidden,height,width)
        return x+self.output(context.to(x.dtype))
