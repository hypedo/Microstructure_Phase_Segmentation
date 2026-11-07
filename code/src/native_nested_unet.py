"""Full-resolution nested residual U-Net with four supervised decoder depths.

Adapted from the established UNet++ nested skip design (Zhou et al., 2018):
https://arxiv.org/abs/1807.10165
Uses this project's residual GroupNorm blocks, widths, four phase logits and
optional boundary logit. This is not an exact paper reproduction or a claim
to invent nested skip connections. All four full-resolution heads receive
equal loss weight during training; inference uses their mean logits.
"""
import torch
from torch import nn
from torch.nn import functional as F
from native_unet import ResidualBlock


class NativeNestedUNet(nn.Module):
    def __init__(self,classes=5,base_channels=32):
        super().__init__()
        if classes not in (4,5) or base_channels<2:
            raise ValueError('Expected four phases, optional boundary and positive feature width')
        self.channels=tuple(base_channels*n for n in (1,2,4,8,12))
        self.stem=ResidualBlock(3,self.channels[0])
        self.pool=nn.MaxPool2d(2,2)
        self.down_blocks=nn.ModuleList(ResidualBlock(a,b) for a,b in zip(self.channels[:-1],self.channels[1:]))
        self.nested=nn.ModuleDict()
        for depth in range(1,5):
            for level in range(5-depth):
                inputs=depth*self.channels[level]+self.channels[level+1]
                self.nested[f'{level}_{depth}']=ResidualBlock(inputs,self.channels[level])
        self.heads=nn.ModuleList(nn.Conv2d(self.channels[0],classes,1) for _ in range(4))

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=3 or min(x.shape[0],*x.shape[-2:])<1:
            raise ValueError('Expected nonempty [B,3,H,W] image tensor')
        height,width=x.shape[-2:]
        ph,pw=(-height)%16,(-width)%16
        if ph or pw:
            x=F.pad(x,(0,pw,0,ph),mode='reflect' if ph<height and pw<width else 'replicate')
        grid={(0,0):self.stem(x)}
        for level,block in enumerate(self.down_blocks,1):
            grid[level,0]=block(self.pool(grid[level-1,0]))
        outputs=[]
        for depth in range(1,5):
            for level in range(5-depth):
                skip=[grid[level,previous] for previous in range(depth)]
                up=F.interpolate(grid[level+1,depth-1],size=grid[level,0].shape[-2:],mode='bilinear',align_corners=False)
                grid[level,depth]=self.nested[f'{level}_{depth}'](torch.cat(skip+[up],1))
            outputs.append(self.heads[depth-1](grid[0,depth])[...,:height,:width])
        stacked=torch.stack(outputs,dim=1)
        return stacked if self.training else stacked.mean(1)
