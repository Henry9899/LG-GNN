"""Reconstructed dual-stream ResNet50/FPN/R-ELAN, not original author code."""
import torch
from torch import nn
from torch.nn import functional as F

from .atlas import pool_regions


class RELAN(nn.Module):
    def __init__(self, channels=256, depth=2):
        super().__init__()
        self.channel = nn.Sequential(nn.Conv2d(channels, channels // 16, 1), nn.ReLU(),
                                     nn.Conv2d(channels // 16, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                                  nn.BatchNorm2d(channels), nn.ReLU())
                                     for _ in range(depth)])
        self.fuse = nn.Conv2d((depth + 1) * channels, channels, 1)

    def forward(self, x):
        ch = self.channel(x.mean((2, 3), keepdim=True) + x.amax((2, 3), keepdim=True)).sigmoid()
        sp = self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)).sigmoid()
        history = [x * ch * sp]
        for block in self.blocks:
            history.append(block(history[-1]))
        return self.fuse(torch.cat(history, 1))


class SliceEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.ops import FeaturePyramidNetwork
        net = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)
        self.scale_weights = nn.Parameter(torch.zeros(4))
        self.relan = RELAN()

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = self.stem(x)
        levels = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            levels[str(i)] = x
        pyramid = self.fpn(levels)
        size = pyramid['0'].shape[-2:]
        fused = sum(w * F.interpolate(level, size=size, mode='bilinear', align_corners=False)
                    for w, level in zip(self.scale_weights.softmax(0), pyramid.values()))
        return self.relan(fused)


class ImageLGGNN(nn.Module):
    """Differentiable image -> atlas pooling -> graph wrapper for custom training."""
    def __init__(self, graph, pretrained=False):
        super().__init__()
        self.graph = graph
        self.mri, self.pet = SliceEncoder(pretrained), SliceEncoder(pretrained)

    def forward(self, mri, pet, assignments, adjacency):
        # Inputs [B,S,C,H,W]; assignments [B,S,Hf,Wf] in graph-region order.
        if mri.shape != pet.shape or mri.ndim != 5:
            raise ValueError("Expected matched MRI/PET [B,S,C,H,W]")
        b, s = mri.shape[:2]
        pooled = []
        for encoder, images in ((self.mri, mri), (self.pet, pet)):
            features = encoder(images.flatten(0, 1))
            if features.shape[-2:] != assignments.shape[-2:]:
                raise ValueError("Atlas assignments must match backbone feature grid")
            features = features.flatten(2).transpose(1, 2).reshape(b, s, -1, 256)
            regions, counts = pool_regions(features, assignments.flatten(2), self.graph.regions)
            pooled.append(regions)
        return self.graph(torch.stack(pooled, 2), counts, adjacency)
