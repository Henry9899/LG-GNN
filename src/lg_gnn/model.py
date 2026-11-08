"""Coverage-masked, multi-head Gaussian diffusion with auxiliary concepts."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def normalized_operator(adjacency, affinities, covered):
    """Rows receive, columns send. Eq. (masked_propagation), per head."""
    r = covered.shape[-1]
    endpoints = covered[:, None, :, None] & covered[:, None, None, :]
    base = adjacency + affinities + torch.eye(r, device=affinities.device)
    edges = torch.where(endpoints, base, 0.0)
    degree = edges.sum(-1)
    inverse = torch.where(degree > 0, degree.clamp_min(1e-12).rsqrt(), 0.0)
    return inverse[..., :, None] * edges * inverse[..., None, :]


@dataclass
class GraphOutput:
    logits: torch.Tensor
    concepts: torch.Tensor
    nodes: torch.Tensor
    pooled: torch.Tensor
    covered: torch.Tensor
    eligible: torch.Tensor


class LGGNN(nn.Module):
    def __init__(self, regions, feature_dim=256, hidden=256, heads=8,
                 steps=3, alpha=0.5, coverage_threshold=20, strict_bottleneck=False):
        super().__init__()
        if hidden % heads or not 0 <= alpha <= 1 or coverage_threshold < 1:
            raise ValueError("Invalid graph configuration")
        self.regions, self.heads, self.steps = regions, heads, steps
        self.alpha, self.coverage_threshold = alpha, coverage_threshold
        self.strict_bottleneck = strict_bottleneck
        self.project = nn.Linear(2 * feature_dim, hidden)
        self.missing = nn.Parameter(torch.zeros(hidden))
        self.identity = nn.Parameter(torch.randn(regions, hidden) * 0.02)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.log_sigma = nn.Parameter(torch.zeros(heads))
        self.transforms = nn.ModuleList([
            nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(heads)])
            for _ in range(steps)])
        self.concept_head = nn.Linear(hidden, 2)
        self.classifier = nn.Linear(regions * 2 if strict_bottleneck else hidden, 3)

    def forward(self, descriptors, counts, adjacency):
        """Descriptors [B,R,2,D]; counts [B,R]; anatomy [R,R].

        Ineligible subjects have NaN logits/concepts and must be withheld.
        """
        if descriptors.shape[1:3] != (self.regions, 2) or counts.shape != descriptors.shape[:2]:
            raise ValueError("Region/modality shape does not match model")
        if adjacency.shape != (self.regions, self.regions) or not torch.isfinite(adjacency).all() or (adjacency < 0).any():
            raise ValueError("Adjacency must be finite, nonnegative and [R,R]")
        covered = counts >= self.coverage_threshold
        clean = torch.where(covered[..., None, None], descriptors, 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("Covered descriptors must be finite")
        observed = self.project(clean.flatten(2))
        nodes = torch.where(covered[..., None], observed, self.missing) + self.identity
        b, r, h = nodes.shape
        q = self.query(nodes).reshape(b, r, self.heads, -1).transpose(1, 2)
        k = self.key(nodes).reshape(b, r, self.heads, -1).transpose(1, 2)
        distance = (q[..., :, None, :] - k[..., None, :, :]).square().sum(-1)
        sigma = F.softplus(self.log_sigma).clamp_min(1e-4)
        affinities = torch.exp(-distance / (2 * sigma[None, :, None, None].square()))
        operator = normalized_operator(adjacency, affinities, covered)
        # Each head retains a full-width state; heads are averaged after diffusion.
        states = nodes[:, None].expand(-1, self.heads, -1, -1)
        states = torch.where(covered[:, None, :, None], states, 0.0)
        for transforms in self.transforms:
            transformed = torch.stack([layer(states[:, i]) for i, layer in enumerate(transforms)], 1)
            states = (1 - self.alpha) * states + self.alpha * (operator @ transformed)
        nodes = states.mean(1)
        pooled = nodes.sum(1) / covered.sum(1).clamp_min(1)[:, None]
        concepts = self.concept_head(nodes)
        readout = (torch.where(covered[..., None], concepts, 0.0).flatten(1)
                   if self.strict_bottleneck else pooled)
        logits = self.classifier(readout)
        eligible = covered.any(1)
        logits = logits.masked_fill(~eligible[:, None], math.nan)
        concepts = concepts.masked_fill(~covered[..., None], math.nan)
        return GraphOutput(logits, concepts, nodes, pooled, covered, eligible)
