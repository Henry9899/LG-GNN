"""Atlas operations on already aligned images; no registration is performed here."""
import itertools

import numpy as np
import torch


def majority_assignment(labels: np.ndarray, region_ids, patch_size=4):
    """[S,H,W] labels -> [S,H/p,W/p] region indices; background/ties -> -1.

    A region must occupy strictly more than half of a patch. Nonconsecutive
    atlas IDs are mapped through the supplied, ordered region manifest.
    """
    labels = np.asarray(labels)
    if patch_size < 1 or len(set(region_ids)) != len(region_ids):
        raise ValueError("Invalid patch size or duplicate region IDs")
    if labels.ndim != 3 or any(n % patch_size for n in labels.shape[1:]):
        raise ValueError("Slice dimensions must be divisible by patch_size")
    s, h, w = labels.shape
    patches = labels.reshape(s, h // patch_size, patch_size, w // patch_size, patch_size)
    result = np.full((s, h // patch_size, w // patch_size), -1, dtype=np.int64)
    for index, region in enumerate(region_ids):
        result[(patches == region).sum(axis=(2, 4)) > patch_size ** 2 / 2] = index
    return result


def pool_regions(features, assignments, regions):
    """Eq. (pool): features [B,S,N,D], assignment [B,S,N], -1=background."""
    if features.shape[:3] != assignments.shape or assignments.dtype != torch.long:
        raise ValueError("Expected matching [B,S,N] assignments of dtype long")
    if ((assignments < -1) | (assignments >= regions)).any():
        raise ValueError("Assignment outside atlas region manifest")
    b, _, _, d = features.shape
    valid = assignments.flatten(1) >= 0
    index = assignments.flatten(1).clamp_min(0)
    flat = features.flatten(1, 2)
    sums = features.new_zeros(b, regions, d)
    # where, rather than multiplication, also excludes invalid NaN backgrounds.
    sums.scatter_add_(1, index[..., None].expand(-1, -1, d),
                      torch.where(valid[..., None], flat, 0.0))
    counts = features.new_zeros(b, regions)
    counts.scatter_add_(1, index, valid.to(features.dtype))
    return sums / counts.clamp_min(1)[..., None], counts


def anatomical_adjacency(volume, region_ids, homologues=()):
    """26-connected atlas contacts plus explicitly declared homologue ID pairs."""
    volume = np.asarray(volume)
    if volume.ndim != 3 or len(set(region_ids)) != len(region_ids):
        raise ValueError("Expected a 3D atlas and unique region IDs")
    lookup = {int(region): i for i, region in enumerate(region_ids)}
    adjacency = np.zeros((len(region_ids), len(region_ids)), dtype=np.float32)
    for offset in itertools.product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        left, right = [], []
        for n, delta in zip(volume.shape, offset):
            left.append(slice(max(0, delta), min(n, n + delta)))
            right.append(slice(max(0, -delta), min(n, n - delta)))
        pairs = np.unique(np.stack([volume[tuple(left)].ravel(),
                                    volume[tuple(right)].ravel()], axis=1), axis=0)
        for a, b in pairs:
            if a != b and int(a) in lookup and int(b) in lookup:
                adjacency[lookup[int(a)], lookup[int(b)]] = 1
    for a, b in homologues:
        adjacency[lookup[a], lookup[b]] = adjacency[lookup[b], lookup[a]] = 1
    np.fill_diagonal(adjacency, 0)  # Graph operator adds self-loops exactly once.
    return adjacency
