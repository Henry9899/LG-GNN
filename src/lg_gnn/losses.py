import torch
from torch.nn import functional as F


def masked_concept_loss(predicted, targets, covered, qc):
    valid = covered[..., None] & qc.bool() & torch.isfinite(targets)
    if not valid.any():
        # Empty slice preserves autograd without allowing NaNs into the loss.
        return predicted.reshape(-1)[:0].sum()
    return F.mse_loss(predicted[valid], targets[valid])


def multiple_positive_alignment(graph, text, positives, temperature=0.07):
    if temperature <= 0 or not positives.any(1).all():
        raise ValueError("Each row requires a positive and positive temperature")
    logits = F.normalize(graph, dim=-1) @ F.normalize(text, dim=-1).T / temperature
    log_prob = logits.log_softmax(-1)
    return -(log_prob * positives).sum(-1).div(positives.sum(-1)).mean()


def supervised_contrastive(embedding, labels, temperature=0.07):
    """Reference supervised pooled-embedding contrast; original DMA variant unknown."""
    n = len(labels)
    if n < 2:
        return embedding.sum() * 0
    eye = torch.eye(n, device=embedding.device, dtype=torch.bool)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    usable = positives.any(-1)
    if not usable.any():
        return embedding.sum() * 0
    logits = F.normalize(embedding, dim=-1) @ F.normalize(embedding, dim=-1).T / temperature
    log_prob = logits.masked_fill(eye, -torch.inf).log_softmax(-1)
    selected = torch.where(positives, log_prob, 0.0)
    return -(selected.sum(-1)[usable] / positives.sum(-1)[usable]).mean()


def graph_objective(output, labels, targets, qc, class_weights, concept_weight=0.5,
                    focal_weight=0.5, contrastive_weight=0.3):
    eligible = output.eligible
    concept = masked_concept_loss(output.concepts, targets, output.covered, qc)
    if not eligible.any():
        return concept
    logits, labels = output.logits[eligible], labels[eligible]
    ce_per = F.cross_entropy(logits, labels, reduction='none')
    weights = class_weights[labels]
    ce = (ce_per * weights).sum() / weights.sum()
    probability = logits.softmax(-1).gather(1, labels[:, None]).squeeze(1)
    focal = (weights * (1 - probability).square() * ce_per).sum() / weights.sum()
    contrast = supervised_contrastive(output.pooled[eligible], labels)
    return ce + focal_weight * focal + contrastive_weight * contrast + concept_weight * concept
