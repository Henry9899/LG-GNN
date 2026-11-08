"""Metrics retain denominator counts and return None for undefined endpoints."""
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.nn import functional as F


def fit_temperature(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if not len(labels) or not torch.isfinite(logits).all():
        raise ValueError("Temperature fitting needs eligible held-out calibration data")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.clamp(-4, 4).exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().clamp(-4, 4).exp())


def classification_metrics(labels, probabilities, bins=15):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    eligible = np.isfinite(probabilities).all(1)
    y, p = labels[eligible], probabilities[eligible]
    result = {'n_total': len(labels), 'n_eligible': int(eligible.sum()),
              'coverage': float(eligible.mean()),
              'class_coverage': [float(eligible[labels == c].mean()) if (labels == c).any() else None
                                 for c in range(3)]}
    if not len(y):
        return {**result, **{k: None for k in ('balanced_accuracy', 'macro_f1', 'mci_recall',
                                             'ad_nc_auroc', 'brier', 'ece')}}
    if (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Expected normalized class probabilities")
    prediction = p.argmax(1)
    # Three-class endpoints are undefined when a retained class has no support.
    all_classes = set(y.tolist()) == {0, 1, 2}
    result['balanced_accuracy'] = float(balanced_accuracy_score(y, prediction)) if all_classes else None
    result['macro_f1'] = float(f1_score(y, prediction, labels=[0, 1, 2], average='macro', zero_division=0)) if all_classes else None
    result['mci_recall'] = float(recall_score(y == 1, prediction == 1, zero_division=0)) if (y == 1).any() else None
    binary = y != 1
    binary_p = p[binary, 2] / np.maximum(p[binary, 0] + p[binary, 2], 1e-15)
    result['ad_nc_auroc'] = float(roc_auc_score(y[binary] == 2, binary_p)) if set(y[binary]) == {0, 2} else None
    result['brier'] = float(np.square(p - np.eye(3)[y]).sum(1).mean())
    confidence = p.max(1)
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    result['ece'] = float(sum((bucket == j).mean() * abs((prediction[bucket == j] == y[bucket == j]).mean()
                                  - confidence[bucket == j].mean())
                             for j in range(bins) if (bucket == j).any()))
    return result


def responsiveness(eligible, changed, revised, omitted, false_revision):
    """Conditional Eq. (phi); omissions never count as successful revisions."""
    eligible, changed, revised, omitted, false_revision = [np.asarray(x, dtype=bool) for x in
        (eligible, changed, revised, omitted, false_revision)]
    change = eligible & changed
    unchanged = eligible & ~changed
    success = change & revised & ~omitted
    return {'eligible': int(eligible.sum()), 'changed': int(change.sum()),
            'successful_revisions': int(success.sum()), 'omitted': int((eligible & omitted).sum()),
            'phi_responsiveness': float(success.sum() / change.sum()) if change.any() else None,
            'concept_change_rate': float(change.sum() / eligible.sum()) if eligible.any() else None,
            'false_revision_rate': float((unchanged & false_revision).sum() / unchanged.sum()) if unchanged.any() else None}


def fault_metrics(faulty, detected, withheld):
    faulty, detected, withheld = [np.asarray(x, dtype=bool) for x in (faulty, detected, withheld)]
    return {'n': len(faulty), 'n_faulty': int(faulty.sum()),
            'fault_recall': float(detected[faulty].mean()) if faulty.any() else None,
            'false_alarm_rate': float(detected[~faulty].mean()) if (~faulty).any() else None,
            'withholding_rate': float(withheld.mean()) if len(withheld) else None,
            'joint_counts': {f'fault_{f}_detected_{d}_withheld_{w}': int(((faulty == f) & (detected == d) & (withheld == w)).sum())
                             for f in (0, 1) for d in (0, 1) for w in (0, 1)}}


def paired_bootstrap_ba(labels, probabilities_a, probabilities_b, resamples=2000, seed=17):
    """Class-stratified paired bootstrap; eligible intersection, no seed uncertainty."""
    labels = np.asarray(labels)
    a, b = np.asarray(probabilities_a), np.asarray(probabilities_b)
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    groups = [np.flatnonzero(valid & (labels == c)) for c in range(3)]
    if any(not len(g) for g in groups):
        raise ValueError("Each class needs paired eligible predictions")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        selected = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        differences.append(balanced_accuracy_score(labels[selected], a[selected].argmax(1))
                           - balanced_accuracy_score(labels[selected], b[selected].argmax(1)))
    return {'n_paired': int(valid.sum()), 'resamples': resamples,
            'difference_ci95': np.quantile(differences, [0.025, 0.975]).tolist()}
