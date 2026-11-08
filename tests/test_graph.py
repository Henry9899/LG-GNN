import numpy as np
import pytest
import torch

from lg_gnn.atlas import anatomical_adjacency, majority_assignment, pool_regions
from lg_gnn.losses import masked_concept_loss, multiple_positive_alignment
from lg_gnn.model import LGGNN, normalized_operator


def test_majority_maps_nonconsecutive_ids_and_rejects_ties():
    labels = np.array([[[10, 10, 10, 10], [10, 99, 99, 99]]])
    assert majority_assignment(labels, [10, 99], 2).tolist() == [[[0, -1]]]


def test_pool_zero_regions_and_background_nan():
    x = torch.tensor([[[[2.], [4.], [float('nan')]]]])
    pooled, counts = pool_regions(x, torch.tensor([[[0, 0, -1]]]), 2)
    torch.testing.assert_close(pooled, torch.tensor([[[3.], [0.]]]))
    assert counts.tolist() == [[2, 0]]


def test_26_connectivity_and_explicit_homologues():
    atlas = np.zeros((2, 2, 2), dtype=int)
    atlas[0, 0, 0], atlas[1, 1, 1] = 10, 20
    a = anatomical_adjacency(atlas, [10, 20, 30], [(20, 30)])
    assert a[0, 1] == a[1, 0] == a[1, 2] == a[2, 1] == 1
    assert np.diag(a).sum() == 0


def test_operator_masks_senders_and_receivers_and_has_safe_degrees():
    operator = normalized_operator(torch.ones(3, 3), torch.ones(1, 2, 3, 3),
                                    torch.tensor([[True, False, True]]))
    assert (operator[:, :, 1, :] == 0).all()
    assert (operator[:, :, :, 1] == 0).all()
    assert torch.isfinite(operator).all()
    empty = normalized_operator(torch.zeros(3, 3), torch.zeros(1, 1, 3, 3),
                                torch.zeros(1, 3, dtype=torch.bool))
    assert empty.count_nonzero() == 0


def test_missing_region_cannot_change_prediction_or_gradient():
    torch.manual_seed(4)
    model = LGGNN(3, feature_dim=4, hidden=16, heads=4)
    x = torch.randn(1, 3, 2, 4, requires_grad=True)
    counts = torch.tensor([[20, 0, 30]])
    first = model(x, counts, torch.ones(3, 3))
    changed = x.detach().clone()
    changed[:, 1] = float('nan')
    second = model(changed, counts, torch.ones(3, 3))
    torch.testing.assert_close(first.logits, second.logits)
    first.logits.sum().backward()
    assert x.grad[:, 1].count_nonzero() == 0
    assert torch.isfinite(x.grad).all()


def test_no_covered_subject_is_withheld():
    model = LGGNN(2, feature_dim=3, hidden=8, heads=2)
    out = model(torch.full((1, 2, 2, 3), float('nan')), torch.zeros(1, 2), torch.zeros(2, 2))
    assert not out.eligible.item() and torch.isnan(out.logits).all()


@pytest.mark.parametrize('target,quality,expected', [
    ([float('nan'), 3.], [False, True], 1.),
    ([float('nan'), float('inf')], [False, False], 0.)])
def test_masked_loss_never_multiplies_nan_by_zero(target, quality, expected):
    predicted = torch.tensor([[[2., 2.]]], requires_grad=True)
    loss = masked_concept_loss(predicted, torch.tensor([[target]]), torch.tensor([[True]]),
                               torch.tensor([[quality]]))
    assert loss.item() == expected
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


def test_multiple_positive_objective_matches_manual_average():
    graph = torch.eye(2, requires_grad=True)
    text = torch.eye(2)
    positives = torch.ones(2, 2, dtype=torch.bool)
    loss = multiple_positive_alignment(graph, text, positives, 1.)
    expected = -torch.eye(2).log_softmax(-1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(graph.grad).all()
