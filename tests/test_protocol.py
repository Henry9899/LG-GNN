import numpy as np
import pytest
import torch

from lg_gnn.agent import TOOLS, TrainingRetriever, execute_policy
from lg_gnn.bridge import FrozenLanguageAdapter, QueryBridge, categorical_positives
from lg_gnn.data import Cohort, synthetic_cohort
from lg_gnn.evaluation import fit_temperature
from lg_gnn.measurements import NCNormalizer, regional_measurements


def test_normalizer_fits_training_nc_only():
    rng = np.random.default_rng(5)
    n = 40
    cov = np.column_stack([rng.uniform(50, 80, n), rng.integers(0, 2, n), rng.uniform(1e6, 2e6, n)])
    values = rng.normal(size=(n, 2, 2))
    nc = np.arange(n) < 20
    qc = np.ones_like(values, dtype=bool)
    norm = NCNormalizer().fit(values, cov, nc, qc)
    original = norm.transform(values, cov)
    values[~nc] += 10000
    second = NCNormalizer().fit(values, cov, nc, qc)
    np.testing.assert_allclose(norm.beta, second.beta)
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original[nc].mean(0), 0., atol=1e-10)


def test_measurements_respect_inverse_jacobian_and_reference():
    atlas = np.ones((2, 2, 2))
    result = regional_measurements(np.ones_like(atlas) * 0.5, np.ones_like(atlas) * 2,
        np.ones_like(atlas) * 4, atlas, [1, 2], atlas.astype(bool), 2., 100.)
    np.testing.assert_allclose(result[0], [0.16, 1.])
    assert np.isnan(result[1]).all()


def test_bridge_masks_missing_memory_and_rejects_empty():
    torch.manual_seed(6)
    bridge = QueryBridge(graph_dim=16, llm_dim=24, queries=4, heads=4)
    memory = torch.randn(2, 3, 16)
    covered = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    before = bridge(memory, covered)
    memory[~covered] = float('nan')
    torch.testing.assert_close(before, bridge(memory, covered))
    with pytest.raises(ValueError):
        bridge(memory, torch.zeros_like(covered))


def test_positive_matching_treats_missing_and_qc_failures_as_absent():
    values = torch.tensor([[[-2., float('nan')]], [[-3., 100.]], [[0., 100.]]])
    valid = torch.tensor([[[1, 0]], [[1, 0]], [[1, 0]]], dtype=torch.bool)
    match = categorical_positives(values, valid)
    assert match.tolist() == [[True, True, False], [True, True, False], [False, False, True]]


def test_frozen_language_gradient_reaches_bridge_only():
    transformers = pytest.importorskip('transformers')
    config = transformers.GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    llm = transformers.GPT2LMHeadModel(config)
    bridge = QueryBridge(graph_dim=8, llm_dim=16, queries=2, heads=2, layers=1)
    adapter = FrozenLanguageAdapter(llm, bridge).train()
    nodes = torch.randn(2, 3, 8)
    covered = torch.ones(2, 3, dtype=torch.bool)
    ids = torch.tensor([[1, 2, 3], [1, 2, 0]])
    loss = adapter.objective(nodes, covered, ids, ids != 0, torch.eye(2, dtype=torch.bool))
    loss.backward()
    assert all(p.grad is None for p in llm.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.parameters())
    assert not llm.training


def test_retriever_rejects_test_subjects_and_diagnosis():
    with pytest.raises(ValueError):
        TrainingRetriever(['test'], [[1., 0.]], [[]], ['train'])
    with pytest.raises(ValueError):
        TrainingRetriever(['train'], [[1., 0.]], [[{'diagnosis': 'AD'}]], ['train'])
    index = TrainingRetriever(['train'], [[1., 0.]], [[{'id': 'a'}]], ['train'])
    assert index.search([1, 0])[0]['subject_id'] == 'train'


def test_bounded_policy_counts_failed_calls_and_fixed_pipeline():
    tools = {name: (lambda: {'ok': True}) for name in TOOLS}
    result = execute_policy(tools, choose=lambda trace: TOOLS[0], budget=6)
    assert result.calls == 6 and result.stop_reason == 'budget_exhausted'
    fixed = execute_policy(tools, fixed=True)
    assert fixed.calls == 4
    def fail():
        raise RuntimeError('synthetic tool error')
    tools[TOOLS[0]] = fail
    result = execute_policy(tools, choose=lambda trace: TOOLS[0], budget=1)
    assert result.calls == 1 and result.trace[0]['error'] == 'RuntimeError'


def test_temperature_improves_calibration_nll():
    logits = torch.tensor([[8., 0., 0.], [8., 0., 0.], [0., 8., 0.], [0., 0., 8.]])
    labels = torch.tensor([0, 1, 1, 2])
    temperature = fit_temperature(logits, labels)
    assert temperature > 0
    assert torch.nn.functional.cross_entropy(logits / temperature, labels) <= torch.nn.functional.cross_entropy(logits, labels)


def test_data_format_rejects_subject_overlap_and_does_not_overwrite(tmp_path):
    path = tmp_path / 'synthetic.npz'
    synthetic_cohort(path)
    data = Cohort.load(path)
    assert data.provenance['synthetic']
    with pytest.raises(FileExistsError):
        synthetic_cohort(path)
    data.subject_ids[0] = data.subject_ids[1]
    with pytest.raises(ValueError):
        data.validate()
