import json

import numpy as np
import pytest

from lg_gnn.data import synthetic_cohort
from lg_gnn.experiment import ExperimentConfig, run_experiment
from lg_gnn.inference import predict


def test_nested_cv_disjointness_checkpoint_inference_and_withholding(tmp_path):
    data_path = tmp_path / 'synthetic.npz'
    synthetic_cohort(data_path, subjects_per_class=24)
    config = ExperimentConfig(outer_folds=3, inner_folds=2, epochs=1,
                              learning_rates=(0.001,), hidden=16, heads=4, diffusion_steps=1)
    output = tmp_path / 'run'
    metrics = run_experiment(data_path, output, config)
    assert metrics['synthetic'] and metrics['n_total'] == 72
    all_tests = []
    for fold in range(3):
        manifest = json.loads((output / f'fold_{fold}' / 'subjects.json').read_text())
        development, calibration, test = [set(manifest[k]) for k in ('development', 'calibration', 'outer_test')]
        assert not (development & calibration or development & test or calibration & test)
        for inner in manifest['inner']:
            train, validation = set(inner['train']), set(inner['validation'])
            assert not train & validation
            assert train | validation == development
        all_tests.extend(test)
    assert len(all_tests) == len(set(all_tests)) == 72
    with np.load(data_path) as data:
        # A prediction-only file needs no labels, targets, QC or covariates.
        fields = {k: data[k][:2] for k in ('subject_ids', 'descriptors', 'counts')}
        fields['adjacency'] = data['adjacency']
    fields['counts'][0] = 0
    prediction_data = tmp_path / 'unlabelled.npz'
    np.savez_compressed(prediction_data, **fields)
    prediction_data.with_suffix('.json').write_text(data_path.with_suffix('.json').read_text())
    result = predict(prediction_data, output / 'fold_0', tmp_path / 'prediction')
    assert result == {'n_subjects': 2, 'n_withheld': 1}
    summaries = json.loads((tmp_path / 'prediction' / 'summaries.json').read_text())
    assert summaries[0]['probabilities'] is None
    assert all(f['source'] == 'predicted' for f in summaries[1]['findings'])
    with pytest.raises(FileExistsError):
        run_experiment(data_path, output, config)
