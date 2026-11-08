from dataclasses import replace

import pytest

from lg_gnn.evaluation import classification_metrics, fault_metrics, responsiveness
from lg_gnn.reporting import (EvidenceRecord, canonical_finding, category,
                              generate_checked, template_summary, validate_finding)


@pytest.fixture
def record():
    return EvidenceRecord('hip-L:MRI:predicted', 'Hippocampus', 'L', 'MRI', -1.8,
                          'z_score', 'predicted', 'pass')


@pytest.mark.parametrize('value,expected', [(-1.5, 'reduced'), (-1.1, 'uncertain'),
    (-1., 'not_reduced'), (8., 'not_reduced'), (float('nan'), 'not_assessed')])
def test_categories(value, expected):
    assert category(value) == expected


def test_predicted_claim_cannot_masquerade_as_measurement(record):
    claim = canonical_finding(record)
    assert validate_finding(claim, [record])
    assert 'estimate' in claim['statement']
    claim['source'] = 'measured'
    assert not validate_finding(claim, [record])


def test_checker_rejects_extra_claim_fields(record):
    claim = canonical_finding(record)
    claim['diagnosis'] = 'Unsupported disease assertion'
    assert not validate_finding(claim, [record])


@pytest.mark.parametrize('field,value', [('laterality', 'R'), ('value', -2.5),
    ('modality', 'PET'), ('category', 'not_reduced'), ('evidence_ids', ['fake']),
    ('statement', 'There is severe Alzheimer pathology.')])
def test_checker_rejects_unsupported_fields(record, field, value):
    claim = canonical_finding(record)
    claim[field] = value
    assert not validate_finding(claim, [record])


def test_uncovered_is_not_normal_and_probabilities_are_copied(record):
    unavailable = replace(record, id='uncovered', value=None, quality='not_assessed')
    summary = template_summary([record, unavailable], {'NC': 0.1, 'MCI': 0.6, 'AD': 0.3})
    assert len(summary['findings']) == 1
    assert 'not_assessed' in summary['limitations'][0]
    assert summary['probabilities']['MCI'] == 0.6
    with pytest.raises(ValueError):
        canonical_finding(unavailable)


def test_generator_retries_once_then_withholds(record):
    prompts = []
    def invalid(prompt):
        prompts.append(prompt)
        return '{bad json'
    result = generate_checked(invalid, [record], {'NC': 0.1, 'MCI': 0.6, 'AD': 0.3})
    assert len(prompts) == 2
    assert result['status'] == 'insufficient_evidence'


def test_responsiveness_denominator_excludes_vacuous_success_and_omissions():
    result = responsiveness([1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0],
                            [0, 1, 0, 0], [0, 0, 1, 0])
    assert result['phi_responsiveness'] == 0.5
    assert result['concept_change_rate'] == 0.5
    assert result['false_revision_rate'] == 0.5
    assert responsiveness([1], [0], [1], [0], [0])['phi_responsiveness'] is None


def test_fault_detection_and_withholding_are_distinct_events():
    result = fault_metrics([1, 0, 0], [1, 1, 0], [0, 0, 1])
    assert result['fault_recall'] == 1
    assert result['false_alarm_rate'] == 0.5
    assert result['withholding_rate'] == 1 / 3
    assert sum(result['joint_counts'].values()) == 3


def test_balanced_accuracy_and_nan_withholding():
    labels = [0, 0, 1, 2]
    p = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [float('nan')] * 3]
    metrics = classification_metrics(labels, p)
    assert metrics['coverage'] == 0.75
    assert metrics['balanced_accuracy'] is None  # No retained AD support.
    p[-1] = [0, 0, 1]
    assert classification_metrics(labels, p)['balanced_accuracy'] == pytest.approx(5 / 6)
