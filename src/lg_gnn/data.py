"""Strict, non-pickled regional-feature exchange format."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class Cohort:
    subject_ids: np.ndarray
    labels: np.ndarray
    descriptors: np.ndarray
    counts: np.ndarray
    measurements: np.ndarray
    qc: np.ndarray
    covariates: np.ndarray
    adjacency: np.ndarray
    regions: list
    provenance: dict

    @classmethod
    def load(cls, path):
        path = Path(path)
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        with np.load(path, allow_pickle=False) as archive:
            data = cls(**{key: archive[key] for key in (
                'subject_ids', 'labels', 'descriptors', 'counts', 'measurements',
                'qc', 'covariates', 'adjacency')}, regions=metadata['regions'],
                provenance=metadata['provenance'])
        data.validate()
        return data

    def validate(self):
        n, r = len(self.subject_ids), len(self.regions)
        if self.subject_ids.dtype.kind not in 'US' or len(set(self.subject_ids)) != n:
            raise ValueError("Exactly one row per unique string subject ID is required")
        if self.labels.shape != (n,) or not np.isin(self.labels, [0, 1, 2]).all():
            raise ValueError("Label order is 0=NC, 1=MCI, 2=AD")
        if self.descriptors.ndim != 4 or self.descriptors.shape[:3] != (n, r, 2):
            raise ValueError("descriptors must have shape [subjects,regions,2,features]")
        expected = {'counts': (n, r), 'measurements': (n, r, 2), 'qc': (n, r, 2),
                    'covariates': (n, 3), 'adjacency': (r, r)}
        for field, shape in expected.items():
            if getattr(self, field).shape != shape:
                raise ValueError(f"{field} must have shape {shape}")
        if not np.isfinite(self.covariates).all():
            raise ValueError("Covariates [age,sex,ICV] must be finite")
        if not np.isin(self.covariates[:, 1], [0, 1]).all() or (self.covariates[:, 2] <= 0).any():
            raise ValueError("Expected binary sex code and positive ICV")
        if not np.isfinite(self.counts).all() or (self.counts < 0).any() or (self.counts != np.floor(self.counts)).any():
            raise ValueError("Counts must be finite nonnegative integers")
        if not np.isfinite(self.adjacency).all() or (self.adjacency < 0).any():
            raise ValueError("Adjacency must be finite and nonnegative")
        if self.qc.dtype != np.bool_:
            raise ValueError("QC must be boolean")
        if not self.provenance.get('feature_origin'):
            raise ValueError("Declare feature_origin and training partition provenance")
        if len({x['id'] for x in self.regions}) != r or any(x.get('laterality') not in ('L', 'R', 'B', 'NA') for x in self.regions):
            raise ValueError("Region IDs must be unique and laterality explicit")


def synthetic_cohort(path, subjects_per_class=30, seed=17):
    """Small artificial data for execution checks. No ADNI data or results."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), subjects_per_class)
    n, r, d = len(labels), 6, 12
    covariates = np.column_stack([rng.uniform(55, 88, n), rng.integers(0, 2, n),
                                  rng.uniform(1.2e6, 1.7e6, n)])
    latent = rng.normal(0, 1, (n, r, 2)) - labels[:, None, None] * 0.5
    values = latent * np.array([0.0003, 0.1]) + np.array([0.004, 1.2])
    values[:, :, 0] += covariates[:, 0, None] * -0.000005
    projection = rng.normal(size=(2, d))
    features = latent[..., None] * projection[None, None] + rng.normal(0, 0.4, (n, r, 2, d))
    counts = rng.integers(25, 55, (n, r))
    counts[::7, -1] = 0
    qc = rng.random((n, r, 2)) > 0.03
    values[~qc] = np.nan
    adjacency = np.eye(r, k=1) + np.eye(r, k=-1)
    regions = [{'id': i + 1, 'name': f'Synthetic_region_{i + 1}',
                'laterality': 'L' if i % 2 == 0 else 'R'} for i in range(r)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(f"Refusing to replace existing data: {path}")
    np.savez_compressed(path, subject_ids=np.array([f'SYN-{i:04}' for i in range(n)]),
                        labels=labels, descriptors=features.astype('float32'), counts=counts,
                        measurements=values, qc=qc, covariates=covariates, adjacency=adjacency.astype('float32'))
    path.with_suffix('.json').write_text(json.dumps({'regions': regions, 'provenance': {
        'synthetic': True, 'seed': seed, 'feature_origin': 'artificial random regional descriptors',
        'note': 'Execution fixture only; no participant data or biomedical interpretation.'}}, indent=2), encoding='utf-8')
