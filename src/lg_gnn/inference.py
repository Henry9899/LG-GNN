"""Imaging-feature-only inference from a saved fold; no labels are required."""
import json
from pathlib import Path

import numpy as np
import torch

from .experiment import write_json
from .model import LGGNN
from .reporting import build_records, template_summary


def predict(data_path, fold_path, output_path, device='cpu', batch_size=16):
    torch.set_num_threads(1)
    data_path, fold_path, output_path = map(Path, (data_path, fold_path, output_path))
    metadata = json.loads(data_path.with_suffix('.json').read_text(encoding='utf-8'))
    checkpoint = torch.load(fold_path / 'model.pt', map_location=device, weights_only=True)
    if metadata['regions'] != checkpoint['regions']:
        raise ValueError('Input region manifest differs from the trained model')
    trained_encoder = checkpoint['feature_provenance'].get('encoder_checkpoint_sha256')
    if trained_encoder and metadata.get('provenance', {}).get('encoder_checkpoint_sha256') != trained_encoder:
        raise ValueError('Input encoder checkpoint differs from the trained feature source')
    regions = metadata['regions']
    with np.load(data_path, allow_pickle=False) as archive:
        ids, x, counts, adjacency = [archive[key] for key in ('subject_ids', 'descriptors', 'counts', 'adjacency')]
    if ids.dtype.kind not in 'US' or len(set(ids)) != len(ids):
        raise ValueError('Unique string subject IDs required')
    expected = (len(ids), len(regions), 2, checkpoint['model_kwargs']['feature_dim'])
    if (x.shape != expected or counts.shape != expected[:2] or not np.isfinite(counts).all()
            or (counts < 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError('Invalid feature/count shape or counts')
    if len(ids) < 1 or batch_size < 1:
        raise ValueError('Nonempty cohort and positive batch size required')
    if not np.array_equal(adjacency, checkpoint['adjacency'].cpu().numpy()):
        raise ValueError('Anatomical adjacency differs from the trained model')
    output_path.mkdir(parents=True, exist_ok=False)
    graph = LGGNN(**checkpoint['model_kwargs']).to(device)
    graph.load_state_dict(checkpoint['state_dict'])
    graph.eval()
    mean, scale = checkpoint['feature_mean'], checkpoint['feature_scale']
    anatomy = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    ps, concepts, masks = [], [], []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            features = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            result = graph((features - mean) / scale,
                           torch.as_tensor(counts[start:start + batch_size], device=device), anatomy)
            ps.append((result.logits / checkpoint['temperature']).softmax(-1).cpu().numpy())
            concepts.append(result.concepts.cpu().numpy())
            masks.append(result.covered.cpu().numpy())
    ps, concepts, masks = map(np.concatenate, (ps, concepts, masks))
    np.savez_compressed(output_path / 'predictions.npz', subject_ids=ids,
                        probabilities=ps, predicted_concepts=concepts, covered=masks)
    summaries = []
    for i, subject in enumerate(ids):
        if not masks[i].any():
            summaries.append({'subject_id': str(subject), 'status': 'insufficient_evidence',
                              'reason': 'No atlas region meets the coverage threshold', 'probabilities': None})
            continue
        # Imaging-only inference: do not load or expose measured biomarker targets.
        predicted_records = [r for r in build_records(regions, np.full_like(concepts[i], np.nan),
            concepts[i], masks[i], np.zeros_like(concepts[i], dtype=bool)) if r.source == 'predicted']
        summary = template_summary(predicted_records, dict(zip(('NC', 'MCI', 'AD'), map(float, ps[i]))))
        summaries.append({'subject_id': str(subject), **summary})
    write_json(output_path / 'summaries.json', summaries)
    return {'n_subjects': len(ids), 'n_withheld': int((~masks.any(1)).sum())}
