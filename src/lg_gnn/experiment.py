"""Nested subject-level CV with isolated model-selection and calibration sets."""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import Cohort
from .evaluation import classification_metrics, fit_temperature
from .losses import graph_objective
from .measurements import NCNormalizer
from .model import LGGNN


@dataclass
class ExperimentConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    calibration_fraction: float = 0.2
    epochs: int = 60
    batch_size: int = 16
    learning_rates: tuple = (0.0001,)
    hidden: int = 256
    heads: int = 8
    diffusion_steps: int = 3
    alpha: float = 0.5
    concept_weight: float = 0.5
    focal_weight: float = 0.5
    contrastive_weight: float = 0.3
    weight_decay: float = 0.01
    coverage_threshold: int = 20
    strict_bottleneck: bool = False
    seed: int = 17
    device: str = 'cpu'
    threads: int = 1

    def validate(self):
        if min(self.outer_folds, self.inner_folds) < 2 or min(self.epochs, self.batch_size, self.threads) < 1:
            raise ValueError("Need >=2 folds and positive epochs, batch size, threads")
        if not 0 < self.calibration_fraction < 1 or not self.learning_rates or min(self.learning_rates) <= 0:
            raise ValueError("Invalid calibration fraction or learning rates")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def fit_training(data, indices, config, lr, seed):
    """All fitted transforms see these training indices only."""
    seed_everything(seed)
    normalizer = NCNormalizer().fit(data.measurements[indices], data.covariates[indices],
                                    data.labels[indices] == 0, data.qc[indices])
    observed = data.counts[indices] >= config.coverage_threshold
    feature_values = data.descriptors[indices][observed]
    if not len(feature_values) or not np.isfinite(feature_values).all():
        raise ValueError("Training requires finite covered regional descriptors")
    mean = feature_values.mean(0)
    scale = feature_values.std(0).clip(1e-6)
    kwargs = {'regions': len(data.regions), 'feature_dim': data.descriptors.shape[-1],
              'hidden': config.hidden, 'heads': config.heads, 'steps': config.diffusion_steps,
              'alpha': config.alpha, 'coverage_threshold': config.coverage_threshold,
              'strict_bottleneck': config.strict_bottleneck}
    model = LGGNN(**kwargs).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    support = np.bincount(data.labels[indices][observed.any(1)], minlength=3)
    if (support == 0).any():
        raise ValueError("Every training class requires at least one covered subject")
    weights = torch.tensor(support.sum() / (3 * support), dtype=torch.float32, device=config.device)
    adjacency = torch.as_tensor(data.adjacency, dtype=torch.float32, device=config.device)
    z = normalizer.transform(data.measurements[indices], data.covariates[indices])
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    for epoch in range(config.epochs):
        epoch_loss, batches = 0., 0
        order = rng.permutation(len(indices))
        for start in range(0, len(order), config.batch_size):
            pos = order[start:start + config.batch_size]
            batch = indices[pos]
            descriptors = (data.descriptors[batch] - mean) / scale
            x = torch.as_tensor(descriptors, dtype=torch.float32, device=config.device)
            counts = torch.as_tensor(data.counts[batch], device=config.device)
            output = model(x, counts, adjacency)
            targets = torch.as_tensor(z[pos], dtype=torch.float32, device=config.device)
            qc = torch.as_tensor(data.qc[batch], device=config.device)
            labels = torch.as_tensor(data.labels[batch], dtype=torch.long, device=config.device)
            loss = graph_objective(output, labels, targets, qc, weights,
                 config.concept_weight, config.focal_weight, config.contrastive_weight)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        history.append({'epoch': epoch + 1, 'mean_batch_loss': epoch_loss / batches})
    return model, normalizer, mean, scale, kwargs, history


@torch.no_grad()
def infer(model, data, indices, mean, scale, config):
    model.eval()
    logits, concepts = [], []
    adjacency = torch.as_tensor(data.adjacency, dtype=torch.float32, device=config.device)
    for start in range(0, len(indices), config.batch_size):
        batch = indices[start:start + config.batch_size]
        x = torch.as_tensor((data.descriptors[batch] - mean) / scale,
                            dtype=torch.float32, device=config.device)
        counts = torch.as_tensor(data.counts[batch], device=config.device)
        result = model(x, counts, adjacency)
        logits.append(result.logits.cpu().numpy())
        concepts.append(result.concepts.cpu().numpy())
    return np.concatenate(logits), np.concatenate(concepts)


def probabilities(logits, temperature=1.):
    return torch.as_tensor(logits / temperature).softmax(-1).numpy()


def run_experiment(data_path, output_path, config):
    config.validate()
    torch.set_num_threads(config.threads)
    data = Cohort.load(data_path)
    if not data.provenance.get('synthetic') and data.provenance.get('feature_origin') != 'external_frozen':
        raise ValueError("CV CLI requires external_frozen descriptors; cohort-trained features require fold-specific end-to-end training")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    write_json(output_path / 'config.json', asdict(config))
    data_path = Path(data_path)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (data_path, data_path.with_suffix('.json'))}
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scikit-learn')},
        'input_sha256': hashes, 'provenance': data.provenance, 'deterministic_algorithms': True}
    write_json(output_path / 'environment.json', environment)
    splitter = StratifiedKFold(config.outer_folds, shuffle=True, random_state=config.seed)
    n = len(data.labels)
    oof = np.full((n, 3), np.nan)
    all_concepts = np.full((n, len(data.regions), 2), np.nan)
    fold_ids = np.full(n, -1)
    fold_results = []
    for fold, (outer_train, test) in enumerate(splitter.split(np.zeros(n), data.labels)):
        development, calibration = train_test_split(outer_train,
            test_size=config.calibration_fraction, stratify=data.labels[outer_train],
            random_state=config.seed + fold)
        inner = list(StratifiedKFold(config.inner_folds, shuffle=True,
                     random_state=config.seed + fold).split(development, data.labels[development]))
        manifest = {'outer_test': data.subject_ids[test].tolist(),
                    'calibration': data.subject_ids[calibration].tolist(),
                    'development': data.subject_ids[development].tolist(),
                    'inner': [{'train': data.subject_ids[development[a]].tolist(),
                               'validation': data.subject_ids[development[b]].tolist()} for a, b in inner]}
        fold_path = output_path / f'fold_{fold}'
        fold_path.mkdir()
        write_json(fold_path / 'subjects.json', manifest)
        candidates = []
        for lr in config.learning_rates:
            scores = []
            for inner_id, (a, b) in enumerate(inner):
                model, norm, mean, scale, _, _ = fit_training(data, development[a], config, lr,
                                                           config.seed + fold * 100 + inner_id)
                logits, _ = infer(model, data, development[b], mean, scale, config)
                score = classification_metrics(data.labels[development[b]], probabilities(logits))['balanced_accuracy']
                if score is None:
                    raise ValueError("Inner validation needs covered subjects in every class")
                scores.append(score)
            candidates.append({'learning_rate': lr, 'inner_balanced_accuracy': scores,
                               'mean': float(np.mean(scores))})
        best = max(candidates, key=lambda x: x['mean'])['learning_rate']
        model, norm, mean, scale, kwargs, history = fit_training(data, development, config, best,
                                                              config.seed + fold * 100 + 99)
        calibration_logits, _ = infer(model, data, calibration, mean, scale, config)
        valid_cal = np.isfinite(calibration_logits).all(1)
        temperature = fit_temperature(calibration_logits[valid_cal], data.labels[calibration][valid_cal])
        logits, concepts = infer(model, data, test, mean, scale, config)
        oof[test] = probabilities(logits, temperature)
        all_concepts[test] = concepts
        fold_ids[test] = fold
        metrics = classification_metrics(data.labels[test], oof[test])
        fold_results.append(metrics)
        write_json(fold_path / 'selection.json', {'candidates': candidates, 'selected_learning_rate': best,
                   'temperature': temperature, 'calibration_n_eligible': int(valid_cal.sum()), 'history': history})
        write_json(fold_path / 'normalizer.json', norm.state_dict())
        write_json(fold_path / 'metrics.json', metrics)
        torch.save({'state_dict': model.cpu().state_dict(), 'model_kwargs': kwargs,
                    'feature_mean': torch.as_tensor(mean), 'feature_scale': torch.as_tensor(scale),
                    'temperature': temperature, 'regions': data.regions,
                    'adjacency': torch.as_tensor(data.adjacency),
                    'feature_provenance': data.provenance}, fold_path / 'model.pt')
        np.savez_compressed(fold_path / 'test_concepts.npz', subject_ids=data.subject_ids[test],
                            measured_z=norm.transform(data.measurements[test], data.covariates[test]),
                            predicted_z=concepts, qc=data.qc[test], covered=data.counts[test] >= config.coverage_threshold)
        print(f'fold {fold + 1}/{config.outer_folds} finished', flush=True)
    if (fold_ids < 0).any():
        raise AssertionError("Missing out-of-fold predictions")
    np.savez_compressed(output_path / 'out_of_fold.npz', subject_ids=data.subject_ids,
        labels=data.labels, probabilities=oof, predicted_concepts=all_concepts, fold=fold_ids)
    metrics = classification_metrics(data.labels, oof)
    defined = [x['balanced_accuracy'] for x in fold_results if x['balanced_accuracy'] is not None]
    metrics['mean_fold_balanced_accuracy'] = float(np.mean(defined)) if defined else None
    metrics['fold_balanced_accuracy_std'] = float(np.std(defined, ddof=1)) if len(defined) > 1 else None
    metrics['synthetic'] = bool(data.provenance.get('synthetic', False))
    write_json(output_path / 'metrics.json', metrics)
    return metrics
