"""Stage 3 on a selected outer fold; graph and local causal LM stay frozen."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lg_gnn.bridge import FrozenLanguageAdapter, QueryBridge, categorical_positives
from lg_gnn.data import Cohort
from lg_gnn.experiment import seed_everything, write_json
from lg_gnn.measurements import NCNormalizer
from lg_gnn.model import LGGNN
from lg_gnn.reporting import build_records, canonical_finding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--fold', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Local, authorized causal-LM directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Positive epochs and batch size required')
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    torch.set_num_threads(1)
    data = Cohort.load(args.data)
    manifests = json.loads((args.fold / 'subjects.json').read_text(encoding='utf-8'))
    lookup = {str(s): i for i, s in enumerate(data.subject_ids)}
    indices = np.array([lookup[s] for s in manifests['development']])
    if set(manifests['development']) & (set(manifests['outer_test']) | set(manifests['calibration'])):
        raise ValueError('Overlapping training and held-out subjects')
    checkpoint = torch.load(args.fold / 'model.pt', map_location=args.device, weights_only=True)
    graph = LGGNN(**checkpoint['model_kwargs']).to(args.device)
    graph.load_state_dict(checkpoint['state_dict'])
    graph.requires_grad_(False).eval()
    norm = NCNormalizer()
    for key, value in json.loads((args.fold / 'normalizer.json').read_text(encoding='utf-8')).items():
        setattr(norm, key, np.array(value))
    mean = checkpoint['feature_mean'].cpu().numpy()
    scale = checkpoint['feature_scale'].cpu().numpy()
    covered = data.counts >= graph.coverage_threshold
    indices = indices[covered[indices].any(1)]
    z = norm.transform(data.measurements, data.covariates)
    valid = covered[..., None] & data.qc & np.isfinite(z)
    # Exclude cases with no valid measured targets from bridge training.
    indices = indices[valid[indices].any((1, 2))]
    if not len(indices):
        raise ValueError('No training subjects with valid measured targets')
    descriptions = {}
    for i in indices:
        records = build_records(data.regions, z[i], np.full_like(z[i], np.nan), covered[i], data.qc[i])
        descriptions[i] = '\n'.join(canonical_finding(r)['statement'] for r in records
                                    if r.source == 'measured' and r.quality == 'pass')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define an EOS token')
    llm = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).to(args.device)
    bridge = QueryBridge(graph_dim=checkpoint['model_kwargs']['hidden'],
                         llm_dim=llm.get_input_embeddings().weight.shape[1],
                         heads=checkpoint['model_kwargs']['heads']).to(args.device)
    adapter = FrozenLanguageAdapter(llm, bridge).to(args.device).train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=2e-4)
    adjacency = torch.as_tensor(data.adjacency, device=args.device)
    rng, history, positive_sizes = np.random.default_rng(args.seed), [], []
    for epoch in range(args.epochs):
        order, losses = rng.permutation(indices), []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            with torch.no_grad():
                output = graph(torch.as_tensor((data.descriptors[batch] - mean) / scale, device=args.device),
                               torch.as_tensor(data.counts[batch], device=args.device), adjacency)
            texts = [descriptions[i] + tokenizer.eos_token for i in batch]
            tokens = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(args.device)
            if tokens.input_ids.shape[1] > args.max_length:
                raise ValueError('Template exceeds --max-length; increase it rather than silently truncating evidence')
            positives = categorical_positives(torch.as_tensor(z[batch], device=args.device),
                                              torch.as_tensor(valid[batch], device=args.device))
            positive_sizes.extend(positives.sum(1).tolist())
            loss = adapter.objective(output.nodes, output.covered, tokens.input_ids, tokens.attention_mask, positives)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite bridge loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch + 1, 'mean_batch_loss': float(np.mean(losses))})
        print(f'bridge epoch {epoch + 1}/{args.epochs} finished', flush=True)
    torch.save(bridge.cpu().state_dict(), args.output / 'bridge.pt')
    write_json(args.output / 'training.json', {'seed': args.seed, 'epochs': args.epochs,
        'history': history, 'training_subjects': data.subject_ids[indices].tolist(),
        'positive_set_sizes': positive_sizes, 'local_model': args.model,
        'graph_frozen': True, 'llm_frozen': True, 'queries': 32,
        'graph_dim': checkpoint['model_kwargs']['hidden'],
        'llm_dim': llm.get_input_embeddings().weight.shape[1], 'layers': 2,
        'heads': checkpoint['model_kwargs']['heads'],
        'note': 'Reference bridge training; does not reproduce manuscript language metrics.'})


if __name__ == '__main__':
    main()
