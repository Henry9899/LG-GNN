# Validation record

Executed locally on 8 November 2026 using Windows, CPython 3.13.13, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, transformers 5.22.2, NumPy 2.5.4, and scikit-learn 1.9.1. Full installed dependency versions are in `requirements-tested.txt`.

## Automated tests

**38 tests passed** in 28.09 seconds on the final source revision before packaging:

```powershell
python -m pytest tests -q
```

Coverage includes atlas majority/tie handling, 26-connected adjacency, missing-region pooling, both-endpoint edge masking, zero degrees, missing-feature invariance and gradients, withholding, nonfinite concept-target exclusion, alignment arithmetic, category thresholds, numeric/laterality/source/ID checking, rejection of extra unsupported fields, bounded regeneration, conditional responsiveness, distinct detection/withholding counts, classification denominators, training-NC reference fitting, inverse-Jacobian volume arithmetic, bridge memory masks, categorical positives, frozen-LM gradient flow, retrieval partition restrictions, bounded/failed tool-call counts, temperature fitting, data overwrite protection, the complete image-to-graph forward path, nested fold disjointness, and label-free checkpoint inference.

The image test uses random ResNet50 weights with small artificial images. The language gradient test uses a tiny randomly initialized GPT-2 with no network access. Neither validates pretrained model behavior or clinical accuracy.

## CLI execution checks

- Generated 90 artificial subjects (30/class), six synthetic regions and 12 features/modality.
- Completed the configured three-outer/two-inner-fold smoke protocol, with two learning-rate candidates and two training epochs per fit: **15 graph fits** total.
- Saved one out-of-fold prediction per subject, fold-specific normalizers, temperature values, full split manifests, model states and training histories.
- Repeated the seeded graph run during development and obtained identical OOF probabilities and metrics. These are execution/determinism checks on artificial data, not biomedical results.
- Rendered the evidence example through the reporting CLI, preserving source tags, a not-assessed limitation, externally supplied probabilities and the measured modality-disagreement record.
- Ran `scripts/train_bridge.py` for one epoch on a saved synthetic outer fold with a local tiny random causal LM and tokenizer. The graph and language model stayed frozen and bridge training completed. No model weights are shipped.

The detailed local execution outputs are stored outside this source archive under the repository's `notes/lg-code-submission-validation/`. This includes JUnit XML, synthetic datasets, fold records, random test models, and checkpoints. These artifacts are excluded from the submission ZIP to avoid confusing execution fixtures with study data or learned study weights.

## Limits of validation

No ADNI data were processed, no original result was regenerated, and no external model weights were downloaded for validation. Full 224-pixel/15-slice end-to-end training, 166-region AAL3 coverage, full-size Llama/MedGemma execution, GPU memory/latency, the clinical-reader study, and injected-fault experiments were not validated. The numerical tables in the manuscript are unchanged.

The source archive includes its own `SHA256SUMS.json`; the adjacent `.zip.sha256` verifies the archive. Hash verification establishes file integrity, not experimental reproducibility.
