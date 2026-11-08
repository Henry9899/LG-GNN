# Implementation and result provenance

This code was created from `../lg_gnn_draft.tex` and `../lg_gnn_draft_review.md`, without modifying either file. It was not recovered from the authors' experiment environment. It must be described as a **reference reconstruction** in a submission unless replaced or reconciled with the original executed implementation.

## Implemented correspondences

| Manuscript element | Reference implementation | Scope |
|---|---|---|
| Atlas pooling and 20-position coverage | `atlas.pool_regions`, `model.LGGNN` | Empty-region zero mean, learned missing/identity embeddings |
| Both-endpoint graph masking | `model.normalized_operator` | Self-loops; safe degree inverse; rows receive, columns send |
| Eight Gaussian affinity heads | `model.LGGNN` | Separate query/key projections; no symmetry claim |
| Concept supervision | `losses.masked_concept_loss` | Covered, QC-passing, finite target indices only |
| Embedding classifier and strict bottleneck | `model.LGGNN` | Auxiliary-concept main branch; bottleneck flag |
| Independent measurement targets | `measurements` | Array arithmetic and training-NC regression; preprocessing supplied upstream |
| Graph-to-language bridge | `bridge`, `scripts/train_bridge.py` | 32 queries, categorical multiple positives, frozen-LM prefix likelihood |
| Source-tagged evidence | `reporting` | Strict canonical claim selection and verification |
| Nested CV and temperature scaling | `experiment`, `evaluation` | New subject splits, isolated calibration, persisted OOF predictions |
| Bounded tools | `agent` | Policy executor and training-only retrieval infrastructure |
| Corrected responsiveness denominator | `evaluation.responsiveness` | Conditional on actual predicted-category change; omission not success |
| Fault detection/withholding accounting | `evaluation.fault_metrics` | Separate events with complete joint counts |

## Explicit reconstruction choices

- Gaussian affinity is `exp(-||Wq h_i - Wk h_j||² / (2 softplus(sigma)²))`. The operator is fixed from initial nodes within each forward pass. Each head maintains a full-width state for three residual diffusion steps; final states are averaged. No extra state activation is inserted. Affinity scaling, depth, and exact original state updates were not supplied.
- FPN has four 256-channel scales, softmax-weighted bilinear fusion at stride four, channel/spatial attention and two sequential ELAN blocks. Attention uses a reduction factor of 16. This reconstructs the described architecture, not an exact unpublished layer specification.
- Stage 2 uses weighted CE plus weighted focal loss (gamma 2), supervised contrastive pooled embeddings, and auxiliary concepts, with AdamW decay. The original DMA contrastive sampling and regularization conventions are unknown. AdamW decay 0.01 is a declared default, not a verified paper hyperparameter. Loss magnitudes are not guaranteed to match the executed DMA objective.
- Feature centering/scaling is fitted on covered development descriptors. The CLI uses cached external-frozen features; it does not optimize image backbones within nested CV.
- NC-reference residual scaling uses regression degrees of freedom. Both ICV division and MRI regression adjustment are explicit. The original measurement convention and sensitivity remain unresolved.
- Stage 3 uses a two-layer PyTorch transformer decoder with cross-attention as the query transformer. This is not the exact BLIP-2 pretrained Q-Former. Parameter count depends on graph/LM width; no 14M-parameter or memory claim is made. Template embedding pooling includes non-padding special tokens. The causal-LM loss is averaged over nonignored target tokens. Positive signatures contain every region/modality category, with all invalid entries represented as absent.
- The checker only validates canonical structured claims and their exact rendered text. It does not adjudicate arbitrary prose, image correctness, biological truth, or clinical utility. This stronger constraint changes the reporting task and cannot be used to substantiate the manuscript's 98.2% result.
- The calibration set is reserved before selection and is not reused for final graph training. This is one explicit leakage-resistant realization; the original split/readout configuration was not available.

## What cannot be reproduced from the supplied project

No original ADNI subject IDs, slice manifest, atlas transform, processed volumes, segmentation/QC results, pretrained experiment checkpoints, generation outputs, seed records, reader assessments, or per-fault outcomes were supplied. The reported 54 covered regions cannot be independently recovered from an illustrative atlas. No 1,384-subject experiment, original grid DMA baseline, external comparator replication, clinical-input comparison, joint-language stage-4 experiment, scanner transfer, or reader study has been executed here.

The paper's agent table still requires reconciliation of detection and withholding, and its saved responsiveness metric version remains unavailable. This package defines new, explicit implementations of those metrics; it does not retroactively validate or replace the reported values. Synthetic results are not copied into the manuscript.

For an archival reproduction of the paper, supply the actual authorized cohort/split manifests, complete preprocessing and atlas configurations, executed layer/loss definitions, weights, OOF predictions, prompts/checker version, and case-level reporting/fault records. The full review already records additional publication requirements. None of these missing artifacts is silently substituted with synthetic participant information.
