# LG-GNN code submission

Runnable reference implementation accompanying **LG-GNN: Anatomy-Grounded MRI–PET Graph Learning with Measurement-Supervised Concepts and Evidence-Linked Summaries**.

**Provenance:** this package was reconstructed from the manuscript and review notes on 8 November 2026. The supplied project contained no original model implementation, trained weights, ADNI participant manifest, or saved experimental predictions. This is a new reference implementation, not the code that produced the manuscript's tables. Synthetic runs verify execution only. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact boundary.

## Install

Use Python 3.10 or newer in a fresh environment. From this directory:

```powershell
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -e ".[test,images,language]"
.venv/Scripts/python -m pytest -q
```

On Linux/macOS use `python3 -m venv .venv` and `.venv/bin/python`. The core graph/CV runner needs only `pip install -e .`; image and language dependencies are optional. The tested environment is recorded in [requirements-tested.txt](requirements-tested.txt). Install the appropriate CUDA-enabled PyTorch build separately for GPU work. Installation and the synthetic examples require no ADNI access or model downloads.

## Run the complete CPU example

After activating the environment:

```powershell
lg-gnn synthetic --output data/synthetic.npz
lg-gnn train --data data/synthetic.npz --config configs/smoke.json --output runs/smoke
lg-gnn predict --data data/synthetic.npz --fold runs/smoke/fold_0 --output runs/predictions
lg-gnn report --evidence examples/evidence.json --output runs/example_summary.json
```

Training performs three outer folds, two inner folds, two learning-rate candidates and two epochs per fit. It saves fold membership, training histories, fitted normalizers, temperature values, checkpoints, and out-of-fold predictions. Every subject has exactly one outer-test prediction. The `predict` command demonstrates checkpoint loading on the same artificial data; its outputs are not a held-out evaluation. The example data and probabilities are artificial and have no clinical interpretation. Output destinations must be new; existing runs are preserved.

## Train on your regional features

Prepare the NPZ and adjacent JSON described in [DATA_FORMAT.md](DATA_FORMAT.md), then run:

```powershell
lg-gnn train --data data/cohort.npz --config configs/reference.json --output runs/reference
```

The CLI trains the **regional graph on cached features from an external, frozen encoder**. It rejects other real-data feature origins to avoid letting cohort-trained feature extraction leak across folds. The dual-stream image model is also implemented, as a differentiable API in `lg_gnn.backbone.ImageLGGNN`, for a separate end-to-end training integration. The CLI does not train that backbone. This distinction prevents cached-feature results from being mistaken for the manuscript's complete image-training experiment.

Change `device` to `cuda` in a copied configuration for GPU execution. The reference configuration uses the manuscript's graph width, eight heads, 20-position coverage threshold, 60 stage-2 epochs, and concept weight 0.5. Diffusion depth, ELAN depth, weight decay, and several loss details are reconstruction choices, listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Setting `concept_weight` to zero gives the atlas-only control; `strict_bottleneck` enables concept-only classification.

Age, sex, and ICV are used only to fit the training-NC measurement reference. They never enter the classification head. MMSE, CDR, diagnosis-derived text, and post-index variables are not accepted as feature fields. The labels are used for supervised classification, class stratification, and identifying training NC reference subjects.

## Language branch

An evidence-only deterministic summary runs without an LLM. To use a locally available, authorized Hugging Face causal LM:

```powershell
lg-gnn report --evidence examples/evidence.json --output runs/llm_summary.json --model PATH_TO_LOCAL_MODEL --device cuda
```

No weights or credentials are bundled or downloaded by these commands. Generated findings must copy canonical structured claims. The checker rejects altered values, laterality, modality, source tags, unsupported IDs, and added prose. It permits one regeneration attempt and then filters invalid findings. Probabilities and limitations are populated by the application. This constrained selector differs from unrestricted free-text generation in the manuscript, so its agreement rates are not comparable to the paper's reporting table.

The 32-query graph-to-language bridge and frozen causal-LM objectives are implemented in `lg_gnn.bridge`. To train stage 3 from one saved outer fold:

```powershell
python scripts/train_bridge.py --data data/cohort.npz --fold runs/reference/fold_0 --model PATH_TO_LOCAL_MODEL --output runs/bridge_fold_0 --device cuda
```

Only that fold's development subjects are used. The graph and LLM remain frozen; gradients flow through the LLM into the bridge. The script saves bridge weights and positive-set counts. This can require substantial GPU memory for an 8B model; it does not use quantization or distributed sharding. Full-size Llama/MedGemma training and generation have not been validated here. A tiny randomly initialized causal LM is used for the automated gradient test. The bridge API also supports visual-prefix generation; the `report` CLI is evidence-only.

## Package contents

| File/module | Purpose |
|---|---|
| `atlas.py` | Patch-majority assignments, regional pooling, 26-connected adjacency and homologue edges |
| `backbone.py` | Separate MRI/PET ResNet50–FPN/R-ELAN encoders and image-to-graph wrapper |
| `model.py` | Both-endpoint coverage masking, Gaussian affinities, diffusion, concepts and classification |
| `measurements.py` | Inverse-Jacobian volume arithmetic, SUVR, training-NC standardization |
| `experiment.py` | Nested CV, training-only transforms, held-out calibration, checkpoints and OOF artifacts |
| `inference.py` | Label-free checkpoint inference and summaries containing predicted estimates only |
| `bridge.py` | Query transformer, multiple-positive alignment, frozen-LM next-token objective |
| `reporting.py` | Source-tagged evidence, canonical claims, checker and bounded regeneration |
| `agent.py` | Training-only retrieval and bounded tool-policy execution |
| `evaluation.py` | Classification/calibration, paired BA bootstrap, conditional responsiveness and fault counts |
| `tests/` | Numerical, leakage, source-consistency, mask and gradient checks |

The agent module supplies infrastructure for four named tools and a six-call budget. An external controller and case-specific tool callables are required; no trained adaptive agent or fault-injection experiment is represented as completed.

## Submission artifacts

[VALIDATION.md](VALIDATION.md) records actual checks. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps code to the manuscript and lists unavailable study artifacts. [DATA_FORMAT.md](DATA_FORMAT.md) defines the data contract. [THIRD_PARTY.md](THIRD_PARTY.md) documents external dependencies and access requirements. `scripts/package_submission.py` creates a source-only ZIP with per-file SHA-256 hashes and excludes environments, datasets, caches, and model weights.
