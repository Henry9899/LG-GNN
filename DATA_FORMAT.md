# Data contract

Use `cohort.npz` with `cohort.json` beside it. NPZ loading always uses `allow_pickle=False`. Each row is one baseline subject with paired MRI/PET; repeated visits and duplicate subject IDs are rejected. Label order is fixed: **0=NC, 1=MCI, 2=AD**. MRI precedes PET in every modality axis.

| NPZ key | Shape | Meaning |
|---|---|---|
| `subject_ids` | `[N]` Unicode/string | Unique subject IDs; use deidentified IDs |
| `labels` | `[N]` integers | NC/MCI/AD class index |
| `descriptors` | `[N,R,2,D]` float32 | Atlas-pooled MRI/PET encoder features; no biomarkers or clinical scores |
| `counts` | `[N,R]` integers | Assigned feature-map positions across all selected slices |
| `measurements` | `[N,R,2]` floats | Raw regional MRI GM/ICV volume fractions and PET SUVR, not precomputed cohort-wide z-scores |
| `qc` | `[N,R,2]` bool | Independent measurement passed QC |
| `covariates` | `[N,3]` floats | Pre-index age, binary sex code, ICV in mm³; declare sex coding in provenance |
| `adjacency` | `[R,R]` nonnegative floats | Anatomical contacts and homologue edges; no diagonal required |

Invalid measurements may be NaN and must fail QC. Covered descriptors must be finite. Counts below the configured threshold are masked before any learned feature use. No-covered-region subjects receive NaN class logits, explicit withheld predictions, and contribute to coverage denominators. Normalizers are fitted to NC subjects in each model's training partition and transform validation, calibration and test subjects without refitting. A regional reference with insufficient NC observations, rank-deficient regression, or zero residual variance produces missing concept targets, never artificial zero z-scores.

Minimal metadata:

```json
{
  "regions": [
    {"id": 123, "name": "Atlas_region_name", "laterality": "L"},
    {"id": 124, "name": "Atlas_region_name", "laterality": "R"}
  ],
  "provenance": {
    "synthetic": false,
    "feature_origin": "external_frozen",
    "encoder_checkpoint_sha256": "RECORD_THE_ACTUAL_CHECKSUM",
    "encoder_training_data": "DOCUMENT_EXTERNAL_TRAINING_SOURCE",
    "atlas_version": "DOCUMENT_ACTUAL_ATLAS_VERSION",
    "atlas_sha256": "RECORD_THE_ACTUAL_CHECKSUM",
    "slice_indices": ["REPLACE_WITH_ACTUAL_INDICES"],
    "registration": "DOCUMENT_TRANSFORMS_AND_JACOBIAN_CONVENTIONS",
    "sex_code": "DOCUMENT_BINARY_CODING",
    "measurement_protocol": "DOCUMENT_SEGMENTATION_PVC_REFERENCE_MASK_AND_ICV_HANDLING"
  }
}
```

The example IDs are illustrative, not an AAL3 lookup table. Obtain the actual ordered atlas manifest under its terms. Metadata fields beyond `feature_origin` are retained as provenance but are not scientifically verified by the loader. The external-frozen check is a declared provenance constraint; it cannot detect falsely described encoders or illicit variables hidden inside descriptor arrays.

## Image API

`ImageLGGNN` accepts matched MRI and PET tensors `[B,S,C,H,W]` (C=1 or 3), integer atlas assignments `[B,S,H/4,W/4]`, and an anatomical adjacency `[R,R]`. The manuscript uses S=15, H=W=224, D=256 and R=166. `majority_assignment` maps ordered, potentially nonconsecutive atlas IDs to zero-based graph indices, with background and non-majority patches set to -1. Tied patches are unassigned. `pool_regions` also exposes the pooled descriptors for cache creation with an external frozen encoder.

Inputs must already be spatially aligned, intensity prepared, and accompanied by verified atlas/slice transforms. Joint image/label spatial augmentation belongs upstream; no spatial augmentation is performed by this package. ImageNet initialization is opt-in (`pretrained=True`) and may download torchvision weights. It does not reproduce an unavailable author checkpoint.

## Measurement API

`regional_measurements` consumes registered arrays and an **absolute inverse-map Jacobian in the same physical coordinate convention**. It multiplies warped native GM probability by this determinant and template voxel volume, then divides by ICV. PET SUVR is regional uptake divided by mean cerebellar-mask uptake. Registration, tissue segmentation, Jacobian calculation, partial-volume correction, reference-mask creation, and independent QC are upstream inputs. This package does not implement FSL or validate their outputs.

`NCNormalizer` regresses MRI volume fractions on intercept, age, sex and ICV; PET on intercept, age and sex. It uses NC-training residual standard deviation with residual degrees of freedom. Simultaneous MRI ICV division and regression follow the written methods but require confirmation against the original experiment. This implementation's explicit convention is not evidence that the original analysis used it.

## Outputs

For `lg-gnn predict`, the NPZ needs only `subject_ids`, `descriptors`, `counts`, and `adjacency`, plus the adjacent JSON region manifest. Region metadata must exactly match the saved model. Labels, measurement targets and covariates are not read. Class probabilities use the checkpoint temperature; summaries contain predicted estimates only. A checkpoint inference run is not an out-of-fold evaluation unless the caller supplies the correct held-out subjects for that fold.

Every outer fold saves `subjects.json`, `normalizer.json`, `selection.json`, `metrics.json`, `model.pt`, and `test_concepts.npz`. The latter retains source-separated measured/predicted z-scores, QC and coverage. Top-level `out_of_fold.npz` contains subject IDs, labels, calibrated probabilities, predicted concepts, and fold IDs. A withheld subject has NaN probabilities. JSON metrics use `null` for undefined quantities; three-way BA/F1 are undefined if a retained class has no support.

The calibration partition is reserved before inner model selection. Final graph fitting and all fitted transforms use development subjects only; calibration subjects fit temperature only. Outer-test data are never used for tuning or normalization. These newly generated deterministic splits are not the unavailable original DMA-GNN splits.
