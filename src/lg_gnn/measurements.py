"""Measurement arithmetic, assuming independent upstream registration/QC."""
import numpy as np


def regional_measurements(gm_in_template, inverse_jacobian, pet, atlas,
                          region_ids, cerebellar_mask, voxel_volume_mm3, icv_mm3):
    """MRI volume fraction and PET SUVR [R,2].

    gm_in_template is rho_N(T^-1(x)); inverse_jacobian must use the same
    physical-coordinate/affine convention. This function does not estimate it.
    PET must already have the intended partial-volume correction applied.
    """
    arrays = [np.asarray(x) for x in (gm_in_template, inverse_jacobian, pet, atlas, cerebellar_mask)]
    gm, jac, uptake, labels, cerebellum = arrays
    if len({x.shape for x in arrays}) != 1 or gm.ndim != 3:
        raise ValueError("All maps must share one registered 3D grid")
    if not np.isfinite([voxel_volume_mm3, icv_mm3]).all() or min(voxel_volume_mm3, icv_mm3) <= 0:
        raise ValueError("Positive physical voxel volume and ICV required")
    if (jac < 0).any():
        raise ValueError("Supply absolute inverse-map Jacobian determinants")
    ref = uptake[cerebellum.astype(bool)]
    if not ref.size or not np.isfinite(ref).all() or ref.mean() <= 0:
        raise ValueError("Invalid PET reference mask/uptake")
    result = np.full((len(region_ids), 2), np.nan)
    for i, region in enumerate(region_ids):
        mask = labels == region
        if not mask.any():
            continue
        if np.isfinite(gm[mask]).all() and np.isfinite(jac[mask]).all():
            result[i, 0] = (gm[mask] * jac[mask]).sum() * voxel_volume_mm3 / icv_mm3
        if np.isfinite(uptake[mask]).all():
            result[i, 1] = uptake[mask].mean() / ref.mean()
    return result


class NCNormalizer:
    """Age/sex adjustment, plus ICV regression for MRI, fitted on NC training only.

    ICV regression is in addition to prior MRI ICV division, following the
    written manuscript; the original measurement configuration is unavailable.
    """
    def fit(self, values, covariates, nc, qc):
        values, covariates = np.asarray(values), np.asarray(covariates)
        if covariates.shape != (len(values), 3) or not np.isfinite(covariates).all():
            raise ValueError("Covariates must be finite [age, sex, ICV]")
        self.mean = covariates[nc].mean(0)
        self.scale = covariates[nc].std(0).clip(1e-8)
        self.beta = np.zeros((*values.shape[1:], 4))
        self.std = np.ones(values.shape[1:])
        self.valid = np.zeros(values.shape[1:], dtype=bool)
        for r in range(values.shape[1]):
            for modality in range(2):
                x = self._design(covariates, modality)
                use = nc & qc[:, r, modality] & np.isfinite(values[:, r, modality])
                if use.sum() <= x.shape[1] or np.linalg.matrix_rank(x[use]) < x.shape[1]:
                    continue
                beta = np.linalg.lstsq(x[use], values[use, r, modality], rcond=None)[0]
                residual = values[use, r, modality] - x[use] @ beta
                sigma = np.sqrt((residual ** 2).sum() / (use.sum() - x.shape[1]))
                if sigma > 1e-10:
                    self.beta[r, modality, :len(beta)] = beta
                    self.std[r, modality] = sigma
                    self.valid[r, modality] = True
        return self

    def _design(self, covariates, modality):
        standardized = (covariates - self.mean) / self.scale
        return np.column_stack([np.ones(len(covariates)), standardized[:, :3 if modality == 0 else 2]])

    def transform(self, values, covariates):
        result = np.full_like(values, np.nan, dtype=np.float64)
        for modality in range(2):
            x = self._design(covariates, modality)
            estimate = x @ self.beta[:, modality, :x.shape[1]].T
            result[:, :, modality] = (values[:, :, modality] - estimate) / self.std[:, modality]
        return np.where(self.valid, result, np.nan)

    def state_dict(self):
        return {name: getattr(self, name).tolist() for name in ('mean', 'scale', 'beta', 'std', 'valid')}
