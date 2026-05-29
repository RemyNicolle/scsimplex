r"""Capture-bias calibration across datasets."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.sparse as sp

try:  # pragma: no cover - optional dependency fallback for local execution
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    from typing import Any

    class _AnnDataModule:
        AnnData = Any

    ad = _AnnDataModule()  # type: ignore[assignment]


def _to_dense_float(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def calibrate_capture_bias(
    adatas: List[ad.AnnData],
    anchor_cluster_obs_key: str,
    anchor_cluster_name: str,
    layer: Optional[str] = None,
) -> list[np.ndarray]:
    r"""Calibrate multiplicative capture-bias vectors from an anchor cell population.

    Let :math:`g_b` denote the geometric mean simplex for the anchor cluster in batch `b`.
    The global reference is computed as the geometric mean across batches:

    .. math::
        g_{ref,g} = \\exp\\left(\\frac{1}{B}\\sum_{b=1}^{B} \\log g_{b,g}\\right)

    The batch-specific multiplicative bias vector is then:

    .. math::
        \\beta_{b,g} = \\frac{g_{ref,g}}{g_{b,g}}.

    Args:
        adatas: Datasets to calibrate. Each dataset receives its own bias vector in `.uns`.
        anchor_cluster_obs_key: Observation column containing cluster labels.
        anchor_cluster_name: Anchor label shared across datasets.
        layer: Optional input layer key.

    Returns:
        The list of calibrated bias vectors, in the same order as `adatas`.
    """

    if len(adatas) == 0:
        return []

    dataset_geometric_means: list[np.ndarray] = []
    anchor_sizes: list[int] = []

    for idx, adata in enumerate(adatas):
        if anchor_cluster_obs_key not in adata.obs:
            raise KeyError(f"Key '{anchor_cluster_obs_key}' not found in adata.obs for dataset index {idx}.")

        mask = np.asarray(adata.obs[anchor_cluster_obs_key] == anchor_cluster_name)
        if not np.any(mask):
            raise ValueError(f"Anchor cluster '{anchor_cluster_name}' not found in dataset index {idx}.")

        matrix = adata.layers[layer] if layer is not None else adata.X
        anchor = _to_dense_float(matrix[mask])
        if anchor.ndim != 2:
            raise ValueError("Anchor slice must be two-dimensional.")

        anchor = anchor + 1e-12
        anchor /= anchor.sum(axis=1, keepdims=True)
        geom_mean = np.exp(np.log(anchor).mean(axis=0))
        geom_mean /= geom_mean.sum()
        dataset_geometric_means.append(geom_mean)
        anchor_sizes.append(anchor.shape[0])

    stacked = np.vstack(dataset_geometric_means)
    global_reference = np.exp(np.log(stacked + 1e-12).mean(axis=0))
    global_reference /= global_reference.sum()

    betas: list[np.ndarray] = []
    eps = 1e-12
    for idx, (adata, dataset_mean) in enumerate(zip(adatas, dataset_geometric_means)):
        beta = (global_reference + eps) / (dataset_mean + eps)
        beta /= np.exp(np.mean(np.log(beta)))
        adata.uns["capture_bias_beta"] = beta
        if hasattr(adata, "var_names"):
            adata.uns["capture_bias_gene_names"] = np.asarray(adata.var_names.astype(str))
        adata.uns["capture_bias_anchor_size"] = int(anchor_sizes[idx])
        adata.uns["capture_bias_global_reference"] = global_reference
        betas.append(beta)

    return betas
