r"""Likelihood-ratio multiplet detection under multinomial mixtures."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

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


def _negative_mixture_log_likelihood(
    lam: float,
    theta0: np.ndarray,
    theta1: np.ndarray,
    cell_counts: np.ndarray,
    eps: float,
) -> float:
    mixture = lam * theta0 + (1.0 - lam) * theta1
    return -float(np.dot(cell_counts, np.log(mixture + eps)))


def detect_multiplets(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    cluster_key: str = "multinomial_kmeans_cluster",
    alpha: float = 0.05,
) -> None:
    r"""Detect multiplets using a multinomial likelihood-ratio test.

    For each cell assigned to cluster :math:`k`, the null model is the cluster profile
    :math:`\\theta_k`. The alternative model is a two-component mixture:

    .. math::
        \\theta(\\lambda) = \\lambda \\theta_k + (1-\\lambda)\\theta_{k'}

    where :math:`k'` ranges over all distinct clusters.

    The test statistic is

    .. math::
        \\Lambda = 2 \\left[\\max_{\\lambda \\in [0,1]} \\log L_1 - \\log L_0 \\right],

    Because the null lies on the boundary of the mixture-weight parameter, each fixed
    alternative uses the one-sided asymptotic calibration
    :math:`0.5\chi^2_1`. A Bonferroni correction accounts for selecting the best alternative
    cluster.

    Args:
        adata: AnnData-like object.
        layer: Optional input layer key.
        cluster_key: Observation key containing cluster assignments.
        alpha: Significance level used for the multiplet classification.

    Returns:
        None. Results are stored in `adata.obs`.
    """

    if cluster_key not in adata.obs:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs.")
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be a finite number strictly between zero and one.")

    matrix = adata.layers[layer] if layer is not None else adata.X
    counts = _to_dense_float(matrix)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("The count matrix must contain at least one cell and one gene.")
    if not np.isfinite(counts).all():
        raise ValueError("The count matrix contains non-finite values.")
    if np.any(counts < 0.0):
        raise ValueError("Multiplet detection requires non-negative counts.")
    clusters = np.asarray(adata.obs[cluster_key].astype(str))
    if clusters.shape[0] != counts.shape[0]:
        raise ValueError("Cluster labels must match the number of count-matrix rows.")
    unique_clusters = np.unique(clusters)

    if unique_clusters.size < 2:
        adata.obs["multiplet_score"] = np.zeros(counts.shape[0], dtype=float)
        adata.obs["multiplet_lrt"] = np.zeros(counts.shape[0], dtype=float)
        adata.obs["multiplet_p_value"] = np.ones(counts.shape[0], dtype=float)
        adata.obs["is_multiplet"] = np.zeros(counts.shape[0], dtype=bool)
        return None

    eps = 1e-12
    profiles = np.zeros((unique_clusters.size, counts.shape[1]), dtype=float)
    cluster_totals = np.zeros_like(profiles)
    cluster_sizes = np.zeros(unique_clusters.size, dtype=int)
    for idx, cluster in enumerate(unique_clusters):
        mask = clusters == cluster
        cluster_sizes[idx] = int(mask.sum())
        cluster_totals[idx] = counts[mask].sum(axis=0)
        cluster_counts = cluster_totals[idx] + eps
        profiles[idx] = cluster_counts / cluster_counts.sum()

    cluster_to_index = {label: i for i, label in enumerate(unique_clusters)}

    lrt = np.zeros(counts.shape[0], dtype=float)
    p_values = np.ones(counts.shape[0], dtype=float)
    multiplet_probability = np.zeros(counts.shape[0], dtype=float)

    for cell_idx in range(counts.shape[0]):
        current_label = clusters[cell_idx]
        current_idx = cluster_to_index[current_label]
        cell_counts = counts[cell_idx]
        if cluster_sizes[current_idx] > 1:
            null_counts = cluster_totals[current_idx] - cell_counts + eps
            theta0 = null_counts / null_counts.sum()
        else:
            theta0 = profiles[current_idx]

        ll_null = float(np.dot(cell_counts, np.log(theta0 + eps)))
        best_stat = 0.0

        for alt_idx, theta1 in enumerate(profiles):
            if alt_idx == current_idx:
                continue

            optimum = minimize_scalar(
                _negative_mixture_log_likelihood,
                bounds=(0.0, 1.0),
                args=(theta0, theta1, cell_counts, eps),
                method="bounded",
            )
            ll_alt = -float(optimum.fun)
            best_stat = max(best_stat, 2.0 * (ll_alt - ll_null))

        stat = max(0.0, best_stat)
        lrt[cell_idx] = stat
        if stat <= 1e-12:
            p_value = 1.0
        else:
            one_sided_p = 0.5 * float(chi2.sf(stat, df=1))
            p_value = min(1.0, one_sided_p * (unique_clusters.size - 1))
        p_values[cell_idx] = p_value
        multiplet_probability[cell_idx] = 1.0 - p_value

    adata.obs["multiplet_lrt"] = lrt
    adata.obs["multiplet_p_value"] = p_values
    adata.obs["multiplet_score"] = multiplet_probability
    adata.obs["is_multiplet"] = p_values < alpha
    return None
