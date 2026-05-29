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

    where :math:`k'` is the closest distinct cluster under multinomial log-likelihood.

    The test statistic is

    .. math::
        \\Lambda = 2 \\left[\\max_{\\lambda \\in [0,1]} \\log L_1 - \\log L_0 \\right],

    with a one degree-of-freedom chi-squared calibration.

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

    matrix = adata.layers[layer] if layer is not None else adata.X
    counts = _to_dense_float(matrix)
    clusters = np.asarray(adata.obs[cluster_key].astype(str))
    unique_clusters = np.unique(clusters)

    if unique_clusters.size < 2:
        adata.obs["multiplet_score"] = np.zeros(counts.shape[0], dtype=float)
        adata.obs["multiplet_lrt"] = np.zeros(counts.shape[0], dtype=float)
        adata.obs["multiplet_p_value"] = np.ones(counts.shape[0], dtype=float)
        adata.obs["is_multiplet"] = np.zeros(counts.shape[0], dtype=bool)
        return None

    eps = 1e-12
    profiles = np.zeros((unique_clusters.size, counts.shape[1]), dtype=float)
    for idx, cluster in enumerate(unique_clusters):
        cluster_counts = counts[clusters == cluster].sum(axis=0) + eps
        profiles[idx] = cluster_counts / cluster_counts.sum()

    log_profiles = np.log(profiles + eps)
    likelihood = counts @ log_profiles.T
    cluster_to_index = {label: i for i, label in enumerate(unique_clusters)}

    lrt = np.zeros(counts.shape[0], dtype=float)
    p_values = np.ones(counts.shape[0], dtype=float)
    multiplet_probability = np.zeros(counts.shape[0], dtype=float)

    for cell_idx in range(counts.shape[0]):
        current_label = clusters[cell_idx]
        current_idx = cluster_to_index[current_label]
        alt_scores = likelihood[cell_idx].copy()
        alt_scores[current_idx] = -np.inf
        alt_idx = int(np.argmax(alt_scores))

        theta0 = profiles[current_idx]
        theta1 = profiles[alt_idx]
        cell_counts = counts[cell_idx]

        ll_null = float(np.dot(cell_counts, np.log(theta0 + eps)))

        def objective(lam: float) -> float:
            mixture = lam * theta0 + (1.0 - lam) * theta1
            return -float(np.dot(cell_counts, np.log(mixture + eps)))

        optimum = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
        ll_alt = -float(optimum.fun)
        stat = max(0.0, 2.0 * (ll_alt - ll_null))

        lrt[cell_idx] = stat
        p_value = float(chi2.sf(stat, df=1))
        p_values[cell_idx] = p_value
        multiplet_probability[cell_idx] = 1.0 - p_value

    adata.obs["multiplet_lrt"] = lrt
    adata.obs["multiplet_p_value"] = p_values
    adata.obs["multiplet_score"] = multiplet_probability
    adata.obs["is_multiplet"] = p_values < alpha
    return None
