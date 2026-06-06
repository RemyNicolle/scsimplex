r"""Multinomial k-means for discrete micro-metacell aggregation."""

from __future__ import annotations

from numbers import Integral
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:  # pragma: no cover - optional dependency fallback for local execution
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    from typing import Any

    class _AnnDataModule:
        AnnData = Any

    ad = _AnnDataModule()  # type: ignore[assignment]


def _as_dense_counts(matrix: object) -> np.ndarray:
    """Convert a matrix-like object to a dense float array."""

    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _validate_counts(matrix: np.ndarray) -> None:
    """Validate the matrix required by the multinomial model."""

    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional count matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Count matrix contains non-finite values.")
    if np.any(matrix < 0):
        raise ValueError("Multinomial k-means requires non-negative raw counts.")


def multinomial_kmeans(
    adata: ad.AnnData,
    n_neighbors: int = 15,
    target_metacell_size: int = 5,
    layer: Optional[str] = None,
    copy: bool = False,
    max_iter: int = 30,
    tol: float = 1e-4,
    random_state: int = 42,
) -> Optional[ad.AnnData]:
    r"""Cluster cells by maximizing the exact multinomial log-likelihood.

    The optimization is performed by Lloyd's algorithm on the simplex:

    .. math::
        \log \mathcal{L}(\vec{n}_c \mid \vec{\theta}_k)
        = \sum_g n_{cg} \log \theta_{kg} + \text{const}(N_c)

    Args:
        adata: AnnData-like object containing a raw UMI count matrix.
        n_neighbors: Reserved for future locality-aware seeding; retained for API compatibility.
        target_metacell_size: Target average number of cells per cluster.
        layer: Layer key to use instead of `adata.X`.
        copy: If `True`, return a modified copy instead of mutating in place.
        max_iter: Maximum number of EM/Lloyd iterations.
        tol: Convergence threshold on the fraction of changed assignments.
        random_state: Seed used for centroid initialization.

    Returns:
        A modified `AnnData` object when `copy=True`, otherwise `None`.
    """

    del n_neighbors

    if isinstance(target_metacell_size, bool) or not isinstance(target_metacell_size, Integral):
        raise TypeError("target_metacell_size must be a positive integer.")
    if target_metacell_size <= 0:
        raise ValueError("target_metacell_size must be a positive integer.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be a positive integer.")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if not np.isfinite(tol) or tol < 0.0 or tol > 1.0:
        raise ValueError("tol must be a finite number between zero and one.")

    rng = np.random.default_rng(random_state)
    work = adata.copy() if copy else adata
    matrix = work.layers[layer] if layer is not None else work.X
    counts = _as_dense_counts(matrix)
    _validate_counts(counts)

    n_cells, n_genes = counts.shape
    if n_cells == 0 or n_genes == 0:
        raise ValueError("The count matrix must contain at least one cell and one gene.")

    k = max(1, n_cells // max(1, target_metacell_size))
    k = min(k, n_cells)

    seed_indices = rng.choice(n_cells, size=k, replace=False)
    centroids = counts[seed_indices].astype(float, copy=True)
    centroids += 1e-12
    centroids /= centroids.sum(axis=1, keepdims=True)

    assignments = np.full(n_cells, -1, dtype=np.int32)
    eps = 1e-12

    for _ in range(max_iter):
        log_centroids = np.log(centroids + eps)
        likelihood = counts @ log_centroids.T
        new_assignments = np.asarray(np.argmax(likelihood, axis=1), dtype=np.int32)

        changed_fraction = float(np.mean(new_assignments != assignments))
        assignments = new_assignments

        updated = np.zeros((k, n_genes), dtype=float)
        for cluster_id in range(k):
            mask = assignments == cluster_id
            if np.any(mask):
                updated[cluster_id] = counts[mask].sum(axis=0)
            else:
                updated[cluster_id] = counts[rng.integers(0, n_cells)]

        updated += eps
        updated /= updated.sum(axis=1, keepdims=True)
        centroids = updated

        if changed_fraction < tol:
            break

    work.obs["multinomial_kmeans_cluster"] = pd.Categorical(assignments.astype(str))
    work.uns["multinomial_kmeans_centroids"] = centroids
    work.uns["multinomial_kmeans_k"] = int(k)
    return work if copy else None
