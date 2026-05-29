r"""Bayesian simplex imputation and CLR transforms."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

try:  # pragma: no cover - optional dependency fallback for local execution
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    from typing import Any

    class _AnnDataModule:
        AnnData = Any

    ad = _AnnDataModule()  # type: ignore[assignment]

try:  # pragma: no cover
    import dask.array as da
except ModuleNotFoundError:  # pragma: no cover
    da = None  # type: ignore[assignment]


def _is_dask_array(matrix: object) -> bool:
    return da is not None and isinstance(matrix, da.Array)


def _to_dense_float(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def bayesian_impute_pseudocounts(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    alpha_prior: float = 1.0,
    out_layer: str = "X_imputed",
) -> None:
    r"""Bayesian posterior mean imputation on the simplex.

    The prior is a dataset-wide global background composition:

    .. math::
        p_g = \\frac{\\sum_c n_{cg}}{\\sum_{c,g} n_{cg}}

    The Dirichlet prior is

    .. math::
        \\alpha_g = \\alpha_{\\mathrm{prior}} p_g

    and the posterior mean for each cell is

    .. math::
        \\theta_{cg} = \\frac{n_{cg} + \\alpha_g}{N_c + \\sum_g \\alpha_g}.

    Args:
        adata: AnnData-like object.
        layer: Optional input layer key.
        alpha_prior: Total prior mass scaling the global background.
        out_layer: Output layer key used to store the imputed simplex.
    """

    matrix = adata.layers[layer] if layer is not None else adata.X

    if _is_dask_array(matrix):
        total_per_gene = matrix.sum(axis=0)
        total_depth = matrix.sum()
        total_depth_value = float(total_depth.compute())
        if total_depth_value == 0.0:
            global_background = da.full((matrix.shape[1],), 1.0 / matrix.shape[1], dtype=float)
        else:
            global_background = total_per_gene / total_depth
            global_background = global_background + 1e-12
            global_background = global_background / global_background.sum()
        alpha_vector = global_background * float(alpha_prior)
        cell_depth = matrix.sum(axis=1, keepdims=True)
        sum_alpha = alpha_vector.sum()
        adata.layers[out_layer] = (matrix + alpha_vector) / (cell_depth + sum_alpha)
        return None

    dense = _to_dense_float(matrix)
    if dense.ndim != 2:
        raise ValueError("Expected a two-dimensional count matrix.")

    total_global = float(dense.sum())
    if total_global == 0.0:
        global_background = np.full(dense.shape[1], 1.0 / dense.shape[1], dtype=float)
    else:
        global_background = dense.sum(axis=0) / total_global
        global_background = global_background + 1e-12
        global_background = global_background / global_background.sum()

    alpha_vector = global_background * float(alpha_prior)
    sum_alpha = float(alpha_vector.sum())
    cell_depth = dense.sum(axis=1, keepdims=True)
    adata.layers[out_layer] = (dense + alpha_vector) / (cell_depth + sum_alpha)
    return None


def clr_transform(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    out_layer: str = "X_clr",
    alpha_prior: float = 1.0,
) -> None:
    r"""Apply a centered log-ratio transform to a simplex-valued matrix.

    If a Bayesian-imputed layer does not exist, it is created automatically.

    .. math::
        \\mathrm{CLR}(\\theta_c) = \\log(\\theta_c) - \\frac{1}{G}\\sum_g \\log(\\theta_{cg})

    Args:
        adata: AnnData-like object.
        layer: Optional input layer key.
        out_layer: Output layer key used to store the CLR coordinates.
        alpha_prior: Prior mass forwarded to Bayesian imputation when needed.
    """

    imputed_layer = "X_imputed"
    if imputed_layer not in adata.layers:
        bayesian_impute_pseudocounts(adata, layer=layer, alpha_prior=alpha_prior, out_layer=imputed_layer)

    imputed_matrix = adata.layers[imputed_layer]
    if _is_dask_array(imputed_matrix):
        log_simplex = da.log(imputed_matrix)
        adata.layers[out_layer] = log_simplex - log_simplex.mean(axis=1, keepdims=True)
        return None

    simplex = _to_dense_float(imputed_matrix)
    if np.any(simplex <= 0):
        raise ValueError("CLR requires strictly positive simplex values.")
    log_simplex = np.log(simplex)
    adata.layers[out_layer] = log_simplex - log_simplex.mean(axis=1, keepdims=True)
    return None
