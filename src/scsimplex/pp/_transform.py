r"""Bayesian simplex imputation and CLR transforms."""

from __future__ import annotations

from typing import Literal, Optional, Union

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
except ImportError:  # pragma: no cover
    da = None  # type: ignore[assignment]


def _is_dask_array(matrix: object) -> bool:
    return da is not None and isinstance(matrix, da.Array)


def _to_dense_float(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


InputType = Literal["auto", "simplex", "counts"]


def _validate_alpha_prior(alpha_prior: float) -> float:
    alpha_prior = float(alpha_prior)
    if not np.isfinite(alpha_prior) or alpha_prior <= 0.0:
        raise ValueError("alpha_prior must be a finite positive number.")
    return alpha_prior


def _validate_dense_matrix(matrix: np.ndarray, *, name: str) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a two-dimensional {name}.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"The {name} must contain at least one row and one column.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"The {name} contains non-finite values.")
    if np.any(matrix < 0):
        raise ValueError(f"The {name} must be non-negative.")


def _resolve_input_type(matrix: np.ndarray, input_type: InputType) -> Literal["simplex", "counts"]:
    if input_type not in {"auto", "simplex", "counts"}:
        raise ValueError("input_type must be one of {'auto', 'simplex', 'counts'}.")
    if input_type != "auto":
        return input_type

    row_sums = matrix.sum(axis=1)
    if np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
        if np.all(matrix > 0.0):
            return "simplex"
        if np.allclose(matrix, np.rint(matrix), atol=1e-12, rtol=0.0):
            return "counts"
        raise ValueError(
            "Rows sum to one but contain zeros, so auto-detection cannot distinguish a boundary simplex "
            "from count data. Pass input_type='simplex' to reject it explicitly or input_type='counts' "
            "to apply Bayesian pseudocount imputation."
        )
    return "counts"


def _bayesian_impute_matrix(matrix: object, alpha_prior: float = 1.0) -> np.ndarray:
    r"""Return the Bayesian posterior-mean simplex for a count matrix."""

    dense = _to_dense_float(matrix)
    _validate_dense_matrix(dense, name="count matrix")
    alpha_prior = _validate_alpha_prior(alpha_prior)

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
    return (dense + alpha_vector) / (cell_depth + sum_alpha)


def _to_simplex_matrix(matrix: object, input_type: InputType = "auto", alpha_prior: float = 1.0) -> np.ndarray:
    """Return an open-simplex matrix, imputing count inputs when necessary."""

    dense = _to_dense_float(matrix)
    _validate_dense_matrix(dense, name="input matrix")
    resolved_type = _resolve_input_type(dense, input_type)
    if resolved_type == "counts":
        return _bayesian_impute_matrix(dense, alpha_prior=alpha_prior)

    if np.any(dense <= 0.0):
        raise ValueError(
            "Simplex inputs must be strictly positive for compositional log-ratio operations. "
            "Use raw counts or input_type='counts' to apply Bayesian pseudocount imputation."
        )
    return dense / dense.sum(axis=1, keepdims=True)


def _clr_matrix(simplex: object) -> np.ndarray:
    """Return CLR coordinates for a strictly positive simplex matrix."""

    dense = _to_dense_float(simplex)
    _validate_dense_matrix(dense, name="simplex matrix")
    if np.any(dense <= 0.0):
        raise ValueError("CLR coordinates require strictly positive simplex values.")
    closed = dense / dense.sum(axis=1, keepdims=True)
    log_simplex = np.log(closed)
    return log_simplex - log_simplex.mean(axis=1, keepdims=True)


def bayesian_impute_pseudocounts(
    adata: Union[ad.AnnData, object],
    layer: Optional[str] = None,
    alpha_prior: float = 1.0,
    out_layer: str = "X_imputed",
) -> Optional[np.ndarray]:
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
        adata: AnnData-like object or a two-dimensional count matrix.
        layer: Optional input layer key.
        alpha_prior: Total prior mass scaling the global background.
        out_layer: Output layer key used to store the imputed simplex for AnnData-like inputs.

    Returns:
        The imputed simplex for matrix input, otherwise ``None`` after storing it in ``adata.layers``.
    """

    alpha_prior = _validate_alpha_prior(alpha_prior)
    if not hasattr(adata, "X") or not hasattr(adata, "layers"):
        if layer is not None:
            raise ValueError("layer can only be used with AnnData-like inputs.")
        return _bayesian_impute_matrix(adata, alpha_prior=alpha_prior)

    matrix = adata.layers[layer] if layer is not None else adata.X

    if _is_dask_array(matrix):
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("The count matrix must contain at least one row and one column.")
        total_per_gene = matrix.sum(axis=0)
        total_depth = matrix.sum()
        total_depth_value, all_finite, all_nonnegative = da.compute(
            total_depth, da.isfinite(matrix).all(), (matrix >= 0).all()
        )
        if not bool(all_finite):
            raise ValueError("The count matrix contains non-finite values.")
        if not bool(all_nonnegative):
            raise ValueError("The count matrix must be non-negative.")
        total_depth_value = float(total_depth_value)
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

    adata.layers[out_layer] = _bayesian_impute_matrix(matrix, alpha_prior=alpha_prior)
    return None


def clr_transform(
    adata: ad.AnnData,
    layer: Optional[str] = None,
    out_layer: str = "X_clr",
    alpha_prior: float = 1.0,
    input_type: InputType = "auto",
) -> None:
    r"""Apply a centered log-ratio transform to a simplex-valued matrix.

    Raw counts are converted to an open simplex with Bayesian pseudocount imputation.
    Strictly positive rows that sum to one are used directly in ``input_type="auto"`` mode.

    .. math::
        \\mathrm{CLR}(\\theta_c) = \\log(\\theta_c) - \\frac{1}{G}\\sum_g \\log(\\theta_{cg})

    Args:
        adata: AnnData-like object.
        layer: Optional input layer key.
        out_layer: Output layer key used to store the CLR coordinates.
        alpha_prior: Prior mass forwarded to Bayesian imputation when needed.
        input_type: ``"simplex"``, ``"counts"``, or ``"auto"``.
    """

    alpha_prior = _validate_alpha_prior(alpha_prior)
    if input_type not in {"auto", "simplex", "counts"}:
        raise ValueError("input_type must be one of {'auto', 'simplex', 'counts'}.")

    matrix = adata.layers[layer] if layer is not None else adata.X
    if _is_dask_array(matrix):
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("The input matrix must contain at least one row and one column.")
        row_sums = matrix.sum(axis=1, keepdims=True)
        all_finite, all_nonnegative, all_positive, sums_to_one, all_integer_like = da.compute(
            da.isfinite(matrix).all(),
            (matrix >= 0).all(),
            (matrix > 0).all(),
            da.isclose(row_sums, 1.0, atol=1e-6, rtol=1e-6).all(),
            da.isclose(matrix, da.rint(matrix), atol=1e-12, rtol=0.0).all(),
        )
        if not bool(all_finite):
            raise ValueError("The input matrix contains non-finite values.")
        if not bool(all_nonnegative):
            raise ValueError("The input matrix must be non-negative.")
        resolved_type = input_type
        if resolved_type == "auto":
            if bool(sums_to_one):
                if bool(all_positive):
                    resolved_type = "simplex"
                elif bool(all_integer_like):
                    resolved_type = "counts"
                else:
                    raise ValueError(
                        "Rows sum to one but contain zeros, so auto-detection cannot distinguish a boundary simplex "
                        "from count data. Set input_type explicitly."
                    )
            else:
                resolved_type = "counts"

        if resolved_type == "simplex":
            if not bool(all_positive):
                raise ValueError(
                    "Simplex inputs must be strictly positive for compositional log-ratio operations. "
                    "Use raw counts or input_type='counts' to apply Bayesian pseudocount imputation."
                )
            simplex = matrix / row_sums
        else:
            bayesian_impute_pseudocounts(adata, layer=layer, alpha_prior=alpha_prior, out_layer="X_imputed")
            simplex = adata.layers["X_imputed"]

        log_simplex = da.log(simplex)
        adata.layers[out_layer] = log_simplex - log_simplex.mean(axis=1, keepdims=True)
        return None

    dense = _to_dense_float(matrix)
    _validate_dense_matrix(dense, name="input matrix")
    resolved_type = _resolve_input_type(dense, input_type)
    simplex = _to_simplex_matrix(dense, input_type=resolved_type, alpha_prior=alpha_prior)
    if resolved_type == "counts":
        adata.layers["X_imputed"] = simplex
    adata.layers[out_layer] = _clr_matrix(simplex)
    return None
