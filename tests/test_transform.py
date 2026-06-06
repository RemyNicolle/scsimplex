from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from scsimplex.pp import bayesian_impute_pseudocounts, clr_transform


def test_bayesian_impute_pseudocounts_matrix_returns_open_simplex() -> None:
    counts = np.array([[4.0, 0.0, 1.0], [0.0, 3.0, 0.0]])

    simplex = bayesian_impute_pseudocounts(counts)

    assert simplex is not None
    assert np.all(simplex > 0.0)
    assert np.allclose(simplex.sum(axis=1), 1.0)


@pytest.mark.parametrize("alpha_prior", [0.0, -1.0, np.inf, np.nan])
def test_bayesian_impute_pseudocounts_rejects_invalid_prior(alpha_prior: float) -> None:
    with pytest.raises(ValueError):
        bayesian_impute_pseudocounts(np.array([[1.0, 2.0]]), alpha_prior=alpha_prior)


def test_clr_transform_uses_requested_simplex_layer_not_stale_imputation(make_adata: Callable[..., object]) -> None:
    adata = make_adata(np.array([[90.0, 10.0], [80.0, 20.0]]))
    bayesian_impute_pseudocounts(adata)
    adata.layers["corrected"] = np.array([[0.2, 0.8], [0.3, 0.7]])

    clr_transform(adata, layer="corrected")

    expected = np.log(adata.layers["corrected"])
    expected -= expected.mean(axis=1, keepdims=True)
    assert np.allclose(adata.layers["X_clr"], expected)


def test_clr_transform_imputes_raw_counts(make_adata: Callable[..., object]) -> None:
    adata = make_adata(np.array([[4.0, 0.0], [0.0, 4.0]]))

    clr_transform(adata)

    assert "X_imputed" in adata.layers
    assert np.isfinite(adata.layers["X_clr"]).all()
    assert np.allclose(adata.layers["X_clr"].sum(axis=1), 0.0)


def test_clr_transform_rejects_boundary_simplex_when_explicit(make_adata: Callable[..., object]) -> None:
    adata = make_adata(np.array([[1.0, 0.0], [0.5, 0.5]]))

    with pytest.raises(ValueError, match="strictly positive"):
        clr_transform(adata, input_type="simplex")


def test_clr_transform_rejects_ambiguous_boundary_simplex_in_auto_mode(make_adata: Callable[..., object]) -> None:
    adata = make_adata(np.array([[0.8, 0.2], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="cannot distinguish"):
        clr_transform(adata)


def test_clr_transform_auto_imputes_one_hot_raw_counts(make_adata: Callable[..., object]) -> None:
    adata = make_adata(np.array([[1.0, 0.0], [0.0, 1.0]]))

    clr_transform(adata)

    assert np.all(adata.layers["X_imputed"] > 0.0)
