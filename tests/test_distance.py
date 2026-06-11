from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from scsimplex.pp import pairwise_intersection_aitchison_distance_matrix


def test_pairwise_intersection_distance_matrix_handles_partial_gene_overlap(
    make_adata: Callable[..., object],
) -> None:
    dataset_a = make_adata(
        np.array(
            [
                [12.0, 3.0, 1.0, 1.0],
                [9.0, 4.0, 2.0, 1.0],
                [10.0, 2.0, 3.0, 1.0],
            ]
        ),
        var_names=["g1", "g2", "g3", "g4"],
    )
    dataset_b = make_adata(
        np.array(
            [
                [4.0, 9.0, 2.0, 1.0],
                [3.0, 10.0, 2.0, 1.0],
                [5.0, 8.0, 2.0, 1.0],
            ]
        ),
        var_names=["g2", "g3", "g4", "g5"],
    )
    dataset_c = make_adata(
        np.array(
            [
                [3.0, 6.0, 9.0, 2.0],
                [2.0, 5.0, 10.0, 2.0],
                [2.0, 7.0, 8.0, 3.0],
            ]
        ),
        var_names=["g3", "g4", "g5", "g6"],
    )

    result = pairwise_intersection_aitchison_distance_matrix(
        [dataset_a, dataset_b, dataset_c],
        dataset_names=["A", "B", "C"],
    )

    assert result.distance_matrix.shape == (9, 9)
    assert np.allclose(result.distance_matrix, result.distance_matrix.T)
    assert np.allclose(np.diag(result.distance_matrix), 0.0)
    assert np.isfinite(result.distance_matrix).all()
    assert result.dataset_names.tolist() == ["A", "B", "C"]
    assert result.dataset_offsets.tolist() == [0, 3, 6, 9]

    diagnostics = result.diagnostics.set_index(["dataset_i", "dataset_j"])
    assert int(diagnostics.loc[("A", "A"), "n_genes"]) == 4
    assert int(diagnostics.loc[("B", "B"), "n_genes"]) == 4
    assert int(diagnostics.loc[("C", "C"), "n_genes"]) == 4
    assert int(diagnostics.loc[("A", "B"), "n_genes"]) == 3
    assert int(diagnostics.loc[("A", "C"), "n_genes"]) == 2
    assert int(diagnostics.loc[("B", "C"), "n_genes"]) == 3


def test_pairwise_intersection_distance_normalization_tightens_cross_pair_medians(
    make_adata: Callable[..., object],
) -> None:
    rng = np.random.default_rng(0)
    latent = rng.dirichlet(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), size=18)
    gene_subsets = {
        "A": np.array([0, 1, 2, 3]),
        "B": np.array([1, 2, 3, 4]),
        "C": np.array([2, 3, 4, 5]),
    }
    capture_bias = {
        "A": np.array([5.0, 1.0, 0.7, 0.4]),
        "B": np.array([0.4, 1.0, 2.5, 4.0]),
        "C": np.array([3.0, 0.6, 1.2, 2.0]),
    }

    datasets = []
    for name in ["A", "B", "C"]:
        subset = gene_subsets[name]
        biased_probs = latent[:, subset] * capture_bias[name]
        biased_probs /= biased_probs.sum(axis=1, keepdims=True)
        counts = np.vstack([rng.multinomial(400, prob) for prob in biased_probs])
        datasets.append(make_adata(counts, var_names=[f"g{gene + 1}" for gene in subset]))

    result = pairwise_intersection_aitchison_distance_matrix(
        datasets,
        dataset_names=["A", "B", "C"],
    )

    cross = result.diagnostics.loc[result.diagnostics["pair_type"] == "cross"].copy()
    raw = cross["median_block_raw"].to_numpy(dtype=float)
    normalized = cross["median_block_normalized"].to_numpy(dtype=float)
    raw_cv = float(np.std(raw, ddof=0) / np.mean(raw))
    normalized_cv = float(np.std(normalized, ddof=0) / np.mean(normalized))

    assert normalized_cv < 0.5 * raw_cv


def test_pairwise_intersection_distance_accepts_aligned_unnamed_matrices() -> None:
    counts_a = np.array([[6.0, 2.0, 1.0], [5.0, 3.0, 1.0], [7.0, 1.0, 1.0]])
    counts_b = np.array([[1.0, 5.0, 2.0], [1.0, 6.0, 2.0], [2.0, 4.0, 2.0]])

    result = pairwise_intersection_aitchison_distance_matrix([counts_a, counts_b], dataset_names=["A", "B"])

    assert result.distance_matrix.shape == (6, 6)
    cross = result.diagnostics.loc[result.diagnostics["pair_type"] == "cross"]
    assert int(cross["n_genes"].iloc[0]) == 3


def test_pairwise_intersection_distance_rejects_single_shared_gene(
    make_adata: Callable[..., object],
) -> None:
    dataset_a = make_adata(np.array([[4.0, 1.0], [3.0, 2.0]]), var_names=["g1", "g2"])
    dataset_b = make_adata(np.array([[5.0, 1.0], [4.0, 2.0]]), var_names=["g2", "g3"])

    with pytest.raises(ValueError, match="at least two intersecting genes"):
        pairwise_intersection_aitchison_distance_matrix([dataset_a, dataset_b], dataset_names=["A", "B"])
