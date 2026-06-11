from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

from scsimplex.pp._metacells import multinomial_kmeans
from scsimplex.tl import detect_multiplets, map_multinomial_nb


def test_map_multinomial_nb_applies_correction_in_raw_query_direction(make_adata: Callable[..., object]) -> None:
    reference = make_adata(
        np.array([[90.0, 10.0], [10.0, 90.0]]),
        obs=pd.DataFrame({"label": ["A", "B"]}),
        var_names=["g1", "g2"],
    )
    query = make_adata(np.array([[90.0, 10.0]]), var_names=["g2", "g1"])
    query.uns["capture_bias_beta"] = np.array([1.0 / 9.0, 9.0])
    query.uns["capture_bias_gene_names"] = np.array(["g2", "g1"])

    prediction = map_multinomial_nb(query, reference, reference_cluster_key="label", use_bias_correction=True)

    assert prediction.tolist() == ["A"]


def test_map_multinomial_nb_rejects_unnamed_mismatched_gene_axes(make_adata: Callable[..., object]) -> None:
    reference = make_adata(np.array([[1.0, 2.0]]), obs=pd.DataFrame({"label": ["A"]}))
    query = make_adata(np.array([[1.0, 2.0, 3.0]]))
    del reference.var
    del query.var

    with pytest.raises(ValueError, match="same number of genes"):
        map_multinomial_nb(query, reference, reference_cluster_key="label", use_bias_correction=False)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("target_metacell_size", 0),
        ("max_iter", 0),
        ("tol", -0.1),
        ("tol", 1.1),
    ],
)
def test_multinomial_kmeans_rejects_invalid_parameters(
    make_adata: Callable[..., object], keyword: str, value: object
) -> None:
    adata = make_adata(np.array([[2.0, 1.0], [1.0, 2.0]]))

    with pytest.raises((TypeError, ValueError)):
        multinomial_kmeans(adata, **{keyword: value})


def test_detect_multiplets_scores_clear_mixture(make_adata: Callable[..., object]) -> None:
    counts = np.array(
        [
            [100.0, 0.0],
            [90.0, 0.0],
            [0.0, 100.0],
            [0.0, 90.0],
            [50.0, 50.0],
        ]
    )
    obs = pd.DataFrame({"cluster": ["A", "A", "B", "B", "A"]})
    adata = make_adata(counts, obs=obs)

    detect_multiplets(adata, cluster_key="cluster")

    assert bool(adata.obs.loc[4, "is_multiplet"])
    assert float(adata.obs.loc[4, "multiplet_p_value"]) < 0.05
    assert np.all((adata.obs["multiplet_p_value"] >= 0.0) & (adata.obs["multiplet_p_value"] <= 1.0))
