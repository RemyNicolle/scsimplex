r"""Batch-adjusted multinomial naive Bayes classification."""

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


def _to_dense_float(matrix: object) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _align_by_var_names(
    query_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    query_matrix: np.ndarray,
    reference_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(query_adata, "var_names") or not hasattr(reference_adata, "var_names"):
        common = np.arange(min(query_matrix.shape[1], reference_matrix.shape[1]))
        return query_matrix[:, common], reference_matrix[:, common], np.asarray(common)

    query_names = np.asarray(query_adata.var_names.astype(str))
    reference_names = np.asarray(reference_adata.var_names.astype(str))
    query_index = {name: idx for idx, name in enumerate(query_names)}
    reference_index = {name: idx for idx, name in enumerate(reference_names)}
    common = [name for name in reference_names if name in query_index]
    if len(common) == 0:
        raise ValueError("Query and reference datasets do not share any genes for classification.")

    query_cols = np.array([query_index[name] for name in common], dtype=int)
    reference_cols = np.array([reference_index[name] for name in common], dtype=int)
    return query_matrix[:, query_cols], reference_matrix[:, reference_cols], np.asarray(common)


def map_multinomial_nb(
    query_adata: ad.AnnData,
    reference_adata: ad.AnnData,
    reference_cluster_key: str,
    use_bias_correction: bool = True,
    query_layer: Optional[str] = None,
    ref_layer: Optional[str] = None,
) -> np.ndarray:
    r"""Classify query cells by maximum-likelihood multinomial naive Bayes.

    The reference cluster profile for class :math:`k` is estimated by summing raw counts
    across reference cells in that class and normalizing onto the simplex:

    .. math::
        \\theta_{kg} = \\frac{\\sum_{r \\in k} n_{rg}}{\\sum_{g'}\\sum_{r \\in k} n_{rg'}}.

    Query cells are assigned by maximizing

    .. math::
        \\arg\\max_k \\sum_g n_{cg} \\log \\theta'_{kg}

    where :math:`\\theta'_{kg}` is optionally bias-corrected with capture-efficiency weights.

    Args:
        query_adata: Query AnnData-like object.
        reference_adata: Reference AnnData-like object.
        reference_cluster_key: Observation key defining reference labels.
        use_bias_correction: If `True`, adjust the reference profiles with `query_adata.uns['capture_bias_beta']`.
        query_layer: Optional query layer key.
        ref_layer: Optional reference layer key.

    Returns:
        Predicted labels as a NumPy array.
    """

    if reference_cluster_key not in reference_adata.obs:
        raise KeyError(f"Reference cluster key '{reference_cluster_key}' not found in reference_adata.obs.")

    query_matrix = _to_dense_float(query_adata.layers[query_layer] if query_layer is not None else query_adata.X)
    reference_matrix = _to_dense_float(reference_adata.layers[ref_layer] if ref_layer is not None else reference_adata.X)
    query_matrix, reference_matrix, common_gene_names = _align_by_var_names(
        query_adata, reference_adata, query_matrix, reference_matrix
    )

    ref_clusters = np.asarray(reference_adata.obs[reference_cluster_key].astype(str))
    unique_labels = np.unique(ref_clusters)

    eps = 1e-12
    prob_matrix = np.zeros((unique_labels.size, reference_matrix.shape[1]), dtype=float)
    for idx, label in enumerate(unique_labels):
        class_counts = reference_matrix[ref_clusters == label].sum(axis=0) + eps
        prob_matrix[idx] = class_counts / class_counts.sum()

    if use_bias_correction:
        beta = query_adata.uns.get("capture_bias_beta")
        if beta is not None:
            beta = np.asarray(beta, dtype=float)
            if beta.shape[0] != prob_matrix.shape[1]:
                beta_gene_names = query_adata.uns.get("capture_bias_gene_names")
                if beta_gene_names is None:
                    raise ValueError("capture_bias_beta length does not match the aligned gene axis.")
                beta_gene_names = np.asarray(beta_gene_names, dtype=str)
                beta_lookup = {name: idx for idx, name in enumerate(beta_gene_names)}
                beta = np.asarray([beta[beta_lookup[name]] for name in common_gene_names], dtype=float)
            prob_matrix = prob_matrix * beta[np.newaxis, :]
            prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)

    log_probs = np.log(prob_matrix + eps)
    scores = query_matrix @ log_probs.T
    predictions = unique_labels[np.argmax(scores, axis=1)]
    query_adata.obs["predicted_cell_state"] = predictions
    query_adata.obsm["predicted_cell_state_scores"] = scores
    return predictions
