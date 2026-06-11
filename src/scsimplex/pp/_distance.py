r"""Pairwise-intersection Aitchison distances across multiple datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform

try:  # pragma: no cover - optional dependency fallback for local execution
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    from typing import Any

    class _AnnDataModule:
        AnnData = Any

    ad = _AnnDataModule()  # type: ignore[assignment]

from ._bias import _extract_matrix, _is_adata_like, calibrate_capture_bias
from ._transform import (
    InputType,
    _clr_matrix,
    _resolve_input_type,
    _to_dense_float,
    _to_simplex_matrix,
    _validate_alpha_prior,
    _validate_dense_matrix,
)


@dataclass(frozen=True)
class PairwiseIntersectionDistanceResult:
    """Global distance matrix and diagnostics for pairwise-intersection geometry."""

    distance_matrix: np.ndarray
    diagnostics: pd.DataFrame
    dataset_names: np.ndarray
    cell_names: np.ndarray
    cell_dataset_names: np.ndarray
    dataset_offsets: np.ndarray


@dataclass(frozen=True)
class _DatasetSpec:
    matrix: np.ndarray
    gene_names: np.ndarray | None
    resolved_input_type: str
    dataset_name: str
    cell_names: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.matrix.shape[1])


def _resolve_dataset_name(data: object, dataset_name: str | None, index: int) -> str:
    if dataset_name is not None:
        return str(dataset_name)
    if _is_adata_like(data):
        uns = getattr(data, "uns", {})
        candidate = uns.get("dataset_name")
        if isinstance(candidate, str) and candidate:
            return candidate
        obs = getattr(data, "obs", None)
        if obs is not None and hasattr(obs, "__contains__") and "dataset" in obs:
            values = pd.Series(obs["dataset"]).astype(str).unique()
            if values.size == 1:
                return str(values[0])
    return f"dataset_{index}"


def _extract_cell_names(data: object, n_cells: int) -> np.ndarray:
    if _is_adata_like(data):
        if hasattr(data, "obs_names"):
            return np.asarray(data.obs_names.astype(str), dtype=str)
        obs = getattr(data, "obs", None)
        if obs is not None and hasattr(obs, "index"):
            return np.asarray(obs.index.astype(str), dtype=str)
    if hasattr(data, "index") and not _is_adata_like(data):
        try:
            index = np.asarray(data.index.astype(str), dtype=str)
            if index.shape[0] == n_cells:
                return index
        except Exception:  # pragma: no cover - defensive fallback
            pass
    return np.asarray([f"cell_{row}" for row in range(n_cells)], dtype=str)


def _prepare_dataset_specs(
    datasets: Sequence[object],
    *,
    layer: str | None,
    dataset_names: Sequence[str] | None,
    input_type: InputType,
) -> list[_DatasetSpec]:
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required.")
    if dataset_names is not None and len(dataset_names) != len(datasets):
        raise ValueError("dataset_names must match the number of datasets.")

    specs: list[_DatasetSpec] = []
    for index, dataset in enumerate(datasets):
        provided_name = None if dataset_names is None else dataset_names[index]
        resolved_name = _resolve_dataset_name(dataset, provided_name, index)
        matrix, gene_names, _ = _extract_matrix(dataset, layer)
        dense = _to_dense_float(matrix)
        _validate_dense_matrix(dense, name=f"input matrix for dataset '{resolved_name}'")
        resolved_input_type = _resolve_input_type(dense, input_type)
        names = None if gene_names is None else np.asarray(gene_names, dtype=str)
        if names is not None:
            if names.shape[0] != dense.shape[1]:
                raise ValueError(f"Gene-name length does not match dataset '{resolved_name}'.")
            if len(set(names.tolist())) != names.shape[0]:
                raise ValueError(f"Gene names must be unique within dataset '{resolved_name}'.")
        specs.append(
            _DatasetSpec(
                matrix=dense,
                gene_names=names,
                resolved_input_type=resolved_input_type,
                dataset_name=resolved_name,
                cell_names=_extract_cell_names(dataset, dense.shape[0]),
            )
        )

    all_dataset_names = [spec.dataset_name for spec in specs]
    if len(set(all_dataset_names)) != len(all_dataset_names):
        raise ValueError("dataset_names must be unique.")
    named_flags = [spec.gene_names is not None for spec in specs]
    if any(named_flags) and not all(named_flags):
        raise ValueError("Either all datasets must provide gene names or none of them must.")
    if not any(named_flags):
        widths = [spec.n_genes for spec in specs]
        if len(set(widths)) != 1:
            raise ValueError(
                "Unnamed matrices must share the same number of columns because pairwise gene intersections "
                "cannot be inferred without gene names."
            )
    return specs


def _pairwise_columns(left: _DatasetSpec, right: _DatasetSpec) -> tuple[np.ndarray, np.ndarray, int]:
    if left.gene_names is None or right.gene_names is None:
        if left.n_genes != right.n_genes:
            raise ValueError(
                f"Unnamed datasets '{left.dataset_name}' and '{right.dataset_name}' must share the same width."
            )
        columns = np.arange(left.n_genes, dtype=int)
        return columns, columns.copy(), int(columns.size)

    if left is right:
        columns = np.arange(left.n_genes, dtype=int)
        return columns, columns.copy(), int(columns.size)

    right_lookup = {name: index for index, name in enumerate(right.gene_names)}
    common_names = [name for name in left.gene_names.tolist() if name in right_lookup]
    if len(common_names) < 2:
        raise ValueError(
            f"Datasets '{left.dataset_name}' and '{right.dataset_name}' share only {len(common_names)} genes; "
            "at least two intersecting genes are required for non-degenerate Aitchison distances."
        )
    left_lookup = {name: index for index, name in enumerate(left.gene_names)}
    left_columns = np.asarray([left_lookup[name] for name in common_names], dtype=int)
    right_columns = np.asarray([right_lookup[name] for name in common_names], dtype=int)
    return left_columns, right_columns, len(common_names)


def _subset_to_simplex(spec: _DatasetSpec, columns: np.ndarray, alpha_prior: float) -> np.ndarray:
    subset = spec.matrix[:, columns]
    return _to_simplex_matrix(subset, input_type=spec.resolved_input_type, alpha_prior=alpha_prior)


def _median_within_distance(clr_features: np.ndarray, dataset_name: str, pair_label: str) -> float:
    distances = pdist(clr_features, metric="euclidean")
    if distances.size == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' needs at least two cells to estimate the within-dataset scale for {pair_label}."
        )
    median_distance = float(np.median(distances))
    if not np.isfinite(median_distance) or median_distance <= 0.0:
        raise ValueError(
            f"Dataset '{dataset_name}' has zero or invalid within-dataset Aitchison scale for {pair_label}."
        )
    return median_distance


def pairwise_intersection_aitchison_distance_matrix(
    datasets: Sequence[object],
    *,
    layer: str | None = None,
    alpha_prior: float = 1.0,
    input_type: InputType = "auto",
    dataset_names: Sequence[str] | None = None,
) -> PairwiseIntersectionDistanceResult:
    r"""Assemble a global distance matrix from pairwise gene intersections.

    For each dataset pair ``(A, B)``, the computation is restricted to the genes shared by those
    two datasets only. Count inputs are converted to the simplex by the existing Bayesian
    posterior-mean pseudocount estimator on that pairwise intersection. Each pair is then
    capture-bias calibrated with the existing dataset-center correction, transformed by CLR,
    and converted to Aitchison distances.

    Cross-dataset blocks are normalized by

    .. math::
        \sqrt{\operatorname{median}(d_{AA}) \operatorname{median}(d_{BB})}

    where both within-dataset medians are computed on the same pairwise intersection and after
    the same pairwise calibration. Self blocks use the analogous ``(A, A)`` scaling on the
    dataset's own gene universe.

    Unnamed matrices are treated as already aligned by column order, so all such matrices must
    share the same width. Mixed named and unnamed inputs are rejected.

    Args:
        datasets: Sequence of AnnData-like objects or matrix-like objects.
        layer: Optional input layer for AnnData-like inputs.
        alpha_prior: Total Dirichlet prior mass for Bayesian simplex imputation of count inputs.
        input_type: ``"simplex"``, ``"counts"``, or ``"auto"``. Auto-detection is resolved on
            each full input before pairwise subsetting so simplex inputs remain simplex after
            intersection trimming.
        dataset_names: Optional names for diagnostics and output ordering.

    Returns:
        A dataclass containing the global symmetric distance matrix, per-pair diagnostics, and
        the dataset and cell ordering used to assemble the matrix.
    """

    alpha_prior = _validate_alpha_prior(alpha_prior)
    specs = _prepare_dataset_specs(
        datasets,
        layer=layer,
        dataset_names=dataset_names,
        input_type=input_type,
    )

    dataset_sizes = np.asarray([spec.n_cells for spec in specs], dtype=int)
    dataset_offsets = np.concatenate(([0], np.cumsum(dataset_sizes)))
    total_cells = int(dataset_offsets[-1])
    distance_matrix = np.zeros((total_cells, total_cells), dtype=float)
    diagnostics_rows: list[dict[str, object]] = []

    cell_names = np.concatenate(
        [
            np.asarray([f"{spec.dataset_name}:{name}" for name in spec.cell_names], dtype=str)
            for spec in specs
        ]
    )
    cell_dataset_names = np.concatenate(
        [np.full(spec.n_cells, spec.dataset_name, dtype=str) for spec in specs]
    )

    for left_index, left in enumerate(specs):
        left_slice = slice(dataset_offsets[left_index], dataset_offsets[left_index + 1])
        left_columns, _, n_genes = _pairwise_columns(left, left)
        left_simplex = _subset_to_simplex(left, left_columns, alpha_prior=alpha_prior)
        corrected_left, _ = calibrate_capture_bias(
            [left_simplex],
            input_type="simplex",
            alpha_prior=alpha_prior,
            out_layer=None,
        )
        left_clr = _clr_matrix(corrected_left[0])
        left_within = pdist(left_clr, metric="euclidean")
        left_scale = _median_within_distance(left_clr, left.dataset_name, f"({left.dataset_name}, {left.dataset_name})")
        distance_matrix[left_slice, left_slice] = squareform(left_within) / left_scale
        diagnostics_rows.append(
            {
                "dataset_i": left.dataset_name,
                "dataset_j": left.dataset_name,
                "pair_type": "self",
                "n_cells_i": left.n_cells,
                "n_cells_j": left.n_cells,
                "n_genes": n_genes,
                "median_within_i_raw": left_scale,
                "median_within_j_raw": left_scale,
                "scale": left_scale,
                "median_block_raw": left_scale,
                "median_block_normalized": 1.0,
            }
        )

        for right_index in range(left_index + 1, len(specs)):
            right = specs[right_index]
            right_slice = slice(dataset_offsets[right_index], dataset_offsets[right_index + 1])
            left_columns, right_columns, n_genes = _pairwise_columns(left, right)
            left_simplex = _subset_to_simplex(left, left_columns, alpha_prior=alpha_prior)
            right_simplex = _subset_to_simplex(right, right_columns, alpha_prior=alpha_prior)
            corrected_pair, _ = calibrate_capture_bias(
                [left_simplex, right_simplex],
                input_type="simplex",
                alpha_prior=alpha_prior,
                out_layer=None,
            )
            left_clr = _clr_matrix(corrected_pair[0])
            right_clr = _clr_matrix(corrected_pair[1])
            pair_label = f"({left.dataset_name}, {right.dataset_name})"
            median_left = _median_within_distance(left_clr, left.dataset_name, pair_label)
            median_right = _median_within_distance(right_clr, right.dataset_name, pair_label)
            scale = float(np.sqrt(median_left * median_right))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Pairwise scale is zero or invalid for {pair_label}.")
            cross_raw = cdist(left_clr, right_clr, metric="euclidean")
            cross_normalized = cross_raw / scale
            distance_matrix[left_slice, right_slice] = cross_normalized
            distance_matrix[right_slice, left_slice] = cross_normalized.T
            diagnostics_rows.append(
                {
                    "dataset_i": left.dataset_name,
                    "dataset_j": right.dataset_name,
                    "pair_type": "cross",
                    "n_cells_i": left.n_cells,
                    "n_cells_j": right.n_cells,
                    "n_genes": n_genes,
                    "median_within_i_raw": median_left,
                    "median_within_j_raw": median_right,
                    "scale": scale,
                    "median_block_raw": float(np.median(cross_raw)),
                    "median_block_normalized": float(np.median(cross_normalized)),
                }
            )

    np.fill_diagonal(distance_matrix, 0.0)
    diagnostics = pd.DataFrame(diagnostics_rows)
    return PairwiseIntersectionDistanceResult(
        distance_matrix=distance_matrix,
        diagnostics=diagnostics,
        dataset_names=np.asarray([spec.dataset_name for spec in specs], dtype=str),
        cell_names=cell_names,
        cell_dataset_names=cell_dataset_names,
        dataset_offsets=dataset_offsets,
    )
