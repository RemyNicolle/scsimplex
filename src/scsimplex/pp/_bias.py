r"""Capture-bias calibration from dataset-level compositional centers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp

try:  # pragma: no cover - optional dependency fallback for local execution
    import anndata as ad
except ModuleNotFoundError:  # pragma: no cover
    from typing import Any

    class _AnnDataModule:
        AnnData = Any

    ad = _AnnDataModule()  # type: ignore[assignment]

from ._transform import InputType, _to_simplex_matrix


@dataclass(frozen=True)
class CaptureBiasCalibration:
    """Reference compositional center used for dataset-level capture-bias calibration."""

    reference_simplex: np.ndarray
    gene_names: Optional[np.ndarray] = None


def _is_adata_like(data: object) -> bool:
    return hasattr(data, "X") and hasattr(data, "layers") and hasattr(data, "uns")


def _extract_matrix(data: object, layer: Optional[str]) -> tuple[object, Optional[np.ndarray], Optional[object]]:
    if _is_adata_like(data):
        if layer is not None and layer not in data.layers:
            raise KeyError(f"Layer '{layer}' not found in AnnData-like input.")
        matrix = data.layers[layer] if layer is not None else data.X
        gene_names = np.asarray(data.var_names.astype(str)) if hasattr(data, "var_names") else None
        return matrix, gene_names, data

    if layer is not None:
        raise ValueError("layer can only be used with AnnData-like inputs.")
    matrix = data.to_numpy() if hasattr(data, "to_numpy") else data
    gene_names = None
    if hasattr(data, "columns"):
        gene_names = np.asarray([str(name) for name in data.columns], dtype=str)
    return matrix, gene_names, None


def _composition_center(simplex: np.ndarray) -> np.ndarray:
    if simplex.ndim != 2 or simplex.shape[0] == 0 or simplex.shape[1] == 0:
        raise ValueError("A compositional center requires a non-empty two-dimensional simplex matrix.")
    if not np.isfinite(simplex).all() or np.any(simplex <= 0.0):
        raise ValueError("A compositional center requires strictly positive finite simplex values.")
    center = np.exp(np.log(simplex).mean(axis=0))
    center /= center.sum()
    return center


def _normalize_beta(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1 or not np.isfinite(beta).all() or np.any(beta <= 0.0):
        raise ValueError("Capture-bias calibration produced an invalid multiplicative correction vector.")
    return beta / np.exp(np.mean(np.log(beta)))


def _closure(matrix: np.ndarray) -> np.ndarray:
    if not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise ValueError("Simplex closure requires strictly positive finite values.")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Cannot close a matrix with non-positive row sums.")
    return matrix / row_sums


def _validate_reference(reference: object) -> np.ndarray:
    reference_array = np.asarray(reference, dtype=float)
    if reference_array.ndim != 1 or reference_array.shape[0] == 0:
        raise ValueError("reference_calibration.reference_simplex must be a non-empty one-dimensional vector.")
    if not np.isfinite(reference_array).all() or np.any(reference_array <= 0.0):
        raise ValueError("reference_calibration.reference_simplex must contain strictly positive finite values.")
    return reference_array / reference_array.sum()


def _is_dataset_sequence(data: object) -> bool:
    if _is_adata_like(data) or sp.issparse(data) or hasattr(data, "shape"):
        return False
    if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
        return False
    if len(data) == 0:
        return True

    first = data[0]
    if _is_adata_like(first) or sp.issparse(first) or hasattr(first, "shape"):
        return True
    try:
        return np.asarray(first).ndim >= 2
    except (TypeError, ValueError):
        return True


def _canonical_gene_order(gene_names: Sequence[Optional[np.ndarray]], n_genes: int) -> Optional[np.ndarray]:
    named = [names for names in gene_names if names is not None]
    if not named:
        return None
    if len(named) != len(gene_names):
        raise ValueError("Either all inputs must provide gene names or none of them must.")

    canonical = np.asarray(named[0], dtype=str)
    if canonical.shape[0] != n_genes:
        raise ValueError("Gene-name length does not match the matrix width.")

    canonical_set = set(canonical.tolist())
    if len(canonical_set) != canonical.shape[0]:
        raise ValueError("Gene names must be unique within each input.")

    for names in named[1:]:
        current = np.asarray(names, dtype=str)
        if current.shape[0] != n_genes:
            raise ValueError("All inputs must have the same number of genes.")
        if set(current.tolist()) != canonical_set:
            raise ValueError("All named inputs must share the same gene set; only gene order may differ.")
        if len(set(current.tolist())) != current.shape[0]:
            raise ValueError("Gene names must be unique within each input.")
    return canonical


def _reindex_to_canonical(
    simplex: np.ndarray,
    gene_names: Optional[np.ndarray],
    canonical_gene_names: Optional[np.ndarray],
) -> np.ndarray:
    if canonical_gene_names is None:
        return simplex
    if gene_names is None:
        raise ValueError("Missing gene names for a calibration that requires named alignment.")
    lookup = {name: idx for idx, name in enumerate(np.asarray(gene_names, dtype=str))}
    order = np.asarray([lookup[name] for name in canonical_gene_names], dtype=int)
    return simplex[:, order]


def _reindex_from_canonical(
    values: np.ndarray,
    gene_names: Optional[np.ndarray],
    canonical_gene_names: Optional[np.ndarray],
) -> np.ndarray:
    if canonical_gene_names is None:
        return np.asarray(values, dtype=float)
    if gene_names is None:
        return np.asarray(values, dtype=float)

    current = np.asarray(gene_names, dtype=str)
    lookup = {name: idx for idx, name in enumerate(canonical_gene_names)}
    order = np.asarray([lookup[name] for name in current], dtype=int)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values[order]
    return values[:, order]


def _store_result(
    container: Optional[object],
    corrected_simplex: np.ndarray,
    beta: np.ndarray,
    dataset_center: np.ndarray,
    reference_simplex: np.ndarray,
    gene_names: Optional[np.ndarray],
    out_layer: Optional[str],
) -> None:
    if container is None:
        return
    if out_layer is not None:
        container.layers[out_layer] = corrected_simplex
    container.uns["capture_bias_beta"] = beta
    if gene_names is not None:
        container.uns["capture_bias_gene_names"] = np.asarray(gene_names, dtype=str)
    container.uns["capture_bias_dataset_center"] = dataset_center
    container.uns["capture_bias_reference_simplex"] = reference_simplex
    container.uns["capture_bias_method"] = "dataset_compositional_center"


def _calibrate_one(
    data: object,
    calibration: CaptureBiasCalibration,
    layer: Optional[str],
    alpha_prior: float,
    input_type: InputType,
    out_layer: Optional[str],
) -> np.ndarray:
    matrix, gene_names, container = _extract_matrix(data, layer)
    simplex = _to_simplex_matrix(matrix, input_type=input_type, alpha_prior=alpha_prior)

    reference = _validate_reference(calibration.reference_simplex)

    if calibration.gene_names is not None:
        canonical_names = np.asarray(calibration.gene_names, dtype=str)
        if reference.shape[0] != canonical_names.shape[0]:
            raise ValueError("reference_calibration gene_names must match reference_simplex length.")
        if len(set(canonical_names.tolist())) != canonical_names.shape[0]:
            raise ValueError("reference_calibration gene_names must be unique.")
    else:
        canonical_names = None

    if canonical_names is None:
        if simplex.shape[1] != reference.shape[0]:
            raise ValueError("Input matrix width does not match the reference calibration.")
        aligned_simplex = simplex
    else:
        if gene_names is None:
            if simplex.shape[1] != reference.shape[0]:
                raise ValueError("Input matrix width does not match the named reference calibration.")
            aligned_simplex = simplex
        else:
            if set(np.asarray(gene_names, dtype=str).tolist()) != set(canonical_names.tolist()):
                raise ValueError("Input gene names do not match the reference calibration gene set.")
            if len(set(np.asarray(gene_names, dtype=str).tolist())) != len(gene_names):
                raise ValueError("Input gene names must be unique.")
            aligned_simplex = _reindex_to_canonical(simplex, gene_names, canonical_names)

    dataset_center = _composition_center(aligned_simplex)
    beta_canonical = _normalize_beta(reference / dataset_center)
    corrected_canonical = _closure(aligned_simplex * beta_canonical[np.newaxis, :])

    corrected = _reindex_from_canonical(corrected_canonical, gene_names, canonical_names)
    beta = _reindex_from_canonical(beta_canonical, gene_names, canonical_names)
    dataset_center_out = _reindex_from_canonical(dataset_center, gene_names, canonical_names)
    reference_out = _reindex_from_canonical(reference, gene_names, canonical_names)
    _store_result(
        container=container,
        corrected_simplex=corrected,
        beta=beta,
        dataset_center=dataset_center_out,
        reference_simplex=reference_out,
        gene_names=gene_names,
        out_layer=out_layer,
    )
    return corrected


def calibrate_capture_bias(
    data: Union[ad.AnnData, object, Sequence[Union[ad.AnnData, object]]],
    reference_calibration: Optional[Union[CaptureBiasCalibration, np.ndarray, Sequence[float]]] = None,
    layer: Optional[str] = None,
    alpha_prior: float = 1.0,
    input_type: InputType = "auto",
    out_layer: Optional[str] = "capture_bias_corrected",
) -> Union[np.ndarray, tuple[list[np.ndarray], CaptureBiasCalibration]]:
    r"""Calibrate dataset-level capture bias without anchor cell types.

    This function supports two modes only:

    1. A list of datasets or matrices with ``reference_calibration=None``. A shared reference
       compositional center is learned from the per-dataset compositional centers, and the
       function returns the calibrated simplex matrices together with that reference object.
    2. A single dataset or matrix with ``reference_calibration`` provided. The input is
       calibrated against the supplied reference and the calibrated simplex matrix is returned.

    The calibration is multiplicative in the simplex. For dataset :math:`b`, let
    :math:`c_b` be the compositional center of its cell-level simplex rows. The shared
    reference is the compositional center across datasets, and the dataset-specific bias
    vector is :math:`\beta_b \propto c_{\mathrm{ref}} / c_b`.

    If the input rows are not already simplex-valued, they are converted first with the same
    Bayesian posterior-mean pseudocount estimator used by
    :func:`scsimplex.pp.bayesian_impute_pseudocounts`.

    Args:
        data: A single AnnData-like object, a single matrix, or a sequence of them.
        reference_calibration: Reference returned by list mode, or a one-dimensional reference
            simplex whose gene order already matches the input.
        layer: Optional input layer for AnnData-like inputs.
        alpha_prior: Prior mass used when raw counts must be converted onto the simplex.
        input_type: ``"simplex"``, ``"counts"``, or ``"auto"``. In auto mode, strictly positive
            rows summing to one are treated as simplex; everything else is treated as counts unless
            it is an ambiguous boundary-simplex input.
        out_layer: Layer used to store the calibrated simplex for AnnData-like inputs.

    Returns:
        In list mode: ``(calibrated_simplex_list, reference_calibration)``.
        In single-input mode: the calibrated simplex matrix.
    """

    if _is_dataset_sequence(data):
        datasets = list(data)
        if reference_calibration is not None:
            raise ValueError(
                "List inputs cannot be used with reference_calibration. Calibrate them one by one instead."
            )
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required to learn a reference calibration.")

        extracted = [_extract_matrix(dataset, layer) for dataset in datasets]
        matrices = [item[0] for item in extracted]
        gene_names = [item[1] for item in extracted]
        containers = [item[2] for item in extracted]

        simplex_list = [
            _to_simplex_matrix(matrix, input_type=input_type, alpha_prior=alpha_prior) for matrix in matrices
        ]
        n_genes = simplex_list[0].shape[1]
        if any(simplex.shape[1] != n_genes for simplex in simplex_list):
            raise ValueError("All inputs must have the same number of genes.")

        canonical_gene_names = _canonical_gene_order(gene_names, n_genes)
        aligned_simplex_list = [
            _reindex_to_canonical(simplex, names, canonical_gene_names)
            for simplex, names in zip(simplex_list, gene_names)
        ]

        dataset_centers = [_composition_center(simplex) for simplex in aligned_simplex_list]
        reference_simplex = _composition_center(np.vstack(dataset_centers))
        calibration = CaptureBiasCalibration(
            reference_simplex=reference_simplex,
            gene_names=None if canonical_gene_names is None else canonical_gene_names.copy(),
        )

        corrected_outputs: list[np.ndarray] = []
        for simplex, names, container, dataset_center in zip(
            aligned_simplex_list, gene_names, containers, dataset_centers
        ):
            beta_canonical = _normalize_beta(reference_simplex / dataset_center)
            corrected_canonical = _closure(simplex * beta_canonical[np.newaxis, :])
            corrected = _reindex_from_canonical(corrected_canonical, names, canonical_gene_names)
            beta = _reindex_from_canonical(beta_canonical, names, canonical_gene_names)
            dataset_center_out = _reindex_from_canonical(dataset_center, names, canonical_gene_names)
            reference_out = _reindex_from_canonical(reference_simplex, names, canonical_gene_names)
            _store_result(
                container=container,
                corrected_simplex=corrected,
                beta=beta,
                dataset_center=dataset_center_out,
                reference_simplex=reference_out,
                gene_names=names,
                out_layer=out_layer,
            )
            corrected_outputs.append(corrected)

        return corrected_outputs, calibration

    if reference_calibration is None:
        raise ValueError("Single inputs require reference_calibration. Use list mode to learn one first.")

    calibration = (
        reference_calibration
        if isinstance(reference_calibration, CaptureBiasCalibration)
        else CaptureBiasCalibration(reference_simplex=np.asarray(reference_calibration, dtype=float))
    )
    return _calibrate_one(
        data=data,
        calibration=calibration,
        layer=layer,
        alpha_prior=alpha_prior,
        input_type=input_type,
        out_layer=out_layer,
    )
