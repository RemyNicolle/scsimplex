from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from scsimplex.pp import CaptureBiasCalibration, calibrate_capture_bias


def _composition_center(simplex: np.ndarray) -> np.ndarray:
    center = np.exp(np.log(simplex + 1e-12).mean(axis=0))
    return center / center.sum()


def test_calibrate_capture_bias_list_mode_learns_reference_from_counts() -> None:
    counts_a = np.array(
        [
            [12.0, 3.0, 1.0],
            [10.0, 4.0, 1.0],
            [11.0, 3.0, 2.0],
        ]
    )
    counts_b = np.array(
        [
            [2.0, 11.0, 1.0],
            [3.0, 12.0, 1.0],
            [2.0, 10.0, 2.0],
        ]
    )

    corrected, calibration = calibrate_capture_bias([counts_a, counts_b])

    assert isinstance(calibration, CaptureBiasCalibration)
    assert len(corrected) == 2
    assert calibration.reference_simplex.shape == (3,)
    assert np.isclose(calibration.reference_simplex.sum(), 1.0)

    for matrix in corrected:
        assert matrix.shape == counts_a.shape
        assert np.all(matrix > 0.0)
        assert np.allclose(matrix.sum(axis=1), 1.0)

    centers = [_composition_center(matrix) for matrix in corrected]
    for center in centers:
        assert np.allclose(center, calibration.reference_simplex, atol=1e-8)


def test_calibrate_capture_bias_single_mode_matches_list_mode() -> None:
    counts_a = np.array(
        [
            [8.0, 2.0, 1.0],
            [9.0, 2.0, 1.0],
            [7.0, 3.0, 1.0],
        ]
    )
    counts_b = np.array(
        [
            [2.0, 8.0, 1.0],
            [2.0, 9.0, 1.0],
            [3.0, 7.0, 1.0],
        ]
    )

    corrected_list, calibration = calibrate_capture_bias([counts_a, counts_b])
    corrected_single = calibrate_capture_bias(counts_b, reference_calibration=calibration)

    assert np.allclose(corrected_single, corrected_list[1], atol=1e-10)


def test_calibrate_capture_bias_ann_data_aligns_gene_names_and_stores_result(make_adata: Callable[..., object]) -> None:
    reference_counts = np.array(
        [
            [9.0, 2.0, 1.0],
            [8.0, 3.0, 1.0],
            [10.0, 2.0, 1.0],
        ]
    )
    query_counts_canonical = np.array(
        [
            [3.0, 7.0, 2.0],
            [2.0, 8.0, 2.0],
            [3.0, 6.0, 2.0],
        ]
    )

    reference = make_adata(reference_counts, var_names=["g1", "g2", "g3"])
    _, calibration = calibrate_capture_bias([reference])

    query = make_adata(query_counts_canonical[:, [2, 0, 1]], var_names=["g3", "g1", "g2"])
    corrected_named = calibrate_capture_bias(query, reference_calibration=calibration)

    corrected_canonical = calibrate_capture_bias(query_counts_canonical, reference_calibration=calibration)
    assert np.allclose(corrected_named[:, [1, 2, 0]], corrected_canonical, atol=1e-10)

    assert "capture_bias_beta" in query.uns
    assert "capture_bias_corrected" in query.layers
    assert np.allclose(query.layers["capture_bias_corrected"], corrected_named)
    assert query.uns["capture_bias_gene_names"].tolist() == ["g3", "g1", "g2"]


def test_calibrate_capture_bias_accepts_list_of_lists_as_single_matrix() -> None:
    corrected = calibrate_capture_bias([[1.0, 3.0], [2.0, 4.0]], reference_calibration=[0.5, 0.5])

    assert corrected.shape == (2, 2)
    assert np.allclose(corrected.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "reference",
    [
        [-0.5, 1.5],
        [0.0, 1.0],
        [np.nan, 1.0],
        [],
    ],
)
def test_calibrate_capture_bias_rejects_invalid_reference(reference: list[float]) -> None:
    with pytest.raises(ValueError):
        calibrate_capture_bias([[1.0, 2.0]], reference_calibration=reference)
