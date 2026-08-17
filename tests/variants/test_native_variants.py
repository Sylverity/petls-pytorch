"""PETLS-independent behavior tests for the Alpha and Rips variants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from petls_pytorch.variants.alpha import Alpha
from petls_pytorch.variants.rips import Rips

pytestmark = pytest.mark.native

POINTS_4 = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
]

POINTS_RECT = np.array(
    [
        [0, 0],
        [0, 3],
        [4, 0],
        [4, 3],
    ]
)

DISTANCES_RECT = np.array(
    [
        [0, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 5, 0, 0],
        [5, 4, 3, 0],
    ],
    dtype=np.float64,
)

OFF_PATH = "tests/variants/data/alpha/input"


def _extract_spectra(complex_):
    return {
        (dim, a, b): np.asarray(eigenvalues, dtype=np.float64)
        for dim, a, b, eigenvalues in complex_.spectra()
    }


def test_alpha_points_and_off_agree_without_reference_package():
    points = _extract_spectra(Alpha(points=POINTS_4, max_dim=3))
    off = _extract_spectra(Alpha(filename=OFF_PATH, max_dim=3))
    assert set(points) == set(off)
    for key in points:
        np.testing.assert_allclose(points[key], off[key], atol=1e-4, rtol=1e-3)


def test_alpha_native_validation_and_request_list():
    with pytest.raises(ValueError, match="requires filename or point set"):
        Alpha()

    result = Alpha(points=POINTS_4, max_dim=3).spectra(
        request_list=[(0, 0.25, 0.5), (1, 0.25, 0.5)]
    )
    assert [(dim, a, b) for dim, a, b, _ in result] == [
        (0, 0.25, 0.5),
        (1, 0.25, 0.5),
    ]


def test_alpha_native_eigenvectors_satisfy_the_laplacian_equation():
    complex_ = Alpha(points=POINTS_4, max_dim=3)
    values, vectors = complex_.eigenpairs(1, 0.25, 0.5)
    if not values:
        pytest.skip("No eigenvalues to verify")
    laplacian = complex_.get_L(1, 0.25, 0.5)
    for index, value in enumerate(values):
        np.testing.assert_allclose(
            (laplacian @ vectors[:, index]).cpu().numpy(),
            (value * vectors[:, index]).cpu().numpy(),
            atol=1e-4,
            rtol=1e-3,
        )


def test_alpha_native_laplacian_is_up_plus_down():
    complex_ = Alpha(points=POINTS_4, max_dim=3)
    filtrations = complex_.get_all_filtrations()
    for dim in range(complex_.top_dim + 1):
        for a, b in zip(filtrations, filtrations[1:]):
            torch.testing.assert_close(
                complex_.get_L(dim, a, b),
                complex_.get_up(dim, a, b) + complex_.get_down(dim, a),
                atol=1e-4,
                rtol=1e-3,
            )


def test_rips_native_validation_and_laplacian_terms():
    with pytest.raises(ValueError, match="requires filename, point set, or distance matrix"):
        Rips()

    complex_ = Rips(points=POINTS_RECT, max_dim=3)
    filtrations = complex_.get_all_filtrations()
    for dim in range(complex_.top_dim + 1):
        for a, b in zip(filtrations, filtrations[1:]):
            torch.testing.assert_close(
                complex_.get_L(dim, a, b),
                complex_.get_up(dim, a, b) + complex_.get_down(dim, a),
                atol=1e-4,
                rtol=1e-3,
            )

    for a, b in [(filtrations[0], filtrations[1]), (filtrations[-2], filtrations[-1])]:
        up = complex_.get_up(complex_.top_dim, a, b)
        assert up.shape[0] == up.shape[1]
        assert torch.allclose(up, torch.zeros_like(up))


def test_rips_native_distance_inputs_agree():
    points = _extract_spectra(Rips(points=POINTS_RECT, max_dim=3))
    distances = _extract_spectra(Rips(distances=DISTANCES_RECT, max_dim=3))
    assert set(points) == set(distances)
    for key in points:
        np.testing.assert_allclose(points[key], distances[key], atol=1e-4, rtol=1e-3)
