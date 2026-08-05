"""Tests for the flipped top-dimension optimization."""

from __future__ import annotations

import numpy as np
import pytest

from petls_pytorch.core.complex import Complex, LaplacianSizeError


ATOL = 1e-4
RTOL = 1e-3


def get_test_complex():
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)
    d2 = np.array([[1], [1], [-1]], dtype=np.float64)
    boundaries = [d1, d2]
    filtrations = [[0, 1, 2], [3, 4, 5], [5]]
    return Complex(boundaries=boundaries, filtrations=filtrations)


def test_get_L_top_dim_flipped_shape():
    pl = get_test_complex()
    L_flip = pl.get_L_top_dim_flipped(5)
    # d2 is 3x1, so B @ B.T is 3x3
    assert L_flip.shape == (3, 3)


def test_flipped_eigenvalues_match_regular():
    """When flipped=True, spectra() for top_dim should match regular spectra()."""
    pl = get_test_complex()

    eigs_regular = pl.spectra(2, 5, 5)
    pl.flipped = True
    eigs_flipped = pl.spectra(2, 5, 5)

    np.testing.assert_allclose(eigs_regular, eigs_flipped, atol=ATOL, rtol=RTOL)


def test_flipped_optimization_preserves_spectrum_when_matrix_is_smaller():
    boundary = np.array([[-1.0, -1.0, 0.0], [1.0, 0.0, 1.0]])
    complex_ = Complex(
        boundaries=[boundary],
        filtrations=[[0.0, 0.0], [0.0, 0.0, 0.0]],
    )

    regular = complex_.spectra(1, 0.0, 0.0)
    assert complex_.get_L_top_dim_flipped(0.0).shape == (2, 2)
    complex_.flipped = True
    flipped = complex_.spectra(1, 0.0, 0.0)

    np.testing.assert_allclose(regular, flipped, atol=ATOL, rtol=RTOL)


def test_flipped_path_guards_actual_size_and_is_used_only_when_smaller():
    boundary = np.ones((10, 1), dtype=np.float64)
    complex_ = Complex(
        boundaries=[boundary],
        filtrations=[[0.0] * 10, [0.0]],
        max_matrix_rows=5,
    )
    complex_.flipped = True

    estimate = complex_.estimate_laplacian(dim=1, a=0.0)
    assert estimate["peak_rows"] == 1
    with pytest.raises(LaplacianSizeError, match="would have 10 rows"):
        complex_.get_L_top_dim_flipped(0.0)

    # spectra() falls back to the guarded 1x1 B.T @ B calculation instead of
    # attempting the larger 10x10 flipped matrix.
    assert complex_.spectra(1, 0.0, 0.0) == pytest.approx([10.0])


def test_flipped_does_not_affect_lower_dims():
    """flipped should only affect the top dimension."""
    pl = get_test_complex()

    eigs_0_regular = pl.spectra(0, 3, 4)
    eigs_1_regular = pl.spectra(1, 4, 5)

    pl.flipped = True
    eigs_0_flipped = pl.spectra(0, 3, 4)
    eigs_1_flipped = pl.spectra(1, 4, 5)

    np.testing.assert_allclose(eigs_0_regular, eigs_0_flipped, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(eigs_1_regular, eigs_1_flipped, atol=ATOL, rtol=RTOL)


def test_flipped_allpairs():
    pl = get_test_complex()
    pl.flipped = True
    result = pl.spectra(allpairs=True)
    assert isinstance(result, list)
    # Check that top-dim entries have correct number of eigenvalues
    for item in result:
        dim, a, b, eigs = item
        if dim == 2:
            # d2 has 1 column, so top-dim Laplacian is 1x1
            assert len(eigs) <= 1
