"""Tests for nonzero_spectra()."""

from __future__ import annotations

import numpy as np

from petls_torch.core.complex import Complex


def get_test_complex():
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)
    d2 = np.array([[1], [1], [-1]], dtype=np.float64)
    boundaries = [d1, d2]
    filtrations = [[0, 1, 2], [3, 4, 5], [5]]
    return Complex(boundaries=boundaries, filtrations=filtrations)


def test_nonzero_spectra_vs_full():
    """nonzero_spectra should return the same nonzero values as spectra()."""
    pl = get_test_complex()
    full = np.array(pl.spectra(1, 4, 5))
    nonzero = pl.nonzero_spectra(1, 4, 5)

    expected = sorted(full[full > 1e-4])
    assert len(nonzero) == len(expected)
    np.testing.assert_allclose(nonzero, expected, atol=1e-4)


def test_nonzero_spectra_with_dummy_basis():
    """Using use_dummy_harmonic_basis=True should still give nonzero eigenvalues."""
    pl = get_test_complex()
    nonzero = pl.nonzero_spectra(0, 3, 4, use_dummy_harmonic_basis=True)
    # L0 at (0,3,4) has eigenvalues [0, 1, 3]
    assert len(nonzero) == 2
    np.testing.assert_allclose(sorted(nonzero), [1, 3], atol=1e-4)


def test_nonzero_spectra_empty_matrix():
    """For an empty matrix, nonzero_spectra should return []."""
    pl = get_test_complex()
    # dim=2, a=0, b=3 has no 2-simplices at b=3
    result = pl.nonzero_spectra(2, 0, 3)
    assert result == []
