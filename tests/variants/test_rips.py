"""Tests for the PyTorch-native Rips variant.

All tests compare against the original C++ ``petls.Rips`` implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

petls = pytest.importorskip("petls", reason="Reference PETLS not available")
from petls_torch.variants.rips import Rips

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
ATOL = 1e-4
RTOL = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_spectra(alpha_obj):
    """Run ``spectra()`` and return a dict keyed by (dim, a, b)."""
    result = {}
    for item in alpha_obj.spectra():
        dim, a, b, eigs = item
        result[(dim, a, b)] = np.asarray(eigs, dtype=np.float64)
    return result


def _compare_spectra(ref_dict, test_dict):
    """Assert that two spectra dicts match within tolerance."""
    assert set(ref_dict.keys()) == set(test_dict.keys()), (
        f"Key mismatch: ref={set(ref_dict.keys())} vs test={set(test_dict.keys())}"
    )
    for key in sorted(ref_dict.keys()):
        ref_eigs = ref_dict[key]
        test_eigs = test_dict[key]
        assert ref_eigs.shape == test_eigs.shape, (
            f"Shape mismatch at {key}: ref={ref_eigs.shape} vs test={test_eigs.shape}"
        )
        np.testing.assert_allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POINTS_RECT = np.array([
    [0, 0],
    [0, 3],
    [4, 0],
    [4, 3],
])

DISTANCES_RECT = np.array([
    [0, 0, 0, 0],
    [3, 0, 0, 0],
    [4, 5, 0, 0],
    [5, 4, 3, 0],
], dtype=np.float64)

FILE_PATH = "tests/variants/data/rips/rect.lower_distance_matrix"


def _rips_points(threshold=None):
    return Rips(points=POINTS_RECT, max_dim=3, threshold=threshold)


def _rips_distances(threshold=None):
    return Rips(distances=DISTANCES_RECT, max_dim=3, threshold=threshold)


def _rips_file(threshold=None):
    return Rips(filename=FILE_PATH, max_dim=3, threshold=threshold)


def _ref_rips_points(threshold=None):
    return petls.Rips(points=POINTS_RECT, max_dim=3, threshold=threshold)


def _ref_rips_distances(threshold=None):
    return petls.Rips(distances=DISTANCES_RECT, max_dim=3, threshold=threshold)


def _ref_rips_file(threshold=None):
    return petls.Rips(filename=FILE_PATH, max_dim=3, threshold=threshold)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestRipsConstruction:
    def test_top_dim_matches_reference_points(self):
        ref = _ref_rips_points()
        test = _rips_points()
        assert test.top_dim == ref.pl.top_dim

    def test_top_dim_matches_reference_distances(self):
        ref = _ref_rips_distances()
        test = _rips_distances()
        assert test.top_dim == ref.pl.top_dim

    def test_top_dim_matches_reference_file(self):
        ref = _ref_rips_file()
        test = _rips_file()
        assert test.top_dim == ref.pl.top_dim

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="requires filename, point set, or distance matrix"):
            Rips()


# ---------------------------------------------------------------------------
# Spectra parity with reference (no threshold)
# ---------------------------------------------------------------------------

class TestRipsSpectraNoThreshold:
    def test_points_vs_reference(self):
        ref = _ref_rips_points()
        test = _rips_points()
        _compare_spectra(_extract_spectra(ref), _extract_spectra(test))

    def test_distances_vs_reference(self):
        ref = _ref_rips_distances()
        test = _rips_distances()
        _compare_spectra(_extract_spectra(ref), _extract_spectra(test))

    def test_file_vs_reference(self):
        ref = _ref_rips_file()
        test = _rips_file()
        _compare_spectra(_extract_spectra(ref), _extract_spectra(test))

    def test_all_three_inputs_agree(self):
        pts = _extract_spectra(_rips_points())
        dist = _extract_spectra(_rips_distances())
        fil = _extract_spectra(_rips_file())
        _compare_spectra(pts, dist)
        _compare_spectra(pts, fil)


# ---------------------------------------------------------------------------
# Spectra parity with reference (with threshold=4.5)
# ---------------------------------------------------------------------------

class TestRipsSpectraThreshold:
    def test_points_vs_reference(self):
        ref = _ref_rips_points(threshold=4.5)
        test = _rips_points(threshold=4.5)
        _compare_spectra(_extract_spectra(ref), _extract_spectra(test))

    def test_distances_vs_reference(self):
        ref = _ref_rips_distances(threshold=4.5)
        test = _rips_distances(threshold=4.5)
        _compare_spectra(_extract_spectra(ref), _extract_spectra(test))

    def test_file_vs_reference(self):
        ref = _ref_rips_file(threshold=4.5)
        test = _rips_file(threshold=4.5)
        _compare_spectra(_extract_spectra(ref), _extract_spectra(test))

    def test_all_three_inputs_agree(self):
        pts = _extract_spectra(_rips_points(threshold=4.5))
        dist = _extract_spectra(_rips_distances(threshold=4.5))
        fil = _extract_spectra(_rips_file(threshold=4.5))
        _compare_spectra(pts, dist)
        _compare_spectra(pts, fil)


# ---------------------------------------------------------------------------
# Laplacian matrix properties
# ---------------------------------------------------------------------------

class TestRipsLaplacian:
    def test_get_L_eigenvalues_match_reference(self):
        ref = _ref_rips_points()
        test = _rips_points()
        filts = ref.pl.get_all_filtrations()
        for dim in range(ref.pl.top_dim + 1):
            for i in range(len(filts) - 1):
                a, b = filts[i], filts[i + 1]
                ref_L = torch.tensor(ref.get_L(dim, a, b), dtype=torch.float64)
                test_L = test.get_L(dim, a, b)
                ref_eigs = np.linalg.eigvalsh(ref_L.cpu().numpy())
                test_eigs = np.linalg.eigvalsh(test_L.cpu().numpy())
                np.testing.assert_allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL)

    def test_get_down_eigenvalues_match_reference(self):
        ref = _ref_rips_points()
        test = _rips_points()
        filts = ref.pl.get_all_filtrations()
        for dim in range(ref.pl.top_dim + 1):
            for a in filts:
                ref_down = torch.tensor(ref.get_down(dim, a), dtype=torch.float64)
                test_down = test.get_down(dim, a)
                ref_eigs = np.linalg.eigvalsh(ref_down.cpu().numpy())
                test_eigs = np.linalg.eigvalsh(test_down.cpu().numpy())
                np.testing.assert_allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL)

    def test_L_equals_up_plus_down(self):
        test = _rips_points()
        filts = test.get_all_filtrations()
        for dim in range(test.top_dim + 1):
            for i in range(len(filts) - 1):
                a, b = filts[i], filts[i + 1]
                L = test.get_L(dim, a, b)
                up = test.get_up(dim, a, b)
                down = test.get_down(dim, a)
                np.testing.assert_allclose(
                    L.cpu().numpy(),
                    (up + down).cpu().numpy(),
                    atol=ATOL,
                    rtol=RTOL,
                )

    def test_get_up_top_dim_is_zero(self):
        test = _rips_points()
        filts = test.get_all_filtrations()
        for a, b in [(filts[0], filts[1]), (filts[-2], filts[-1])]:
            up = test.get_up(test.top_dim, a, b)
            assert up.shape[0] == up.shape[1]
            assert torch.allclose(up, torch.zeros_like(up))
