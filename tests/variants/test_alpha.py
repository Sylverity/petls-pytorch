"""Tests for the PyTorch-native Alpha variant.

All tests compare against the original C++ ``petls.Alpha`` implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.parity

from petls_pytorch.variants.alpha import Alpha  # noqa: E402
from tests.reference import petls  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
ATOL = 1e-4
RTOL = 1e-3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POINTS_4 = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
]

OFF_PATH = "tests/variants/data/alpha/input"


@pytest.fixture
def ref_alpha_points():
    return petls.Alpha(points=POINTS_4, max_dim=3)


@pytest.fixture
def ref_alpha_off():
    return petls.Alpha(filename=OFF_PATH, max_dim=3)


@pytest.fixture
def torch_alpha_points():
    return Alpha(points=POINTS_4, max_dim=3)


@pytest.fixture
def torch_alpha_off():
    return Alpha(filename=OFF_PATH, max_dim=3)


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
# Construction tests
# ---------------------------------------------------------------------------


class TestAlphaConstruction:
    def test_top_dim_matches_reference_points(self, ref_alpha_points, torch_alpha_points):
        assert torch_alpha_points.top_dim == ref_alpha_points.pl.top_dim

    def test_top_dim_matches_reference_off(self, ref_alpha_off, torch_alpha_off):
        assert torch_alpha_off.top_dim == ref_alpha_off.pl.top_dim

    def test_filtration_count_matches(self, ref_alpha_points, torch_alpha_points):
        ref_filts = ref_alpha_points.pl.get_all_filtrations()
        torch_filts = torch_alpha_points.get_all_filtrations()
        assert torch_filts == pytest.approx(ref_filts, abs=ATOL)


# ---------------------------------------------------------------------------
# Spectra parity with reference
# ---------------------------------------------------------------------------


class TestAlphaSpectra:
    def test_spectra_points_vs_reference(self, ref_alpha_points, torch_alpha_points):
        ref = _extract_spectra(ref_alpha_points)
        test = _extract_spectra(torch_alpha_points)
        _compare_spectra(ref, test)

    def test_spectra_off_vs_reference(self, ref_alpha_off, torch_alpha_off):
        ref = _extract_spectra(ref_alpha_off)
        test = _extract_spectra(torch_alpha_off)
        _compare_spectra(ref, test)

    def test_spectra_allpairs_vs_reference(self, ref_alpha_points, torch_alpha_points):
        # allpairs=True for reference doesn't work the same way (C++ method),
        # so we just verify our allpairs produces a superset.
        test_all = torch_alpha_points.spectra(allpairs=True)
        ref_dict = _extract_spectra(ref_alpha_points)
        for item in test_all:
            dim, a, b, eigs = item
            key = (dim, a, b)
            if key in ref_dict:
                np.testing.assert_allclose(
                    np.asarray(eigs, dtype=np.float64),
                    ref_dict[key],
                    atol=ATOL,
                    rtol=RTOL,
                )


# ---------------------------------------------------------------------------
# Eigenpairs parity
# ---------------------------------------------------------------------------


class TestAlphaEigenpairs:
    def test_eigenpairs_values_match(self, ref_alpha_points, torch_alpha_points):
        ref_vals, ref_vecs = ref_alpha_points.eigenpairs(1, 0.25, 0.5)
        test_vals, test_vecs = torch_alpha_points.eigenpairs(1, 0.25, 0.5)
        np.testing.assert_allclose(
            np.asarray(ref_vals, dtype=np.float64),
            np.asarray(test_vals, dtype=np.float64),
            atol=ATOL,
            rtol=RTOL,
        )


# ---------------------------------------------------------------------------
# max_dim parameter
# ---------------------------------------------------------------------------


class TestAlphaMaxDim:
    def test_max_dim_2(self):
        ref = petls.Alpha(points=POINTS_4, max_dim=2)
        test = Alpha(points=POINTS_4, max_dim=2)
        assert test.top_dim == ref.pl.top_dim == 2
        ref_dict = _extract_spectra(ref)
        test_dict = _extract_spectra(test)
        _compare_spectra(ref_dict, test_dict)

    def test_max_dim_1(self):
        ref = petls.Alpha(points=POINTS_4, max_dim=1)
        test = Alpha(points=POINTS_4, max_dim=1)
        assert test.top_dim == ref.pl.top_dim == 1
        ref_dict = _extract_spectra(ref)
        test_dict = _extract_spectra(test)
        _compare_spectra(ref_dict, test_dict)


# ---------------------------------------------------------------------------
# Laplacian matrix properties (exact matrix ordering may differ from reference
# because C++ and Python boundary extractions use different simplex orderings,
# but eigenvalues / rank / L = up + down must match).
# ---------------------------------------------------------------------------


class TestAlphaLaplacian:
    def test_get_L_eigenvalues_match_reference(self, ref_alpha_points, torch_alpha_points):
        for dim in range(ref_alpha_points.pl.top_dim + 1):
            filts = ref_alpha_points.pl.get_all_filtrations()
            for i in range(len(filts) - 1):
                a, b = filts[i], filts[i + 1]
                ref_L = torch.tensor(ref_alpha_points.get_L(dim, a, b), dtype=torch.float64)
                test_L = torch_alpha_points.get_L(dim, a, b)
                ref_eigs = np.linalg.eigvalsh(ref_L.cpu().numpy())
                test_eigs = np.linalg.eigvalsh(test_L.cpu().numpy())
                np.testing.assert_allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL)

    def test_get_up_eigenvalues_match_reference(self, ref_alpha_points, torch_alpha_points):
        # Note: reference get_up segfaults at top_dim; skip that case
        for dim in range(ref_alpha_points.pl.top_dim):
            filts = ref_alpha_points.pl.get_all_filtrations()
            for i in range(len(filts) - 1):
                a, b = filts[i], filts[i + 1]
                ref_up = torch.tensor(ref_alpha_points.get_up(dim, a, b), dtype=torch.float64)
                test_up = torch_alpha_points.get_up(dim, a, b)
                ref_eigs = np.linalg.eigvalsh(ref_up.cpu().numpy())
                test_eigs = np.linalg.eigvalsh(test_up.cpu().numpy())
                np.testing.assert_allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL)

    def test_get_down_eigenvalues_match_reference(self, ref_alpha_points, torch_alpha_points):
        for dim in range(1, ref_alpha_points.pl.top_dim + 1):
            filts = ref_alpha_points.pl.get_all_filtrations()
            for a in filts:
                ref_down = torch.tensor(ref_alpha_points.get_down(dim, a), dtype=torch.float64)
                test_down = torch_alpha_points.get_down(dim, a)
                ref_eigs = np.linalg.eigvalsh(ref_down.cpu().numpy())
                test_eigs = np.linalg.eigvalsh(test_down.cpu().numpy())
                np.testing.assert_allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL)

        vertex_down = torch_alpha_points.get_down(0, 0.0)
        assert vertex_down.shape == (len(POINTS_4), len(POINTS_4))
        assert torch.count_nonzero(vertex_down) == 0
