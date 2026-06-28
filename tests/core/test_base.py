"""
Port of PETLS tests/core/test_base.py to the PyTorch implementation.

Validates that our Laplacian construction (get_down, get_up, get_L) matches
the C++ reference exactly on the small canonical test complex.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from petls_torch.core.complex import Complex
from tests.conftest import assert_tensors_close


# ---------------------------------------------------------------------------
# Fixture helpers (same data as original test_base.py::get_pl())
# ---------------------------------------------------------------------------


def get_small_complex():
    """Exact data from original test_base.py::get_pl()."""
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float32)
    d2 = np.array([[1], [1], [-1]], dtype=np.float32)
    boundaries = [d1, d2]
    filtrations = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [5.0]]
    return Complex(boundaries=boundaries, filtrations=filtrations)


# ---------------------------------------------------------------------------
# Construction & data integrity
# ---------------------------------------------------------------------------


def test_construction_matches_reference(ref_small_complex):
    """Our Complex stores the same topological data as the reference."""
    pl = get_small_complex()
    assert pl.top_dim == ref_small_complex.pl.top_dim
    assert pl.top_dim == 2


def test_get_all_filtrations_matches_reference(ref_small_complex):
    """Filtration extraction should match reference exactly."""
    pl = get_small_complex()
    our_filts = pl.get_all_filtrations()
    ref_filts = ref_small_complex.pl.get_all_filtrations()
    assert our_filts == ref_filts


def test_filtration_list_to_spectra_request():
    """Request generation should match reference logic."""
    pl = get_small_complex()
    filtrations = [0.0, 3.0, 5.0]
    dims = [0, 1, 2]
    requests = pl.filtration_list_to_spectra_request(filtrations, dims)

    expected_len = (len(filtrations) - 1) * len(dims) + len(dims)
    assert len(requests) == expected_len
    assert requests[0] == (0, 0.0, 3.0)
    assert requests[-1] == (2, 5.0, 5.0)


# ---------------------------------------------------------------------------
# Laplacian construction — compared against C++ reference
# ---------------------------------------------------------------------------


def test_get_down_matches_reference(ref_small_complex):
    """get_down(1, 5) should match reference exactly."""
    pl = get_small_complex()
    our_down = pl.get_down(1, 5.0).cpu().numpy()
    ref_down = ref_small_complex.pl.get_down(1, 5.0)
    assert_tensors_close(torch.from_numpy(our_down), ref_down, atol=1e-5, rtol=1e-4)


def test_get_up_matches_reference(ref_small_complex):
    """get_up(0, 1, 3) should match reference exactly."""
    pl = get_small_complex()
    our_up = pl.get_up(0, 1.0, 3.0).cpu().numpy()
    ref_up = ref_small_complex.pl.get_up(0, 1.0, 3.0)
    assert_tensors_close(torch.from_numpy(our_up), ref_up, atol=1e-5, rtol=1e-4)


def test_get_L_matches_reference(ref_small_complex):
    """get_L(1, 5, 5) should match reference exactly."""
    pl = get_small_complex()
    our_L = pl.get_L(1, 5.0, 5.0).cpu().numpy()
    ref_L = ref_small_complex.pl.get_L(1, 5.0, 5.0)
    assert_tensors_close(torch.from_numpy(our_L), ref_L, atol=1e-5, rtol=1e-4)


def test_sum_up_down(ref_small_complex):
    """L = up + down (the defining property of the Laplacian)."""
    pl = get_small_complex()
    down = pl.get_down(1, 5.0)
    up = pl.get_up(1, 5.0, 5.0)
    L = pl.get_L(1, 5.0, 5.0)
    assert_tensors_close(L, up + down, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# Spectra — compared against C++ reference
# ---------------------------------------------------------------------------


def test_spectra_specific(ref_small_complex):
    """spectra(0, 5, 6) should match reference exactly."""
    pl = get_small_complex()
    our = pl.spectra(0, 5.0, 6.0)
    ref = ref_small_complex.pl.spectra(0, 5.0, 6.0)
    ref_list = ref.tolist() if hasattr(ref, "tolist") else ref
    assert pytest.approx(our, abs=1e-4) == ref_list


def test_spectra_single_dim_0(ref_small_complex):
    """spectra(0, 1.2, 4.5) should match reference."""
    pl = get_small_complex()
    our = pl.spectra(0, 1.2, 4.5)
    ref = ref_small_complex.pl.spectra(0, 1.2, 4.5)
    ref_list = ref.tolist() if hasattr(ref, "tolist") else ref
    assert pytest.approx(our, abs=1e-4) == ref_list


def test_spectra_single_dim_1(ref_small_complex):
    """spectra(1, 3, 4) should match reference."""
    pl = get_small_complex()
    our = pl.spectra(1, 3.0, 4.0)
    ref = ref_small_complex.pl.spectra(1, 3.0, 4.0)
    ref_list = ref.tolist() if hasattr(ref, "tolist") else ref
    assert pytest.approx(our, abs=1e-4) == ref_list


def test_spectra_all(ref_small_complex):
    """spectra() with no args — all successive filtrations."""
    pl = get_small_complex()
    our = pl.spectra()
    ref = ref_small_complex.pl.spectra()

    assert len(our) == len(ref)
    for o, r in zip(our, ref):
        assert o[0] == r[0]  # dim
        assert abs(o[1] - r[1]) < 1e-6  # a
        assert abs(o[2] - r[2]) < 1e-6  # b
        ref_eigs = r[3].tolist() if hasattr(r[3], "tolist") else r[3]
        assert pytest.approx(o[3], abs=1e-4) == ref_eigs


def test_spectra_allpairs(ref_small_complex):
    """spectra(allpairs=True) — all filtration combinations."""
    pl = get_small_complex()
    our = pl.spectra(allpairs=True)
    ref = ref_small_complex.pl.spectra_allpairs()

    assert len(our) == len(ref)
    for o, r in zip(our, ref):
        assert o[0] == r[0]
        assert abs(o[1] - r[1]) < 1e-6
        assert abs(o[2] - r[2]) < 1e-6
        ref_eigs = r[3].tolist() if hasattr(r[3], "tolist") else r[3]
        assert pytest.approx(o[3], abs=1e-4) == ref_eigs


def test_spectra_request_list(ref_small_complex):
    """spectra(request_list=...) should match reference."""
    pl = get_small_complex()
    req = [[0, 5.0, 6.0], [1, 3.0, 4.0], [2, 5.0, 5.0]]
    our = pl.spectra(request_list=req)
    ref = ref_small_complex.pl.spectra(req)

    assert len(our) == len(ref)
    for o, r in zip(our, ref):
        assert o[0] == r[0]
        assert abs(o[1] - r[1]) < 1e-6
        assert abs(o[2] - r[2]) < 1e-6
        ref_eigs = r[3].tolist() if hasattr(r[3], "tolist") else r[3]
        assert pytest.approx(o[3], abs=1e-4) == ref_eigs


# ---------------------------------------------------------------------------
# Eigenpairs — compared against C++ reference
# ---------------------------------------------------------------------------


def test_eigenpairs_single(ref_small_complex):
    """eigenpairs(0, 5, 6) values should match reference."""
    pl = get_small_complex()
    our_vals, our_vecs = pl.eigenpairs(0, 5.0, 6.0)
    ref_vals, ref_vecs = ref_small_complex.pl.eigenpairs(0, 5.0, 6.0)

    ref_vals_list = ref_vals.tolist() if hasattr(ref_vals, "tolist") else ref_vals
    assert pytest.approx(our_vals, abs=1e-4) == ref_vals_list
    assert our_vecs.shape == tuple(ref_vecs.shape)

    # Verify eigenvector property: L v = lambda v
    L = pl.get_L(0, 5.0, 6.0)
    for i, val in enumerate(our_vals):
        if abs(val) < 1e-4:
            continue  # skip near-zero for numerical stability
        lhs = L @ our_vecs[:, i]
        rhs = val * our_vecs[:, i]
        assert_tensors_close(lhs, rhs, atol=1e-3, rtol=1e-3)


def test_eigenpairs_request_list(ref_small_complex):
    """eigenpairs(request_list=...) should match reference values."""
    pl = get_small_complex()
    req = [[0, 5.0, 6.0], [1, 3.0, 4.0]]
    our = pl.eigenpairs(request_list=req)
    ref = ref_small_complex.pl.eigenpairs(req)

    assert len(our) == len(ref)
    for o, r in zip(our, ref):
        assert o[0] == r[0]
        assert abs(o[1] - r[1]) < 1e-6
        assert abs(o[2] - r[2]) < 1e-6
        ref_eigs = r[3].tolist() if hasattr(r[3], "tolist") else r[3]
        assert pytest.approx(o[3], abs=1e-4) == ref_eigs
        assert o[4].shape == tuple(r[4].shape)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_get_L_dim_zero_is_up_only(ref_small_complex):
    """dim=0: L = L_up only (no down-Laplacian)."""
    pl = get_small_complex()
    L = pl.get_L(0, 0.0, 3.0)
    up = pl.get_up(0, 0.0, 3.0)
    assert_tensors_close(L, up, atol=1e-5, rtol=1e-4)


def test_get_L_top_dim_is_down_only(ref_small_complex):
    """dim=top_dim: L = L_down only (no up-Laplacian)."""
    pl = get_small_complex()
    L = pl.get_L(2, 5.0, 5.0)
    down = pl.get_down(2, 5.0)
    assert_tensors_close(L, down, atol=1e-5, rtol=1e-4)


def test_get_L_beyond_top_dim_is_empty():
    """dim > top_dim returns empty matrix."""
    pl = get_small_complex()
    L = pl.get_L(10, 0.0, 1.0)
    assert L.shape == (0, 0)


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------


def test_eigenvalues_summarize():
    """Matches C++ eigenvalues_summarize with 1e-4 tolerance."""
    pl = get_small_complex()

    betti, lam = pl.eigenvalues_summarize([0.0, 0.0, 0.0, 0.0])
    assert betti == 4
    assert lam == 0.0

    betti, lam = pl.eigenvalues_summarize([0.0, 0.0, 1.5])
    assert betti == 2
    assert abs(lam - 1.5) < 1e-6

    betti, lam = pl.eigenvalues_summarize([1.0, 2.0, 3.0])
    assert betti == 0
    assert abs(lam - 1.0) < 1e-6

    betti, lam = pl.eigenvalues_summarize([])
    assert betti == 0
    assert lam == 0.0


def test_eigenvalues_summarize_matches_reference(ref_small_complex):
    """Against reference on known spectra."""
    pl = get_small_complex()
    ref = ref_small_complex

    test_cases = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [],
    ]
    for eigs in test_cases:
        our_betti, our_lam = pl.eigenvalues_summarize(eigs)
        ref_betti, ref_lam = ref.pl.eigenvalues_summarize(eigs)
        assert our_betti == ref_betti
        assert abs(our_lam - ref_lam) < 1e-6


# ---------------------------------------------------------------------------
# Regression: exact small example from original docstring
# ---------------------------------------------------------------------------


def test_small_example_data():
    """
    The exact example from the original Complex docstring.
    """
    pl = get_small_complex()
    assert pl.top_dim == 2
    assert pl.filtered_boundaries[1].shape == (3, 3)  # d1
    assert pl.filtered_boundaries[2].shape == (3, 1)  # d2

    assert torch.allclose(
        pl.filtered_boundaries[1].domain_filtrations.cpu(),
        torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64),
    )
    assert torch.allclose(
        pl.filtered_boundaries[1].range_filtrations.cpu(),
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
    )
