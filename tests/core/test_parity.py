"""
Feature-parity tests against the original PETLS Python API.

These tests verify that petls_torch exposes the same public signatures and
convenience names as the original ``petls`` package.
"""

import numpy as np
import pytest
import torch

import petls_torch


@pytest.fixture
def small_complex():
    """Small test complex with known spectra."""
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float32)
    d2 = np.array([[1], [1], [-1]], dtype=np.float32)
    filtrations = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [5.0]]
    return petls_torch.Complex([d1, d2], filtrations)


def test_eigenpairs_allpairs_signature(small_complex):
    """eigenpairs accepts allpairs keyword for API parity."""
    result = small_complex.eigenpairs(allpairs=True)
    assert isinstance(result, list)
    assert len(result) > 0
    for item in result:
        assert len(item) == 5
        dim, a, b, vals, vecs = item
        assert isinstance(dim, int)
        assert isinstance(vals, list)
        assert isinstance(vecs, torch.Tensor)


def test_eigenpairs_single_positional_request_list(small_complex):
    """A single positional argument is treated as a request list."""
    request = [[0, 3.0, 4.0]]
    result = small_complex.eigenpairs(request)
    assert isinstance(result, list)
    assert len(result) == 1
    dim, a, b, vals, vecs = result[0]
    assert dim == 0
    assert a == pytest.approx(3.0)
    assert b == pytest.approx(4.0)


def test_set_eigs_algorithm_kwargs(small_complex):
    """set_eigs_algorithm accepts num_eigenvalues and eigenvalue_order."""
    small_complex.set_eigs_algorithm("sparse", num_eigenvalues=2, eigenvalue_order="LM")
    assert small_complex._num_eigenvalues == 2
    assert small_complex._eigenvalue_order == "LM"


def test_print_boundaries_does_not_raise(small_complex, capsys):
    """print_boundaries is exposed and prints boundary information."""
    small_complex.print_boundaries()
    captured = capsys.readouterr()
    assert "d_0" in captured.out


def test_profile_wrap_up(small_complex):
    """Profile.wrap_up records one computation and computes Betti/λ."""
    profile = petls_torch.Profile()
    profile.wrap_up(dim=0, a=3.0, b=4.0, L_rows=3, eigs=[0.0, 1.0, 2.0])
    assert profile.dims == [0]
    assert profile.filtration_a == [3.0]
    assert profile.filtration_b == [4.0]
    assert profile.L_rows == [3]
    assert profile.bettis == [1]
    assert profile.lambdas == [1.0]


def test_eigvalsh_export():
    """petls_torch exposes scipy.linalg.eigvalsh like the original package."""
    from scipy.linalg import eigvalsh as scipy_eigvalsh

    assert petls_torch.eigvalsh is scipy_eigvalsh


def test_coo_matrix_export():
    """petls_torch exposes scipy.sparse.coo_matrix like the original package."""
    from scipy.sparse import coo_matrix as scipy_coo_matrix

    assert petls_torch.coo_matrix is scipy_coo_matrix
