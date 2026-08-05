"""Focused API behavior that complements numerical PETLS parity tests."""

import numpy as np
import pytest
import torch

import petls_pytorch


@pytest.fixture
def small_complex():
    """Small test complex with known spectra."""
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float32)
    d2 = np.array([[1], [1], [-1]], dtype=np.float32)
    filtrations = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [5.0]]
    return petls_pytorch.Complex([d1, d2], filtrations)


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


def test_set_eigs_algorithm_kwargs(small_complex):
    """set_eigs_algorithm accepts num_eigenvalues and eigenvalue_order."""
    small_complex.set_eigs_algorithm("sparse", num_eigenvalues=2, eigenvalue_order="LM")
    assert small_complex._num_eigenvalues == 2
    assert small_complex._eigenvalue_order == "LM"


@pytest.mark.parametrize("algorithm", ["selfadjoint", "eigensolver", "bdcsvd", "spectra"])
def test_cpp_solver_aliases_are_not_supported(small_complex, algorithm):
    with pytest.raises(ValueError, match="algorithm must be"):
        small_complex.set_eigs_algorithm(algorithm)


def test_print_boundaries_does_not_raise(small_complex, capsys):
    """print_boundaries is exposed and prints boundary information."""
    small_complex.print_boundaries()
    captured = capsys.readouterr()
    assert "d_0" in captured.out


def test_profile_wrap_up(small_complex):
    """Profile.wrap_up records one computation and computes Betti/λ."""
    profile = petls_pytorch.Profile()
    profile.wrap_up(dim=0, a=3.0, b=4.0, L_rows=3, eigs=[0.0, 1.0, 2.0])
    assert profile.dims == [0]
    assert profile.filtration_a == [3.0]
    assert profile.filtration_b == [4.0]
    assert profile.L_rows == [3]
    assert profile.bettis == [1]
    assert profile.lambdas == [1.0]


def test_profile_uses_scale_aware_object_tolerance():
    profile = petls_pytorch.Profile(zero_atol=0.1, zero_rtol=0.0)
    profile.wrap_up(dim=0, a=0.0, b=0.0, L_rows=3, eigs=[0.0, 0.05, 1.0])

    assert profile.bettis == [2]
    assert profile.lambdas == [1.0]

    complex_ = petls_pytorch.Complex(
        boundaries=[],
        filtrations=[[0.0]],
        zero_atol=0.2,
        zero_rtol=0.03,
    )
    assert complex_.profile.zero_atol == pytest.approx(0.2)
    assert complex_.profile.zero_rtol == pytest.approx(0.03)
