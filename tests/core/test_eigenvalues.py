"""
Tests for eigenvalue solvers.

These can run immediately since solve_eigenvalues is already implemented
using torch.linalg.eigvalsh (GPU-accelerated via cuSOLVER).
"""

from __future__ import annotations

import pytest
import torch

from petls_torch.core.eigenvalues import solve_eigenvalues, solve_eigenpairs
from tests.conftest import assert_tensors_close


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_eigvalsh_on_diagonal():
    """Eigenvalues of a diagonal matrix are the diagonal entries."""
    L = torch.diag(torch.tensor([3.0, 1.0, 2.0]))
    vals = solve_eigenvalues(L, algorithm="eigvalsh")
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert_tensors_close(vals, expected)


def test_eigvalsh_on_tridiagonal():
    """Laplacian of a path graph: known spectrum."""
    n = 5
    L = torch.zeros(n, n)
    for i in range(n):
        L[i, i] = 2.0 if 0 < i < n - 1 else 1.0
        if i > 0:
            L[i, i - 1] = -1.0
            L[i - 1, i] = -1.0

    vals = solve_eigenvalues(L, algorithm="eigvalsh")
    # First eigenvalue of connected graph Laplacian is 0
    assert abs(vals[0].item()) < 1e-5
    assert vals[-1].item() > 3.0


def test_eigvalsh_on_empty():
    """Empty matrix should return empty eigenvalues."""
    L = torch.empty(0, 0)
    vals = solve_eigenvalues(L, algorithm="eigvalsh")
    assert vals.numel() == 0


def test_eigvalsh_on_1x1():
    """1x1 matrix returns its single element."""
    L = torch.tensor([[5.0]])
    vals = solve_eigenvalues(L, algorithm="eigvalsh")
    assert vals.numel() == 1
    assert abs(vals[0].item() - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# GPU execution (if available)
# ---------------------------------------------------------------------------


def test_eigvalsh_on_gpu():
    """Verify solver runs on CUDA when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    L = torch.diag(torch.tensor([3.0, 1.0, 2.0])).cuda()
    vals = solve_eigenvalues(L, algorithm="eigvalsh")
    assert vals.device.type == "cuda"
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert_tensors_close(vals, expected)


# ---------------------------------------------------------------------------
# Eigenpairs
# ---------------------------------------------------------------------------


def test_eigenpairs_basic():
    """eigenpairs returns sorted values and orthonormal vectors."""
    L = torch.diag(torch.tensor([3.0, 1.0, 2.0]))
    vals, vecs = solve_eigenpairs(L, algorithm="eigh")
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert_tensors_close(vals, expected)

    # Check orthonormality
    identity = vecs.T @ vecs
    assert_tensors_close(identity, torch.eye(3))


# ---------------------------------------------------------------------------
# Callable algorithm
# ---------------------------------------------------------------------------


def test_custom_algorithm():
    """Pass a custom callable as algorithm."""

    def custom_solver(L: torch.Tensor) -> torch.Tensor:
        return torch.linalg.eigvalsh(L) * 2  # nonsense but testable

    L = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    vals = solve_eigenvalues(L, algorithm=custom_solver)
    # Should be doubled
    assert abs(vals[0].item() - 2.0) < 1e-6
