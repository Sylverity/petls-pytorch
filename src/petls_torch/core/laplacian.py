"""
Laplacian construction: get_up, get_down, get_L.

GPU-native implementations of the persistent Laplacian formulas:
  L_down^{dim}(a)  = B_{dim}^a^T @ B_{dim}^a
  L_up^{dim}(a,b)  = Schur complement on B_{dim+1}
  L^{dim}(a,b)     = L_up + L_down   (except edge cases dim=0 and dim=top_dim)

Reference: Memoli, Wan, Wang 2020 (Schur complement algorithm).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from petls_torch._config import get_device, get_dtype

if TYPE_CHECKING:
    from petls_torch.core.filtered_boundary import FilteredBoundaryMatrix


def _symmetrize_lower(L: torch.Tensor) -> torch.Tensor:
    """
    Copy lower triangle to upper triangle, matching Eigen::selfadjointView<Lower>.
    """
    L_tril = torch.tril(L)
    return L_tril + L_tril.T - torch.diag(L_tril.diagonal())


def get_down(
    fbm: FilteredBoundaryMatrix,
    a: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the persistent down-Laplacian L_down^{dim}(a).

    Formula: B^T @ B restricted to filtration a.

    Parameters
    ----------
    fbm : FilteredBoundaryMatrix
        Boundary matrix B_{dim} (maps dim-simplices to (dim-1)-simplices).
    a : float
        Filtration value.
    device : torch.device, optional
        Override target device.

    Returns
    -------
    torch.Tensor
        Dense symmetric matrix of shape (n, n) where n is the number of
        dim-simplices present at filtration a.
    """
    B = fbm.submatrix_at_filtration(a)

    if B.shape[0] == 0 or B.shape[1] == 0:
        # Empty or degenerate boundary → zero Laplacian
        target_device = device if device is not None else B.device
        return torch.zeros(
            B.shape[1],
            B.shape[1],
            dtype=get_dtype(),
            device=target_device,
        )

    # Sparse → dense, then dense GEMM (fast on GPU)
    B_dense = B.to_dense().to(dtype=get_dtype())
    L_down = B_dense.T @ B_dense
    return L_down


def get_up(
    fbm: FilteredBoundaryMatrix,
    a: float,
    b: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the persistent up-Laplacian L_up^{dim}(a,b) via Schur complement.

    This is the GPU-native replacement for C++ schur_algorithm().

    Algorithm (Memoli-Wan-Wang):
      1. Extract B_pers = submatrix of B_{dim+1} at filtration b.
      2. Compute L_up^b = B_pers @ B_pers^T.
      3. If no new rows between a and b: return L_up^b.
      4. Partition L_up^b into blocks [A B; C D] at row index a.
      5. Solve D X = C for X via Cholesky (D is SPD).
      6. L_up = A - B @ X.
      7. Symmetrize from lower triangle.

    Parameters
    ----------
    fbm : FilteredBoundaryMatrix
        Boundary matrix B_{dim+1} (maps (dim+1)-simplices to dim-simplices).
    a, b : float
        Filtration values with b >= a.
    device : torch.device, optional
        Override target device.

    Returns
    -------
    torch.Tensor
        Dense symmetric matrix of shape (n, n) where n is the number of
        dim-simplices present at filtration a.
    """
    a_row = fbm.index_of_filtration(use_domain=False, a=a)
    b_row = fbm.index_of_filtration(use_domain=False, a=b)
    b_col = fbm.index_of_filtration(use_domain=True, a=b)

    target_device = device if device is not None else fbm.device
    dtype = get_dtype()

    # Edge cases (match C++ exactly)
    if a_row == b_row:
        # No new dim-simplices between a and b → standard up-Laplacian at b
        if b_col < 0:
            return torch.zeros(a_row + 1, a_row + 1, dtype=dtype, device=target_device)
        B_pers = fbm.submatrix_at_filtration(b)
        B_dense = B_pers.to_dense().to(dtype=dtype)
        return B_dense @ B_dense.T

    if a_row == -1:
        # No dim-simplices at a
        return torch.empty(0, 0, dtype=dtype, device=target_device)

    if b_col == -1:
        # No (dim+1)-simplices at b → zero matrix sized to dim-simplices at a
        return torch.zeros(a_row + 1, a_row + 1, dtype=dtype, device=target_device)

    B_pers = fbm.submatrix_at_filtration(b)
    B_dense = B_pers.to_dense().to(dtype=dtype)

    L_up_b = B_dense @ B_dense.T

    a_rows = a_row + 1
    b_rows = b_row + 1

    A = L_up_b[:a_rows, :a_rows]
    C = L_up_b[a_rows:b_rows, :a_rows]
    B_block = C.T
    D = L_up_b[a_rows:b_rows, a_rows:b_rows]

    # D is SPD in theory but can be singular when some rows of B_pers are
    # zero. The original C++ uses LDLT which handles semidefinite matrices;
    # we fall back to the pseudoinverse for numerical stability.
    try:
        L_chol = torch.linalg.cholesky(D)
        X = torch.cholesky_solve(C, L_chol)
    except RuntimeError:
        X = torch.linalg.pinv(D) @ C

    L_up = A - B_block @ X
    L_up = _symmetrize_lower(L_up)

    return L_up


def get_L(
    dim: int,
    a: float,
    b: float,
    filtered_boundaries: list[FilteredBoundaryMatrix],
    top_dim: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the persistent Laplacian L^{dim}(a,b) = L_up + L_down.

    Handles edge cases:
      dim == 0       → L = L_up only
      dim == top_dim → L = L_down only
      dim > top_dim  → empty matrix
    """
    target_device = device if device is not None else get_device()
    dtype = get_dtype()

    if dim == 0:
        return get_up(filtered_boundaries[1], a, b, target_device)

    if dim == top_dim:
        return get_down(filtered_boundaries[top_dim], a, target_device)

    if dim > top_dim:
        return torch.empty(0, 0, dtype=dtype, device=target_device)

    L_up = get_up(filtered_boundaries[dim + 1], a, b, target_device)
    L_down = get_down(filtered_boundaries[dim], a, target_device)

    if L_up.numel() == 0:
        return L_down
    if L_down.numel() == 0:
        return L_up
    return L_up + L_down
