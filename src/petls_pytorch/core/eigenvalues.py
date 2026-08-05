"""Dense and partial symmetric eigenvalue solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch

_CUDA_CPU_FALLBACK_ROWS = 512


def _full_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return torch.empty(0, device=matrix.device, dtype=matrix.dtype)
    if matrix.shape[0] == 1:
        return matrix.diagonal().real
    if matrix.device.type == "cuda" and matrix.shape[0] <= _CUDA_CPU_FALLBACK_ROWS:
        return cast(
            torch.Tensor,
            torch.linalg.eigvalsh(matrix.cpu()).to(device=matrix.device),
        )
    return cast(torch.Tensor, torch.linalg.eigvalsh(matrix))


def _full_eigenpairs(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.numel() == 0:
        empty = torch.empty(0, device=matrix.device, dtype=matrix.dtype)
        return empty, torch.empty(0, 0, device=matrix.device, dtype=matrix.dtype)
    if matrix.shape[0] == 1:
        values = matrix.diagonal().real
        return values, torch.ones_like(values).unsqueeze(0)
    if matrix.device.type == "cuda" and matrix.shape[0] <= _CUDA_CPU_FALLBACK_ROWS:
        values, vectors = torch.linalg.eigh(matrix.cpu())
        return values.to(device=matrix.device), vectors.to(device=matrix.device)
    return cast(tuple[torch.Tensor, torch.Tensor], torch.linalg.eigh(matrix))


def solve_eigenvalues(
    L: torch.Tensor,
    algorithm: str | Callable[[torch.Tensor], torch.Tensor] = "eigvalsh",
) -> torch.Tensor:
    """
    Compute sorted eigenvalues of a symmetric matrix.

    Parameters
    ----------
    L : torch.Tensor
        Dense symmetric matrix (real).
    algorithm : str or callable
        Solver name or callable that accepts L and returns eigenvalues.

    Returns
    -------
    torch.Tensor
        Sorted real eigenvalues.
    """
    if callable(algorithm):
        return cast(torch.Tensor, algorithm(L))

    if algorithm != "eigvalsh":
        raise ValueError("algorithm must be 'eigvalsh' or a callable")
    return _full_eigenvalues(L)


def solve_eigenpairs(
    L: torch.Tensor,
    algorithm: str | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = "eigh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sorted eigenvalues and eigenvectors of a symmetric matrix.

    Returns
    -------
    eigenvalues : torch.Tensor
    eigenvectors : torch.Tensor  (columns are eigenvectors)
    """
    if callable(algorithm):
        return algorithm(L)

    if algorithm != "eigh":
        raise ValueError("algorithm must be 'eigh' or a callable")
    return _full_eigenpairs(L)


def solve_sparse_eigenvalues(
    matrix: torch.Tensor,
    num_eigenvalues: int = 10,
    which: str = "SM",
) -> torch.Tensor:
    """Return a partial symmetric spectrum for an already-dense matrix."""
    import scipy.linalg
    import scipy.sparse.linalg

    if num_eigenvalues < 1:
        raise ValueError("num_eigenvalues must be positive")
    if which not in {"SM", "SA", "LM", "LA", "BE"}:
        raise ValueError("which must be one of 'SM', 'SA', 'LM', 'LA', or 'BE'")
    if matrix.numel() == 0:
        return torch.empty(0, device=matrix.device, dtype=matrix.dtype)

    dense = matrix.detach().cpu().numpy()
    rows = dense.shape[0]
    requested = min(num_eigenvalues, rows)
    complete = rows == 1 or requested == rows
    diagonal = np.count_nonzero(dense - np.diag(np.diag(dense))) == 0
    if complete:
        values = scipy.linalg.eigvalsh(dense)
    elif diagonal:
        values = np.sort(np.diag(dense))
    else:
        values = scipy.sparse.linalg.eigsh(
            dense,
            k=requested,
            which=which,
            return_eigenvectors=False,
        )
        values.sort()

    if len(values) > requested:
        if which in {"SM", "SA"}:
            values = values[:requested]
        elif which in {"LM", "LA"}:
            values = values[-requested:]
        else:
            lower = requested // 2
            values = np.concatenate((values[:lower], values[-(requested - lower) :]))
    return torch.as_tensor(values, dtype=matrix.dtype, device=matrix.device)
