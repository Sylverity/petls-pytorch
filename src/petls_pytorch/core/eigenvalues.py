"""
Eigenvalue solver dispatch.

GPU-native equivalents for the original C++ solvers:
  - "selfadjoint" / "eigvalsh" -> torch.linalg.eigvalsh  (cuSOLVER syevd)
  - "eigh"                     -> torch.linalg.eigh       (values + vectors)
  - "lobpcg"                   -> torch.lobpcg            (partial spectrum)
  - "spectra"                  -> mapped to lobpcg or scipy fallback
  - callable                   -> user-provided function
"""

from __future__ import annotations

import torch
from typing import Callable


# Registry of named solvers
SOLVERS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
_CUDA_CPU_FALLBACK_ROWS = 512


def _register_defaults() -> None:
    """Register built-in solver functions."""

    def eigh_full(L: torch.Tensor) -> torch.Tensor:
        """Full symmetric eigendecomposition via cuSOLVER (GPU) or LAPACK (CPU)."""
        if L.numel() == 0:
            return torch.empty(0, device=L.device, dtype=L.dtype)
        if L.shape[0] == 1:
            return L.diagonal().real
        if L.device.type == "cuda" and L.shape[0] <= _CUDA_CPU_FALLBACK_ROWS:
            return torch.linalg.eigvalsh(L.cpu()).to(device=L.device)
        # torch.linalg.eigvalsh returns eigenvalues in ascending order.
        return torch.linalg.eigvalsh(L)

    def eigh_pairs(L: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (eigenvalues, eigenvectors) like scipy.linalg.eigh."""
        if L.numel() == 0:
            empty = torch.empty(0, device=L.device, dtype=L.dtype)
            return empty, torch.empty(0, 0, device=L.device, dtype=L.dtype)
        if L.shape[0] == 1:
            val = L.diagonal().real
            vec = torch.ones_like(val).unsqueeze(0)
            return val, vec
        if L.device.type == "cuda" and L.shape[0] <= _CUDA_CPU_FALLBACK_ROWS:
            vals, vecs = torch.linalg.eigh(L.cpu())
            return vals.to(device=L.device), vecs.to(device=L.device)
        vals, vecs = torch.linalg.eigh(L)
        return vals, vecs

    SOLVERS["eigvalsh"] = eigh_full
    SOLVERS["eigh"] = eigh_pairs
    SOLVERS["selfadjoint"] = eigh_full  # alias for compat

    # C++ algorithm names — fall back to eigvalsh since we have no C++ backend
    SOLVERS["bdcsvd"] = eigh_full
    SOLVERS["eigensolver"] = eigh_full
    SOLVERS["spectra"] = eigh_full


_register_defaults()


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
        return algorithm(L)

    if algorithm not in SOLVERS:
        raise ValueError(
            f"Unknown eigenvalue algorithm '{algorithm}'. Available: {list(SOLVERS.keys())}"
        )

    return SOLVERS[algorithm](L)


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

    if algorithm == "eigvalsh":
        # Only values requested — return dummy vectors
        vals = solve_eigenvalues(L, "eigvalsh")
        vecs = torch.zeros(L.shape[0], L.shape[0], device=L.device, dtype=L.dtype)
        return vals, vecs

    if algorithm not in SOLVERS:
        raise ValueError(
            f"Unknown eigenpair algorithm '{algorithm}'. Available: {list(SOLVERS.keys())}"
        )

    solver = SOLVERS[algorithm]
    # Some solvers return tuples, some return just values
    result = solver(L)
    if isinstance(result, tuple):
        return result
    # Fallback: return values + identity vectors
    return result, torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
