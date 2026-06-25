"""
PETLS — GPU-Native Persistent Topological Laplacians in PyTorch.

This is the public API. It mirrors the original PETLS interface where possible,
but is implemented entirely in Python with PyTorch tensors.
"""

import enum

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse import coo_matrix

from petls_torch._config import get_device, get_dtype, set_device, set_dtype
from petls_torch.core.complex import Complex
from petls_torch.core.profile import Profile, Timer

from petls_torch.variants.alpha import Alpha
from petls_torch.variants.dflag import dFlag
from petls_torch.variants.rips import Rips
from petls_torch.variants.sheaf import sheaf_simplex_tree, PersistentSheafLaplacian
from petls_torch.utils.plotting import summaries, plot_summary
from petls_torch.utils.simplex_tree import simplex_tree_boundaries_filtrations

__version__ = "1.0.0"


class UpAlgorithms(enum.Enum):
    """Enum to choose the up-Laplacian algorithm (mirrors original PETLS)."""

    schur = 1


# Backwards-compatible alias used by the original PETLS API.
up_Algorithms = UpAlgorithms


def matrix_is_diagonal(L: np.ndarray) -> bool:
    """Fast check whether a square numpy array is diagonal."""
    i, j = L.shape
    assert i == j
    test = L.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return not np.any(test[:, 1:])


def eigvalsh_wrapper(L: np.ndarray) -> np.ndarray:
    """Wrapper around scipy.linalg.eigvalsh with diagonal short-circuit."""
    if matrix_is_diagonal(L):
        return np.array(sorted(np.diag(L)))
    return eigvalsh(L)


def sparse_wrapper(
    L: np.ndarray, num_eigs: int = 10, which_eigs: str = "SM", ncv: int = 20
) -> np.ndarray:
    """Wrapper around scipy.sparse.linalg.eigs with fallbacks."""
    import scipy.sparse.linalg

    if matrix_is_diagonal(L):
        return np.array(sorted(np.diag(L)))

    num_rows = L.shape[0]
    num_eigs = min(num_rows - 1, num_eigs)
    ncv = min(max(2 * num_eigs, ncv), num_rows)

    try:
        eigs = scipy.sparse.linalg.eigs(
            L, k=num_eigs, ncv=ncv, which=which_eigs, return_eigenvectors=False
        )
        return np.array(sorted(eigs.real))
    except Exception:
        all_eigs = eigvalsh(L)
        if which_eigs in ("SM", "SA"):
            return np.array(all_eigs[:num_eigs])
        elif which_eigs in ("LM", "LA"):
            return all_eigs[-num_eigs:]
        elif which_eigs == "BE":
            lowest = all_eigs[: num_eigs // 2]
            if num_eigs % 2 == 1:
                highest = all_eigs[-(num_eigs // 2 + 1) :]
            else:
                highest = all_eigs[-(num_eigs // 2) :]
            return np.concatenate((lowest, highest))
        else:
            return all_eigs


__all__ = [
    "Complex",
    "Profile",
    "Timer",
    "get_device",
    "get_dtype",
    "set_device",
    "set_dtype",
    "Alpha",
    "dFlag",
    "Rips",
    "sheaf_simplex_tree",
    "PersistentSheafLaplacian",
    "summaries",
    "plot_summary",
    "UpAlgorithms",
    "up_Algorithms",
    "eigvalsh",
    "eigvalsh_wrapper",
    "sparse_wrapper",
    "matrix_is_diagonal",
    "coo_matrix",
    "simplex_tree_boundaries_filtrations",
]
