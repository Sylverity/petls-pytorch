"""
dFlag (directed flag complex) variant — PyTorch-native replacement for petls::dFlag.

Reads a weighted directed graph from a ``.flag`` file, enumerates the directed
flag complex up to ``max_dim``, and delegates all persistent-Laplacian
computations to :class:`petls_torch.core.complex.Complex`.
"""

from __future__ import annotations

import numpy as np
import torch

from petls_torch.core.complex import Complex


def _read_flag_file(path: str) -> np.ndarray:
    """Parse a flagser ``.flag`` file and return a dense weighted adjacency matrix.

    Diagonal entries store vertex weights; off-diagonal entries store directed
    edge weights. Missing edges are represented by ``0``.
    """
    try:
        from pyflagser.flagio import load_weighted_flag
        adj = load_weighted_flag(path)
        return adj.toarray()
    except ImportError as exc:
        raise ImportError(
            "pyflagser is required for dFlag complex construction. "
            "Install it with: pip install pyflagser"
        ) from exc


def _enumerate_directed_simplices(adj: np.ndarray, max_dim: int):
    """Enumerate all directed simplices in the graph.

    A *k*-simplex is an ordered tuple ``(v0, v1, ..., vk)`` such that a directed
    edge ``vi -> vj`` exists for every ``0 <= i < j <= k``.

    Returns
    -------
    simplices : list[list[tuple[int, ...]]]
        ``simplices[d]`` contains all *d*-simplices.
    filtrations : list[list[float]]
        ``filtrations[d]`` contains the filtration value of each *d*-simplex.
        Vertex filtrations are taken from the diagonal of *adj*;
        higher-dimensional filtrations use the ``"max"`` algorithm (largest edge
        weight in the simplex).
    """
    n = adj.shape[0]
    simplices = [[] for _ in range(max_dim + 1)]
    filtrations = [[] for _ in range(max_dim + 1)]

    # 0-simplices: vertices with their diagonal weights
    # Cast to float32 to match C++ flagser's value_t = float precision.
    for v in range(n):
        simplices[0].append((v,))
        filtrations[0].append(float(np.float32(adj[v, v])))

    # Higher dimensions — brute-force enumeration (sufficient for small test graphs)
    for dim in range(1, max_dim + 1):
        # Enumerate all ordered (dim+1)-tuples of distinct vertices
        # and keep those that form a directed simplex.
        seen = set()
        for code in range(n ** (dim + 1)):
            tup = []
            tmp = code
            for _ in range(dim + 1):
                tup.append(tmp % n)
                tmp //= n
            tup = tuple(tup)

            if len(set(tup)) != dim + 1:
                continue
            if tup in seen:
                continue

            valid = True
            max_weight = 0.0
            for i in range(dim + 1):
                for j in range(i + 1, dim + 1):
                    w = adj[tup[i], tup[j]]
                    if w <= 0 or w == np.inf:
                        valid = False
                        break
                    max_weight = max(max_weight, float(np.float32(w)))
                if not valid:
                    break

            if valid:
                seen.add(tup)
                simplices[dim].append(tup)
                filtrations[dim].append(max_weight)

        # Sort by filtration value (ties broken by original enumeration order)
        indexed = list(enumerate(zip(simplices[dim], filtrations[dim])))
        indexed.sort(key=lambda x: (x[1][1], x[0]))
        simplices[dim] = [s for _, (s, _) in indexed]
        filtrations[dim] = [f for _, (_, f) in indexed]

    # Also sort 0-simplices by filtration
    indexed = list(enumerate(zip(simplices[0], filtrations[0])))
    indexed.sort(key=lambda x: (x[1][1], x[0]))
    simplices[0] = [s for _, (s, _) in indexed]
    filtrations[0] = [f for _, (_, f) in indexed]

    return simplices, filtrations


def _build_boundaries(simplices):
    """Build dense boundary matrices from sorted simplex lists.

    ``boundaries[d]`` is :math:`d_{d+1}` with shape
    ``(n_d, n_{d+1})``.
    """
    max_dim = len(simplices) - 1
    boundaries = []
    for dim in range(1, max_dim + 1):
        n_rows = len(simplices[dim - 1])
        n_cols = len(simplices[dim])
        B = np.zeros((n_rows, n_cols), dtype=np.float64)
        idx_map = {s: i for i, s in enumerate(simplices[dim - 1])}

        for j, simplex in enumerate(simplices[dim]):
            for i in range(dim + 1):
                face = simplex[:i] + simplex[i + 1 :]
                sign = 1 if i % 2 == 0 else -1
                if face in idx_map:
                    B[idx_map[face], j] = sign

        boundaries.append(B)
    return boundaries


class dFlag(Complex):
    """Directed flag complex from a directed graph read from a ``.flag`` file.

    This is a drop-in PyTorch replacement for ``petls.dFlag``.

    Parameters
    ----------
    filename : str
        Path to a ``.flag`` file describing a weighted directed graph.
    max_dim : int
        Maximum simplex dimension to retain.
    device : torch.device, optional
        Override global compute device.

    Raises
    ------
    ImportError
        If ``pyflagser`` is not installed.
    """

    def __init__(
        self,
        filename: str,
        max_dim: int = 3,
        device: torch.device | None = None,
        eigs_Algorithm: str = "eigvalsh",
        up_Algorithm: str = "schur",
    ):
        adj = _read_flag_file(filename)
        simplices, filtrations = _enumerate_directed_simplices(adj, max_dim)
        boundaries = _build_boundaries(simplices)

        # Truncate to actual top dimension (highest dim with simplices),
        # matching C++ flagser behaviour.
        actual_top_dim = max(
            (d for d in range(max_dim + 1) if len(simplices[d]) > 0),
            default=0,
        )
        filtrations = filtrations[: actual_top_dim + 1]
        boundaries = boundaries[:actual_top_dim]

        super().__init__(
            boundaries=boundaries,
            filtrations=filtrations,
            device=device,
            eigs_Algorithm=eigs_Algorithm,
            up_Algorithm=up_Algorithm,
        )
