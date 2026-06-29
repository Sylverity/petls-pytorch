"""
Rips complex variant — PyTorch-native replacement for petls::Rips.

Constructs a simplicial complex from a point cloud, distance matrix, or
lower-triangular distance matrix file using Gudhi's RipsComplex, extracts
boundary matrices and filtrations, and delegates all persistent-Laplacian
computations to :class:`petls_pytorch.core.complex.Complex`.
"""

from __future__ import annotations

import numpy as np
import torch

from petls_pytorch.core.complex import Complex
from petls_pytorch.utils.simplex_tree import simplex_tree_boundaries_filtrations


def _read_lower_distance_matrix(path: str) -> np.ndarray:
    """Parse a lower-triangular distance matrix file.

    Each line *i* contains *i* comma-separated values representing
    distances from vertex *i* to vertices *0, 1, ..., i-1*.

    Example
    -------
    >>> # 4 vertices
    >>> # line 0: empty
    >>> 3,
    >>> 4,5,
    >>> 5,4,3

    Returns a dense symmetric ``(n, n)`` numpy array with zeros on the
    diagonal.
    """
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#"):
                continue
            # Remove trailing comma if present
            if line.endswith(","):
                line = line[:-1]
            if not line:
                # Empty line = vertex with no lower-triangular entries
                rows.append([])
                continue
            vals = [float(v.strip()) for v in line.split(",") if v.strip()]
            rows.append(vals)

    n = len(rows)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        for j in range(len(rows[i])):
            dist[i, j] = rows[i][j]
            dist[j, i] = rows[i][j]
    return dist


def _pad_to_max_dim(
    boundaries: list[np.ndarray],
    filtrations: list[list[float]],
    max_dim: int,
) -> tuple[list[np.ndarray], list[list[float]]]:
    """Pad boundaries and filtrations so that top_dim == max_dim.

    The original C++ Rips class always creates empty boundary matrices and
    empty filtration lists for dimensions that have no simplices. This ensures
    ``spectra()`` returns entries like ``(dim, a, a, [])`` for every
    ``dim <= max_dim``.
    """
    # Pad filtrations up to max_dim
    while len(filtrations) <= max_dim:
        filtrations.append([])

    # Pad boundaries up to max_dim
    while len(boundaries) < max_dim:
        if len(boundaries) == 0:
            # d_1 with no edges: shape (n_vertices, 0)
            n_vertices = len(filtrations[0])
            boundaries.append(np.zeros((n_vertices, 0), dtype=np.float64))
        else:
            prev_n_cols = boundaries[-1].shape[1]
            boundaries.append(np.zeros((prev_n_cols, 0), dtype=np.float64))

    return boundaries, filtrations


class Rips(Complex):
    """Rips complex from a point cloud or distance matrix, using Gudhi's RipsComplex.

    This is a drop-in PyTorch replacement for ``petls.Rips``.

    Parameters
    ----------
    filename : str, optional
        Path to a lower-triangular distance matrix file.
    points : array-like, optional
        List of point coordinates ``[[x1, y1, ...], [x2, y2, ...], ...]``.
    distances : array-like, optional
        Dense symmetric distance matrix.
    max_dim : int, optional
        Maximum simplex dimension (default 3).
    threshold : float, optional
        Max edge length. If ``None``, uses infinity.
    device : torch.device, optional
        Override global compute device.

    Raises
    ------
    ValueError
        If none of *filename*, *points*, or *distances* is provided.
    ImportError
        If ``gudhi`` is not installed.
    """

    def __init__(
        self,
        filename: str | None = None,
        points: list[list[float]] | np.ndarray | None = None,
        distances: list[list[float]] | np.ndarray | None = None,
        max_dim: int = 3,
        threshold: float | None = None,
        device: torch.device | None = None,
        eigs_Algorithm: str = "eigvalsh",
        up_Algorithm: str = "schur",
    ):
        try:
            import gudhi
        except ImportError as exc:
            raise ImportError(
                "Gudhi is required for Rips complex construction. "
                "Install it with: pip install gudhi"
            ) from exc

        if filename is not None:
            distances = _read_lower_distance_matrix(filename)
            # Fall through to the distance_matrix branch below
            points = None
        elif points is not None:
            distances = None
        elif distances is not None:
            distances = np.asarray(distances, dtype=np.float64)
            points = None
        else:
            raise ValueError("Rips requires filename, point set, or distance matrix as input")

        max_edge_length = float("inf") if threshold is None else threshold

        if points is not None:
            rips = gudhi.RipsComplex(
                points=np.asarray(points, dtype=np.float64),
                max_edge_length=max_edge_length,
            )
        else:
            rips = gudhi.RipsComplex(
                distance_matrix=distances,
                max_edge_length=max_edge_length,
            )

        # PETLS interprets max_dim as the largest Laplacian dimension to
        # query. Computing L_dim needs boundary d_{dim+1}, so build one
        # simplex dimension higher when it exists.
        build_dim = max_dim + 1
        simplex_tree = rips.create_simplex_tree(max_dimension=build_dim)

        boundaries, filtrations = simplex_tree_boundaries_filtrations(
            simplex_tree, sign_convention="python"
        )
        boundaries, filtrations = _pad_to_max_dim(boundaries, filtrations, max_dim)

        super().__init__(
            boundaries=boundaries,
            filtrations=filtrations,
            device=device,
            eigs_Algorithm=eigs_Algorithm,
            up_Algorithm=up_Algorithm,
        )
