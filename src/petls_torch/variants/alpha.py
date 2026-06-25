"""
Alpha complex variant — PyTorch-native replacement for petls::Alpha.

Constructs a simplicial complex from a 3-D point cloud using Gudhi's
AlphaComplex, extracts boundary matrices and filtrations, and delegates
all persistent-Laplacian computations to :class:`petls_torch.core.complex.Complex`.
"""

from __future__ import annotations

import numpy as np
import torch

from petls_torch.core.complex import Complex


def _read_off_file(path: str) -> list[list[float]]:
    """Parse an OFF file and return a list of 3-D points.

    Falls back to Gudhi's ``read_points_from_off_file`` if available,
    otherwise uses a minimal pure-Python parser.
    """
    try:
        import gudhi
        points = gudhi.read_points_from_off_file(path)
        # Gudhi returns list of tuples; normalise to list of lists
        return [list(p) for p in points]
    except Exception:
        pass

    points = []
    with open(path, "r") as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.strip().startswith("#")]
    # First non-comment line should be "OFF"
    idx = 0
    if lines[idx] == "OFF":
        idx += 1
    n_vertices, n_faces, n_edges = map(int, lines[idx].split())
    idx += 1
    for i in range(n_vertices):
        coords = list(map(float, lines[idx + i].split()))
        points.append(coords[:3])  # only first 3 coordinates
    return points


def _simplex_tree_boundaries_filtrations(simplex_tree):
    """Extract dense boundary matrices and per-dimension filtrations from a Gudhi simplex tree.

    This is a PyTorch-native reimplementation of the logic in
    ``petls.PLutil.simplex_tree_boundaries_filtrations``.

    Returns
    -------
    boundaries : list[np.ndarray]
        ``boundaries[d]`` is the boundary matrix :math:`d_{d+1}` with shape
        ``(n_d, n_{d+1})``.
    filtrations : list[list[float]]
        ``filtrations[d]`` contains the filtration values for dimension *d*.
    """
    # 1. Assign a unique global index to every simplex (in filtration order).
    indices = {}
    for simplex, filtration in simplex_tree.get_filtration():
        indices[tuple(simplex)] = len(indices)

    max_dim = simplex_tree.dimension()

    # 2. Collect boundary triples [global_face_idx, global_simplex_idx, coeff]
    #    and filtrations per dimension.
    boundaries_triples = [[] for _ in range(max_dim + 1)]
    filtrations = [[] for _ in range(max_dim + 1)]

    for simplex, filtration in simplex_tree.get_filtration():
        dim = len(simplex) - 1
        filtrations[dim].append(filtration)

        if dim == 0:
            continue

        # Match C++ sign convention: sign = 1 - 2*(dim % 2), alternating.
        sign = 1 - 2 * (dim % 2)
        for face, _ in simplex_tree.get_boundaries(simplex):
            boundaries_triples[dim].append(
                [indices[tuple(face)], indices[tuple(simplex)], sign]
            )
            sign = -sign

    # 3. Re-index to dimension-local indices [0 .. N_dim-1].
    index_mappings = [{} for _ in range(max_dim + 1)]

    for dim in range(max_dim):
        face_set = set(triple[0] for triple in boundaries_triples[dim + 1])
        simplex_set = set(triple[1] for triple in boundaries_triples[dim])
        actual = list(face_set | simplex_set)
        for i, idx in enumerate(actual):
            index_mappings[dim][idx] = i

    top_set = set(triple[1] for triple in boundaries_triples[max_dim])
    for i, idx in enumerate(list(top_set)):
        index_mappings[max_dim][idx] = i

    # 4. Build dense numpy boundary matrices.
    boundaries = []
    for dim in range(1, max_dim + 1):
        rows = []
        cols = []
        data = []
        for triple in boundaries_triples[dim]:
            rows.append(index_mappings[dim - 1][triple[0]])
            cols.append(index_mappings[dim][triple[1]])
            data.append(triple[2])

        n_rows = len(index_mappings[dim - 1])
        n_cols = len(index_mappings[dim])
        B = np.zeros((n_rows, n_cols), dtype=np.float64)
        B[rows, cols] = data
        boundaries.append(B)

    return boundaries, filtrations


class Alpha(Complex):
    """Alpha complex from a 3-D point cloud, using Gudhi's AlphaComplex.

    This is a drop-in PyTorch replacement for ``petls.Alpha``.

    Parameters
    ----------
    filename : str, optional
        Path to an OFF file containing the point cloud.
    points : list[list[float]], optional
        List of 3-D point coordinates ``[[x, y, z], ...]``.
    max_dim : int, optional
        Maximum simplex dimension to retain (default 3).
    device : torch.device, optional
        Override global compute device.

    Raises
    ------
    ValueError
        If neither *filename* nor *points* is provided.
    ImportError
        If ``gudhi`` is not installed.
    """

    def __init__(
        self,
        filename: str | None = None,
        points: list[list[float]] | None = None,
        max_dim: int = 3,
        device: torch.device | None = None,
    ):
        try:
            import gudhi
        except ImportError as exc:
            raise ImportError(
                "Gudhi is required for Alpha complex construction. "
                "Install it with: pip install gudhi"
            ) from exc

        if filename is not None:
            points = _read_off_file(filename)
        elif points is None:
            raise ValueError("Alpha complex requires filename or point set as input")

        alpha = gudhi.AlphaComplex(points=points)
        simplex_tree = alpha.create_simplex_tree()

        if max_dim is not None and simplex_tree.dimension() > max_dim:
            simplex_tree.prune_above_dimension(max_dim)

        boundaries, filtrations = _simplex_tree_boundaries_filtrations(simplex_tree)

        super().__init__(boundaries=boundaries, filtrations=filtrations, device=device)
