"""
Alpha complex variant — PyTorch-native replacement for petls::Alpha.

Constructs a simplicial complex from a point cloud using Gudhi's
AlphaComplex, extracts boundary matrices and filtrations, and delegates
all persistent-Laplacian computations to :class:`petls_pytorch.core.complex.Complex`.
"""

from __future__ import annotations

import torch

from petls_pytorch.core.complex import Complex
from petls_pytorch.utils.simplex_tree import simplex_tree_boundaries_filtrations


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


class Alpha(Complex):
    """Alpha complex from a point cloud, using Gudhi's AlphaComplex.

    This is a drop-in PyTorch replacement for ``petls.Alpha``.

    Parameters
    ----------
    filename : str, optional
        Path to an OFF file containing the point cloud.
    points : list[list[float]], optional
        List of point coordinates such as ``[[x, y], ...]`` or
        ``[[x, y, z], ...]``.
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

        boundaries, filtrations = simplex_tree_boundaries_filtrations(
            simplex_tree, sign_convention="cpp"
        )

        super().__init__(boundaries=boundaries, filtrations=filtrations, device=device)
