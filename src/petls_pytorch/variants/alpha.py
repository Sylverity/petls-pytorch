"""
Alpha complex variant — PyTorch-native replacement for petls::Alpha.

Constructs a simplicial complex from a point cloud using Gudhi's
AlphaComplex, extracts boundary matrices and filtrations, and delegates
all persistent-Laplacian computations to :class:`petls_pytorch.core.complex.Complex`.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
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
    points : array-like, optional
        List of point coordinates such as ``[[x, y], ...]`` or
        ``[[x, y, z], ...]``.
    max_dim : int, optional
        Maximum simplex dimension to retain (default 3).
    weights : array-like, optional
        General power weights, one finite value per point.
    precision : {"fast", "safe", "exact"}, optional
        Gudhi alpha-complex precision mode (default ``"safe"``).
    max_alpha_square : float, optional
        Largest alpha-square filtration value to construct.
    point_labels : sequence of hashable, optional
        Labels retained separately from topology for downstream interpretation.

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
        points: Sequence[Sequence[float]] | np.ndarray | torch.Tensor | None = None,
        weights: Sequence[float] | np.ndarray | torch.Tensor | None = None,
        max_dim: int = 3,
        precision: str = "safe",
        max_alpha_square: float = float("inf"),
        point_labels: Sequence[Hashable] | None = None,
        device: torch.device | str | None = "cpu",
        dtype: torch.dtype | str | None = torch.float64,
        zero_atol: float = 1e-8,
        zero_rtol: float = 1e-7,
        max_matrix_rows: int | None = 12_000,
        max_matrix_bytes: int | None = 4_000_000_000,
        on_oversize: str = "raise",
        eigs_algorithm: str = "eigvalsh",
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

        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim != 2 or point_array.shape[0] == 0 or point_array.shape[1] == 0:
            raise ValueError("points must be a non-empty two-dimensional array")
        if not np.all(np.isfinite(point_array)):
            raise ValueError("points must contain only finite values")
        if max_dim < 0:
            raise ValueError("max_dim must be non-negative")
        if precision not in {"fast", "safe", "exact"}:
            raise ValueError("precision must be 'fast', 'safe', or 'exact'")
        if np.isnan(max_alpha_square):
            raise ValueError("max_alpha_square must not be NaN")

        weight_array = None
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
            weight_array = np.asarray(weights, dtype=np.float64)
            if weight_array.ndim != 1:
                raise ValueError("weights must be one-dimensional")
            if len(weight_array) != len(point_array):
                raise ValueError(
                    f"len(weights)={len(weight_array)} must equal len(points)={len(point_array)}"
                )
            if not np.all(np.isfinite(weight_array)):
                raise ValueError("weights must contain only finite values")

        labels = None if point_labels is None else list(point_labels)
        if labels is not None and len(labels) != len(point_array):
            raise ValueError(
                f"len(point_labels)={len(labels)} must equal len(points)={len(point_array)}"
            )

        alpha_kwargs = {
            "points": point_array.tolist(),
            "precision": precision,
        }
        if weight_array is not None:
            alpha_kwargs["weights"] = weight_array.tolist()
        alpha = gudhi.AlphaComplex(**alpha_kwargs)
        simplex_tree = alpha.create_simplex_tree(max_alpha_square=float(max_alpha_square))

        if simplex_tree.dimension() > max_dim:
            simplex_tree.prune_above_dimension(max_dim)

        boundaries, filtrations, _ = simplex_tree_boundaries_filtrations(
            simplex_tree,
            sign_convention="cpp",
            return_simplices=True,
        )

        super().__init__(
            boundaries=boundaries,
            filtrations=filtrations,
            simplex_tree=simplex_tree,
            device=device,
            dtype=dtype,
            zero_atol=zero_atol,
            zero_rtol=zero_rtol,
            max_matrix_rows=max_matrix_rows,
            max_matrix_bytes=max_matrix_bytes,
            on_oversize=on_oversize,
            eigs_algorithm=eigs_algorithm,
        )
        self.alpha_complex = alpha
        self.points = point_array.copy()
        self.weights = None if weight_array is None else weight_array.copy()
        self.precision = precision
        self.max_alpha_square = float(max_alpha_square)
        self.point_labels = labels
        if labels is not None:
            self.simplex_labels_by_dimension = [
                [tuple(labels[vertex] for vertex in simplex) for simplex in simplices]
                for simplices in self.simplices_by_dimension
            ]
