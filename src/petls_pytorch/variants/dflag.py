"""
dFlag (directed flag complex) variant — PyTorch-native replacement for petls::dFlag.

Reads a weighted directed graph from a ``.flag`` file, enumerates the directed
flag complex up to ``max_dim``, and delegates all persistent-Laplacian
computations to :class:`petls_pytorch.core.complex.Complex`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from petls_pytorch.core.complex import Complex


@dataclass(frozen=True)
class _FlagData:
    """Parsed weighted directed graph with explicit edge presence."""

    vertex_weights: np.ndarray
    edge_weights: np.ndarray
    edge_present: np.ndarray


def _read_flag_file(path: str) -> _FlagData:
    """Parse a weighted ``.flag`` file without conflating absent and zero edges.

    ``edge_present`` records whether a directed edge appeared in the file;
    ``edge_weights`` may therefore contain the default value ``0`` for absent
    edges without changing the graph. All parsed weights must be finite.
    """
    lines = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if line:
            lines.append(line)

    if len(lines) < 2:
        raise ValueError(".flag file must contain a 'dim 0' header and vertex weights")

    if not _is_dim_header(lines[0], 0):
        raise ValueError(".flag file must start with a 'dim 0' header")

    try:
        vertices = [float(x) for x in lines[1].split()]
    except ValueError as exc:
        raise ValueError("Invalid vertex weights in .flag file") from exc

    if not vertices:
        raise ValueError(".flag file must contain at least one vertex weight")
    vertex_weights = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(vertex_weights).all():
        raise ValueError("Vertex weights in .flag file must be finite")

    edge_weights = np.zeros((len(vertices), len(vertices)), dtype=np.float64)
    edge_present = np.zeros((len(vertices), len(vertices)), dtype=bool)

    if len(lines) == 2:
        return _FlagData(vertex_weights, edge_weights, edge_present)
    if not _is_dim_header(lines[2], 1):
        raise ValueError(".flag edge list must be preceded by a 'dim 1' header")

    for line_number, line in enumerate(lines[3:], start=4):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"Invalid .flag edge on line {line_number}: expected 'source target weight'"
            )
        try:
            src = int(parts[0])
            dst = int(parts[1])
            weight = float(parts[2])
        except ValueError as exc:
            raise ValueError(f"Invalid .flag edge on line {line_number}") from exc
        if src == dst:
            raise ValueError(f"Self-loops are not supported in .flag files: line {line_number}")
        if src < 0 or src >= len(vertices) or dst < 0 or dst >= len(vertices):
            raise ValueError(f".flag edge endpoint out of range on line {line_number}")
        if not np.isfinite(weight):
            raise ValueError(f".flag edge weight must be finite on line {line_number}")
        if edge_present[src, dst]:
            raise ValueError(f"Duplicate .flag edge declaration on line {line_number}")
        edge_weights[src, dst] = weight
        edge_present[src, dst] = True

    return _FlagData(vertex_weights, edge_weights, edge_present)


def _is_dim_header(line: str, dim: int) -> bool:
    """Return whether *line* is a flagser-style dimension header."""
    normalized = line.lower().replace(" ", "")
    return normalized == f"dim{dim}"


def _enumerate_directed_simplices(graph: _FlagData, max_dim: int):
    """Enumerate all directed simplices in the graph.

    A *k*-simplex is an ordered tuple ``(v0, v1, ..., vk)`` such that a directed
    edge ``vi -> vj`` exists for every ``0 <= i < j <= k``.

    Returns
    -------
    simplices : list[list[tuple[int, ...]]]
        ``simplices[d]`` contains all *d*-simplices.
    filtrations : list[list[float]]
        ``filtrations[d]`` contains the filtration value of each *d*-simplex.
        Vertex filtrations use ``vertex_weights``. Every simplex filtration is
        the maximum of all constituent vertex weights and directed edge weights,
        which guarantees that every face appears no later than its cofaces.
    """
    n = len(graph.vertex_weights)
    simplices: list[list[tuple[int, ...]]] = [[] for _ in range(max_dim + 1)]
    filtrations: list[list[float]] = [[] for _ in range(max_dim + 1)]

    # 0-simplices: vertices with their diagonal weights
    # Cast to float32 to match C++ flagser's value_t = float precision.
    vertex_filtrations = [float(np.float32(weight)) for weight in graph.vertex_weights]
    for v in range(n):
        simplices[0].append((v,))
        filtrations[0].append(vertex_filtrations[v])

    out_neighbors = {
        v: {
            w: float(np.float32(graph.edge_weights[v, w]))
            for w in range(n)
            if v != w and graph.edge_present[v, w]
        }
        for v in range(n)
    }

    def expand(simplex: tuple[int, ...], candidates: set[int], max_weight: float) -> None:
        simplex_dim = len(simplex) - 1
        if simplex_dim > max_dim:
            return
        if simplex_dim > 0:
            simplices[simplex_dim].append(simplex)
            filtrations[simplex_dim].append(max_weight)
        if simplex_dim == max_dim:
            return

        used = set(simplex)
        for vertex in sorted(candidates):
            if vertex in used:
                continue
            edge_weights = [out_neighbors[existing][vertex] for existing in simplex]
            next_weight = max(max_weight, vertex_filtrations[vertex], *edge_weights)
            next_candidates = candidates.intersection(out_neighbors[vertex]).difference(used)
            next_candidates.discard(vertex)
            expand((*simplex, vertex), next_candidates, next_weight)

    for v in range(n):
        expand((v,), set(out_neighbors[v]), vertex_filtrations[v])

    for dim in range(1, max_dim + 1):
        # Sort by filtration value (ties broken by the old brute-force scan order)
        indexed = list(enumerate(zip(simplices[dim], filtrations[dim])))
        indexed.sort(key=lambda x: (x[1][1], _bruteforce_rank(x[1][0], n)))
        simplices[dim] = [s for _, (s, _) in indexed]
        filtrations[dim] = [f for _, (_, f) in indexed]

    # Also sort 0-simplices by filtration
    indexed = list(enumerate(zip(simplices[0], filtrations[0])))
    indexed.sort(key=lambda x: (x[1][1], x[0]))
    simplices[0] = [s for _, (s, _) in indexed]
    filtrations[0] = [f for _, (_, f) in indexed]

    return simplices, filtrations


def _bruteforce_rank(simplex: tuple[int, ...], n_vertices: int) -> int:
    """Tie-break key matching the previous base-n tuple scan order."""
    return sum(vertex * (n_vertices**index) for index, vertex in enumerate(simplex))


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

    The supported ``.flag`` format has a ``dim 0`` section containing one
    whitespace-separated vertex weight per vertex, followed by an optional
    ``dim 1`` section containing ``source target weight`` directed edges.
    Every edge row declares an edge even when its weight is zero or negative;
    duplicate rows and non-finite weights are rejected. A simplex filtration is
    the maximum of its vertex and directed-edge weights.
    """

    def __init__(
        self,
        filename: str,
        max_dim: int = 3,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
        zero_atol: float = 1e-8,
        zero_rtol: float = 1e-7,
        max_matrix_rows: int | None = 12_000,
        max_matrix_bytes: int | None = 4_000_000_000,
        on_oversize: str = "raise",
        eigs_algorithm: str = "eigvalsh",
    ):
        graph = _read_flag_file(filename)
        simplices, filtrations = _enumerate_directed_simplices(graph, max_dim)
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
            dtype=dtype,
            zero_atol=zero_atol,
            zero_rtol=zero_rtol,
            max_matrix_rows=max_matrix_rows,
            max_matrix_bytes=max_matrix_bytes,
            on_oversize=on_oversize,
            eigs_algorithm=eigs_algorithm,
        )
        self.simplices_by_dimension = simplices[: actual_top_dim + 1]
        self.simplex_to_index = [
            {simplex: index for index, simplex in enumerate(values)}
            for values in self.simplices_by_dimension
        ]
