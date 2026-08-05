"""
Gudhi simplex tree -> sparse boundary extraction.

Shared by Alpha and Rips variants.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix


def simplex_tree_boundaries_filtrations(
    simplex_tree,
    sign_convention: str = "python",
    return_simplices: bool = False,
):
    """Extract sparse boundary matrices and per-dimension filtrations from a Gudhi simplex tree.

    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        The Gudhi simplex tree.
    sign_convention : {"python", "cpp"}, optional
        ``"python"`` starts with ``+1`` for every simplex (matches
        ``petls.PLutil.simplex_tree_boundaries_filtrations``).
        ``"cpp"`` uses ``sign = 1 - 2*(dim % 2)`` (matches the C++ Alpha
        extraction).

    Returns
    -------
    boundaries : list[scipy.sparse.coo_matrix]
        ``boundaries[d]`` is the boundary matrix :math:`d_{d+1}` with shape
        ``(n_d, n_{d+1})``.
    filtrations : list[list[float]]
        ``filtrations[d]`` contains the filtration values for dimension *d*.
    """
    if sign_convention not in ("python", "cpp"):
        raise ValueError("sign_convention must be 'python' or 'cpp'")

    max_dim = simplex_tree.dimension()
    if max_dim < 0:
        if return_simplices:
            return [], [[]], [[]]
        return [], [[]]
    filtrations: list[list[float]] = [[] for _ in range(max_dim + 1)]
    simplices: list[list[tuple[int, ...]]] = [[] for _ in range(max_dim + 1)]

    for simplex, filtration in simplex_tree.get_filtration():
        dim = len(simplex) - 1
        simplices[dim].append(tuple(int(vertex) for vertex in simplex))
        filtrations[dim].append(float(filtration))

    index_mappings = [
        {simplex: index for index, simplex in enumerate(simplices_dim)}
        for simplices_dim in simplices
    ]

    boundaries: list[coo_matrix] = []
    for dim in range(1, max_dim + 1):
        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        for col, simplex in enumerate(simplices[dim]):
            # Gudhi returns codimension-one faces in its own stable order.
            # Keeping that order preserves the historical PETLS sign choice.
            sign = 1 - 2 * (dim % 2) if sign_convention == "cpp" else 1
            for face, _ in simplex_tree.get_boundaries(list(simplex)):
                rows.append(index_mappings[dim - 1][tuple(face)])
                cols.append(col)
                data.append(sign)
                sign = -sign

        n_rows = len(simplices[dim - 1])
        n_cols = len(simplices[dim])
        B = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
        boundaries.append(B)

    if return_simplices:
        return boundaries, filtrations, simplices
    return boundaries, filtrations
