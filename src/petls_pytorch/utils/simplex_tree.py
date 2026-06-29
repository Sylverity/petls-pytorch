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
) -> tuple[list[coo_matrix], list[list[float]]]:
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

    indices = {}
    for simplex, filtration in simplex_tree.get_filtration():
        indices[tuple(simplex)] = len(indices)

    max_dim = simplex_tree.dimension()

    boundaries_triples = [[] for _ in range(max_dim + 1)]
    filtrations = [[] for _ in range(max_dim + 1)]

    for simplex, filtration in simplex_tree.get_filtration():
        dim = len(simplex) - 1
        filtrations[dim].append(filtration)

        if dim == 0:
            continue

        if sign_convention == "cpp":
            sign = 1 - 2 * (dim % 2)
        else:
            sign = 1

        for face, _ in simplex_tree.get_boundaries(simplex):
            boundaries_triples[dim].append([indices[tuple(face)], indices[tuple(simplex)], sign])
            sign = -sign

    index_mappings = [{} for _ in range(max_dim + 1)]

    for dim in range(max_dim):
        face_set = set(triple[0] for triple in boundaries_triples[dim + 1])
        simplex_set = set(triple[1] for triple in boundaries_triples[dim])
        actual = sorted(list(face_set | simplex_set))
        for i, idx in enumerate(actual):
            index_mappings[dim][idx] = i

    top_set = set(triple[1] for triple in boundaries_triples[max_dim])
    for i, idx in enumerate(sorted(list(top_set))):
        index_mappings[max_dim][idx] = i

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
        B = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float32)
        boundaries.append(B)

    return boundaries, filtrations
