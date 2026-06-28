"""
Sheaf support — PyTorch-native replacement for petls::sheaf_simplex_tree
and petls::PersistentSheafLaplacian.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix

from petls_pytorch.core.complex import Complex

if TYPE_CHECKING:
    import gudhi


class sheaf_simplex_tree:
    """Wrap a Gudhi simplex tree with extra data and a restriction function.

    Parameters
    ----------
    st : gudhi.SimplexTree
        Underlying simplex tree.
    extra_data : dict
        Keys are simplices (tuple of int), values can be anything.
    restriction : Callable[[list[int], list[int], "sheaf_simplex_tree"], float]
        Restriction function mapping (simplex, coface, self) → float.
    """

    def __init__(
        self,
        st: "gudhi.SimplexTree",
        extra_data: dict,
        restriction: Callable[[list[int], list[int], "sheaf_simplex_tree"], float],
    ):
        self.st = st
        self.extra_data = extra_data
        self.restriction = restriction
        self.complex_dim = st.dimension()

        # Give each simplex a unique index
        index = 0
        indices = {}
        for simplex_with_filtration in self.st.get_filtration():
            indices[tuple(simplex_with_filtration[0])] = index
            index += 1
        self.indices = indices

    def coface_index(self, simplex: list[int], coface: list[int]) -> int:
        """Index of the missing vertex, e.g. coface_index([0,1,3], [0,1,2,3]) = 2."""
        if len(simplex) != len(coface) - 1:
            raise ValueError(
                f"len(simplex) != len(coface)-1. len(simplex)={len(simplex)}, "
                f"len(coface)={len(coface)}"
            )
        for i in range(len(simplex)):
            if simplex[i] != coface[i]:
                return i
        return len(simplex)

    def apply_restriction_function(self) -> tuple[list[np.ndarray], list[list[float]]]:
        """Build coboundaries and filtrations from the sheaf data.

        Returns
        -------
        coboundaries : list[np.ndarray]
            Coboundary matrices (maps k-simplices to (k+1)-simplices).
        filtrations : list[list[float]]
            Filtration values per dimension.
        """
        coboundaries_triples = [[] for _ in range(self.complex_dim)]
        filtrations = [[] for _ in range(self.complex_dim + 1)]

        for simplex_with_filtration in self.st.get_filtration():
            simplex = simplex_with_filtration[0]
            filtration = simplex_with_filtration[1]
            dim = len(simplex) - 1
            filtrations[dim].append(filtration)
            if dim == self.complex_dim:
                continue
            for coface_with_filtration in self.st.get_cofaces(simplex, 1):
                coface = coface_with_filtration[0]
                sign = (-1) ** (self.coface_index(simplex, coface) % 2)
                coeff = sign * self.restriction(simplex, coface, self)
                coboundaries_triples[dim].append(
                    [self.indices[tuple(coface)], self.indices[tuple(simplex)], coeff]
                )

        return self.reindex_coboundaries(coboundaries_triples), filtrations

    def reindex_coboundaries(self, coboundaries_triples):
        """Re-index coboundary triples to per-dimension dense matrices."""
        indices_of_actual_simplices_set = [set() for _ in range(self.complex_dim + 1)]
        for dim in range(self.complex_dim):
            for triple in coboundaries_triples[dim]:
                indices_of_actual_simplices_set[dim].add(triple[1])
                indices_of_actual_simplices_set[dim + 1].add(triple[0])
        indices_of_actual_simplices = [list(s) for s in indices_of_actual_simplices_set]

        index_mappings = [{} for _ in range(self.complex_dim + 1)]
        for dim in range(self.complex_dim + 1):
            for i, idx in enumerate(indices_of_actual_simplices[dim]):
                index_mappings[dim][idx] = i

        coboundaries = []
        for dim in range(self.complex_dim):
            row = []
            col = []
            data = []
            for triple in coboundaries_triples[dim]:
                row.append(index_mappings[dim + 1][triple[0]])
                col.append(index_mappings[dim][triple[1]])
                data.append(triple[2])
            shape = (
                len(indices_of_actual_simplices[dim + 1]),
                len(indices_of_actual_simplices[dim]),
            )
            coboundary = coo_matrix((data, (row, col)), shape=shape).toarray()
            coboundaries.append(coboundary)

        return coboundaries


class PersistentSheafLaplacian(Complex):
    """Persistent Laplacian built from a cellular sheaf.

    Parameters
    ----------
    sst : sheaf_simplex_tree, optional
        Sheaf simplex tree to build from.
    boundaries : list of np.ndarray, optional
        Pre-computed boundary matrices (alternative to ``sst``).
    filtrations : list of list of float, optional
        Pre-computed filtrations (alternative to ``sst``).
    device : torch.device, optional
        Override global compute device.
    """

    def __init__(
        self,
        sst: sheaf_simplex_tree | None = None,
        boundaries=None,
        filtrations=None,
        device=None,
    ):
        if sst is not None:
            coboundaries, filtrations = sst.apply_restriction_function()
            boundaries = [x.T for x in coboundaries]
        elif boundaries is None or filtrations is None:
            raise TypeError(
                "PersistentSheafLaplacian requires either a sheaf_simplex_tree "
                "or both boundaries and filtrations."
            )
        super().__init__(boundaries=boundaries, filtrations=filtrations, device=device)
