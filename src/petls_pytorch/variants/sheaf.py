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

        # Keep one deterministic order per dimension. Matrix coordinates and
        # filtration entries must use these same lists, including isolated
        # simplices that never occur in a restriction triple.
        self.simplices_by_dimension: list[list[tuple[int, ...]]] = [
            [] for _ in range(self.complex_dim + 1)
        ]
        self.filtrations_by_dimension: list[list[float]] = [[] for _ in range(self.complex_dim + 1)]
        indices: dict[tuple[int, ...], int] = {}
        index = 0
        for simplex_with_filtration in self.st.get_filtration():
            simplex = tuple(int(vertex) for vertex in simplex_with_filtration[0])
            dim = len(simplex) - 1
            self.simplices_by_dimension[dim].append(simplex)
            self.filtrations_by_dimension[dim].append(float(simplex_with_filtration[1]))
            indices[simplex] = index
            index += 1
        self.indices = indices
        self.simplex_indices_by_dimension = [
            {simplex: index for index, simplex in enumerate(simplices)}
            for simplices in self.simplices_by_dimension
        ]

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

        Every simplex contributes a cochain coordinate, including isolated
        simplices and coordinates whose restriction entries are all zero.
        """
        coboundaries_triples: list[list[tuple[tuple[int, ...], tuple[int, ...], float]]] = [
            [] for _ in range(self.complex_dim)
        ]

        for dim in range(self.complex_dim):
            for simplex in self.simplices_by_dimension[dim]:
                for coface_with_filtration in self.st.get_cofaces(list(simplex), 1):
                    coface = tuple(int(vertex) for vertex in coface_with_filtration[0])
                    simplex_list = list(simplex)
                    coface_list = list(coface)
                    sign = (-1) ** (self.coface_index(simplex_list, coface_list) % 2)
                    coeff = sign * self.restriction(simplex_list, coface_list, self)
                    coboundaries_triples[dim].append((coface, simplex, coeff))

        return self.reindex_coboundaries(coboundaries_triples), [
            list(values) for values in self.filtrations_by_dimension
        ]

    def reindex_coboundaries(self, coboundaries_triples):
        """Build dense matrices in the deterministic simplex-tree order."""
        coboundaries = []
        for dim in range(self.complex_dim):
            row = []
            col = []
            data = []
            local_rows = self.simplex_indices_by_dimension[dim + 1]
            local_columns = self.simplex_indices_by_dimension[dim]
            for coface, simplex, coeff in coboundaries_triples[dim]:
                row.append(local_rows[coface])
                col.append(local_columns[simplex])
                data.append(coeff)
            shape = (
                len(self.simplices_by_dimension[dim + 1]),
                len(self.simplices_by_dimension[dim]),
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
        dtype=None,
        zero_atol: float = 1e-8,
        zero_rtol: float = 1e-7,
        max_matrix_rows: int | None = 12_000,
        max_matrix_bytes: int | None = 4_000_000_000,
        on_oversize: str = "raise",
    ):
        if sst is not None:
            coboundaries, filtrations = sst.apply_restriction_function()
            boundaries = [x.T for x in coboundaries]
        elif boundaries is None or filtrations is None:
            raise TypeError(
                "PersistentSheafLaplacian requires either a sheaf_simplex_tree "
                "or both boundaries and filtrations."
            )
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
        )
