"""
FilteredBoundaryMatrix — PyTorch replacement for C++ FilteredBoundaryMatrix<FBMcoeff>.

Stores a boundary matrix d_n as a torch sparse tensor together with filtration
values for domain (n-simplices) and range ((n-1)-simplices).

Key design decisions:
  - Internal storage is torch.sparse_coo_tensor (most flexible for manipulation).
  - All tensors live on the configured device.
  - submatrix_at_filtration filters and reindexes COO entries (O(nnz)).
"""

from __future__ import annotations

import torch

from petls_pytorch._config import get_device


class FilteredBoundaryMatrix:
    """
    Boundary matrix d_n with per-simplex filtration metadata.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse COO tensor of shape (m, n) where m = #range simplices,
        n = #domain simplices.
    domain_filtrations : torch.Tensor
        1-D tensor of shape (n,) with filtration values for n-simplices.
    range_filtrations : torch.Tensor
        1-D tensor of shape (m,) with filtration values for (n-1)-simplices.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        domain_filtrations: torch.Tensor,
        range_filtrations: torch.Tensor,
        device: torch.device | str | None = None,
    ):
        if not matrix.is_sparse:
            raise ValueError("matrix must be a sparse tensor (COO or CSR)")

        target_device = torch.device(device) if device is not None else get_device()
        self.matrix = matrix.to(device=target_device)
        self.domain_filtrations = domain_filtrations.to(device=target_device, dtype=torch.float64)
        self.range_filtrations = range_filtrations.to(device=target_device, dtype=torch.float64)

        self.num_rows = self.range_filtrations.shape[0]
        self.num_cols = self.domain_filtrations.shape[0]
        self._coalesced_matrix = (
            self.matrix.coalesce() if self.matrix.layout == torch.sparse_coo else self.matrix
        )
        self._submatrix_cache: dict[tuple[float, bool], torch.Tensor] = {}
        self._incidence_data: tuple[torch.Tensor, torch.Tensor] | bool | None = None

        # Ensure sorted filtrations (required for index_of_filtration correctness)
        if not torch.all(self.domain_filtrations[:-1] <= self.domain_filtrations[1:]):
            raise ValueError("domain_filtrations must be non-decreasing")
        if not torch.all(self.range_filtrations[:-1] <= self.range_filtrations[1:]):
            raise ValueError("range_filtrations must be non-decreasing")

        if (
            self.device.type == "cpu"
            and self.num_cols > 0
            and self._coalesced_matrix._nnz() == 2 * self.num_cols
        ):
            self._get_incidence_data()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    def index_of_filtration(self, use_domain: bool, a: float) -> int:
        """
        Largest index where filtration <= a.

        Returns -1 if no simplices exist at or before a.
        """
        filts = self.domain_filtrations if use_domain else self.range_filtrations
        # searchsorted(..., right=True) gives first index where filts[idx] > a
        idx = torch.searchsorted(filts, a, right=True).item()
        return int(idx) - 1

    def submatrix_at_filtration(self, a: float, return_coo: bool = True) -> torch.Tensor:
        """
        Extract top-left submatrix where both row and col indices are <= a.

        Returns a sparse COO tensor on the same device.
        """
        cache_key = (float(a), return_coo)
        cached = self._submatrix_cache.get(cache_key)
        if cached is not None:
            return cached

        col_idx = self.index_of_filtration(use_domain=True, a=a)
        row_idx = self.index_of_filtration(use_domain=False, a=a)

        if col_idx < 0 or row_idx < 0:
            # Empty submatrix
            empty = torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long, device=self.device),
                values=torch.empty(0, dtype=self.matrix.dtype, device=self.device),
                size=(max(0, row_idx + 1), max(0, col_idx + 1)),
            )
            result = empty if return_coo else empty.to_sparse_csr()
            self._submatrix_cache[cache_key] = result
            return result

        n_rows = row_idx + 1
        n_cols = col_idx + 1

        # Convert to COO for easy filtering
        coo = self._coalesced_matrix
        indices = coo.indices()  # shape (2, nnz)
        values = coo.values()

        mask = (indices[0] < n_rows) & (indices[1] < n_cols)
        sub_indices = indices[:, mask]
        sub_values = values[mask]

        # No reindexing needed — we keep original indices within the submatrix
        # because the submatrix starts at (0,0) by definition
        # (the top-left block already has indices starting at 0)
        submatrix = torch.sparse_coo_tensor(
            indices=sub_indices,
            values=sub_values,
            size=(n_rows, n_cols),
            dtype=sub_values.dtype,
            device=self.device,
        )

        if not return_coo:
            submatrix = submatrix.to_sparse_csr()

        result = submatrix.coalesce() if return_coo else submatrix
        self._submatrix_cache[cache_key] = result
        return result

    def incidence_laplacian(
        self,
        max_row: int,
        max_col: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | None:
        """Return ``B @ B.T`` for cached two-entry incidence columns, if applicable."""
        target_device = torch.device(device) if device is not None else self.device
        if target_device.type != "cpu":
            return None

        if max_col < 0:
            n_rows = max_row + 1
            return torch.zeros(n_rows, n_rows, dtype=dtype, device=target_device)

        incidence_data = self._get_incidence_data()
        if incidence_data is None:
            return None

        row_pairs, value_pairs = incidence_data
        n_rows = max_row + 1
        n_cols = min(max_col + 1, row_pairs.shape[0])

        rows = row_pairs[:n_cols].to(device=target_device)
        values = value_pairs[:n_cols].to(device=target_device, dtype=dtype)
        valid = (rows[:, 0] < n_rows) & (rows[:, 1] < n_rows)
        if not bool(torch.all(valid)):
            rows = rows[valid]
            values = values[valid]

        L = torch.zeros(n_rows, n_rows, dtype=dtype, device=target_device)
        if rows.numel() == 0:
            return L

        r0 = rows[:, 0]
        r1 = rows[:, 1]
        v0 = values[:, 0]
        v1 = values[:, 1]
        L.index_put_((r0, r0), v0 * v0, accumulate=True)
        L.index_put_((r1, r1), v1 * v1, accumulate=True)
        offdiag = v0 * v1
        L.index_put_((r0, r1), offdiag, accumulate=True)
        L.index_put_((r1, r0), offdiag, accumulate=True)
        return L

    def _get_incidence_data(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._incidence_data is False:
            return None
        if self._incidence_data is not None:
            return self._incidence_data

        if self.num_cols == 0:
            empty_rows = torch.empty((0, 2), dtype=torch.long, device=self.device)
            empty_values = torch.empty((0, 2), dtype=self.matrix.dtype, device=self.device)
            self._incidence_data = (empty_rows, empty_values)
            return self._incidence_data

        coo = self._coalesced_matrix
        indices = coo.indices()
        cols = indices[1]
        counts = torch.bincount(cols, minlength=self.num_cols)
        if not bool(torch.all(counts == 2)):
            self._incidence_data = False
            return None

        order = torch.argsort(cols)
        row_pairs = indices[0, order].reshape(self.num_cols, 2)
        value_pairs = coo.values()[order].reshape(self.num_cols, 2)
        self._incidence_data = (row_pairs, value_pairs)
        return self._incidence_data

    def transpose(self) -> FilteredBoundaryMatrix:
        """Return the transposed boundary matrix with swapped filtrations."""
        return FilteredBoundaryMatrix(
            matrix=self.matrix.t(),
            domain_filtrations=self.range_filtrations.clone(),
            range_filtrations=self.domain_filtrations.clone(),
            device=self.device,
        )

    def __repr__(self) -> str:
        return (
            f"FilteredBoundaryMatrix("
            f"shape={self.shape}, "
            f"device={self.device}, "
            f"dtype={self.matrix.dtype}, "
            f"nnz={self.matrix._nnz()}, "
            f"domain_filts={self.num_cols}, "
            f"range_filts={self.num_rows})"
        )
