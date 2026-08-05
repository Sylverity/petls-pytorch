"""
Complex — main PETLS class.

This is the PyTorch-native replacement for petls::Complex (C++).
It stores a list of FilteredBoundaryMatrix objects and provides
get_L, get_up, get_down, spectra, and eigenpairs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Hashable, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from petls_pytorch._config import resolve_device, resolve_dtype
from petls_pytorch.core.filtered_boundary import FilteredBoundaryMatrix
from petls_pytorch.core.profile import Profile


class LaplacianSizeError(MemoryError):
    """Raised before a requested dense Laplacian would exceed its guard."""


class Complex:
    """
    Primary class for computing persistent Laplacian matrices and eigenvalues.

    Parameters
    ----------
    boundaries : list of np.ndarray or torch.Tensor, optional
        List of boundary matrices d_1, d_2, ..., d_N.
        Each should be a dense array or sparse matrix.
    filtrations : list of list of float, optional
        Filtration values per dimension. filtrations[dim] is a list of
        filtration values for simplices in dimension dim.
        Must satisfy len(filtrations) == len(boundaries) + 1.
    device : torch.device, optional
        Override global device.
    """

    def __init__(
        self,
        boundaries: Optional[List[Any]] = None,
        filtrations: Optional[List[List[float]]] = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
        simplex_tree=None,
        eigs_algorithm: str | Callable = "eigvalsh",
        zero_atol: float = 1e-8,
        zero_rtol: float = 1e-7,
        max_matrix_rows: int | None = 12_000,
        max_matrix_bytes: int | None = 4_000_000_000,
        on_oversize: str = "raise",
    ):
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)
        if zero_atol < 0 or zero_rtol < 0:
            raise ValueError("zero_atol and zero_rtol must be non-negative")
        if max_matrix_rows is not None and max_matrix_rows < 1:
            raise ValueError("max_matrix_rows must be positive or None")
        if max_matrix_bytes is not None and max_matrix_bytes < 1:
            raise ValueError("max_matrix_bytes must be positive or None")
        if on_oversize not in {"raise", "homology_only"}:
            raise ValueError("on_oversize must be 'raise' or 'homology_only'")
        self.zero_atol = float(zero_atol)
        self.zero_rtol = float(zero_rtol)
        self.max_matrix_rows = max_matrix_rows
        self.max_matrix_bytes = max_matrix_bytes
        self.on_oversize = on_oversize
        self._verbose = False
        self._flipped = False
        self._logger = logging.getLogger(__name__)

        self.top_dim: int = 0
        self.filtered_boundaries: List[FilteredBoundaryMatrix] = []
        self.profile = Profile(zero_atol=self.zero_atol, zero_rtol=self.zero_rtol)
        self.simplex_tree = simplex_tree
        self.simplices_by_dimension: list[list[tuple[int, ...]]] = []
        self.simplex_filtrations: list[list[float]] = []
        self.simplex_to_index: list[dict[tuple[int, ...], int]] = []
        self.point_labels: list[Hashable] | None = None
        self.simplex_labels_by_dimension: list[list[tuple[Hashable, ...]]] | None = None
        self._persistence_computed = False

        self._eigs_algorithm: str | Callable = "eigvalsh"
        self._num_eigenvalues: int = 10
        self._eigenvalue_order: str = "SM"

        if simplex_tree is not None:
            from petls_pytorch.utils.simplex_tree import simplex_tree_boundaries_filtrations

            extracted = simplex_tree_boundaries_filtrations(
                simplex_tree,
                return_simplices=True,
            )
            extracted_boundaries, extracted_filtrations, simplices = extracted
            if boundaries is None:
                boundaries = extracted_boundaries
            if filtrations is None:
                filtrations = extracted_filtrations
            self.simplices_by_dimension = simplices
            self.simplex_filtrations = [list(values) for values in extracted_filtrations]
            self.simplex_to_index = [
                {simplex: index for index, simplex in enumerate(values)} for values in simplices
            ]

        if boundaries is not None and filtrations is not None:
            self.set_boundaries_filtrations(boundaries, filtrations)
            if not self.simplex_filtrations:
                self.simplex_filtrations = [list(map(float, values)) for values in filtrations]
        else:
            # Empty complex — user will set later
            self.filtered_boundaries = []
            self.top_dim = 0

        self.set_eigs_algorithm(eigs_algorithm)

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._verbose = value

    @property
    def flipped(self) -> bool:
        return self._flipped

    @flipped.setter
    def flipped(self, value: bool) -> None:
        self._flipped = value

    def set_boundaries_filtrations(
        self,
        boundaries: List[Any],
        filtrations: List[List[float]],
    ) -> None:
        """
        Set boundary matrices and filtrations.

        boundaries[i] is d_{i+1} with shape (n_{i}, n_{i+1}).
        filtrations[i] is the filtration list for dimension i.
        """
        if len(filtrations) != len(boundaries) + 1:
            raise ValueError(
                f"len(filtrations)={len(filtrations)} must be len(boundaries)+1={len(boundaries) + 1}"
            )

        self.filtered_boundaries = []
        self.top_dim = len(boundaries)

        # d_0 placeholder aligns indexing with the original PETLS Complex, but
        # its metadata must contain every actual vertex birth (including
        # negative power-filtration values).
        vertex_filtrations = torch.tensor(filtrations[0], device=self.device, dtype=torch.float64)
        n_vertices = len(vertex_filtrations)
        with torch.sparse.check_sparse_tensor_invariants():
            dummy_mat = torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long, device=self.device),
                values=torch.empty(0, dtype=self.dtype, device=self.device),
                size=(n_vertices, n_vertices),
            ).coalesce()
        dummy = FilteredBoundaryMatrix(
            matrix=dummy_mat,
            domain_filtrations=vertex_filtrations,
            range_filtrations=vertex_filtrations,
            device=self.device,
        )
        self.filtered_boundaries.append(dummy)

        for dim, (B, f_dom, f_rng) in enumerate(
            zip(boundaries, filtrations[1:], filtrations[:-1]), start=1
        ):
            B_t = self._ensure_sparse_tensor(B)
            domain_f = torch.tensor(f_dom, device=self.device, dtype=torch.float64)
            range_f = torch.tensor(f_rng, device=self.device, dtype=torch.float64)

            if B_t.shape[0] != len(f_rng):
                raise ValueError(
                    f"boundaries[{dim - 1}].shape[0]={B_t.shape[0]} != len(filtrations[{dim - 1}])={len(f_rng)}"
                )
            if B_t.shape[1] != len(f_dom):
                raise ValueError(
                    f"boundaries[{dim - 1}].shape[1]={B_t.shape[1]} != len(filtrations[{dim}])={len(f_dom)}"
                )

            fbm = FilteredBoundaryMatrix(
                matrix=B_t,
                domain_filtrations=domain_f,
                range_filtrations=range_f,
                device=self.device,
            )
            self.filtered_boundaries.append(fbm)

    def _ensure_sparse_tensor(self, x) -> torch.Tensor:
        """Convert numpy array, scipy sparse, or torch dense to torch sparse COO."""
        import scipy.sparse

        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                sparse = x.coalesce() if x.layout == torch.sparse_coo else x
                return sparse.to(dtype=self.dtype)
            # Dense tensor -> COO
            return cast(torch.Tensor, x.to(dtype=self.dtype).to_sparse_coo())
        if isinstance(x, np.ndarray):
            return cast(torch.Tensor, torch.from_numpy(x).to(dtype=self.dtype).to_sparse_coo())
        if scipy.sparse.issparse(x):
            coo = x.tocoo()
            indices = torch.stack(
                [
                    torch.from_numpy(coo.row).long(),
                    torch.from_numpy(coo.col).long(),
                ]
            )
            values = torch.from_numpy(coo.data).to(dtype=self.dtype)
            return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()
        raise TypeError(f"Cannot convert type {type(x)} to sparse tensor")

    def set_eigs_algorithm(
        self,
        algorithm: str | Callable,
        num_eigenvalues: int = 10,
        eigenvalue_order: str = "SM",
    ) -> None:
        """Set eigenvalue solver.

        Parameters
        ----------
        algorithm : str or callable
            Solver name or callable that accepts a matrix and returns eigenvalues.
        num_eigenvalues : int, optional
            Number of eigenvalues for sparse solvers (default 10).
        eigenvalue_order : str, optional
            Which eigenvalues to target for sparse solvers (default "SM").
        """
        if not callable(algorithm) and algorithm not in {"eigvalsh", "sparse"}:
            raise ValueError("algorithm must be 'eigvalsh', 'sparse', or a callable")
        if num_eigenvalues < 1:
            raise ValueError("num_eigenvalues must be positive")
        if eigenvalue_order not in {"SM", "SA", "LM", "LA", "BE"}:
            raise ValueError("eigenvalue_order must be one of 'SM', 'SA', 'LM', 'LA', or 'BE'")
        self._eigs_algorithm = algorithm
        self._num_eigenvalues = num_eigenvalues
        self._eigenvalue_order = eigenvalue_order

    def _laplacian_rows(self, dim: int, scale: float) -> int:
        if dim < 0:
            raise ValueError("dim must be non-negative")
        if dim == 0:
            if self.top_dim == 0:
                return self.filtered_boundaries[0].index_of_filtration(True, scale) + 1
            return self.filtered_boundaries[1].index_of_filtration(False, scale) + 1
        if dim > self.top_dim:
            return 0
        return self.filtered_boundaries[dim].index_of_filtration(True, scale) + 1

    def estimate_laplacian(self, dim: int, a: float, b: float | None = None) -> dict[str, Any]:
        """Estimate final and peak-intermediate dense Laplacian allocations."""
        b = a if b is None else b
        if b < a:
            raise ValueError("b must be greater than or equal to a")
        rows = self._laplacian_rows(dim, a)
        intermediate_rows = rows
        if dim < self.top_dim:
            coboundary = self.filtered_boundaries[dim + 1]
            if coboundary.index_of_filtration(True, b) >= 0:
                intermediate_rows = coboundary.index_of_filtration(False, b) + 1
        peak_rows = max(rows, intermediate_rows)
        element_size = torch.empty((), dtype=self.dtype).element_size()
        dense_bytes = rows * rows * element_size
        intermediate_dense_bytes = intermediate_rows * intermediate_rows * element_size
        peak_dense_bytes = max(dense_bytes, intermediate_dense_bytes)
        exceeds_rows = self.max_matrix_rows is not None and peak_rows > self.max_matrix_rows
        exceeds_bytes = (
            self.max_matrix_bytes is not None and peak_dense_bytes > self.max_matrix_bytes
        )
        within_dense_limits = not (exceeds_rows or exceeds_bytes)
        if within_dense_limits:
            backend = "dense"
        elif a == b:
            backend = "sparse"
        else:
            backend = "homology_only"
        return {
            "dim": int(dim),
            "filtration_a": float(a),
            "filtration_b": float(b),
            "rows": rows,
            "intermediate_rows": intermediate_rows,
            "peak_rows": peak_rows,
            "dtype": str(self.dtype).removeprefix("torch."),
            "dense_bytes": dense_bytes,
            "intermediate_dense_bytes": intermediate_dense_bytes,
            "peak_dense_bytes": peak_dense_bytes,
            "within_dense_limits": within_dense_limits,
            "exceeds_max_matrix_rows": exceeds_rows,
            "exceeds_max_matrix_bytes": exceeds_bytes,
            "recommended_backend": backend,
        }

    def _guard_dense_laplacian(self, dim: int, a: float, b: float) -> None:
        estimate = self.estimate_laplacian(dim, a, b)
        if estimate["within_dense_limits"]:
            return
        message = (
            f"Dense L_{dim}({a}, {b}) would have {estimate['rows']} output rows, "
            f"require up to {estimate['intermediate_rows']} intermediate rows, and allocate "
            f"approximately {estimate['peak_dense_bytes']} bytes for its largest dense "
            "matrix, exceeding this object's matrix guard. Use an ordinary sparse spectrum "
            "at a == b, raise the limits explicitly, or use "
            "topology_summary(..., on_oversize='homology_only')."
        )
        raise LaplacianSizeError(message)

    @staticmethod
    def _to_scipy_sparse(matrix: torch.Tensor):
        import scipy.sparse

        coo = matrix.to_sparse_coo().coalesce().cpu()
        indices = coo.indices().numpy()
        values = coo.values().numpy()
        return scipy.sparse.coo_matrix(
            (values, (indices[0], indices[1])), shape=tuple(coo.shape)
        ).tocsr()

    def get_L_sparse(self, dim: int, scale: float):
        """Construct an ordinary Hodge Laplacian directly as SciPy CSR.

        This path never materializes a dense boundary matrix or Laplacian.
        Persistent Schur-complement Laplacians (``a < b``) are intentionally
        excluded because their complement can be dense.
        """
        import scipy.sparse

        rows = self._laplacian_rows(dim, scale)
        result = scipy.sparse.csr_matrix(
            (rows, rows), dtype=np.dtype(str(self.dtype).split(".")[-1])
        )
        if rows == 0 or dim > self.top_dim:
            return result
        if dim > 0:
            boundary = self.filtered_boundaries[dim].submatrix_at_filtration(scale)
            boundary_scipy = self._to_scipy_sparse(boundary)
            result = result + boundary_scipy.T @ boundary_scipy
        if dim < self.top_dim:
            coboundary = self.filtered_boundaries[dim + 1].submatrix_at_filtration(scale)
            coboundary_scipy = self._to_scipy_sparse(coboundary)
            result = result + coboundary_scipy @ coboundary_scipy.T
        return result.tocsr()

    def _sparse_ordinary_eigenpairs(
        self,
        dim: int,
        scale: float,
        num_eigenvalues: int | None = None,
        return_eigenvectors: bool = False,
        augment_for_betti: bool = True,
        eigenvalue_order: str | None = None,
    ):
        from petls_pytorch.core.eigenvalues import (
            EIGENVALUE_ORDERS,
            eigenvalue_indices,
            select_eigenpairs,
        )

        import scipy.sparse.linalg

        laplacian = self.get_L_sparse(dim, scale)
        rows = laplacian.shape[0]
        if rows == 0:
            empty_values = np.empty(0, dtype=np.float64)
            empty_vectors = np.empty((0, 0), dtype=np.float64)
            return (empty_values, empty_vectors) if return_eigenvectors else empty_values

        requested = self._num_eigenvalues if num_eigenvalues is None else int(num_eigenvalues)
        if requested < 1:
            raise ValueError("num_eigenvalues must be positive")
        which = self._eigenvalue_order if eigenvalue_order is None else eigenvalue_order
        if which not in EIGENVALUE_ORDERS:
            raise ValueError("eigenvalue_order must be one of 'SM', 'SA', 'LM', 'LA', or 'BE'")
        known_betti = self._gudhi_betti_at(dim, scale)
        # A few eigenvalues beyond a known nullity usually expose the gap. Do
        # not let a very large Betti number silently turn a lowest-spectrum
        # query into an almost-complete eigendecomposition.
        if (
            augment_for_betti
            and which in {"SM", "SA"}
            and known_betti is not None
            and known_betti < rows
        ):
            requested = max(requested, min(known_betti + 3, 256))
        requested = min(requested, rows)

        if laplacian.nnz == 0:
            values = np.zeros(rows, dtype=np.float64)
            selected = eigenvalue_indices(values, requested, which)
            vectors = np.zeros((rows, len(selected)), dtype=np.float64)
            vectors[selected, np.arange(len(selected))] = 1.0
            values = values[selected]
        elif np.count_nonzero(laplacian.diagonal()) == laplacian.nnz:
            values = np.asarray(laplacian.diagonal(), dtype=np.float64)
            selected = eigenvalue_indices(values, requested, which)
            vectors = np.zeros((rows, len(selected)), dtype=np.float64)
            vectors[selected, np.arange(len(selected))] = 1.0
            values = values[selected]
        elif rows <= 256 and requested == rows:
            # Complete output is dense by definition, but this fallback is
            # deliberately limited to small matrices.
            dense = laplacian.toarray()
            values, vectors = np.linalg.eigh(dense)
        else:
            requested = min(requested, rows - 1)
            solver_which = "LA" if which == "BE" and requested == 1 else which
            values, vectors = scipy.sparse.linalg.eigsh(
                laplacian,
                k=requested,
                which=solver_which,
                return_eigenvectors=True,
            )
        values, selected_vectors = select_eigenpairs(values, vectors, requested, which)
        assert selected_vectors is not None
        if return_eigenvectors:
            return values, selected_vectors
        return values

    def ordinary_spectrum(
        self,
        dim: int,
        scale: float,
        num_eigenvalues: int | None = None,
        eigenvalue_order: str | None = None,
    ) -> list[float]:
        """Return a selected ordinary Hodge spectrum via a sparse path."""
        values = self._sparse_ordinary_eigenpairs(
            dim,
            scale,
            num_eigenvalues,
            eigenvalue_order=eigenvalue_order,
        )
        return cast(list[float], np.asarray(values, dtype=np.float64).tolist())

    def get_L(self, dim: int, a: float, b: float) -> torch.Tensor:
        """Get persistent Laplacian matrix L^{dim}(a,b) as dense tensor."""
        from petls_pytorch.core.laplacian import get_L

        self._guard_dense_laplacian(dim, a, b)
        return get_L(
            dim,
            a,
            b,
            self.filtered_boundaries,
            self.top_dim,
            self.device,
            self.dtype,
        )

    def get_L_top_dim_flipped(self, a: float) -> torch.Tensor:
        """Get the flipped top-dimension Laplacian B @ B^T.

        When ``flipped=True``, ``spectra()`` uses this matrix for the top
        dimension because the nonzero eigenvalues of ``B @ B^T`` (shape
        m×m) are the same as those of ``B^T @ B`` (shape n×n), but the
        former may be smaller.
        """
        if self.top_dim == 0:
            return torch.empty(0, 0, dtype=self.dtype, device=self.device)
        self._guard_dense_laplacian(self.top_dim, a, a)
        fbm = self.filtered_boundaries[self.top_dim]
        B = fbm.submatrix_at_filtration(a)
        if B.shape[0] == 0 or B.shape[1] == 0:
            return torch.empty(0, 0, dtype=self.dtype, device=self.device)
        B_dense = B.to_dense().to(dtype=self.dtype)
        return B_dense @ B_dense.T

    def get_up(self, dim: int, a: float, b: float) -> torch.Tensor:
        """Get persistent up-Laplacian."""
        from petls_pytorch.core.laplacian import get_up

        self._guard_dense_laplacian(dim, a, b)
        if dim >= self.top_dim:
            # No higher-dimensional simplices → zero matrix sized to dim-simplices at a
            if dim == 0 and len(self.filtered_boundaries) == 1:
                # Edge case: no 1-simplices at all
                n = self.filtered_boundaries[0].index_of_filtration(True, a) + 1
                return torch.zeros(n, n, dtype=self.dtype, device=self.device)
            fbm = self.filtered_boundaries[dim]
            n = fbm.index_of_filtration(True, a) + 1
            return torch.zeros(n, n, dtype=self.dtype, device=self.device)
        return get_up(self.filtered_boundaries[dim + 1], a, b, self.device, self.dtype)

    def get_down(self, dim: int, a: float) -> torch.Tensor:
        """Get persistent down-Laplacian."""
        from petls_pytorch.core.laplacian import get_down

        self._guard_dense_laplacian(dim, a, a)
        return cast(
            torch.Tensor,
            get_down(self.filtered_boundaries[dim], a, self.device, self.dtype),
        )

    def _solve_eigs(self, L: torch.Tensor) -> torch.Tensor:
        """Dispatch to eigenvalue solver."""
        from petls_pytorch.core.eigenvalues import (
            solve_eigenvalues,
            solve_sparse_eigenvalues,
        )

        algorithm = self._eigs_algorithm
        if algorithm == "sparse":
            return solve_sparse_eigenvalues(
                L,
                num_eigenvalues=self._num_eigenvalues,
                which=self._eigenvalue_order,
            )
        return cast(torch.Tensor, solve_eigenvalues(L, algorithm=algorithm))

    def _solve_eigenpairs(self, L: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to eigenpair solver."""
        from petls_pytorch.core.eigenvalues import solve_eigenpairs

        return solve_eigenpairs(L, algorithm="eigh")

    def spectra(
        self,
        dim: Optional[int] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        request_list: Optional[List[Tuple[int, float, float]]] = None,
        allpairs: bool = False,
    ):
        """
        Compute eigenvalues of L_{dim}^{a,b}.

        Parameters
        ----------
        dim, a, b : optional
            Single (dim, a, b) query.
        request_list : list of (dim, a, b), optional
            Multiple queries.
        allpairs : bool
            If True and no other args given, compute all (a_i, a_j) pairs
            with j >= i for all dimensions.

        Returns
        -------
        list[float]
            If single (dim, a, b) passed.
        list[tuple]
            If request_list or no args passed: [(dim, a, b, eigenvalues), ...]
        """
        # Build request list
        if dim is not None and a is not None and b is not None:
            requests = [(dim, a, b)]
        elif request_list is not None:
            requests = [(int(r[0]), float(r[1]), float(r[2])) for r in request_list]
        else:
            dims = list(range(self.top_dim + 1))
            filtrations = self.get_all_filtrations()
            if allpairs:
                requests = self.filtration_list_to_spectra_request_allpairs(filtrations, dims)
            else:
                requests = self.filtration_list_to_spectra_request(filtrations, dims)

        responses = []
        for d, fa, fb in requests:
            self.profile.start_all()

            # Edge case: no 1-simplices and dim == 0
            if d == 0 and len(self.filtered_boundaries) == 1:
                betti0 = self.filtered_boundaries[0].index_of_filtration(True, fa) + 1
                eigs_list = [0.0] * betti0
                self.profile.durations_L.append(0.0)
                self.profile.durations_eigs.append(0.0)
                self.profile.stop_all()
                self.profile.dims.append(d)
                self.profile.filtration_a.append(fa)
                self.profile.filtration_b.append(fb)
                self.profile.L_rows.append(betti0)
                self.profile.bettis.append(betti0)
                self.profile.lambdas.append(0.0)
                responses.append((d, fa, fb, eigs_list))
                continue

            # Determine matrix size for profiling
            if d == 0:
                l_rows = self.filtered_boundaries[1].index_of_filtration(False, fa) + 1
            else:
                l_rows = self.filtered_boundaries[d].index_of_filtration(True, fa) + 1
            self.profile.L_rows.append(l_rows)

            self.profile.start_L()
            use_sparse_ordinary = self._eigs_algorithm == "sparse" and fa == fb
            if use_sparse_ordinary:
                # Construction and solve happen together inside SciPy's sparse
                # operator path; no dense Laplacian is created.
                self.profile.stop_L()
                self.profile.start_eigs()
                eigs_list = self.ordinary_spectrum(d, fa, self._num_eigenvalues)
                self.profile.stop_eigs()
            else:
                # Flipped top-dimension optimization
                if d == self.top_dim and self.flipped:
                    self._guard_dense_laplacian(d, fa, fb)
                    L = self.get_L_top_dim_flipped(fa)
                else:
                    L = self.get_L(d, fa, fb)
                self.profile.stop_L()

                if L.numel() == 0:
                    eigs = torch.empty(0, device=L.device, dtype=L.dtype)
                    self.profile.durations_eigs.append(0.0)
                else:
                    self.profile.start_eigs()
                    eigs = self._solve_eigs(L)
                    self.profile.stop_eigs()
                eigs_list = eigs.cpu().tolist() if isinstance(eigs, torch.Tensor) else list(eigs)

            self.profile.stop_all()

            # Zero-pad flipped top-dimension eigenvalues to match true Laplacian size
            if d == self.top_dim and self.flipped and not use_sparse_ordinary:
                fbm = self.filtered_boundaries[self.top_dim]
                B = fbm.submatrix_at_filtration(fa)
                m, n = B.shape
                expected = l_rows
                actual = len(eigs_list)
                if actual < expected:
                    eigs_list = [0.0] * (expected - actual) + sorted(eigs_list)
                elif actual > expected:
                    eigs_list = sorted(eigs_list)[actual - expected :]

            betti, lam = self.eigenvalues_summarize(eigs_list)
            self.profile.dims.append(d)
            self.profile.filtration_a.append(fa)
            self.profile.filtration_b.append(fb)
            self.profile.bettis.append(betti)
            self.profile.lambdas.append(lam)

            if self.verbose:
                self._logger.debug(
                    "dim=%s a=%s b=%s | size=%s | betti=%s",
                    d,
                    fa,
                    fb,
                    l_rows,
                    betti,
                )

            responses.append((d, fa, fb, eigs_list))

        if len(responses) == 1:
            return responses[0][3]
        return responses

    def eigenpairs(
        self,
        dim: Optional[int] = None,
        a: Optional[float] = None,
        b: Optional[float] = None,
        request_list: Optional[List[Tuple[int, float, float]]] = None,
        allpairs: bool = False,
    ):
        """
        Compute eigenvalues and eigenvectors of L_{dim}^{a,b}.

        Parameters
        ----------
        dim, a, b : optional
            Single (dim, a, b) query.
        request_list : list of (dim, a, b), optional
            Multiple queries.
        allpairs : bool
            If True and no other args given, compute all (a_i, a_j) pairs
            with j >= i for all dimensions.

        Returns
        -------
        (eigenvalues, eigenvectors)
            If single (dim, a, b) passed.
        list[tuple]
            If request_list or no args passed: [(dim, a, b, eigenvalues, eigenvectors), ...]
        """
        single_query = dim is not None and a is not None and b is not None
        if dim is not None and a is not None and b is not None:
            requests = [(dim, a, b)]
        elif request_list is not None:
            requests = [(int(r[0]), float(r[1]), float(r[2])) for r in request_list]
        else:
            dims = list(range(self.top_dim + 1))
            filtrations = self.get_all_filtrations()
            if allpairs:
                requests = self.filtration_list_to_spectra_request_allpairs(filtrations, dims)
            else:
                requests = self.filtration_list_to_spectra_request(filtrations, dims)

        responses = []
        for d, fa, fb in requests:
            L = self.get_L(d, fa, fb)
            if L.numel() == 0:
                vals = torch.empty(0, device=L.device, dtype=L.dtype)
                vecs = torch.empty(0, 0, device=L.device, dtype=L.dtype)
            else:
                vals, vecs = self._solve_eigenpairs(L)
            vals_list = vals.cpu().tolist()
            responses.append((d, fa, fb, vals_list, vecs))

        if single_query and len(responses) == 1:
            return responses[0][3], responses[0][4]
        return responses

    def eigenvalues_summarize(
        self,
        eigenvalues: list[float] | np.ndarray | torch.Tensor,
    ) -> Tuple[int, float]:
        """Compute ``(nullity, least_nonzero)`` with object-local tolerances."""

        eigenvalue_array = (
            eigenvalues.cpu().numpy()
            if isinstance(eigenvalues, torch.Tensor)
            else np.array(eigenvalues)
        )

        if eigenvalue_array.size == 0:
            return 0, 0.0
        scale = float(np.max(np.abs(eigenvalue_array)))
        tol = self.zero_atol + self.zero_rtol * scale
        absolute = np.abs(eigenvalue_array)
        betti = int(np.sum(absolute <= tol))
        nonzeros = absolute[absolute > tol]
        least = float(nonzeros.min()) if len(nonzeros) > 0 else 0.0
        return betti, least

    def _zero_tolerance(self, eigenvalues: Sequence[float] | np.ndarray) -> float:
        values = np.asarray(eigenvalues, dtype=np.float64)
        spectral_scale = float(np.max(np.abs(values))) if values.size else 0.0
        return self.zero_atol + self.zero_rtol * spectral_scale

    def _ensure_persistence(self) -> None:
        if self.simplex_tree is None:
            raise RuntimeError("This operation requires construction from a Gudhi SimplexTree")
        if not self._persistence_computed:
            self.simplex_tree.persistence(
                min_persistence=-1.0,
                persistence_dim_max=True,
            )
            self._persistence_computed = True

    def persistence_intervals(self, dim: int) -> np.ndarray:
        """Return Gudhi persistence intervals ``[birth, death)`` for a dimension."""
        if dim < 0:
            raise ValueError("dim must be non-negative")
        self._ensure_persistence()
        intervals = self.simplex_tree.persistence_intervals_in_dimension(dim)
        return np.asarray(intervals, dtype=np.float64).copy()

    def persistent_betti(self, dim: int, birth_scale: float, death_scale: float) -> int:
        """Return the rank of the homology map from ``birth_scale`` to ``death_scale``."""
        if death_scale < birth_scale:
            raise ValueError("death_scale must be greater than or equal to birth_scale")
        intervals = self.persistence_intervals(dim)
        if intervals.size == 0:
            return 0
        return int(np.sum((intervals[:, 0] <= birth_scale) & (intervals[:, 1] > death_scale)))

    def betti_numbers_at(self, scale: float) -> dict[int, int]:
        """Return authoritative Gudhi Betti numbers for the sublevel complex."""
        return {dim: self.persistent_betti(dim, scale, scale) for dim in range(self.top_dim + 1)}

    def _gudhi_betti_at(self, dim: int, scale: float) -> int | None:
        if self.simplex_tree is None:
            return None
        return self.persistent_betti(dim, scale, scale)

    def topology_summary(
        self,
        dimensions: Sequence[int] | None = None,
        a: float = 0.0,
        b: float = 0.0,
        on_oversize: str | None = None,
        smallest_eigenvalues: int = 10,
    ) -> dict[str, Any]:
        """Summarize Betti numbers and persistent-Laplacian spectral gaps.

        Gudhi persistence intervals are authoritative when a simplex tree is
        available.  Spectral nullity is retained as an auditable numerical
        companion rather than silently replacing homology.
        """
        if b < a:
            raise ValueError("b must be greater than or equal to a")
        policy = self.on_oversize if on_oversize is None else on_oversize
        if policy not in {"raise", "homology_only"}:
            raise ValueError("on_oversize must be 'raise' or 'homology_only'")
        if smallest_eigenvalues < 0:
            raise ValueError("smallest_eigenvalues must be non-negative")
        if dimensions is None:
            dimensions = range(self.top_dim + 1)

        betti: dict[int, int] = {}
        spectral_nullity: dict[int, int | None] = {}
        least_nonzero: dict[int, float | None] = {}
        matrix_rows: dict[int, int] = {}
        tolerances: dict[int, float | None] = {}
        smallest: dict[int, list[float]] = {}
        statuses: dict[int, str] = {}
        estimates: dict[int, dict[str, Any]] = {}
        betti_source: dict[int, str] = {}

        for raw_dim in dimensions:
            dim = int(raw_dim)
            if dim < 0:
                raise ValueError("dimensions must contain non-negative integers")
            estimate = self.estimate_laplacian(dim, a, b)
            estimates[dim] = estimate
            matrix_rows[dim] = estimate["rows"]

            authoritative = None
            if self.simplex_tree is not None:
                authoritative = self.persistent_betti(dim, a, b)
                betti[dim] = authoritative
                betti_source[dim] = "gudhi_persistence"

            if not estimate["within_dense_limits"] and a < b:
                if policy == "raise":
                    self._guard_dense_laplacian(dim, a, b)
                if authoritative is None:
                    statuses[dim] = "skipped_oversize_no_homology_backend"
                else:
                    statuses[dim] = "homology_only_oversize"
                spectral_nullity[dim] = None
                least_nonzero[dim] = None
                tolerances[dim] = None
                smallest[dim] = []
                continue

            try:
                if a == b and (
                    not estimate["within_dense_limits"] or self._eigs_algorithm == "sparse"
                ):
                    eigenvalues = self.ordinary_spectrum(dim, a, eigenvalue_order="SM")
                    statuses[dim] = "sparse_lowest_spectrum"
                else:
                    eigenvalues = self.spectra(dim, a, b)
                    statuses[dim] = "complete"
            except LaplacianSizeError:
                if policy == "raise":
                    raise
                statuses[dim] = "homology_only_oversize"
                spectral_nullity[dim] = None
                least_nonzero[dim] = None
                tolerances[dim] = None
                smallest[dim] = []
                continue

            values = np.asarray(eigenvalues, dtype=np.float64)
            nullity, least = self.eigenvalues_summarize(values)
            tolerance = self._zero_tolerance(values)
            spectral_nullity[dim] = nullity
            if statuses[dim] == "sparse_lowest_spectrum" and (
                least == 0.0 or (authoritative is not None and authoritative >= len(values))
            ):
                least_nonzero[dim] = None
                statuses[dim] = "sparse_null_modes_only"
            else:
                least_nonzero[dim] = least
            tolerances[dim] = tolerance
            smallest[dim] = np.sort(values)[:smallest_eigenvalues].tolist()
            if authoritative is None:
                betti[dim] = nullity
                betti_source[dim] = "laplacian_nullity"

        return {
            "filtration_a": float(a),
            "filtration_b": float(b),
            "betti_kind": "ordinary" if a == b else "persistent",
            "betti": betti,
            "betti_source": betti_source,
            "spectral_nullity": spectral_nullity,
            "least_nonzero_eigenvalue": least_nonzero,
            "matrix_rows": matrix_rows,
            "zero_tolerance": tolerances,
            "zero_atol": self.zero_atol,
            "zero_rtol": self.zero_rtol,
            "smallest_eigenvalues": smallest,
            "calculation_status": statuses,
            "estimates": estimates,
            "method": "gudhi_homology_with_persistent_laplacian_spectrum",
        }

    def harmonic_features(
        self,
        dim: int,
        a: float,
        b: float | None = None,
        coefficient_atol: float = 0.0,
        max_features: int | None = None,
    ) -> dict[str, Any]:
        """Return numerical harmonic representatives mapped to simplices."""
        b = a if b is None else b
        if not self.simplices_by_dimension:
            raise RuntimeError("Simplex mappings require construction from a Gudhi SimplexTree")
        if dim < 0 or dim >= len(self.simplices_by_dimension):
            raise ValueError(f"dim must be between 0 and {len(self.simplices_by_dimension) - 1}")
        if coefficient_atol < 0:
            raise ValueError("coefficient_atol must be non-negative")
        if max_features is not None and max_features < 1:
            raise ValueError("max_features must be positive or None")

        authoritative = self.persistent_betti(dim, a, b) if self.simplex_tree is not None else None
        estimate = self.estimate_laplacian(dim, a, b)
        if a < b and not estimate["within_dense_limits"]:
            raise LaplacianSizeError(
                "Persistent harmonic localization requires a dense Schur-complement "
                "Laplacian, and this request exceeds the matrix guard. Harmonic "
                "localization has a sparse oversized path only for ordinary a == b "
                "calculations."
            )
        feature_target = authoritative
        calculation_status = "complete"
        if authoritative is not None and max_features is not None:
            feature_target = min(authoritative, max_features)
        elif authoritative is not None and not estimate["within_dense_limits"]:
            feature_target = min(authoritative, 10)
            if feature_target < authoritative:
                calculation_status = "truncated_for_scale"
        if a == b and (not estimate["within_dense_limits"] or self._eigs_algorithm == "sparse"):
            requested = max(self._num_eigenvalues, (feature_target or 0) + 3)
            values_np, vectors_np = self._sparse_ordinary_eigenpairs(
                dim,
                a,
                requested,
                return_eigenvectors=True,
                augment_for_betti=False,
                eigenvalue_order="SM",
            )
        else:
            values, vectors = self.eigenpairs(dim, a, b)
            values_np = np.asarray(values, dtype=np.float64)
            vectors_np = vectors.detach().cpu().numpy()

        tolerance = self._zero_tolerance(values_np)
        harmonic_indices = np.flatnonzero(np.abs(values_np) <= tolerance)
        numerical_nullity = len(harmonic_indices)
        if feature_target is not None:
            harmonic_indices = harmonic_indices[:feature_target]
        simplex_count = self._laplacian_rows(dim, a)
        simplices = self.simplices_by_dimension[dim][:simplex_count]
        labels = (
            self.simplex_labels_by_dimension[dim][:simplex_count]
            if self.simplex_labels_by_dimension is not None
            else None
        )
        features = []
        for eigen_index in harmonic_indices:
            coefficients = []
            for simplex_index, (simplex, coefficient) in enumerate(
                zip(simplices, vectors_np[:, eigen_index])
            ):
                coefficient_float = float(coefficient)
                if abs(coefficient_float) < coefficient_atol:
                    continue
                item: dict[str, Any] = {
                    "simplex_index": simplex_index,
                    "simplex": list(simplex),
                    "coefficient": coefficient_float,
                }
                if labels is not None:
                    item["labels"] = list(labels[simplex_index])
                coefficients.append(item)
            features.append(
                {
                    "eigenvalue": float(values_np[eigen_index]),
                    "simplex_coefficients": coefficients,
                }
            )
        return {
            "dimension": dim,
            "filtration_a": float(a),
            "filtration_b": float(b),
            "betti": authoritative if authoritative is not None else len(features),
            "spectral_nullity": numerical_nullity,
            "returned_features": len(features),
            "features_complete": authoritative is None or len(features) == authoritative,
            "calculation_status": calculation_status,
            "zero_tolerance": tolerance,
            "features": features,
        }

    def nonzero_spectra(
        self,
        dim: int,
        a: float,
        b: float,
        PH_basis=None,
        use_dummy_harmonic_basis: bool = True,
    ):
        """Compute only the nonzero eigenvalues of L^{dim}(a,b).

        Parameters
        ----------
        dim, a, b : int, float, float
            Dimension and filtration values.
        PH_basis : np.ndarray or torch.Tensor, optional
            Basis for the null space of the Laplacian (e.g. from persistent
            homology). If given, the Laplacian is projected onto the
            orthogonal complement of this basis.
        use_dummy_harmonic_basis : bool
            If True and ``PH_basis`` is None, compute the null space of the
            Laplacian directly and project onto its orthogonal complement.

        Returns
        -------
        list[float]
            Nonzero eigenvalues, sorted ascending.
        """

        L = self.get_L(dim, a, b)
        if L.numel() == 0:
            return []

        L_np = L.cpu().numpy()

        if PH_basis is not None:
            basis = np.atleast_2d(PH_basis)
            # Project onto orthogonal complement of basis columns
            Q, _ = np.linalg.qr(basis)
            P = np.eye(L_np.shape[0]) - Q @ Q.T
            L_proj = P @ L_np @ P.T
        elif use_dummy_harmonic_basis:
            # Compute null space via SVD of L
            u, s, vh = np.linalg.svd(L_np)
            tol = 1e-8
            rank = int(np.sum(s > tol))
            null_dim = L_np.shape[0] - rank
            if null_dim == 0:
                L_proj = L_np
            else:
                Q = u[:, rank:]
                P = np.eye(L_np.shape[0]) - Q @ Q.T
                L_proj = P @ L_np @ P.T
        else:
            L_proj = L_np

        # Eigenvalues of projected matrix
        eigs = np.linalg.eigvalsh(L_proj)
        tol = 1e-4
        nonzero = eigs[eigs > tol]
        return nonzero.tolist()

    def store_L(self, dim: int, a: float, b: float, prefix: str) -> None:
        """Save the Laplacian matrix L^{dim}(a,b) to a Matrix Market file.

        File is written to ``{prefix}.mtx``.
        """
        import scipy.io

        L = self.get_L(dim, a, b)
        L_np = L.cpu().numpy()
        scipy.io.mmwrite(f"{prefix}.mtx", scipy.sparse.csr_matrix(L_np))

    def store_spectra(self, spectra_list, file_prefix: str) -> None:
        """Store eigenvalues to text files.

        One file per dimension: ``{file_prefix}_spectra_{dim}.txt``. Each line
        contains the eigenvalues for one (dim, a, b) record, space-separated,
        matching the original PETLS format.
        """
        by_dim: dict[int, list[list[float]]] = {}
        for item in spectra_list:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                dim, eigs = int(item[0]), item[3]
                by_dim.setdefault(dim, []).append(eigs)

        for dim, entries in by_dim.items():
            with open(f"{file_prefix}_spectra_{dim}.txt", "w") as fh:
                for eigs in entries:
                    line = " ".join(str(x) for x in eigs) + "\n"
                    fh.write(line)

    def store_spectra_summary(self, spectra_list, file_prefix: str) -> None:
        """Store eigenvalue summaries to a text file.

        File is written to ``{file_prefix}_spectra_summary.txt``. Each line
        corresponds to a unique (a, b) pair and contains Betti numbers and
        least nonzero eigenvalues for every dimension, matching the original
        PETLS format.
        """
        top_dim = self.top_dim
        items_per_line = 2 + 2 * (top_dim + 1)

        unique_pairs: list[tuple[float, float]] = []
        pair_index: dict[tuple[float, float], int] = {}
        for item in spectra_list:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                a, b = float(item[1]), float(item[2])
                pair = (a, b)
                if pair not in pair_index:
                    pair_index[pair] = len(unique_pairs)
                    unique_pairs.append(pair)

        output_lines = []
        for a, b in unique_pairs:
            line = [0.0] * items_per_line
            line[0] = a
            line[1] = b
            output_lines.append(line)

        for item in spectra_list:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                dim = int(item[0])
                a, b = float(item[1]), float(item[2])
                eigs = item[3]
                betti, lam = self.eigenvalues_summarize(eigs)
                idx = pair_index[(a, b)]
                output_lines[idx][2 + dim] = float(betti)
                output_lines[idx][3 + top_dim + dim] = float(lam)

        with open(f"{file_prefix}_spectra_summary.txt", "w") as fh:
            header = ["a", "b"]
            for d in range(top_dim + 1):
                header.append(f"betti_{d}")
            for d in range(top_dim + 1):
                header.append(f"lambda_{d}")
            fh.write("\t".join(header) + "\n")

            for line in output_lines:
                fh.write("\t".join(str(x) for x in line) + "\n")

    def time_to_csv(self, filename: str) -> None:
        """Store profiling data to a CSV file."""
        self.profile.to_csv(filename)

    def get_all_filtrations(
        self,
        merge_tolerance: float = 1e-10,
        include_vertex_filtrations: bool = True,
    ) -> List[float]:
        """Return sorted filtration values, optionally merging near duplicates.

        Values within ``merge_tolerance`` of the preceding retained value are
        represented by the first value in that cluster.  Vertex births are
        included by default and may be negative for weighted alpha complexes.
        """
        if merge_tolerance < 0:
            raise ValueError("merge_tolerance must be non-negative")
        values: list[float] = []
        if include_vertex_filtrations and self.filtered_boundaries:
            values.extend(self.filtered_boundaries[0].domain_filtrations.cpu().tolist())
        for fbm in self.filtered_boundaries[1:]:
            values.extend(fbm.domain_filtrations.cpu().tolist())
        if not values:
            return []
        ordered = sorted(set(float(value) for value in values))
        merged = [ordered[0]]
        for value in ordered[1:]:
            if value - merged[-1] > merge_tolerance:
                merged.append(value)
        return merged

    def filtration_list_to_spectra_request(
        self, filtrations: List[float], dims: List[int]
    ) -> List[Tuple[int, float, float]]:
        """Generate (dim, a, b) for successive filtrations."""
        if not filtrations:
            return []
        requests = []
        for i in range(len(filtrations) - 1):
            a, b = filtrations[i], filtrations[i + 1]
            for dim in dims:
                requests.append((dim, a, b))
        # Final (a, a) case
        a = filtrations[-1]
        for dim in dims:
            requests.append((dim, a, a))
        return requests

    def filtration_list_to_spectra_request_allpairs(
        self, filtrations: List[float], dims: List[int]
    ) -> List[Tuple[int, float, float]]:
        """Generate (dim, a, b) for all filtration pairs with b >= a."""
        requests = []
        for i in range(len(filtrations)):
            for j in range(i, len(filtrations)):
                a, b = filtrations[i], filtrations[j]
                for dim in dims:
                    requests.append((dim, a, b))
        return requests

    def print_boundaries(self) -> None:
        """Print boundary matrices and their filtrations for debugging."""
        for i, fbm in enumerate(self.filtered_boundaries):
            print(f"\n--- d_{i} ---")
            print(fbm)
            print(f"Domain filtrations: {fbm.domain_filtrations[:10]}...")
            print(f"Range filtrations:  {fbm.range_filtrations[:10]}...")
