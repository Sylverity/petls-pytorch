"""
Complex — main PETLS class.

This is the PyTorch-native replacement for petls::Complex (C++).
It stores a list of FilteredBoundaryMatrix objects and provides
get_L, get_up, get_down, spectra, and eigenpairs.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from petls_pytorch._config import get_device, get_dtype, get_sparse_dtype
from petls_pytorch.core.filtered_boundary import FilteredBoundaryMatrix
from petls_pytorch.core.profile import Profile


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

    cpp_algorithms_list = ["selfadjoint", "eigensolver", "bdcsvd", "spectra"]

    def __init__(
        self,
        boundaries: Optional[List[torch.Tensor]] = None,
        filtrations: Optional[List[List[float]]] = None,
        device: Optional[torch.device] = None,
        simplex_tree=None,
        eigs_Algorithm: str | Callable = "eigvalsh",
        up_Algorithm: str = "schur",
    ):
        self.device = device if device is not None else get_device()
        self.dtype = get_dtype()
        self._verbose = False
        self._flipped = False
        self._logger = logging.getLogger(__name__)

        self.top_dim: int = 0
        self.filtered_boundaries: List[FilteredBoundaryMatrix] = []
        self.profile = Profile()

        self._eigs_algorithm: str | Callable = "eigvalsh"
        self._up_algorithm: str = "schur"
        self._num_eigenvalues: int = 10
        self._eigenvalue_order: str = "SM"

        if simplex_tree is not None:
            from petls_pytorch.utils.simplex_tree import simplex_tree_boundaries_filtrations

            boundaries, filtrations = simplex_tree_boundaries_filtrations(simplex_tree)

        if boundaries is not None and filtrations is not None:
            self.set_boundaries_filtrations(boundaries, filtrations)
        else:
            # Empty complex — user will set later
            self.filtered_boundaries = []
            self.top_dim = 0

        self.set_eigs_Algorithm(eigs_Algorithm)
        self.set_up_Algorithm(up_Algorithm)

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
        boundaries: List[torch.Tensor],
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

        # d_0 placeholder to align indexing with the original PETLS Complex.
        # For vertex-only complexes, keep the real 0-simplex filtrations so
        # spectra() can report the correct Betti-0 multiplicity.
        if boundaries:
            vertex_filtrations = torch.tensor([0.0], device=self.device, dtype=torch.float64)
        else:
            vertex_filtrations = torch.tensor(
                filtrations[0], device=self.device, dtype=torch.float64
            )
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

    @staticmethod
    def _ensure_sparse_tensor(x) -> torch.Tensor:
        """Convert numpy array, scipy sparse, or torch dense to torch sparse COO."""
        import scipy.sparse

        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                return x.coalesce() if x.layout == torch.sparse_coo else x
            # Dense tensor -> COO
            return x.to_sparse_coo()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to_sparse_coo()
        if scipy.sparse.issparse(x):
            coo = x.tocoo()
            indices = torch.stack(
                [
                    torch.from_numpy(coo.row).long(),
                    torch.from_numpy(coo.col).long(),
                ]
            )
            values = torch.from_numpy(coo.data).to(dtype=get_sparse_dtype())
            return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()
        raise TypeError(f"Cannot convert type {type(x)} to sparse tensor")

    def set_eigs_algorithm(
        self,
        algorithm: str | Callable,
        num_eigenvalues: int = 10,
        eigenvalue_order: str = "SM",
        **kwargs,
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
        self._eigs_algorithm = algorithm
        self._num_eigenvalues = num_eigenvalues
        self._eigenvalue_order = eigenvalue_order

    # Alias matching original PETLS camelCase API
    set_eigs_Algorithm = set_eigs_algorithm

    def set_up_algorithm(self, algorithm: str) -> None:
        """Set up-Laplacian algorithm. Currently only 'schur' is supported."""
        if algorithm != "schur":
            raise ValueError("Only 'schur' up-algorithm is currently supported")
        self._up_algorithm = algorithm

    # Alias matching original PETLS camelCase API
    set_up_Algorithm = set_up_algorithm

    def get_L(self, dim: int, a: float, b: float) -> torch.Tensor:
        """Get persistent Laplacian matrix L^{dim}(a,b) as dense tensor."""
        from petls_pytorch.core.laplacian import get_L

        return get_L(dim, a, b, self.filtered_boundaries, self.top_dim, self.device)

    def get_L_top_dim_flipped(self, a: float) -> torch.Tensor:
        """Get the flipped top-dimension Laplacian B @ B^T.

        When ``flipped=True``, ``spectra()`` uses this matrix for the top
        dimension because the nonzero eigenvalues of ``B @ B^T`` (shape
        m×m) are the same as those of ``B^T @ B`` (shape n×n), but the
        former may be smaller.
        """
        if self.top_dim == 0:
            return torch.empty(0, 0, dtype=self.dtype, device=self.device)
        fbm = self.filtered_boundaries[self.top_dim]
        B = fbm.submatrix_at_filtration(a)
        if B.shape[0] == 0 or B.shape[1] == 0:
            return torch.empty(0, 0, dtype=self.dtype, device=self.device)
        B_dense = B.to_dense().to(dtype=self.dtype)
        return B_dense @ B_dense.T

    def get_up(self, dim: int, a: float, b: float) -> torch.Tensor:
        """Get persistent up-Laplacian."""
        from petls_pytorch.core.laplacian import get_up

        if dim >= self.top_dim:
            # No higher-dimensional simplices → zero matrix sized to dim-simplices at a
            if dim == 0 and len(self.filtered_boundaries) == 1:
                # Edge case: no 1-simplices at all
                n = self.filtered_boundaries[0].index_of_filtration(True, a) + 1
                return torch.zeros(n, n, dtype=get_dtype(), device=self.device)
            fbm = self.filtered_boundaries[dim]
            n = fbm.index_of_filtration(True, a) + 1
            return torch.zeros(n, n, dtype=get_dtype(), device=self.device)
        return get_up(self.filtered_boundaries[dim + 1], a, b, self.device)

    def get_down(self, dim: int, a: float) -> torch.Tensor:
        """Get persistent down-Laplacian."""
        from petls_pytorch.core.laplacian import get_down

        return get_down(self.filtered_boundaries[dim], a, self.device)

    def _solve_eigs(self, L: torch.Tensor) -> torch.Tensor:
        """Dispatch to eigenvalue solver."""
        from petls_pytorch.core.eigenvalues import solve_eigenvalues
        from petls_pytorch import sparse_wrapper

        algorithm = self._eigs_algorithm
        if algorithm == "sparse":

            def sparse_algorithm(matrix: torch.Tensor) -> np.ndarray:
                return sparse_wrapper(
                    matrix.cpu().numpy(),
                    num_eigs=self._num_eigenvalues,
                    which_eigs=self._eigenvalue_order,
                )

            return torch.asarray(sparse_algorithm(L), dtype=L.dtype, device=L.device)
        return solve_eigenvalues(L, algorithm=algorithm)

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

            # Flipped top-dimension optimization
            if d == self.top_dim and self.flipped:
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

            self.profile.stop_all()

            eigs_list = eigs.cpu().tolist() if isinstance(eigs, torch.Tensor) else list(eigs)

            # Zero-pad flipped top-dimension eigenvalues to match true Laplacian size
            if d == self.top_dim and self.flipped:
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
            Multiple queries. May also be passed as the only positional argument.
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
        if (
            not single_query
            and dim is not None
            and a is None
            and b is None
            and request_list is None
        ):
            request_list = dim
            dim = None

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

    def eigenvalues_summarize(self, eigenvalues: list[float] | torch.Tensor) -> Tuple[int, float]:
        """Compute (betti, least_nonzero) from a list of eigenvalues."""

        if isinstance(eigenvalues, torch.Tensor):
            eigenvalues = eigenvalues.cpu().numpy()
        else:
            eigenvalues = np.array(eigenvalues)

        tol = 1e-4
        betti = int(np.sum(eigenvalues < tol))
        nonzeros = eigenvalues[eigenvalues > tol]
        least = float(nonzeros.min()) if len(nonzeros) > 0 else 0.0
        return betti, least

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
        by_dim = {}
        for item in spectra_list:
            if isinstance(item, (list, tuple)) and len(item) >= 4:
                dim, eigs = item[0], item[3]
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

        unique_pairs = []
        pair_index = {}
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

    def get_all_filtrations(self) -> List[float]:
        """Return sorted list of all unique filtration values in the complex."""
        filts = set()
        # d_0 domain
        filts.update(self.filtered_boundaries[0].domain_filtrations.cpu().tolist())
        # All other domain filtrations
        for fbm in self.filtered_boundaries[1:]:
            filts.update(fbm.domain_filtrations.cpu().tolist())
        return sorted(filts)

    def filtration_list_to_spectra_request(
        self, filtrations: List[float], dims: List[int]
    ) -> List[Tuple[int, float, float]]:
        """Generate (dim, a, b) for successive filtrations."""
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
