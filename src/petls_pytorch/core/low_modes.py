"""Device-resident low modes of sparse ordinary and persistent Laplacians.

Geometry/boundary indexing may use CPU sparse matrices. Operator applications,
Schur solves, Rayleigh--Ritz steps and residuals stay on the requested device.
No dense full-size persistent Laplacian or projector is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import torch


def _tensor(matrix, device, dtype):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.as_tensor(np.stack((coo.row, coo.col)), device=device),
        torch.as_tensor(coo.data, device=device, dtype=dtype),
        coo.shape,
    ).coalesce()


@dataclass
class LaplacianOperator:
    base: torch.Tensor
    cross: torch.Tensor | None
    inverse: torch.Tensor | None
    scale: float
    schur_rows: int

    def __post_init__(self):
        self.cross_t = self.cross.transpose(0, 1).coalesce() if self.cross is not None else None

    def __call__(self, vectors):
        result = torch.sparse.mm(self.base, vectors)
        if self.cross is not None:
            rhs = torch.sparse.mm(self.cross, vectors)
            result = result - torch.sparse.mm(self.cross_t, self.inverse @ rhs)
        return result


def build_operator(complex_, dim, a, b):
    """Construct sparse base minus exact small Schur correction."""
    from petls_pytorch.core.complex import LaplacianSizeError

    rows = complex_._laplacian_rows(dim, a)
    base = sp.csr_matrix((rows, rows), dtype=np.float64)
    if rows and dim > 0:
        boundary = complex_._to_scipy_sparse(
            complex_.filtered_boundaries[dim].submatrix_at_filtration(a)
        )
        base = base + boundary.T @ boundary
    cross = inverse = None
    schur_rows = 0
    if rows and dim + 1 < len(complex_.filtered_boundaries):
        boundary = complex_._to_scipy_sparse(
            complex_.filtered_boundaries[dim + 1].submatrix_at_filtration(b)
        )
        gram = (boundary @ boundary.T).tocsr()
        base = base + gram[:rows, :rows]
        if gram.shape[0] > rows:
            complement = gram[rows:, rows:].tocsr()
            active = complement.diagonal() > 0
            complement = complement[active][:, active]
            schur_rows = complement.shape[0]
            # Budget includes simultaneous factorization/workspace allocations.
            needed = schur_rows * schur_rows * 8 * 8
            if (complex_.max_matrix_rows is not None and schur_rows > complex_.max_matrix_rows) or (
                complex_.max_matrix_bytes is not None and needed > complex_.max_matrix_bytes
            ):
                raise LaplacianSizeError("Persistent Schur block exceeds the allocation budget")
            if schur_rows:
                cross = _tensor(gram[rows:, :rows][active], complex_.device, torch.float64)
                dense = torch.as_tensor(
                    complement.toarray(), device=complex_.device, dtype=torch.float64
                )
                factor, info = torch.linalg.cholesky_ex(dense)
                # The inverse is only of the small Schur block, never the full L.
                # A singular Gram block can acquire a tiny positive Cholesky
                # pivot through roundoff, especially on CUDA. Inverting that
                # pivot destroys the Schur correction despite info == 0.
                pivot_floor = (
                    torch.finfo(dense.dtype).eps * schur_rows * dense.diagonal().abs().max()
                )
                if int(info.item()) == 0 and bool((factor.diagonal().square() > pivot_floor).all()):
                    inverse = torch.cholesky_inverse(factor)
                else:
                    inverse = torch.linalg.pinv(dense, hermitian=True)
    base = base.tocsr()
    scale = max(1.0, float(np.asarray(abs(base).sum(axis=1)).max(initial=0)))
    return LaplacianOperator(
        _tensor(base, complex_.device, torch.float64).to_sparse_csr(),
        cross,
        inverse,
        scale,
        schur_rows,
    )


def _range_orthogonalize(preimages, against_preimages, against, operator):
    """Orthonormalize LY while carrying Y through every basis transformation.

    Merely initializing in range(L) lets floating-point kernel contamination
    grow during iteration. Reconstructing LY after projections prevents that
    drift without constructing or solving for a harmonic basis.
    """
    vectors = operator(preimages)
    for _ in range(2):
        coefficients = against.T @ vectors
        preimages = preimages - against_preimages @ coefficients
        vectors = vectors - against @ coefficients
    norms = torch.linalg.vector_norm(vectors, dim=0)
    # A preimage is nonunique up to ker(L). Near-dependent residual directions
    # can accumulate enormous invisible kernel components and lose digits in
    # LY. Discard those directions before normalization amplifies the error.
    preimage_norms = torch.linalg.vector_norm(preimages, dim=0)
    keep = (norms > 1e-12) & (preimage_norms * operator.scale <= 1e8 * norms)
    preimages = preimages[:, keep] / norms[keep]
    vectors = vectors[:, keep] / norms[keep]
    for _ in range(2):
        gram = vectors.T @ vectors
        values, rotation = torch.linalg.eigh((gram + gram.T) * 0.5)
        keep = values > 1e-10
        transform = rotation[:, keep] / torch.sqrt(values[keep])
        preimages, vectors = preimages @ transform, vectors @ transform
    vectors = operator(preimages)
    keep = torch.linalg.vector_norm(preimages, dim=0) * operator.scale <= 1e8
    return preimages[:, keep], vectors[:, keep]


def _range_preimages(operator, vectors, values):
    """Refresh bounded preimages by consistent, device-resident block RHS CG.

    Each column uses independent CG coefficients. A near-eigenpair starts at
    x/lambda, and the iterates stay in range(L) rather than accumulating an
    arbitrary kernel component. This is a linear solve, not a kernel solve.
    """
    y = vectors / values.clamp_min(torch.finfo(vectors.dtype).eps * operator.scale)
    residual = vectors - operator(y)
    direction = residual.clone()
    squared = (residual * residual).sum(dim=0)
    for _ in range(2000):
        active = squared > 1e-16
        if not bool(active.any()):
            break
        applied = operator(direction)
        denominator = (direction * applied).sum(dim=0)
        valid = active & (denominator > 0)
        step = torch.where(valid, squared / denominator.clamp_min(1e-300), 0)
        y = y + direction * step
        residual = residual - applied * step
        updated = (residual * residual).sum(dim=0)
        ratio = torch.where(valid, updated / squared.clamp_min(1e-300), 0)
        direction = residual + direction * ratio
        squared = updated
    return y


def block_lowest(operator, count, max_iterations, relative_tolerance):
    """Leading *positive* Ritz pairs restricted to range(L), on the device.

    For symmetric L, range(L) is orthogonal to ker(L). Carrying preimages Y
    makes X=LY explicit at every iteration. A small oversampled block explores
    positive modes, independent of Betti multiplicity. No kernel eigenpairs,
    inverse, dense projector, or private Torch hooks are needed.
    """
    n = operator.base.shape[0]
    width = min(n, 2 * count)
    generator = torch.Generator(device=operator.base.device).manual_seed(0)
    y = torch.randn(n, width, generator=generator, device=operator.base.device, dtype=torch.float64)
    y, x = _range_orthogonalize(y, y[:, :0], y[:, :0], operator)
    previous = y[:, :0]
    for iteration in range(1, max_iterations + 1):
        if not x.shape[1]:
            break
        x = operator(y)
        ax = operator(x)
        h = x.T @ ax
        values, rotation = torch.linalg.eigh((h + h.T) * 0.5)
        y, x, ax = y @ rotation, x @ rotation, ax @ rotation
        if iteration % 100 == 0:
            y = _range_preimages(operator, x, values)
            y, x = _range_orthogonalize(y, y[:, :0], y[:, :0], operator)
            previous = y[:, :0]
            continue
        residuals = torch.linalg.vector_norm(ax - x * values, dim=0)
        if len(values) and bool(
            (
                (values[:count] > 0) & (residuals[:count] <= relative_tolerance * values[:count])
            ).all()
        ):
            break
        # L(X - Y lambda) is the eigenpair residual, in range(L).
        directions_y, directions = _range_orthogonalize(
            torch.cat((x - y * values, previous), dim=1), y, x, operator
        )
        if not directions.shape[1]:
            y = _range_preimages(operator, x, values)
            y, x = _range_orthogonalize(y, y[:, :0], y[:, :0], operator)
            previous = y[:, :0]
            continue
        trial_y = torch.cat((y, directions_y), dim=1)
        trial = torch.cat((x, directions), dim=1)
        applied = torch.cat((ax, operator(directions)), dim=1)
        h = trial.T @ applied
        _, rotation = torch.linalg.eigh((h + h.T) * 0.5)
        rotation = rotation[:, : y.shape[1]]
        previous = directions_y @ rotation[y.shape[1] :]
        y = trial_y @ rotation
    # Reconstruct, reorthogonalize and audit the actual operator, not cached AX.
    y, x = _range_orthogonalize(y, y[:, :0], y[:, :0], operator)
    ax = operator(x)
    h = x.T @ ax
    values, rotation = torch.linalg.eigh((h + h.T) * 0.5)
    rotation = rotation[:, :count]
    x, y = x @ rotation, y @ rotation
    range_error = float(torch.linalg.matrix_norm(x - operator(y)))
    return values[:count], x, iteration, range_error


def positive_modes(complex_, dim, a, b, count=8, max_iterations=1000, relative_tolerance=1e-3):
    """Return leading modes with authoritative homology and per-mode quality."""
    from petls_pytorch.core.complex import LaplacianSizeError

    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 128:
        raise ValueError("positive_modes must be an integer between 1 and 128")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must be between zero and one")
    if b < a or dim < 0:
        raise ValueError("Require a non-negative dimension and b >= a")
    if complex_.simplex_tree is None:
        raise ValueError("Partial positive spectra require authoritative simplex-tree homology")
    betti = complex_.persistent_betti(dim, a, b)
    rows = complex_._laplacian_rows(dim, a)
    # Modular homology is reported independently; it is not the real nullity.
    # Range orthogonalization discovers the available numerical search space.
    requested = min(count, rows)
    width = min(rows, 2 * requested)
    budget = complex_.max_matrix_bytes
    started = perf_counter()
    op = build_operator(complex_, dim, a, b)
    if op.base.device.type == "cuda":
        torch.cuda.synchronize(op.base.device)
    construction_s = perf_counter() - started
    started = perf_counter()
    dense = rows <= max(64, 3 * width)
    range_error = 0.0
    nullity = None
    solver = "torch_range_partial"
    if op.base._nnz() == 0:
        values = torch.empty(0, device=complex_.device, dtype=torch.float64)
        vectors = torch.empty(rows, 0, device=complex_.device, dtype=torch.float64)
        raw_values = [0.0] * min(rows, 256)
        nullity, iterations, solver = rows, 0, "exact_zero"
        zero_tolerance = complex_._zero_tolerance(np.asarray(raw_values))
    elif dense:
        if budget is not None and rows * rows * 8 * 8 > budget:
            raise LaplacianSizeError("Small complete solve exceeds allocation budget")
        matrix = op(torch.eye(rows, device=complex_.device, dtype=torch.float64))
        all_values, all_vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
        # Use the complete real spectrum for a count-independent zero test.
        zero_tolerance = complex_._zero_tolerance(all_values.cpu().numpy())
        nullity = int((all_values.abs() <= zero_tolerance).sum())
        positive = torch.nonzero(all_values > zero_tolerance).flatten()[:requested]
        values, vectors = all_values[positive], all_vectors[:, positive]
        raw_values = all_values[: nullity + len(positive)].cpu().tolist()
        iterations, solver = 1, "torch_small_complete"
    else:
        if budget is not None and rows * max(1, width) * 8 * 16 > budget:
            raise LaplacianSizeError("Partial eigensolver work vectors exceed allocation budget")
        values, vectors, iterations, range_error = block_lowest(
            op, requested, max_iterations, relative_tolerance
        )
        raw_values = values.cpu().tolist()
        zero_tolerance = complex_._zero_tolerance(np.asarray(raw_values))
    residual = torch.linalg.vector_norm(op(vectors) - vectors * values, dim=0)
    orthogonality = (
        float(
            torch.linalg.matrix_norm(
                vectors.T @ vectors
                - torch.eye(len(values), device=complex_.device, dtype=torch.float64)
            )
        )
        if len(values)
        else 0.0
    )
    if op.base.device.type == "cuda":
        torch.cuda.synchronize(op.base.device)
    elapsed = perf_counter() - started
    vals, errors = values.detach().cpu().numpy(), residual.detach().cpu().numpy()
    strict = max(1e-8, complex_.zero_atol, complex_.zero_rtol * 0.1)
    valid_basis = orthogonality <= 1e-6 and np.isfinite(range_error) and range_error <= 1e-8
    if not np.all(np.isfinite(vals)) or not np.all(np.isfinite(errors)):
        raise RuntimeError("Partial eigensolver produced non-finite results")
    modes = []
    for value, error in zip(vals, errors):
        relative = float(error / max(abs(value), np.finfo(float).tiny))
        quality = "unavailable"
        if valid_basis and value > zero_tolerance:
            if error / op.scale <= strict:
                quality = "certified"
            elif relative <= relative_tolerance:
                quality = "approximate"
        modes.append(
            {
                "value": float(value),
                "residual": float(error),
                "relative_residual": relative,
                "quality": quality,
            }
        )
    usable = bool(modes and modes[0]["quality"] != "unavailable")
    return {
        "betti": betti,
        "homology_coeff_field": 11,
        "spectral_coeff_field": "real",
        "homology_spectral_agreement": nullity == betti if nullity is not None else None,
        "spectral_nullity": nullity,
        "least_nonzero_eigenvalue": modes[0]["value"] if usable else None,
        "positive_modes": modes,
        "smallest_eigenvalues": raw_values,
        "zero_tolerance": zero_tolerance,
        "matrix_rows": rows,
        "spectrum_solver": solver,
        "spectrum_device": str(op.base.device),
        "spectrum_certified": bool(usable and modes[0]["quality"] == "certified"),
        "spectrum_max_normalized_residual": float(errors.max(initial=0) / op.scale),
        "calculation_status": (
            "partial_spectrum"
            if usable
            else "null_modes_only"
            if nullity == rows
            else "partial_spectrum_unconverged"
        ),
        "iterations": iterations,
        "construction_s": construction_s,
        "solve_s": elapsed,
        "schur_rows": op.schur_rows,
        "requested_positive_modes": count,
        "working_block_size": width if solver == "torch_range_partial" else rows,
        "nullspace_handling": "range_restriction"
        if solver == "torch_range_partial"
        else "explicit",
        "range_reconstruction_error": range_error if solver == "torch_range_partial" else None,
    }
