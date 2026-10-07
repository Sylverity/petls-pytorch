# PETLS-PyTorch technical notes

This page collects numerical and implementation details kept out of the main
README. It is intended for users selecting filtration, solver, precision, and
allocation settings for larger workloads.

## Filtration semantics

### Alpha complexes

`Alpha` accepts Gudhi power weights directly. Weights must be finite, and point
and weight shapes are validated. Omit `weights` for an ordinary Alpha complex.

```python
alpha = petls_pytorch.Alpha(
    points=positions,
    weights=power_weights,
    max_dim=3,
    precision="safe",
    max_alpha_square=maximum_scale,
    point_labels=labels,
    device="cpu",
    dtype=torch.float64,
)

scales = alpha.get_all_filtrations(
    merge_tolerance=1e-10,
    include_vertex_filtrations=True,
)
```

Weighted vertex births, including negative filtration values, are retained.
`max_alpha_square` limits Gudhi construction. Filtration enumeration includes
vertex births and merges near-duplicate scales by default.

### Rips complexes

For `Rips`, `max_dim` is the highest dimension reported through the public API.
One additional simplex dimension may be retained internally so the
top-dimensional Laplacian includes its up term.

### Directed-flag files

`dFlag` reads weighted directed graphs from `.flag` files:

```text
dim 0
0.0 0.0 0.0 0.0
dim 1
0 1 1.25
1 2 2.50
0 2 3.00
```

The `dim 0` line contains one vertex weight per vertex. Each `dim 1` row is
`source target weight`, using zero-based indices. A listed edge exists even if
its weight is zero or negative; a missing edge remains absent. Self-loops,
duplicate rows, and non-finite weights are rejected. A simplex filtration is
the maximum of its vertex and directed-edge weights, so every face appears no
later than its cofaces.

## Topology and localization

Gudhi-backed complexes retain `simplex_tree`, `simplices_by_dimension`,
`simplex_filtrations`, and `simplex_to_index`. Optional `point_labels` follow
the same stable simplex order. This makes Laplacian coordinates and harmonic
features traceable to their geometry.

Gudhi persistence is the authoritative homology source when available.
`topology_summary()` reports Betti numbers separately from numerical nullity,
spectral gaps, tolerances, matrix sizes, and calculation status. Queries above
`top_dim` return a zero Betti number and an empty spectrum.

Construction takes a snapshot of the supplied simplex tree. Replacing boundaries
with `set_boundaries_filtrations()` replaces the complex and clears its previous
simplex mappings, labels, persistence state, and profile. Subsequent summaries
use the new matrices; invalid replacement data leaves the original object intact.

Gudhi homology uses coefficients in the field of integers modulo 11. Laplacian
nullity is computed numerically over the reals; the two quantities are reported
separately because their coefficient fields differ.

`positive_spectrum()`, `topology_summary()`, and localization results expose
`homology_coeff_field` (11), `spectral_coeff_field` (`"real"`), and
`homology_spectral_agreement`. Agreement is `True` or `False` when the complete
real numerical nullity is available, and `None` when it is not. A disagreement
can arise from torsion, coefficient-dependent persistent maps, or numerical
tolerances; it does not by itself invalidate real eigenpairs. Complete solves
select positive modes with a zero tolerance based on the complete real spectrum.
Range solves determine their working subspace from the operator, without using
the modular Betti count as the real rank or number of eigenvalues to skip.

### Basis-invariant harmonic participation

`harmonic_features()` returns an orthonormal numerical real kernel basis. Its
individual vectors can change sign or rotate within a repeated zero eigenspace.
They are not canonical persistence-interval representatives. For localization
of the whole harmonic space, use:

```python
participation = alpha.harmonic_participation(dim=1, a=0.0, b=0.0)
if participation["participation_complete"]:
    simplices = participation["simplex_participation"]
    points = participation["point_participation"]
```

For an orthonormal harmonic basis `H`, the simplex scores are
`diag(H @ H.T)`, computed as row sums of squares without allocating the dense
projector. A rotation `H @ Q` with orthogonal `Q` leaves these scores unchanged.
Scores are not normalized to probabilities: their sum is the number of recovered
harmonic directions. Each simplex's score is divided equally among its vertices
for `point_participation`, preserving the total. Simplex labels and point labels
(for example, molecule identifiers) are included when available. Repeated labels
remain separate point records and can be summed by the caller.

`max_features` limits the returned harmonic directions. Oversized ordinary
localization defaults to at most ten directions; persistent localization retains
its dense allocation guard. If the full numerical kernel is not recovered,
`participation_complete=False` and `participation_scope="computed_subspace"`.
Such scores are invariant only within that selected subspace, whose selection
can change between solves. They must not be compared as full harmonic-space
invariants. `spectral_nullity` is `None` when the total numerical kernel dimension
is unknown; `recovered_nullity` records the zero modes found before the output
cap. `features_complete` refers to the real kernel, independently of modular
Betti counts. The coefficient display cutoff in `harmonic_features()` is not
applied to participation scores.

These scores localize the harmonic space collectively, not individual holes.
Across frames or scales, coordinate correspondence is still required; basis
invariance alone is not a stability or feature-tracking theorem.

Ordinary oversized localization can use the sparse path and limit automatic
representatives with `max_features`. Persistent localization requires a dense
Schur-complement calculation and observes the configured allocation guard.

## Devices, precision, and allocation

- `device` defaults to CPU. Pass `"cuda"` explicitly or `"auto"` to use an
  available GPU. Settings are object-local.
- `dtype` accepts `torch.float32` or `torch.float64`; weighted Alpha defaults to
  `float64`.
- `zero_atol` and `zero_rtol` define the scale-aware zero test
  `abs(λ) <= zero_atol + zero_rtol * max(abs(spectrum))`.
- `max_matrix_rows` and `max_matrix_bytes` guard dense allocations before
  construction. The byte limit includes rectangular boundary matrices and
  bounds the largest individual dense construction matrix, including a possible
  dense fallback; it is not a bound on total process memory or solver workspace.
- `on_oversize` accepts `"raise"` or `"homology_only"` for oversized persistent
  requests.

```python
estimate = alpha.estimate_laplacian(dim=1, a=0.0, b=0.0)

guarded = petls_pytorch.Alpha(
    points=positions,
    max_matrix_rows=12_000,
    max_matrix_bytes=4_000_000_000,
    on_oversize="homology_only",
)
```

## Solver selection

`nonzero_spectra()` returns the complete positive spectrum, preserving
multiplicity and using the same zero tolerance as the spectral summaries. It
filters the full spectrum directly without constructing a dummy harmonic basis.
A supplied `PH_basis` is checked for harmonicity and cannot remove positive modes.

Scalar `spectra(dim, a, b)` calls return eigenvalues. Batch calls with
`request_list`, or automatic calls without arguments, always return records
`(dim, a, b, eigenvalues)`, including batches of zero or one request. These records
can be passed directly to the spectrum export methods.

Use `set_eigs_algorithm("eigvalsh")` for a complete dense spectrum or
`set_eigs_algorithm("sparse", num_eigenvalues=...)` for a partial ordinary
spectrum. Sparse orders `SM`, `SA`, `LM`, `LA`, and `BE` are supported.
`ordinary_spectrum(dim, scale, num_eigenvalues)` provides a bounded sparse solve
without constructing the full dense operator.

Gudhi-backed `topology_summary()` defaults to `spectral_mode="partial"`.
`positive_spectrum(dim, a, b, positive_modes=8, max_iterations=1000,
relative_tolerance=1e-3)` exposes the same implementation. Sparse operator
applications, Rayleigh–Ritz iterations, and residual checks run on the requested
device in float64; CPU sparse geometry/indexing is still used during construction.
Small problems use a complete device solve internally and return the requested
leading modes. Larger problems use a block iteration through public Torch APIs.

Large partial solves restrict the search to `range(L)`, which is orthogonal to
`ker(L)` for a symmetric Laplacian. They carry preimages through every basis
transformation and explicitly reconstruct `X = L Y`; initializing a random
block in the range alone would allow roundoff to leak back into the kernel.
The working block has at most twice the requested positive-mode count,
independent of the Betti multiplicity. Ill-conditioned preimage directions are
dropped; every 100 iterations, and when search directions are exhausted,
device CG solves consistent
`L Y = X` equations to refresh bounded preimages. These solves do not construct
a nullspace basis. All operator products, block operations, and CG iterations
remain on the selected device. This is not shift-invert or randomized spectral
compression.

For persistent queries the operator is a sparse base minus `C.T @ pinv(D) @ C`.
Only the active eliminated block `D` is dense; a Cholesky inverse is used when
positive definite, otherwise a Hermitian pseudoinverse. The complete persistent
Laplacian is never materialized on the large-problem path.

`relative_tolerance` now controls stopping as well as acceptance: all requested
positive modes must meet their relative residual threshold, or the iteration
budget ends. Final residuals are evaluated again against the original operator.
Each mode records value, absolute residual, relative residual, and quality.
Strict residuals, orthogonality, and range reconstruction are required for
`certified`; `approximate` retains a mode meeting the relative residual check.
These labels are numerical checks, not eigenvalue enclosures or physical
validation. A difficult later mode does not invalidate a usable first gap.

For range solves `spectral_nullity` is `None` and `smallest_eigenvalues` contains
only the computed positive Ritz values. They do not pretend to have numerically
recovered zero modes. GUDHI Betti counts are reported independently. Small
complete solves and explicit full solves retain their numerical-nullity checks.
Returned diagnostics include `nullspace_handling`, `working_block_size`,
`range_reconstruction_error`, construction/solve time, device, iterations, and
Schur block size. Oversized requests raise `LaplacianSizeError`, or summaries
with `on_oversize="homology_only"` retain homology without a spectrum.

The explicit full path (`spectral_mode="full"`, `spectra()`, or localization)
keeps its earlier semantics. Its persistent Schur complements can become dense;
allocation guards count both final and intermediate matrices. Its legacy sparse
ordinary solver uses SciPy on CPU. These explicit APIs do not select the new
partial path through `set_eigs_algorithm()`.

## Benchmarks

Install the benchmark dependencies and run the standard workload on CPU or
CUDA:

```bash
uv run --extra benchmark python -m benchmark --preset standard \
  --package petls-pytorch --algorithm eigvalsh --device cpu

uv run --extra benchmark python -m benchmark --preset standard \
  --package petls-pytorch --algorithm eigvalsh --device cuda --dtype float32
```

Larger GPU stress run:

```bash
uv run --extra benchmark python -m benchmark --preset stress \
  --package petls-pytorch --algorithm eigvalsh --device cuda
```

Reference PETLS, when it is available on the platform:

```bash
uv run --extra benchmark --with petls python -m benchmark --preset standard \
  --package petls --algorithm selfadjoint
```

Custom workload:

```bash
uv run --extra benchmark python -m benchmark \
  --dataset torus --n_points 2000 --complex alpha --max_dim 3 \
  --package petls-pytorch --algorithm eigvalsh --device cuda --dtype float32
```

Use `--output_dir benchmark-results/<run-name>` to keep named results. The CLI
defaults to `float32` for benchmark continuity; pass `--dtype float64` for
higher-precision weighted-Alpha measurements. Check `nvidia-smi` and
`torch.cuda.is_available()` before interpreting a GPU run.

## Development checks

The full development and parity-test commands live in
[CONTRIBUTING.md](../CONTRIBUTING.md).

To benchmark the new ordinary and persistent partial solver with the existing
harness, select `--algorithm partial`, for example:

```bash
uv run --extra benchmark python -m benchmark --preset quick \
  --package petls-pytorch --algorithm partial --device cuda --dtype float64
```

Saved results include construction time, solve time, complete request wall time,
per-mode residual/quality records, iterations, and the actual device. The row
cap applies to the dense eliminated block for this algorithm. Keep `eigvalsh`
as the full-spectrum reference; comparisons must match filtration pairs and
positive eigenvalues, not the length of a partial spectrum. Timing checks should
run on controlled hardware separately from ordinary correctness CI.
