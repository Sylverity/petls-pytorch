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
  construction.
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

Use `set_eigs_algorithm("eigvalsh")` for a complete dense spectrum or
`set_eigs_algorithm("sparse", num_eigenvalues=...)` for a partial ordinary
spectrum. Sparse orders `SM`, `SA`, `LM`, `LA`, and `BE` are supported.
`ordinary_spectrum(dim, scale, num_eigenvalues)` provides a bounded sparse solve
without constructing the full dense operator.

Gudhi-backed ordinary summaries use block LOBPCG when repeated zero modes need
more reliable nullspace recovery. Residuals, orthogonality, and numerical
nullity are audited before a spectral gap is reported. Certified summaries
expose `spectrum_solver`, `spectrum_certified`, and
`spectrum_max_normalized_residual`; incomplete recovery leaves
`least_nonzero_eigenvalue` unset and records an explicit status.

Persistent `a < b` Schur complements can become dense. Allocation guards count
both the final matrix at `a` and the larger intermediate at `b`. The flipped
top-dimensional optimization is restricted to complete `eigvalsh` solves where
the algebraic top boundary has no higher-dimensional up term. Partial spectra
use the ordinary sparse path so kernel dimensions remain correct.

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
