# petls-pytorch

[![PyPI](https://img.shields.io/pypi/v/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![CI](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![License](https://img.shields.io/pypi/l/petls-pytorch.svg)](LICENSE)

Persistent topological Laplacians in PyTorch for weighted Alpha, Rips, directed
flag, and cellular-sheaf complexes.

Status: beta

PETLS-PyTorch brings persistent-Laplacian calculations into ordinary Python and
PyTorch workflows while retaining the geometric and topological information
needed to interpret the results. Its main capabilities include:

- weighted and unweighted Gudhi Alpha complexes with general power weights;
- Gudhi-backed Betti numbers and persistence intervals alongside
  persistent-Laplacian spectra;
- stable simplex identities and simplex-mapped harmonic representatives;
- explicit CPU, CUDA, dtype, and numerical-tolerance controls;
- guarded dense calculations and sparse lowest-spectrum analysis for larger
  ordinary Hodge Laplacians; and
- Rips, directed flag, and cellular-sheaf complex support.

The original C++/pybind11 PETLS implementation remains a numerical reference
for shared persistent-Laplacian algorithms. The Python API itself is focused on
PyTorch conventions and does not reproduce compatibility-only C++ names.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Alpha Complexes and Power Weights](#alpha-complexes-and-power-weights)
- [Topology, Persistence, and Localization](#topology-persistence-and-localization)
- [Precision, Devices, and Scaling](#precision-devices-and-scaling)
- [Relationship to PETLS](#relationship-to-petls)
- [Supported API](#supported-api)
- [Directed Flag Files](#directed-flag-files)
- [Benchmark Notes](#benchmark-notes)
- [Test Suite](#test-suite)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

```bash
pip install petls-pytorch
```

PETLS-PyTorch supports CPython 3.10 through 3.14. For GPU acceleration,
install a CUDA-enabled PyTorch build that matches your system using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).

The runtime dependencies are `torch`, `numpy`, `scipy`, `gudhi`, `pandas`, and
`matplotlib`. Benchmark datasets use the optional `benchmark` extra, which adds
`tadasets` without expanding the runtime install closure.

## Quick Start

```python
import petls_pytorch
import gudhi
import torch

# Weighted Alpha complex. Weights are general Gudhi power weights.
alpha = petls_pytorch.Alpha(
    points=[[0, 0], [1, 0], [1, 1], [0, 1]],
    weights=[0.20, 0.18, 0.20, 0.18],
    max_dim=2,
    max_alpha_square=1.0,
    device="cpu",
    dtype=torch.float64,
)

# Gudhi homology is authoritative; PETLS adds persistent-Laplacian spectra.
summary = alpha.topology_summary(dimensions=(0, 1), a=0.0, b=0.0)
intervals = alpha.persistence_intervals(dim=1)

# Advanced callers can inspect spectra and harmonic representatives directly.
eigenvalues = alpha.spectra(dim=1, a=0.0, b=0.0)
features = alpha.harmonic_features(dim=1, a=0.0, b=0.0)

# Rips complex from distance matrix
rips = petls_pytorch.Rips(distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], max_dim=2)

# Directed flag complex from .flag file
dflag = petls_pytorch.dFlag("graph.flag", max_dim=3)

# Sheaf Laplacian
st = gudhi.SimplexTree()
st.insert([0], filtration=0.0)
st.insert([1], filtration=0.0)
st.insert([0, 1], filtration=1.0)
extra_data = {(0,): 1.0, (1,): 1.0, (0, 1): 1.0}


def restriction(simplex, coface, sst):
    return 1.0


sst = petls_pytorch.sheaf_simplex_tree(st, extra_data, restriction)
psl = petls_pytorch.PersistentSheafLaplacian(sst)
filtrations = psl.get_all_filtrations()
sheaf_eigenvalues = psl.spectra(dim=0, a=filtrations[0], b=filtrations[-1])
```

For `Rips`, `max_dim` is the highest homology/Laplacian dimension reported by
the public API. PETLS may retain one additional simplex dimension internally so
the top-dimensional Laplacian includes the up term needed to distinguish cycles
from boundaries.

For any complex, choose `a <= b` from `get_all_filtrations()`.
`spectra(dim, a, b)` computes the persistent-Laplacian spectrum; at `a == b`
it is the ordinary Hodge spectrum.

## Alpha Complexes and Power Weights

`Alpha` accepts Gudhi power weights directly. A weight is a finite scalar
associated with a point; PETLS does not assign domain-specific meaning to it.
Callers are responsible for deriving weights appropriate to their application.

```python
alpha = petls_pytorch.Alpha(
    points=positions,
    weights=power_weights,
    max_dim=3,
    precision="safe",
    max_alpha_square=maximum_scale,
    point_labels=labels,  # optional; kept separate from topology
    device="cpu",
    dtype=torch.float64,
    zero_atol=1e-8,
    zero_rtol=1e-7,
)
```

The constructor validates point and weight shapes and finite values. Weighted
vertex births, including negative filtration values, are retained. The
`max_alpha_square` cutoff is passed to Gudhi during construction so simplices
beyond the analysis range are never built. Passing no weights retains ordinary
unweighted-alpha behavior.

Filtration enumeration includes real vertex births and merges numerical
near-duplicates by default:

```python
scales = alpha.get_all_filtrations(
    merge_tolerance=1e-10,
    include_vertex_filtrations=True,
)
```

Use `include_vertex_filtrations=False` only when reproducing workflows that
intentionally ignore zero-dimensional birth scales.

## Topology, Persistence, and Localization

Complexes constructed from Gudhi retain `simplex_tree`,
`simplices_by_dimension`, `simplex_filtrations`, and `simplex_to_index`.
Consequently, every Laplacian row and eigenvector coordinate has a stable
simplex identity. If `point_labels` are supplied to `Alpha`, corresponding
label tuples are available in `simplex_labels_by_dimension` and harmonic
feature results.

```python
summary = alpha.topology_summary(dimensions=(0, 1, 2), a=0.0, b=0.0)
intervals = alpha.persistence_intervals(dim=1)
bettis = alpha.betti_numbers_at(scale=0.0)
rank = alpha.persistent_betti(dim=1, birth_scale=0.0, death_scale=0.5)
features = alpha.harmonic_features(dim=1, a=0.0, b=0.0)
```

At `a == b`, `topology_summary()` reports ordinary Betti numbers. At `a < b`,
it reports persistent Betti numbers. Gudhi persistence is the authoritative
homology source when available; persistent-Laplacian nullity, least nonzero
eigenvalues, the effective zero tolerance, smallest eigenvalues, matrix rows,
cost estimates, and per-dimension calculation status are returned separately
for numerical auditing. Requested dimensions above the complex's `top_dim`
return Betti number zero and an empty spectrum.

`harmonic_features()` returns each numerical zero-mode eigenvector as
simplex/coefficient records. Harmonic bases need not be unique, but their
coordinates always follow the retained simplex order. For oversized ordinary
calculations (`a == b`), the sparse path caps automatic localization at ten
representatives and reports `truncated_for_scale`; pass `max_features=` to
request a different explicit limit. Persistent localization (`a < b`) requires
the dense Schur-complement Laplacian and raises `LaplacianSizeError` when either
its output or peak intermediate allocation exceeds the configured guard.

## Precision, Devices, and Scaling

Device and dtype are object-specific. The library default device is CPU;
`device="auto"` opts in to CUDA when available, and `device="cuda"` requests it
explicitly. Constructing one object never changes another object's placement.
`Alpha` defaults to `torch.float64`; callers can select `float32` or
`float64` explicitly on all primary complex variants.

Zero eigenvalues use the scale-aware test
`abs(lambda) <= zero_atol + zero_rtol * max(abs(spectrum))`. Summary results
report the effective tolerance and smallest eigenvalues so borderline modes can
be inspected.

Dense allocations are guarded before construction. Estimates report the final
matrix at filtration `a` and the larger Schur-complement intermediate that may
be required at filtration `b`. The defaults are 12,000 rows and 4 GB for the
largest dense matrix in the calculation, and both are configurable per object:

```python
estimate = alpha.estimate_laplacian(dim=1, a=0.0, b=0.0)

guarded = petls_pytorch.Alpha(
    points=positions,
    weights=power_weights,
    max_matrix_rows=12_000,
    max_matrix_bytes=4_000_000_000,
    on_oversize="homology_only",  # or "raise"
)
```

`ordinary_spectrum(dim, scale, num_eigenvalues)` builds sparse boundary Gram
matrices and calls SciPy's sparse symmetric eigensolver without first creating
a dense Laplacian. The same path is used by `spectra()` for `a == b` after
`set_eigs_algorithm("sparse", ..., eigenvalue_order="SM")`; `SM`, `SA`, `LM`,
`LA`, and `BE` selection is honored consistently.

For an ordinary lowest-spectrum summary backed by a Gudhi simplex tree, PETLS
uses the authoritative Betti number to size a block eigensolve. This is
important when the zero eigenvalue is highly degenerate: scalar ARPACK can
return converged positive eigenpairs without recovering every independent zero
mode. PETLS uses block LOBPCG for feasible repeated kernels, checks normalized
residuals and eigenvector orthogonality, and reports a spectral gap only when
the numerical nullity agrees with Gudhi homology. The summary exposes
`spectrum_solver`, `spectrum_certified`, and
`spectrum_max_normalized_residual` for auditing. If recovery is incomplete or
cannot be certified, `least_nonzero_eigenvalue` is `None` and
`calculation_status` is `spectral_nullity_mismatch`,
`sparse_spectrum_unverified`, or `sparse_null_modes_only` as appropriate.

Persistent Schur complements can become dense, so oversized `a < b` requests
either raise `LaplacianSizeError` or return Gudhi homology-only status through
`topology_summary()`. This distinction is intentional: large systems retain
authoritative homology and ordinary sparse Hodge spectra without pretending
that full persistent spectra are sparse.

## Relationship to PETLS

`petls-pytorch` is an independent PyTorch-native implementation of the PETLS
methods, with attribution to the original PETLS paper and project. Correctness
tests compare numerical results against the original implementation on shared
inputs. The supported interface follows Python and PyTorch conventions instead
of carrying duplicate aliases for historical C++ and pybind11 names.

The full test suite includes reference/parity tests against original PETLS
fixtures and variants, with default comparison tolerances of `atol=1e-4` and
`rtol=1e-3` unless a test specifies a stricter tolerance.

This project is based on the public PETLS API, PETLS documentation, and the
algorithms described in the PETLS paper. No source code from the original PETLS
implementation is included in this repository.

## Supported API

The package intentionally exposes one supported path for each operation:

| Area | API |
|------|-----|
| Complexes | `Complex`, `Alpha`, `Rips`, `dFlag`, `PersistentSheafLaplacian` |
| Persistent Laplacians | `get_L()`, `get_up()`, `get_down()`, `spectra()`, `eigenpairs()` |
| Topology | `persistence_intervals()`, `betti_numbers_at()`, `persistent_betti()`, `topology_summary()` |
| Localization | `harmonic_features()`, simplex mappings, optional point labels |
| Scaling | `estimate_laplacian()`, allocation guards, `ordinary_spectrum()` |
| Analysis and output | `Profile`, `Timer`, `summaries()`, `plot_summary()`, storage helpers |

Solver selection uses `set_eigs_algorithm("eigvalsh")` for complete dense
spectra or `set_eigs_algorithm("sparse", num_eigenvalues=...)` for a partial
ordinary spectrum. Device and dtype are supplied to constructors; there is no
mutable package-wide configuration.

## Directed Flag Files

`dFlag` reads weighted directed graphs from `.flag` files with a `dim 0`
vertex-weight section and an optional `dim 1` directed-edge section:

```text
dim 0
0.0 0.0 0.0 0.0
dim 1
0 1 1.25
1 2 2.50
0 2 3.00
```

The `dim 0` line after the header contains one whitespace-separated vertex
weight per vertex. Each `dim 1` row is `source target weight`, using zero-based
vertex indices. An edge is present whenever its row appears, including zero or
negative weights; missing directed edges remain absent. Self-loops are not
supported, duplicate rows are rejected, and all weights must be finite.
Filtration values are the maximum of all vertex and directed-edge weights in a
simplex, ensuring every face appears no later than its cofaces.

## Benchmark Notes

The benchmark runner compares `petls-pytorch` and the original PETLS package on
identical synthetic inputs. It streams progress for every trial and writes all
CSV, JSON, and plot outputs under `benchmark-results/` by default.

The `standard` preset is the main comparison workload. It exercises Alpha
complexes on torus, sphere, swiss roll, and Klein bottle point clouds, plus a
bounded Rips-complex case. It samples dimensions 0, 1, and 2 and completes all
standard rows by default, including empty Laplacians and the largest sampled
Alpha matrices. Use `--max_matrix_rows` only for custom capped runs.

Final benchmark on our Windows 11 Pro machine:

- CPU: Intel Core i7-13700K, 16 cores / 24 logical processors
- GPU: NVIDIA GeForce RTX 4070 Ti, 12GB
- PyTorch: `2.10.0+cu130`
- Original PETLS: `petls==1.0.1`, native Windows C++ build

| Package | Device | Completed | Skipped | Trial Time | Mean Trial | Mean Eigs | Complex Builds | Max Completed Matrix |
|---------|--------|----------:|--------:|-----------:|-----------:|----------:|---------------:|---------------------:|
| `petls` | CPU | 78 | 0 | 8.05 s | 103.2 ms | 97.7 ms | 0.52 s | 2399 x 2399 |
| `petls-pytorch` | CPU | 78 | 0 | 2.20 s | 28.2 ms | 24.2 ms | 0.57 s | 2399 x 2399 |
| `petls-pytorch` | CUDA | 78 | 0 | 1.05 s | 13.4 ms | 9.7 ms | 0.66 s | 2399 x 2399 |

On this workload, `petls-pytorch` CPU is `3.65x` faster by trial time and
`4.03x` faster on eigensolves than native PETLS. On the RTX 4070 Ti,
`petls-pytorch` CUDA is `7.70x` faster by trial time and `10.04x` faster on
eigensolves. A few tiny rows are still slower row-by-row because fixed overhead
dominates, but the standard CPU and CUDA aggregate comparisons are both clear
wins with no skipped benchmark rows.

## Running Benchmarks

From a source checkout, run the benchmark module with `uv`. The benchmark
harness needs the `benchmark` extra, which provides `tadasets`; add
`--with petls` when benchmarking against the original PETLS package.

```bash
# Representative CPU/GPU comparison
uv run --extra benchmark python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cpu
uv run --extra benchmark python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cuda --dtype float32

# Reference PETLS, if installed for your platform
uv run --extra benchmark --with petls python -m benchmark --preset standard --package petls --algorithm selfadjoint

# Larger GPU stress run
uv run --extra benchmark python -m benchmark --preset stress --package petls-pytorch --algorithm eigvalsh --device cuda

# Custom single run
uv run --extra benchmark python -m benchmark \
    --dataset torus --n_points 2000 --complex alpha --max_dim 3 \
    --package petls-pytorch --algorithm eigvalsh --device cuda --dtype float32 \
    --max_matrix_rows 12000
```

Benchmark device and dtype are passed directly to each PETLS-PyTorch object and
recorded in CSV output. The benchmark CLI defaults to `float32` to preserve the
historical comparison workload; pass `--dtype float64` to benchmark the
higher-precision weighted-Alpha default used by the scientific API. Original
PETLS does not expose the PyTorch dtype selection, so its result rows and
dataset metadata report `dtype="native"`.

By default, benchmark files are written under `benchmark-results/results`. Use
`--output_dir benchmark-results/<run-name>` to keep named runs together.

Verify CUDA before benchmarking:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If you are not using `uv`, install from the source checkout first:

```bash
python -m pip install -e ".[benchmark]"
python -m pip install petls  # only needed for --package petls
python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cpu
```

## Test Suite

From a source checkout, run the default test suite with development
dependencies:

```bash
uv run --extra dev pytest tests/ -v
```

Run the same release-quality checks used by CI from the active supported Python
environment:

```bash
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src/petls_pytorch benchmark
```

Tests are explicitly separated with `native`, `parity`, and `benchmark` markers.
The native suite is independent of the optional reference package:

```bash
uv run --extra dev pytest -m "not parity"
```

Run the PETLS comparison suite separately. It requires the installed reference
package and never changes native-test collection:

```bash
uv run --extra dev --with petls pytest -m parity -v
```

Benchmark harness configuration tests are also selectable independently:

```bash
uv run --extra dev pytest -m benchmark -v
```

On Windows, PETLS `get_down()` reference calls can trigger access violations in
the original native package for the Rips, Alpha, and dFlag comparison cases. To
avoid those platform-specific crashes while still running the rest of the suite,
exclude those reference checks:

```bash
uv run --extra dev --with petls pytest -m parity -v -k "not test_get_down_eigenvalues_match_reference and not test_get_down_eigenvalues_match_mwe"
```

If you are not using `uv`, install the package and test dependencies first:

```bash
python -m pip install -e ".[dev]"
python -m pip install petls  # only needed for the full parity suite
pytest -m "not parity" -v
```

The full parity suite covers core functionality, Rips complexes, alpha
complexes, directed flag complexes, sheaf support, eigenvalue utilities, and
I/O helpers.

## Contributing

Issues and pull requests are welcome at
https://github.com/Sylverity/petls-pytorch/issues. Contributions that expand
coverage against the reference PETLS implementation are especially helpful.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

`petls_pytorch` is an independent, clean-room implementation and contains no
source code from the original PETLS project. It depends on third-party packages
with their own licenses, including `gudhi` (MIT, with GPL-licensed dependencies
such as CGAL used by some modules, e.g. alpha complexes). These are installed
separately and are not redistributed as part of this project.

## Citation

If you use `petls-pytorch` in research, please cite both this PyTorch
implementation and the original PETLS paper.

### petls-pytorch

```bibtex
@software{marston2026petlspytorch,
  title        = {petls-pytorch: A PyTorch-native implementation of persistent topological Laplacians},
  author       = {Marston, Sumner K.},
  year         = {2026},
  publisher    = {Sylverity Research},
  url          = {https://github.com/Sylverity/petls-pytorch}
}
```

### Original PETLS paper

```bibtex
@misc{jones2025petlspersistenttopologicallaplacian,
    title={PETLS: PErsistent Topological Laplacian Software},
    author={Benjamin Jones and Guo-Wei Wei},
    year={2025},
    eprint={2508.11560},
    archivePrefix={arXiv},
    primaryClass={math.AT},
    url={https://arxiv.org/abs/2508.11560},
}
```

This project is an independent PyTorch-native implementation based on the
public PETLS API, PETLS documentation, and the PETLS paper. No source code from
the original PETLS implementation is included in this repository.

- Original PETLS repository: https://github.com/bdjones13/PETLS
- PETLS documentation: https://www.benjones-math.com/software/PETLS/
- PETLS paper: https://arxiv.org/abs/2508.11560
