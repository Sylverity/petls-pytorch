# petls-pytorch

[![PyPI](https://img.shields.io/pypi/v/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![CI](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![License](https://img.shields.io/pypi/l/petls-pytorch.svg)](LICENSE)

PyTorch-native persistent topological Laplacians, with CUDA support for large
dense eigendecompositions.

Status: beta

A PyTorch-native implementation of persistent topological Laplacians, based on the
public PETLS API, PETLS documentation, and the algorithms described in the
**PETLS** paper.

The goal of this project is to make PETLS-style computations easier to use in
Python and PyTorch workflows, especially when downstream code already works with
torch tensors or CUDA devices. The original C++/pybind11 PETLS implementation
remains the reference for correctness.

## Contents

- [Quick Start](#quick-start)
- [Weighted Alpha Complexes](#weighted-alpha-complexes)
- [Topology and Simplex Inspection](#topology-and-simplex-inspection)
- [Numerical and Scaling Controls](#numerical-and-scaling-controls)
- [Relationship to PETLS](#relationship-to-petls)
- [API Coverage](#api-coverage)
- [Directed Flag Files](#directed-flag-files)
- [Benchmark Notes](#benchmark-notes)
- [Installation](#installation)
- [Test Suite](#test-suite)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Quick Start

```python
import petls_pytorch
import gudhi
import torch

# Alpha complex from point cloud
alpha = petls_pytorch.Alpha(points=[[0, 0], [1, 0], [0.5, 1]], max_dim=2)
eigs = alpha.spectra(0, 0.0, 1.0)

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
sheaf_eigs = psl.spectra(dim=0, a=filtrations[0], b=filtrations[-1])
```

For `spectra(dim, a, b)`, choose `a` and `b` from the filtration values in
`psl.get_all_filtrations()` with `a <= b`.

## Weighted Alpha Complexes

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

## Topology and Simplex Inspection

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
for numerical auditing.

`harmonic_features()` returns each numerical zero-mode eigenvector as
simplex/coefficient records. Harmonic bases need not be unique, but their
coordinates always follow the retained simplex order. On an oversized complex,
the method caps automatic localization at ten representatives and reports
`truncated_for_scale`; pass `max_features=` to request a different explicit
limit.

## Numerical and Scaling Controls

Device and dtype are object-specific. The library default device is CPU;
`device="auto"` opts in to CUDA when available, and `device="cuda"` requests it
explicitly. Constructing one object never changes another object's placement.
Weighted `Alpha` defaults to `torch.float64`; callers can select `float32` or
`float64` explicitly on all primary complex variants.

Zero eigenvalues use the scale-aware test
`abs(lambda) <= zero_atol + zero_rtol * max(abs(spectrum))`. Summary results
report the effective tolerance and smallest eigenvalues so borderline modes can
be inspected.

Dense allocations are guarded before construction. The defaults are 12,000
rows and 4 GB per Laplacian, and both are configurable per object:

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
`set_eigs_algorithm("sparse", ...)`. Persistent Schur complements can become
dense, so oversized `a < b` requests either raise `LaplacianSizeError` or return
Gudhi homology-only status through `topology_summary()`. This distinction is
intentional: large systems retain authoritative homology and ordinary sparse
Hodge spectra without pretending that full persistent spectra are sparse.

## Relationship to PETLS

`petls-pytorch` is an independent PyTorch-native implementation of the PETLS
methods and public API behavior, with attribution to the original PETLS paper
and project. Correctness tests compare against the original implementation using
shared inputs.

The full test suite includes 65 reference/parity tests against original PETLS
fixtures and variants, with default comparison tolerances of `atol=1e-4` and
`rtol=1e-3` unless a test specifies a stricter tolerance.

This project is based on the public PETLS API, PETLS documentation, and the
algorithms described in the PETLS paper. No source code from the original PETLS
implementation is included in this repository.

## API Coverage

| Original PETLS API | PyTorch API | Status |
|--------------------|-------------|--------|
| `Complex` | `Complex` | Implemented |
| `Alpha` | `Alpha` | Implemented |
| `Rips` | `Rips` | Implemented |
| `dFlag` | `dFlag` | Implemented |
| `sheaf_simplex_tree` | `sheaf_simplex_tree` | Implemented |
| `PersistentSheafLaplacian` | `PersistentSheafLaplacian` | Implemented |
| `Profile`, `timer` | `Profile`, `Timer` | Implemented |
| `summaries`, `plot_summary` | `summaries`, `plot_summary` | Implemented |
| `up_Algorithms` enum | `up_Algorithms` enum | Implemented |
| `eigvalsh_wrapper`, `sparse_wrapper`, `matrix_is_diagonal` | Same | Implemented |
| `flipped` optimization | `flipped` + `get_L_top_dim_flipped` | Implemented |
| `nonzero_spectra()` | `nonzero_spectra()` | Implemented |
| `store_L()`, `store_spectra()`, `store_spectra_summary()` | Same | Implemented |
| `time_to_csv()` | `time_to_csv()` | Implemented |
| Weighted alpha / power weights | `Alpha(weights=..., max_alpha_square=...)` | Implemented |
| Gudhi persistence inspection | `persistence_intervals()`, `betti_numbers_at()`, `persistent_betti()` | Implemented |
| Topology result object | `topology_summary()` | Implemented |
| Harmonic localization | `harmonic_features()` + simplex mappings | Implemented |
| Allocation safeguards | `estimate_laplacian()` + row/byte guards | Implemented |
| Sparse ordinary low spectrum | `ordinary_spectrum()` | Implemented |

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
vertex indices. Missing directed edges are absent from the file; self-loops are
not supported. Filtration values for higher-dimensional directed simplices are
the maximum edge weight in each simplex.

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

From a source checkout, run the benchmark module with `uv`. Add `--with petls`
when benchmarking against the original PETLS package.

```bash
# Representative CPU/GPU comparison
uv run python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cpu
uv run python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cuda

# Reference PETLS, if installed for your platform
uv run --with petls python -m benchmark --preset standard --package petls --algorithm selfadjoint

# Larger GPU stress run
uv run python -m benchmark --preset stress --package petls-pytorch --algorithm eigvalsh --device cuda

# Custom single run
uv run python -m benchmark \
    --dataset torus --n_points 2000 --complex alpha --max_dim 3 \
    --package petls-pytorch --algorithm eigvalsh --device cuda \
    --max_matrix_rows 12000
```

By default, benchmark files are written under `benchmark-results/results`. Use
`--output_dir benchmark-results/<run-name>` to keep named runs together.

Verify CUDA before benchmarking:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If you are not using `uv`, install from the source checkout first:

```bash
python -m pip install -e .
python -m pip install petls  # only needed for --package petls
python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cpu
```

## Installation

```bash
pip install petls-pytorch
```

Then import it in Python with `import petls_pytorch`.

Requires CPython 3.10, 3.11, 3.12, 3.13, or 3.14.

For GPU acceleration, install a CUDA-enabled PyTorch build that matches your
system using the official selector at https://pytorch.org/get-started/locally/.

Release notes are tracked in [CHANGELOG.md](CHANGELOG.md).

Dependencies:

- `torch`
- `numpy`
- `scipy`
- `gudhi`
- `tadasets` for benchmarks

## Test Suite

From a source checkout, run the default test suite with development
dependencies:

```bash
uv run --extra dev pytest tests/ -v
```

By default, tests that compare against the original PETLS package are skipped
when `petls` is not installed. To run the full parity suite against the
reference implementation, use `uv run --with petls` to add PETLS to the
temporary run environment:

```bash
uv run --extra dev --with petls pytest tests/ -v
```

On Windows, PETLS `get_down()` reference calls can trigger access violations in
the original native package for the Rips, Alpha, and dFlag comparison cases. To
avoid those platform-specific crashes while still running the rest of the suite,
exclude those reference checks:

```bash
uv run --extra dev --with petls pytest tests/ -v -k "not test_get_down_eigenvalues_match_reference and not test_get_down_eigenvalues_match_mwe"
```

If you are not using `uv`, install the package and test dependencies first:

```bash
python -m pip install -e ".[dev]"
python -m pip install petls  # only needed for the full parity suite
pytest tests/ -v
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
