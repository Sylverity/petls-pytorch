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

The benchmark results below are included to give a sense of current runtime
behavior in one local WSL2 environment. They should be read as preliminary
measurements, not as a comprehensive performance study.

The benchmark suite is a performance comparison against the reference PETLS
implementation on identical synthetic inputs. The shared runner is
parameterized with `--package petls` or `--package petls-pytorch`.

### Hardware

- CPU: Intel i7-13700K as reported by WSL2
- GPU: NVIDIA RTX 4070 Ti 12GB
- PyTorch 2.12.1+cu130 with CUDA available

### Quick Preset

Configuration: torus `n=500` plus sphere `n=300`, `max_dim=2`, 8 filtrations
per dataset.

| Package | Device | Total Time | Mean Trial | Mean Build | Mean Eigs |
|---------|--------|-----------:|-----------:|-----------:|----------:|
| `petls` | CPU | 442.88 s | 9226.6 ms | 25.0 ms | 9201.7 ms |
| `petls-pytorch` | CPU | 419.67 s | 8743.0 ms | 284.3 ms | 8458.8 ms |
| `petls-pytorch` | CUDA | 2.77 s | 57.8 ms | 9.5 ms | 48.3 ms |

On this machine, the CPU runs are similar on this workload. The CUDA run is much
faster because the dense eigendecompositions dominate the total runtime.

### Medium Workload

Configuration: torus `n=500`, `max_dim=3`, 16 filtrations.

| Package | Device | Total Time | Mean Trial | Max Matrix | Status |
|---------|--------|-----------:|-----------:|-----------:|--------|
| `petls` | CPU | > 300 s | unavailable | unavailable | Reached benchmark timeout |
| `petls-pytorch` | CPU | > 300 s | unavailable | unavailable | Reached benchmark timeout |
| `petls-pytorch` | CUDA | 4.91 s | 76.7 ms | 5990 x 5990 | Completed |

For this larger dense-eigendecomposition workload, the CUDA path completed
within the benchmark timeout. Both CPU runs reached the configured 300-second
timeout in this WSL2 test configuration.

### Operation Breakdown

Quick preset timings:

```text
petls (CPU)          : build=25.0 ms   eigs=9201.7 ms
petls-pytorch (CPU)  : build=284.3 ms  eigs=8458.8 ms
petls-pytorch (CUDA) : build=9.5 ms    eigs=48.3 ms
```

In these measurements, most time is spent in eigendecomposition. The PyTorch
implementation benefits from PyTorch's dense linear algebra backends on CPU and
from CUDA support on larger matrix workloads.

## Running Benchmarks

From a source checkout, run the benchmark module with `uv`. Add `--with petls`
when benchmarking against the original PETLS package:

```bash
# Original PETLS
uv run --with petls python -m benchmark --preset quick --package petls --algorithm eigvalsh

# petls-pytorch on CUDA
uv run python -m benchmark --preset quick --package petls-pytorch --algorithm eigvalsh --device cuda

# petls-pytorch on CPU
uv run python -m benchmark --preset quick --package petls-pytorch --algorithm eigvalsh --device cpu

# Custom single run
uv run python -m benchmark \
    --dataset torus --n_points 2000 --complex alpha --max_dim 3 \
    --package petls-pytorch --algorithm eigvalsh
```

If you are not using `uv`, install from the source checkout first:

```bash
python -m pip install -e .
python -m pip install petls  # only needed for --package petls
python -m benchmark --preset quick --package petls --algorithm eigvalsh
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
reference implementation:

```bash
uv run --extra dev --with petls pytest tests/ -v
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
