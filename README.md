# PETLS-PyTorch

A PyTorch-native implementation of persistent topological Laplacians, based on the
public PETLS API, PETLS documentation, and the algorithms described in the
**PETLS** paper.

The goal of this project is to make PETLS-style computations easier to use in
Python and PyTorch workflows, especially when downstream code already works with
torch tensors or CUDA devices. The original C++/pybind11 PETLS implementation
remains the reference for correctness.

## Quick Start

```python
import petls_torch

# Alpha complex from point cloud
alpha = petls_torch.Alpha(points=[[0, 0], [1, 0], [0.5, 1]], max_dim=2)
eigs = alpha.spectra(0, 0.0, 1.0)

# Rips complex from distance matrix
rips = petls_torch.Rips(distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], max_dim=2)

# Directed flag complex from .flag file
dflag = petls_torch.dFlag("graph.flag", max_dim=3)

# Sheaf Laplacian
sst = petls_torch.sheaf_simplex_tree(st, extra_data, restriction)
psl = petls_torch.PersistentSheafLaplacian(sst)
filtrations = psl.get_all_filtrations()
sheaf_eigs = psl.spectra(dim=0, a=filtrations[0], b=filtrations[-1])
```

For `spectra(dim, a, b)`, choose `a` and `b` from the filtration values in
`psl.get_all_filtrations()` with `a <= b`.

## Relationship to PETLS

`petls_torch` is an independent PyTorch-native implementation of the PETLS
methods and public API behavior, with attribution to the original PETLS paper
and project. Correctness tests compare against the original implementation using
shared inputs and the default tolerances used in this repository.

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
behavior on one local machine. They should be read as preliminary measurements,
not as a comprehensive performance study.

The benchmark suite is a performance comparison against the reference PETLS
implementation on identical synthetic inputs. The shared runner is
parameterized with `--package petls` or `--package petls_torch`.

### Hardware

- CPU: Intel i7-14700K
- GPU: NVIDIA RTX 4070 Ti 12GB
- PyTorch 2.x with CUDA 12.x

### Quick Preset

Configuration: torus `n=500` plus sphere `n=300`, `max_dim=2`, 8 filtrations
per dataset.

| Package | Device | Total Time | Mean Trial | Mean Build | Mean Eigs |
|---------|--------|-----------:|-----------:|-----------:|----------:|
| `petls` | CPU | 8.29 s | 172.6 ms | 15.2 ms | 157.4 ms |
| `petls_torch` | CPU | 1.82 s | 37.9 ms | 4.8 ms | 33.1 ms |
| `petls_torch` | CUDA | 1.85 s | 38.5 ms | 4.7 ms | 33.8 ms |

On this workload, the PyTorch implementation is about 4.5x faster than the
reference PETLS run on the same CPU. The CUDA timing is similar to the CPU
timing for this preset, which suggests that eigendecomposition and data sizes
are still small enough that GPU overhead can offset part of the benefit.

### Medium Workload

Configuration: torus `n=500`, `max_dim=3`, 16 filtrations.

| Package | Device | Total Time | Mean Trial | Max Matrix | Status |
|---------|--------|-----------:|-----------:|-----------:|--------|
| `petls` | CPU | > 300 s | unavailable | unavailable | Reached benchmark timeout |
| `petls_torch` | CUDA | 8.80 s | 137.5 ms | 5990 x 5990 | Completed |

For this larger dense-eigendecomposition workload, the PyTorch/CUDA path
completed within the benchmark timeout. The PETLS CPU run reached the configured
300-second timeout in this test configuration.

### Operation Breakdown

Quick preset timings:

```text
petls (CPU)          : build=15.2 ms  eigs=157.4 ms
petls_torch (CPU)    : build=4.8 ms   eigs=33.1 ms
petls_torch (CUDA)   : build=4.7 ms   eigs=33.8 ms
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

# PETLS-PyTorch on CUDA
uv run python -m benchmark --preset quick --package petls_torch --algorithm eigvalsh --device cuda

# PETLS-PyTorch on CPU
uv run python -m benchmark --preset quick --package petls_torch --algorithm eigvalsh --device cpu

# Custom single run
uv run python -m benchmark \
    --dataset torus --n_points 2000 --complex alpha --max_dim 3 \
    --package petls_torch --algorithm eigvalsh
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

Requires CPython 3.10, 3.11, 3.12, 3.13, or 3.14.

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

## Citation

If you use `petls_torch` in research, please cite both this PyTorch
implementation and the original PETLS paper.

### PETLS-PyTorch

```bibtex
@software{marston2026petlstorch,
  title        = {PETLS-PyTorch: A PyTorch-native implementation of persistent topological Laplacians},
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
