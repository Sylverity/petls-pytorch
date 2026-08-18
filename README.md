# petls-pytorch

[![PyPI](https://img.shields.io/pypi/v/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![CI](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![License](https://img.shields.io/pypi/l/petls-pytorch.svg)](LICENSE)

PETLS-PyTorch brings persistent topological Laplacians directly into the PyTorch ecosystem, delivering fast, GPU-accelerated eigensolves for high-dimensional data. Built for molecular-dynamics trajectories, complex point clouds, and other geometric data, it builds on Ben Jones's foundational PETLS library with native CUDA execution and seamless Gudhi integration. Full support for weighted Alpha complexes, Rips, directed-flag, and cellular-sheaf complexes—and harmonic localization—is included.

Highlights:

- PyTorch-native tensors with CPU and CUDA execution.
- Weighted Gudhi Alpha complexes, including negative vertex births and power weights.
- Persistent spectra, Betti numbers, persistence intervals, and harmonic localization.
- Rips, directed flag, and cellular-sheaf constructions.
- Sparse ordinary spectra, dense-allocation guards, and auditable topology summaries.
- Object-local device, dtype, solver, and numerical-tolerance controls.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmark Notes](#benchmark-notes)
- [Running Benchmarks](#running-benchmarks)
- [Alpha Complexes](#alpha-complexes)
- [Topology, Persistence, and Localization](#topology-persistence-and-localization)
- [Precision, Devices, and Scaling](#precision-devices-and-scaling)
- [Supported API](#supported-api)
- [Test Suite](#test-suite)
- [File Formats](#file-formats)
- [Relationship to PETLS & Acknowledgments](#relationship-to-petls--acknowledgments)
- [Advanced Performance & Algorithm Notes](#advanced-performance--algorithm-notes)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

```bash
pip install petls-pytorch
```

PETLS-PyTorch supports CPython 3.10–3.14. Install a CUDA-enabled PyTorch build that matches your system using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Runtime dependencies are `torch`, `numpy`, `scipy`, and `gudhi`.

Optional extras:

- `pip install "petls-pytorch[benchmark]"` adds benchmark datasets (`tadasets`).
- `pip install "petls-pytorch[analysis]"` adds benchmark plotting (`matplotlib`).

## Quick Start

```python
import torch
import petls_pytorch

alpha = petls_pytorch.Alpha(
    points=[[0, 0], [1, 0], [1, 1], [0, 1]],
    weights=[0.20, 0.18, 0.20, 0.18],
    max_dim=2,
    max_alpha_square=1.0,
    device="cuda",  # use "cpu" when CUDA is unavailable
    dtype=torch.float64,
)

summary = alpha.topology_summary(dimensions=(0, 1), a=0.0, b=0.0)
spectrum = alpha.spectra(dim=1, a=0.0, b=0.0)
intervals = alpha.persistence_intervals(dim=1)
features = alpha.harmonic_features(dim=1, a=0.0, b=0.0)
```

Construct other supported variants just as directly:

```python
rips = petls_pytorch.Rips(distances=[[0, 1, 1], [1, 0, 1], [1, 1, 0]], max_dim=2)
dflag = petls_pytorch.dFlag("graph.flag", max_dim=3)
```

For `Rips`, `max_dim` is the highest dimension reported by the public API. One additional simplex dimension may be retained internally so the top-dimensional Laplacian includes its up term. Single `spectra()` and `eigenpairs()` queries require `dim`, `a`, and `b` together; use `request_list` for multiple queries.

## Benchmark Notes

The benchmark harness measures complex construction, Laplacian construction, and eigensolving on the same inputs. It records device, dtype, skipped rows, and failed requests under `benchmark-results/`.

Representative workload (78 completed trials, RTX 4070 Ti, PyTorch 2.10.0+cu130):

| Backend | Device | Total trial time | Mean trial | Mean eigensolve |
|---------|--------|-----------------:|-----------:|----------------:|
| `petls-pytorch` | CPU | 2.20 s | 28.2 ms | 24.2 ms |
| `petls-pytorch` | CUDA | 1.05 s | 13.4 ms | 9.7 ms |

CUDA reduced aggregate trial time by about 52% and mean eigensolve time by about 60% on this workload. Results depend on matrix sizes, GPU model, dtype, and batching/parallelism.

## Running Benchmarks

From a source checkout:

```bash
# CPU and CUDA workloads
uv run --extra benchmark python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cpu
uv run --extra benchmark python -m benchmark --preset standard --package petls-pytorch --algorithm eigvalsh --device cuda --dtype float32

# Larger GPU stress run
uv run --extra benchmark python -m benchmark --preset stress --package petls-pytorch --algorithm eigvalsh --device cuda

# Reference PETLS, when available on your platform
uv run --extra benchmark --with petls python -m benchmark --preset standard --package petls --algorithm selfadjoint
```

Use `--output_dir benchmark-results/<run-name>` for named runs. Verify CUDA before a GPU run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The CLI defaults to `float32` for benchmark continuity; pass `--dtype float64` for higher-precision weighted-Alpha runs. A custom workload looks like:

```bash
uv run --extra benchmark python -m benchmark \
    --dataset torus --n_points 2000 --complex alpha --max_dim 3 \
    --package petls-pytorch --algorithm eigvalsh --device cuda --dtype float32
```

## Alpha Complexes

`Alpha` accepts Gudhi power weights directly. Weights must be finite, and the constructor validates point and weight shapes. Omit `weights` for ordinary Alpha behavior.

```python
alpha = petls_pytorch.Alpha(
    points=positions,
    weights=power_weights,
    max_dim=3,
    precision="safe",
    max_alpha_square=maximum_scale,
    point_labels=labels,  # optional, kept separate from topology
    device="cpu",
    dtype=torch.float64,
)

scales = alpha.get_all_filtrations(
    merge_tolerance=1e-10,
    include_vertex_filtrations=True,
)
```

Weighted vertex births, including negative filtration values, are retained. `max_alpha_square` limits Gudhi construction; filtration enumeration includes vertex births and merges near-duplicate scales by default.

## Topology, Persistence, and Localization

Gudhi-backed complexes retain simplex identities and stable simplex-to-matrix mappings, so Laplacian coordinates and harmonic features can be interpreted geometrically:

```python
summary = alpha.topology_summary(dimensions=(0, 1, 2), a=0.0, b=0.0)
bettis = alpha.betti_numbers_at(scale=0.0)
persistent = alpha.persistent_betti(dim=1, birth_scale=0.0, death_scale=0.5)
features = alpha.harmonic_features(dim=1, a=0.0, b=0.0)
```

Gudhi persistence is the authoritative homology source when available. Summaries report Betti numbers separately from numerical diagnostics such as nullity, spectral gaps, tolerances, matrix sizes, and calculation status. Queries above `top_dim` return zero Betti number and an empty spectrum.

Gudhi-backed objects also retain `simplex_tree`, `simplices_by_dimension`, `simplex_filtrations`, and `simplex_to_index`. If `point_labels` are supplied, label tuples follow the same stable simplex order. Ordinary oversized localization uses a sparse path and can cap automatic representatives with `max_features`; persistent localization requires the dense Schur-complement path and observes the configured allocation guard.

## Precision, Devices, and Scaling

- `device`: CPU by default; use `"cuda"` explicitly or `"auto"` to opt into available CUDA. Settings are object-local.
- `dtype`: choose `torch.float32` or `torch.float64`; weighted Alpha defaults to `float64`.
- `zero_atol`, `zero_rtol`: define the scale-aware zero test `abs(lambda) <= zero_atol + zero_rtol * max(abs(spectrum))`.
- `max_matrix_rows`, `max_matrix_bytes`: guard dense allocations before construction.
- `on_oversize`: use `"raise"` or `"homology_only"` for oversized persistent requests.

```python
estimate = alpha.estimate_laplacian(dim=1, a=0.0, b=0.0)
guarded = petls_pytorch.Alpha(
    points=positions,
    max_matrix_rows=12_000,
    max_matrix_bytes=4_000_000_000,
    on_oversize="homology_only",
)
```

## Supported API

| Area | API |
|------|-----|
| Complexes | `Complex`, `Alpha`, `Rips`, `dFlag`, `PersistentSheafLaplacian` |
| Laplacians | `get_L()`, `get_up()`, `get_down()`, `spectra()`, `eigenpairs()` |
| Topology | `persistence_intervals()`, `betti_numbers_at()`, `persistent_betti()`, `topology_summary()` |
| Localization | `harmonic_features()`, simplex mappings, point labels |
| Scaling | `estimate_laplacian()`, `ordinary_spectrum()`, allocation guards |
| Output | `Profile`, `Timer`, `summaries()`, `plot_summary()`, storage helpers |

Use `set_eigs_algorithm("eigvalsh")` for complete dense spectra or `set_eigs_algorithm("sparse", num_eigenvalues=...)` for partial ordinary spectra. Sparse orders `SM`, `SA`, `LM`, `LA`, and `BE` are supported.

## Test Suite

```bash
uv run --extra dev pytest -m "not parity"
uv run --extra dev --with petls pytest -m parity -v
uv run --extra dev pytest -m benchmark -v
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .
uv run --extra dev mypy src/petls_pytorch benchmark
```

Native tests do not require the optional reference package. The parity job installs `petls==1.0.1` separately. On Windows, a small set of native `get_down()` comparisons may need a focused exclusion because of platform-specific behavior; the parity test documentation shows that command.

## File Formats

`dFlag` reads weighted directed graphs from `.flag` files:

```text
dim 0
0.0 0.0 0.0 0.0
dim 1
0 1 1.25
1 2 2.50
0 2 3.00
```

The `dim 0` line contains one vertex weight per vertex. Each `dim 1` row is `source target weight` with zero-based indices. A listed edge exists even when its weight is zero or negative; missing edges remain absent. Self-loops and duplicate rows are rejected, weights must be finite, and simplex filtration is the maximum of its vertex and directed-edge weights so every face appears no later than its cofaces.

## Relationship to PETLS & Acknowledgments

This project is a GPU-accelerated, PyTorch-native port that builds on the foundational persistent-Laplacian work of Ben Jones and the PETLS project. PETLS remains an important reference for shared numerical behavior, while this package follows Python and PyTorch conventions.

The implementation is clean-room and independent: it uses the public PETLS API, documentation, and paper, and includes no source code from the original PETLS implementation. Numerical parity tests compare shared workflows against the reference package.

- [Original PETLS repository](https://github.com/bdjones13/PETLS)
- [PETLS documentation](https://www.benjones-math.com/software/PETLS/)
- [PETLS paper](https://arxiv.org/abs/2508.11560)

## Advanced Performance & Algorithm Notes

- Ordinary `a == b` spectra can use sparse boundary Gram matrices and SciPy's sparse symmetric eigensolver without materializing a dense Laplacian.
- Use `ordinary_spectrum(dim, scale, num_eigenvalues)` directly when you want a bounded sparse solve without constructing the full dense operator for large ordinary Hodge problems.
- Gudhi-backed ordinary summaries use block LOBPCG when repeated zero modes require more reliable nullspace recovery. Residuals, orthogonality, and numerical nullity are audited before reporting a spectral gap.
- Certified summaries expose `spectrum_solver`, `spectrum_certified`, and `spectrum_max_normalized_residual`; incomplete recovery leaves `least_nonzero_eigenvalue` unset and records an explicit calculation status.
- Persistent `a < b` Schur complements can become dense. Allocation guards account for both the final matrix at `a` and the larger intermediate at `b`; oversized requests either raise `LaplacianSizeError` or return authoritative homology-only status.
- The flipped top-dimensional optimization is limited to complete `eigvalsh` solves where the algebraic top boundary has no higher-dimensional up term. Partial spectra use the ordinary sparse path so kernel dimensions remain correct.

## Contributing

Issues and pull requests are welcome at https://github.com/Sylverity/petls-pytorch/issues.

## License

Apache License 2.0; see [LICENSE](LICENSE). Third-party dependencies retain their own licenses.

## Citation

If you use `petls-pytorch` in research, cite this software and the original PETLS paper.

```bibtex
@software{marston2026petlspytorch,
  title        = {petls-pytorch: A PyTorch-native implementation of persistent topological Laplacians},
  author       = {Marston, Sumner K.},
  year         = {2026},
  publisher    = {Sylverity Research},
  url          = {https://github.com/Sylverity/petls-pytorch}
}

@misc{jones2025petlspersistenttopologicallaplacian,
  title={PETLS: PErsistent Topological Laplacian Software},
  author={Benjamin Jones and Guo-Wei Wei},
  year={2025},
  eprint={2508.11560},
  archivePrefix={arXiv},
  primaryClass={math.AT},
  url={https://arxiv.org/abs/2508.11560}
}
```
