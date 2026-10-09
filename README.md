# petls-pytorch

[![PyPI](https://img.shields.io/pypi/v/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![CI](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/Sylverity/petls-pytorch/actions/workflows/ci.yml)
[![Security](https://github.com/Sylverity/petls-pytorch/actions/workflows/security.yml/badge.svg)](https://github.com/Sylverity/petls-pytorch/actions/workflows/security.yml)
[![Python](https://img.shields.io/pypi/pyversions/petls-pytorch.svg)](https://pypi.org/project/petls-pytorch/)
[![License](https://img.shields.io/pypi/l/petls-pytorch.svg)](https://github.com/Sylverity/petls-pytorch/blob/main/LICENSE)

<p align="center">
  <img src="https://raw.githubusercontent.com/Sylverity/petls-pytorch/main/docs/assets/petls-topology-hero.png" alt="Alpha complex on a torus with two independent cycles highlighted" width="800" />
</p>

<p align="center"><em>One component, two independent tunnels, one enclosed surface: β = (1, 2, 1).</em></p>

**PETLS-PyTorch turns point clouds, graphs, and molecular trajectories into
multiscale topological signatures, with PyTorch-native CPU and CUDA
eigensolvers.** It answers three complementary questions:

- **What persists?** Betti numbers and persistence intervals track components,
  tunnels, and voids across scale.
- **How is it organized?** Persistent-Laplacian spectra distinguish structures
  that have the same hole counts but different geometry or connectivity.
- **Where is it?** Harmonic representatives and basis-invariant participation
  scores map topological structure back to supporting points, simplices, or
  molecules.

The hero is an actual Alpha complex at `α² = 0.30`; the magenta and cyan paths
represent its two independent one-dimensional classes. The
[figure source](https://github.com/Sylverity/petls-pytorch/blob/main/examples/torus_hero/render_hero.py) is fully reproducible.

## Performance

Measured **October 7, 2026** on an Intel Core i7-13700K and RTX 4070 Ti
(WSL2, PyTorch 2.13.0, CUDA 13.0). Both devices use **float64**, the same
seeded inputs, and four CPU threads. Times are medians of three measured runs
after one warmup per solver/device.

| Workload | Solver | CPU pipeline | CUDA pipeline |
|---|---|---:|---:|
| Standard: 78 queries, up to 2,399 rows | Default: up to 8 positive modes | 3.02 s | 2.38 s |
| Standard: 78 queries, up to 2,399 rows | Full spectrum | 2.61 s | 1.99 s |
| Larger matrices: 48 queries, up to 5,990 rows | Default: up to 8 positive modes | 5.11 s | 6.03 s |
| Larger matrices: 48 queries, up to 5,990 rows | Full spectrum | 9.76 s | 6.39 s |

Pipeline time includes dataset/complex construction once plus all query work:
operator construction, eigensolving, homology, and result processing. Warmup,
validation, and file writes are excluded. All **1,512 measured requests** across
both workloads completed without failures or skips; every requested positive
mode passed comparison with the matching full CPU spectrum.

The default solver uses complete device solves for small matrices and range
iteration for larger ones. Full spectra are faster on the standard workload;
the default reduces CPU pipeline time by about **48%** on the larger-matrix
workload. CUDA full-spectrum timings include the existing CPU eigensolver route
for matrices of 2–512 rows; the default solver stays on the selected device.

Reproduce the comparison from a checkout with [uv](https://docs.astral.sh/uv/):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --extra benchmark python -m benchmark.compare --preset standard \
  --threads 4 --warmups 1 --repeats 3 \
  --output-dir benchmark-results/standard-comparison
```

Use `--preset quick` for the larger-matrix workload above. Results include raw
requests, repeat ranges, solver quality, and hardware/source provenance under
`benchmark-results/`. See the [measurement notes](https://github.com/Sylverity/petls-pytorch/blob/main/docs/benchmarks-2026-10-07.md)
for the workload breakdown and optimization results, or
[technical notes](https://github.com/Sylverity/petls-pytorch/blob/main/docs/technical-notes.md#benchmarks) for other benchmark commands.

## Install

Install the published release from PyPI:

```bash
pip install --upgrade petls-pytorch
```

This README documents **1.2.0**. For development, or to run the included
examples and benchmarks, install a repository checkout:

```bash
git clone https://github.com/Sylverity/petls-pytorch.git
cd petls-pytorch
python -m pip install -e .
```

CPython 3.10–3.14 is supported. For GPU execution, install the CUDA-enabled
PyTorch build appropriate for your system using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).

Optional extras add benchmark datasets (`petls-pytorch[benchmark]`) or the CIF
and plotting dependencies used by the crystal demo
(`petls-pytorch[analysis]`).

## Quick start

```python
import torch
import petls_pytorch

alpha = petls_pytorch.Alpha(
    points=[[0, 0], [1, 0], [1, 1], [0, 1]],
    weights=[0.20, 0.18, 0.20, 0.18],
    max_dim=2,
    max_alpha_square=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float64,
)

scale = 0.10  # The square's edges have entered; its interior is still unfilled.
summary = alpha.topology_summary(dimensions=(0, 1), a=scale, b=scale)
modes = alpha.positive_spectrum(dim=1, a=scale, positive_modes=8)
# Explicit full spectrum, when needed:
# spectrum = alpha.spectra(dim=1, a=scale, b=scale)
intervals = alpha.persistence_intervals(dim=1)
features = alpha.harmonic_features(dim=1, a=scale)
participation = alpha.harmonic_participation(dim=1, a=scale)

print(summary["betti"])  # {0: 1, 1: 1}
if participation["participation_complete"]:
    point_scores = participation["point_participation"]
```

Here `a == b` describes topology at one scale; `a < b` requests a persistent
Laplacian between scales. Alpha scales are squared-radius/power filtration
values. Up to eight positive modes are requested; this example has only three,
with eigenvalues approximately `2, 2, 4`.

Supported constructions:

| Construction | Class | Typical input |
|---|---|---|
| Weighted or ordinary Alpha | `Alpha` | Point coordinates and optional power weights |
| Vietoris–Rips | `Rips` | Point coordinates or a distance matrix |
| Directed flag | `dFlag` | Weighted directed graphs in `.flag` format |
| Cellular sheaf | `PersistentSheafLaplacian` | Filtered complexes with restriction maps |

All constructions provide `get_L()`, `spectra()`, `eigenpairs()`,
`topology_summary()`, and `estimate_laplacian()`. `persistence_intervals()` and
`positive_spectrum()` require a retained Gudhi simplex tree, as in `Alpha`,
`Rips`, or `Complex(simplex_tree=...)`. Directed-flag and sheaf summaries use the
existing spectral path and infer Betti counts from numerical Laplacian nullity.
Harmonic localization requires simplex mappings: these are retained by `Alpha`,
`Rips`, and `dFlag`, but not by `PersistentSheafLaplacian`.

## From a crystal to an interpretable signature

![Weighted-Alpha filtration of a 160-molecule theobromine crystal](https://raw.githubusercontent.com/Sylverity/petls-pytorch/main/docs/assets/theobromine-crystal-topology.gif)

This example reduces a public experimental **160-molecule,
1,920-heavy-atom crystal** to a scale-dependent signature:

- **Left:** the violet Alpha scaffold grows through the crystal; the magenta
  cage localizes one selected harmonic 2-cycle to the molecules supporting it.
- **Top right:** `β₀`, `β₁`, and `β₂` count connected regions, tunnels, and
  enclosed voids. The white cursor marks the scale rendered on the left.
- **Bottom right:** the smallest positive 1- and 2-Laplacian eigenvalues reveal
  organization that Betti numbers alone cannot distinguish.

Applied frame-by-frame, the same workflow creates interpretable time series for
assembly transitions, packing changes, persistent scale ranges, and the
molecules driving each change.

The [complete demo](https://github.com/Sylverity/petls-pytorch/blob/main/examples/theobromine_crystal/render_demo.py),
[reproduction guide](https://github.com/Sylverity/petls-pytorch/blob/main/examples/theobromine_crystal/README.md), and
[public experimental CIF](https://github.com/Sylverity/petls-pytorch/blob/main/examples/theobromine_crystal/7235246.cif) are included.
This is a finite-supercell demonstration, not periodic homology or a binding
energetics calculation.

## Numerical behavior

- CPU is the default; select `"cuda"` explicitly or use `"auto"`.
- `float32` and `float64` are supported; `Alpha` defaults to `float64` with or
  without weights. The partial positive-mode solver always computes in `float64`.
- Gudhi persistence is the authoritative homology source for Gudhi-backed
  complexes, while numerical nullity and spectral diagnostics are reported
  separately.
- Gudhi homology uses coefficients modulo 11; the Laplacian uses real coefficients.
  Modular Betti counts never select or discard real positive modes. Results
  expose both fields and report their agreement separately from solver quality.
- `harmonic_participation()` localizes the harmonic space using projector-diagonal
  scores, invariant to rotations of a complete harmonic basis. Each simplex's
  score is shared equally among its vertices; the total equals the number of
  recovered harmonic directions. Labels are retained when available. These
  scores localize the harmonic space collectively; individual basis vectors are
  not canonical persistence-interval representatives. For truncated results,
  check `participation_complete` before comparing whole-space scores.
- Gudhi-backed `topology_summary()` defaults to requesting eight leading positive
  modes. `positive_spectrum()` exposes the same solver directly; small problems
  use a complete device eigensolve internally.
- In the partial solver, sparse operator applications, projected eigensolves,
  and residual checks stay on the selected device in float64, with no CPU
  eigensolver fallback. Geometry and boundary indexing can still use CPU.
- Large partial persistent solves apply a Schur correction without materializing
  the full dense Laplacian. The active eliminated block is dense and factored
  on the device; numerically singular blocks use a pseudoinverse. Allocation
  guards still apply.
- Large partial solves work in `range(L)` with a block independent of Betti
  multiplicity. They stop when the requested positive modes meet the configured
  relative residual tolerance, or reach the iteration limit; they do not
  rediscover the harmonic kernel.
- Each partial-solver mode reports its residual and `certified`, `approximate`, or
  `unavailable` quality. Range reconstruction and orthogonality are audited.
  These labels describe numerical checks, not formal eigenvalue enclosures.
  Numerical nullity is `None` for range solves; authoritative homology remains
  available. Small complete solves still compare numerical nullity to homology.
- `spectra()` and `eigenpairs()` retain their configured solver behavior; the
  default `eigvalsh` route computes a complete spectrum. Use
  `spectral_mode="full"` for the previous summary path; boundary-only complexes
  also use that path. Its legacy sparse ordinary solver uses SciPy on CPU.
- Harmonic localization uses a complete solve within the dense limits by default.
  Oversized ordinary localization can use a sparse solve with truncated output;
  persistent localization still requires a dense Laplacian and observes the
  allocation guards.

See [technical notes](https://github.com/Sylverity/petls-pytorch/blob/main/docs/technical-notes.md) for filtration semantics, solver
selection, localization behavior, allocation controls, the `.flag` format, and
the full benchmark command set.

## Project

PETLS-PyTorch is a clean-room, PyTorch-native implementation built on the
foundational persistent-Laplacian work of Ben Jones, Guo-Wei Wei, and the
[PETLS project](https://github.com/bdjones13/PETLS). The reference
[documentation](https://www.benjones-math.com/software/PETLS/) and
[paper](https://arxiv.org/abs/2508.11560) define the shared mathematical context;
parity tests cover common workflows without incorporating original source code.

Issues and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/Sylverity/petls-pytorch/blob/main/CONTRIBUTING.md) for
development setup and checks. Licensed under [Apache-2.0](https://github.com/Sylverity/petls-pytorch/blob/main/LICENSE).

## Citation

If you use `petls-pytorch` in research, cite this software and the original
PETLS paper.

```bibtex
@software{marston2026petlspytorch,
  title     = {petls-pytorch: A PyTorch-native implementation of persistent topological Laplacians},
  author    = {Marston, Sumner K.},
  year      = {2026},
  publisher = {Sylverity Research},
  url       = {https://github.com/Sylverity/petls-pytorch}
}

@misc{jones2025petlspersistenttopologicallaplacian,
  title         = {PETLS: PErsistent Topological Laplacian Software},
  author        = {Benjamin Jones and Guo-Wei Wei},
  year          = {2025},
  eprint        = {2508.11560},
  archivePrefix = {arXiv},
  primaryClass  = {math.AT},
  url           = {https://arxiv.org/abs/2508.11560}
}
```
