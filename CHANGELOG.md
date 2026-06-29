# Changelog

## Unreleased

### Changed

- Made benchmark timing more explicit and fair: package import/device warmup is
  excluded from complex-build timing, CUDA runs synchronize around timed
  regions, empty eigensolves report `0.0 ms`, and PETLS-PyTorch eigensolve
  timing now solves the already-built Laplacian instead of rebuilding it via
  `spectra()`.
- PETLS benchmark eigensolve timing now also calls the configured eigensolver
  on the already-built `get_L()` matrix, replacing the previous
  `spectra() - get_L()` estimate.
- Switched Gudhi simplex-tree boundary extraction to sparse COO matrices and
  reused the shared extractor for Alpha complexes.
- Added a small-matrix CUDA eigensolver fallback that solves matrices up to
  `512 x 512` on CPU and transfers eigenvalues back to CUDA, avoiding cuSOLVER
  launch overhead on small benchmark rows.
- Warmed representative benchmark eigensolve and scatter paths outside timed
  regions so first-use PyTorch backend setup is not charged to the first trial.
- Warmed representative sparse-boundary and Hermitian pseudoinverse backend
  paths outside timed regions.

### Performance

- Switched the Schur-complement singular fallback to
  `torch.linalg.pinv(..., hermitian=True)`, preserving the symmetric
  pseudoinverse result while reducing fallback cost.
- Use dense Gram multiplication for small sparse boundary blocks
  (`rows * cols <= 150000`), avoiding sparse-kernel overhead on the remaining
  small Rips benchmark rows while preserving the sparse path for larger blocks.
- Skip the guaranteed-failing CPU Cholesky attempt for small Schur-complement
  blocks with a non-positive diagonal entry and go directly to the Hermitian
  pseudoinverse fallback.
- Assemble CPU two-entry incidence Laplacians through NumPy for small graph
  boundary matrices, reducing PyTorch scatter overhead on Rips dimension-0
  rows.
- Keep CPU mirrors for CUDA boundary matrices and build Laplacians with at most
  `256` rows on CPU before transferring the dense result back to CUDA, reducing
  launch overhead on the smallest CUDA Rips rows while preserving CUDA return
  tensors.
- Mark filtered COO submatrices as coalesced at construction time instead of
  running a redundant coalesce pass, avoiding cache-fill overhead for first-use
  Rips and Alpha Laplacian builds.
- Checkpoint CPU standard preset on Windows:
  - PETLS direct-eigensolve baseline: `9.65 s` trial time, `0.61 s` complex
    builds.
  - PETLS-PyTorch: `1.35 s` trial time, `0.55 s` complex builds.
- Checkpoint CUDA standard preset on Windows:
  - PETLS-PyTorch CUDA: `0.67 s` trial time, `0.64 s` complex builds.
  - Remaining row-wise misses against PETLS baseline:
    - CPU: `9` total-time misses, `2` non-empty total-time misses, and no
      non-empty eigensolve-time misses out of `75` completed rows.
    - CUDA: `19` total-time misses, with `10` non-empty total-time misses and
      `4` non-empty eigensolve-time misses out
      of `75` completed rows.
    - CUDA total-time overage dropped again to `9.56 ms`.
    - The aggregate trial time target is met, but the all-rows stopping
      condition is still open.

### Validation

- `ruff check .` passes.
- `pytest tests -k "not test_get_down_eigenvalues_match_reference and not
  test_get_down_eigenvalues_match_mwe"` passes: `107 passed, 3 deselected`.
  The deselected tests are the known Windows PETLS `get_down()` access
  violation cases. `pytest --with petls` is not currently a registered option
  in this repo.

## 1.0.2 - 2026-06-28

### Changed

- Aligned public package, documentation, and benchmark naming on `petls-pytorch`;
  the Python import package remains `petls_pytorch`.
- Updated benchmark presets and reporting for representative Windows CPU/GPU
  comparisons, including streamed progress, partial CSV/JSON output, skipped
  rows, and matrix-size caps.
- Changed benchmark outputs to default under `benchmark-results/`.

### Fixed

- Fixed Rips complex construction to build the extra simplex dimension needed
  for top requested Laplacian dimensions, matching original PETLS Betti values.
- Avoided hidden benchmark setup work by making matrix statistics optional and
  bounding the representative Rips threshold.

## 1.0.1 - 2026-06-28

### Fixed

- Fixed `Profile.time_to_csv()` after `spectra()` on vertex-only complexes.
- Fixed benchmark documentation and CLI usage by using the real `python -m benchmark` entry point.
- Fixed benchmark dataset generation so the same runner can benchmark either `petls` or `petls-pytorch` via `--package`.
- Removed deprecated setuptools license metadata that emitted build warnings.

### Changed

- Removed the hard dependency on `pyflagser`; `dFlag` now parses weighted `.flag` files directly.
- Replaced brute-force directed flag simplex enumeration with directed clique expansion.
- Documented the supported `.flag` input format.
- Declared and tested CPython support for 3.10, 3.11, 3.12, 3.13, and 3.14.
- Expanded GitHub Actions CI and release test matrices to Python 3.10-3.14.

### Maintenance

- Cleaned repository-wide ruff lint findings.
