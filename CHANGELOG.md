# Changelog

## Unreleased

### Highlights

- Improved PETLS-PyTorch benchmark performance on the Windows standard preset
  while preserving API compatibility.
- The standard benchmark now completes every sampled row by default, including
  empty Laplacians and the largest sampled Alpha matrices.
- Final no-skip Windows standard-preset results:
  - PETLS CPU baseline: `8.05 s` trial time, `0.52 s` complex builds,
    `78` completed rows and `0` skipped rows.
  - PETLS-PyTorch CPU: `2.20 s` trial time, `0.57 s` complex builds,
    `78` completed rows and `0` skipped rows.
  - PETLS-PyTorch CUDA: `1.05 s` trial time, `0.66 s` complex builds,
    `78` completed rows and `0` skipped rows.
  - Aggregate speedups against native PETLS are `3.65x` CPU trial time,
    `4.03x` CPU eigensolve time, `7.70x` CUDA trial time, and `10.04x`
    CUDA eigensolve time.

### Changed

- Made benchmark timing fairer and more explicit: package import/device warmup
  is excluded from complex-build timing, CUDA runs synchronize around timed
  regions, and PETLS/PETLS-PyTorch eigensolve timing now solves an already-built
  Laplacian instead of measuring `spectra()` side effects.
- Empty benchmark Laplacians now complete as `0.0 ms` eigensolves instead of
  being skipped, and the standard preset no longer applies a matrix-size cap.
- Switched Gudhi simplex-tree boundary extraction to sparse COO matrices and
  reused the shared extractor for Alpha complexes.

### Performance

- Reduced small-matrix overhead with CPU-backed CUDA fallbacks, CPU mirrors for
  small CUDA boundary matrices, NumPy assembly for small graph Laplacians, and
  dense Gram multiplication for small sparse boundary blocks.
- Reduced Schur-complement fallback cost by using Hermitian pseudoinverses,
  skipping guaranteed-failing Cholesky attempts, trimming zero diagonal rows,
  and returning known-empty Laplacians directly.
- Avoided redundant work by marking filtered COO submatrices as coalesced,
  warming representative backend paths outside timed regions, and removing the
  extra sort after `torch.linalg.eigvalsh()` / `eigh()`.

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
