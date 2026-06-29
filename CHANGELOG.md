# Changelog

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
