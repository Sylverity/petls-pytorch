# Changelog

## Unreleased

### Changed

- Renamed the import package and benchmark backend selector to `petls_pytorch`
  so the project name, install name, imports, and documentation use one name.

## 1.0.1 - 2026-06-28

### Fixed

- Fixed `Profile.time_to_csv()` after `spectra()` on vertex-only complexes.
- Fixed benchmark documentation and CLI usage by using the real `python -m benchmark` entry point.
- Fixed benchmark dataset generation so the same runner can benchmark either `petls` or `petls_pytorch` via `--package`.
- Removed deprecated setuptools license metadata that emitted build warnings.

### Changed

- Removed the hard dependency on `pyflagser`; `dFlag` now parses weighted `.flag` files directly.
- Replaced brute-force directed flag simplex enumeration with directed clique expansion.
- Documented the supported `.flag` input format.
- Declared and tested CPython support for 3.10, 3.11, 3.12, 3.13, and 3.14.
- Expanded GitHub Actions CI and release test matrices to Python 3.10-3.14.

### Maintenance

- Cleaned repository-wide ruff lint findings.
