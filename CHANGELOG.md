# Changelog

## Unreleased

### Added

- Contribution guidance for issues, development, and pull requests.
- Structured issue forms, a pull-request template, and default code owners.
- Dependabot updates for Python and GitHub Actions dependencies.
- CodeQL, dependency review, OpenSSF Scorecard, and coverage enforcement in CI.
- Build-provenance attestations and automatic distribution attachments for GitHub releases.
- Reproducible, lockfile-backed CI and release environments using `uv`.
- A patched Pillow floor for the optional analysis stack.

### Changed

- Expanded package links and corrected the 1.1.2 citation release date.

## 1.1.2 - 2026-08-17

### Changed

- Separated native implementation tests from optional PETLS parity and benchmark suites, with CI coverage for each.
- Made benchmark runs failure-aware and reproducible, with consistent output paths and optional analysis tooling.
- Made profiling device-aware, added synchronized CUDA timing, and tightened core input/query validation.
- Expanded sparse boundary support and reduced the core installation to its required runtime dependencies.
- Reworked the README around GPU-enabled workflows, benchmarks, supported APIs, file formats, and advanced algorithm guidance.

## 1.1.1 - 2026-08-05

### Changed

- Added certified sparse spectral analysis with repeated-nullspace handling and topology cross-checks.
- Strengthened correctness across weighted Alpha, Rips, directed-flag, sheaf, and flipped top-dimensional Laplacian workflows.

## 1.1.0 - 2026-08-04

### Added

- Introduced weighted Alpha complexes, persistence/topology inspection, harmonic localization, allocation guards, and sparse ordinary spectra.

### Changed

- Consolidated the supported API around native PyTorch objects with object-local device, dtype, tolerance, and solver controls.
- Improved benchmark coverage, typing, and Python-version support.

### Removed

- Removed legacy compatibility aliases, global configuration, and unused solver/dependency shims.

## 1.0.2 - 2026-06-28

### Changed

- Standardized package naming, benchmark reporting, and output organization.

### Fixed

- Corrected Rips top-dimensional construction and reduced hidden benchmark setup work.

## 1.0.1 - 2026-06-28

### Changed

- Replaced the `pyflagser` dependency with direct weighted `.flag` parsing and directed clique expansion.
- Expanded CI and documented the supported Python versions and input format.

### Fixed

- Corrected profile export, benchmark entry points, and backend selection.
