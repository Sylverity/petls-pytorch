# Changelog

## 1.2.0 - 2026-10-08

### Added

- `positive_spectrum()` returns leading real positive modes with per-mode
  residuals, quality labels, and construction/solve diagnostics on CPU or CUDA.
- `harmonic_participation()` provides basis-invariant simplex and point scores,
  optional labels, and explicit completeness flags.
- Repeated CPU/CUDA benchmark comparisons record hardware and source provenance
  and validate every requested mode against a full CPU spectrum.
- Reproducible torus and molecular-crystal demos, with updated documentation.
- Distribution checks validate release metadata, README rendering, source-archive
  contents, and the installed wheel before publication.

### Changed

- Gudhi-backed `topology_summary()` defaults to eight leading positive modes.
  Use `spectral_mode="full"` for the previous summary route. Explicit spectrum
  and harmonic-localization APIs retain their configured behavior.
- Large partial solves operate in the Laplacian range, with a working block
  independent of Betti multiplicity. Numerical nullity remains unknown when the
  kernel was not computed; the configured residual tolerance controls stopping.
- Complete device solves handle small matrices using measured CPU/CUDA size
  cutoffs. Tight allocation guards retain range iteration when its workspace fits.
- Alpha boundaries are extracted once, preserving their orientation convention.
- README benchmarks now measure the current default and full-spectrum solvers
  in float64, including homology and result processing on both timing paths.

### Fixed

- Real positive-mode selection and rank no longer depend on modulo-11 Betti
  counts. Coefficient-field disagreement is reported separately from solver
  quality, so torsion does not skip modes or invalidate correct real spectra.
- Harmonic-localization completeness describes the recovered real kernel,
  independently of modular Betti counts, including unknown and truncated states.
- Persistent Schur corrections reject numerically singular Cholesky pivots,
  including CUDA factorizations that report success on a singular Gram block.
- `nonzero_spectra()` preserves small positive eigenvalues and multiplicities
  using the configured zero tolerance.
- Replacing boundaries clears stale topology and geometric mappings; invalid
  updates preserve the original complex.
- Dense allocation guards include rectangular boundary matrices.
- Empty and single-request spectrum batches retain consistent results and exports.
- Source distributions include the full test suite, fixtures, benchmark tools,
  and release documentation. README links and images work on GitHub and PyPI.

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
