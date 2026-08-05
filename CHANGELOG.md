# Changelog

## 1.1.0 - 2026-08-04

### Added

- Added weighted Gudhi alpha complexes with general power weights, finite-value
  validation, negative vertex births, precision selection, construction
  cutoffs, and optional point labels.
- Retained Gudhi simplex trees, per-dimension simplex identities, filtration
  values, and stable simplex-to-matrix index mappings.
- Added Gudhi-backed `persistence_intervals()`, `betti_numbers_at()`, and
  `persistent_betti()` inspection APIs.
- Added `topology_summary()` for authoritative homology, persistent-Laplacian
  nullity, spectral gaps, tolerances, matrix sizes, and calculation status.
- Added simplex-mapped harmonic representatives through `harmonic_features()`.
- Added `estimate_laplacian()`, configurable dense-allocation guards, and a
  genuinely sparse ordinary Hodge lowest-spectrum path.

### Changed

- Simplified the supported API around native PyTorch concepts. Constructor
  options use snake case, solver selection has one canonical setter, and
  device and dtype configuration is exclusively object-local.
- Filtration enumeration now includes complete zero-dimensional births and
  merges nearly equal scales with a configurable tolerance.
- Device, dtype, eigenvalue tolerances, and allocation policies are now
  object-specific. CPU is the safe default, `device="auto"` is opt-in, and
  Alpha calculations default to `float64`.
- Oversized persistent summaries can return authoritative Gudhi homology with
  explicit `homology_only` status instead of attempting unsafe allocations.
- Benchmark timing now isolates complex construction, Laplacian construction,
  and eigensolving. Device and dtype are passed directly to each object and
  recorded in output rows.
- Benchmark `--dtype` selection defaults to `float32` for continuity with the
  published comparison workload.
- Moved the benchmark-only `tadasets` dependency from the runtime install to
  dedicated `benchmark` and development extras.
- Completed static typing cleanup while retaining Python 3.10 as the minimum
  runtime and analysis target.

### Performance

- Reduced small-matrix overhead with CPU-backed CUDA fallbacks, cached CPU
  mirrors, NumPy incidence assembly, and efficient sparse boundary handling.
- Reduced Schur-complement fallback cost with Hermitian pseudoinverses and
  singular-block trimming.
- Removed redundant eigenvalue sorting and repeated sparse conversions.

### Fixed

- Corrected weighted zero-dimensional filtration bookkeeping throughout
  Laplacian construction and filtration enumeration.
- Guarded the peak Schur-complement intermediate at filtration `b`, not only
  the final persistent-Laplacian output at filtration `a`.
- Made default topology summaries dimension-aware and honored every supported
  sparse eigenvalue-order selection.
- Clarified oversized harmonic localization behavior and corrected original
  PETLS benchmark dtype metadata to report `native`.
- Aligned profiling summaries with object-specific, scale-aware zero
  tolerances.
- Ensured benchmark Alpha complexes honor requested device and dtype settings.

### Removed

- Removed unused C++ solver-name aliases and the no-op up-Laplacian algorithm
  selector; Schur complementation remains the single implemented method.
- Removed camel-case solver aliases, the positional `eigenpairs()` request-list
  overload, and permissive unused keyword handling.
- Removed mutable process-global device and dtype settings. Pass `device=` and
  `dtype=` to each complex instead.
- Removed top-level SciPy re-exports, legacy eigensolver wrappers, and the
  compatibility-only `up_Algorithms` enum. NumPy, SciPy, and the focused
  helpers in `petls_pytorch.core` remain directly available from their owning
  modules.

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
