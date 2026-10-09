# CPU and CUDA benchmarks — October 7, 2026

These measurements cover the current 1.2.0 development implementation after
updating the complete/iterative solver crossover. The README reports median
pipeline time; the table below also separates query time and its observed range.

## Machine and method

- CPU: 13th Gen Intel(R) Core(TM) i7-13700K; four intra-op threads and one inter-op thread.
- GPU: NVIDIA GeForce RTX 4070 Ti, 12 GB; CUDA 13.0.
- Platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39; Python 3.13.13.
- Packages: petls-pytorch 1.2.0, torch 2.13.0, numpy 2.5.0, scipy 1.18.0, gudhi 3.12.0, tadasets 0.2.2.
- Float64 throughout; seed 42; OMP, MKL, and OpenBLAS thread limits set to four.
- One complete warmup suite per solver/device, then three measured repeats.
  Variant order rotates between repeats; CPU and GPU benchmarks run sequentially.
- CUDA is synchronized at timing boundaries. Query time includes operator
  construction, eigensolving, homology, and result processing. Pipeline time adds
  each dataset/complex build once. Backend preparation, warmups, validation, and
  result-file writes are excluded. Failed-attempt time is retained in comparisons.
- CPU full spectra are the reference. Comparisons check request identity,
  ordered positive eigenvalues including multiplicities, homology, and available
  real nullities. They reject missing or unavailable modes. Tolerances are
  `rtol=1e-3`, `atol=1e-7`; a correct empty positive spectrum is valid.
- These are measurements on one workstation, not guarantees for other hardware.

## Workloads

Both presets query dimensions 0, 1, and 2. Successive sampled scales form the
persistent pairs; selected final scales also contribute ordinary queries.

| Preset | Dataset | Points | Construction | Sampled scales |
|---|---|---:|---|---:|
| `standard` | Torus | 300 | Alpha | 6 |
| `standard` | Sphere | 250 | Alpha | 6 |
| `standard` | Swiss roll | 300 | Alpha | 6 |
| `standard` | Projected Klein bottle | 250 | Alpha | 6 |
| `standard` | Torus | 90 | Rips, distance quantile 0.12 cutoff | 6 |
| `quick` | Torus | 500 | Alpha | 8 |
| `quick` | Sphere | 300 | Alpha | 8 |

`standard` has 75 persistent and 3 ordinary queries, with matrices up to 2,399
rows. `quick` has 42 persistent and 6 ordinary queries, reaching 5,990 rows.
Despite its name, `quick` contains larger matrices than `standard` because it
uses larger point clouds and includes the final Alpha filtration.

## Results

| Preset | Solver | Device | Median query time | Query min–max | Median pipeline | Validated requests |
|---|---|---|---:|---:|---:|---:|
| `standard` | `partial` | CPU | 2.833 s | 2.828–2.873 s | 3.021 s | 234/234 |
| `standard` | `partial` | CUDA | 2.192 s | 2.006–2.274 s | 2.377 s | 234/234 |
| `standard` | `eigvalsh` | CPU | 2.455 s | 2.422–2.537 s | 2.611 s | 234/234 |
| `standard` | `eigvalsh` | CUDA | 1.826 s | 1.685–1.900 s | 1.987 s | 234/234 |
| `quick` | `partial` | CPU | 5.023 s | 4.961–5.743 s | 5.112 s | 144/144 |
| `quick` | `partial` | CUDA | 5.871 s | 5.848–5.979 s | 6.032 s | 144/144 |
| `quick` | `eigvalsh` | CPU | 9.672 s | 9.617–9.861 s | 9.763 s | 144/144 |
| `quick` | `eigvalsh` | CUDA | 6.275 s | 6.204–6.421 s | 6.387 s | 144/144 |

All 1,512 measured requests completed with zero failed, skipped, or
unavailable-mode requests. The largest observed relative eigenvalue difference
from the CPU full-spectrum reference was `2.49e-07`. CPU reference rows
are included in the request count; they are not an independent accuracy check.

The default API asks for up to eight positive modes and checks their residuals.
The full path returns every eigenvalue. Full CUDA spectra use CPU
eigendecomposition for matrices of 2–512 rows; the default partial API uses
PyTorch eigensolvers on the selected device for both complete and iterative solves.
In `standard`, the default CPU solver used range iteration for 7 of 78 queries;
CUDA used complete solves or exact-zero handling. In `quick`, range iteration
handled 14 of 48 CPU queries and 7 of 48 CUDA queries. Thus the second workload
also validates the large-problem path.

## Changes driven by the measurements

The old 64-row crossover sent many small problems through numerous iterations,
where CUDA launch and synchronization overhead outweighed the saved arithmetic.
Crossover trials on the standard matrices favored complete solves up to roughly
1,500 CPU rows and through the largest tested CUDA matrices (about 2,400 rows).
The implementation now uses limits of 1,536 CPU rows and 2,560 CUDA rows.
Tighter row or workspace guards force range iteration when its workspace fits;
the change does not remove allocation guards or introduce a CPU spectral fallback.

With the same improved timing harness, seed, dtype, and thread settings:

| Default solver, standard workload | Before tuning, median queries | After tuning, median queries | Speedup |
|---|---:|---:|---:|
| CPU | 5.442 s | 2.833 s | 1.92× |
| CUDA | 18.056 s | 2.192 s | 8.24× |

The comparison harness now preserves failed-attempt time, uses equivalent query
boundaries for partial and full methods, records actual eigensolver devices,
and checks all requested modes rather than accepting a block solely because its
first mode is usable. It records both modular homology and real spectral nullity.

## Reproduction and provenance

From the repository checkout:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --extra benchmark python -m benchmark.compare --preset standard \
  --threads 4 --warmups 1 --repeats 3 \
  --output-dir benchmark-results/standard-comparison

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  uv run --extra benchmark python -m benchmark.compare --preset quick \
  --threads 4 --warmups 1 --repeats 3 \
  --output-dir benchmark-results/quick-comparison
```

Raw measurements from this session are retained locally in
`benchmark-results/2026-10-07-standard-final/` and
`benchmark-results/2026-10-07-quick/`. The before-tuning comparison is in
`benchmark-results/2026-10-07-standard/`. Generated raw outputs are ignored by Git.
Each directory contains CSV requests, structured JSON requests for measured
repeats, and `comparison.json` with timings, validation details, exact package
versions, source-file hashes, Git state, and hardware information.

The measured source tree is based on commit
`c78f8fd7809815a12fbb73d036e002398bc6e62f` plus this benchmark/dispatch change.
Both final workloads used this aggregate SHA-256 of the source, benchmark
Python files, and lockfile (documentation and tests are excluded):

`60b4ddfa863d1e53d65111e682ff9b8f27bd1cc9bc335aa87f4d50733773d545`

Subsequent release preparation added Twine to the development dependencies and
updated the lockfile. The measured library and benchmark Python sources, and
the runtime package versions listed above, remain unchanged.
