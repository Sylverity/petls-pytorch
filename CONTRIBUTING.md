# Contributing to petls-pytorch

Thank you for helping improve `petls-pytorch`. Bug reports, documentation fixes, tests,
benchmarks, and focused implementation changes are all welcome.

## Before opening an issue

- Search existing issues and discussions for related work.
- Use a minimal, reproducible example for bugs.

## Development setup

The project supports CPython 3.10 through 3.14. The quickest setup uses
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Sylverity/petls-pytorch.git
cd petls-pytorch
uv sync --locked --extra dev
```

When changing dependencies in `pyproject.toml`, run `uv lock` and commit the updated
`uv.lock` alongside the manifest. CI rejects an out-of-date lockfile.

An ordinary virtual environment also works:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Running checks

Run the same core checks used by continuous integration:

```bash
uv run --frozen pytest -m "not parity"
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy src/petls_pytorch benchmark
```

The parity suite requires the reference PETLS package:

```bash
uv sync --locked --extra dev --extra parity
uv run --frozen pytest -m parity
```

To measure coverage locally:

```bash
uv run --frozen pytest -m "not parity" --cov=petls_pytorch --cov-report=term-missing
```

## Pull requests

- Keep each pull request focused on one coherent change.
- Add or update tests for behavior changes and regressions.
- Update `README.md`, `CHANGELOG.md`, and public docstrings when user-facing behavior changes.
- Preserve CPU behavior when changing CUDA paths, and test device and dtype handling explicitly.
- Include benchmark evidence for performance claims.
- Do not commit generated distributions, caches, profiles, or benchmark output.

Maintainers may request changes to keep the numerical API, PETLS parity, and sparse-allocation
guarantees stable.

## Release preparation

Release validation uses Python 3.11 or newer. Before publishing, install the
development and reference-test dependencies and run the complete suite:

```bash
uv sync --locked --extra dev --extra parity
uv run --frozen pytest
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy src/petls_pytorch benchmark
uv run --frozen python scripts/check_release.py
```

Build into an empty, dedicated output directory so older distributions cannot
be uploaded accidentally:

```bash
uv run --frozen python -m build --no-isolation --outdir dist/release
uv run --frozen python scripts/check_release.py --dist-dir dist/release
uv run --frozen python -m twine check --strict dist/release/*
uv pip install --no-deps --target /tmp/petls-wheel-check dist/release/*.whl
uv run --no-sync python scripts/smoke_wheel.py /tmp/petls-wheel-check
```

Use a fresh target directory for each wheel check. `--no-sync` keeps the smoke
test from replacing the installed artifact with the editable checkout.
CI runs distribution checks and a wheel smoke test on Python 3.13; the release
workflow repeats them before attestation and publication.

When publishing a version:

1. Keep `pyproject.toml`, `src/petls_pytorch/__init__.py`, `CITATION.cff`, and
   `uv.lock` consistent. Turn its `Unreleased` changelog heading into
   `## VERSION - YYYY-MM-DD`, add the same `date-released` to `CITATION.cff`, and
   update the README installation section to describe the release.
2. Run `uv run --frozen python scripts/check_release.py --tag vVERSION`.
   Merge the complete release change after CI and parity checks pass; the README
   and the implementation it describes must travel together.
3. Create an annotated `vVERSION` tag on the reviewed release commit and push
   that tag. A tag push triggers the release workflow: version checks, the
   Python test matrix, parity tests, distribution checks, build provenance,
   trusted publishing to PyPI, and GitHub release artifacts.
4. Confirm the release workflow succeeds and the version is available on both
   PyPI and GitHub before announcing it. Published PyPI artifacts are immutable;
   corrections need a new version rather than a moved tag.

Generated distributions and raw benchmark results remain outside Git. Preserve
benchmark methodology and representative measurements in the documentation.
