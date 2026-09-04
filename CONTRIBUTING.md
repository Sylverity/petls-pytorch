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
uv sync --frozen --extra dev
```

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
uv sync --frozen --extra dev --extra parity
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
