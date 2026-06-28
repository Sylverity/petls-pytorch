"""
pytest configuration and shared fixtures.

Provides:
  - reference_petls: imported original PETLS package for ground-truth tests.
  - small_complex_fixtures: exact boundary matrices from PETLS tests/core/test_base.py
  - comparison helpers: assert_tensors_close against reference outputs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

# Try to import reference PETLS from sibling directory
_ref_petls = None

try:
    import sys
    from pathlib import Path

    ref_path = Path(__file__).resolve().parent.parent.parent / "PETLS"
    if str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))
    # The original package is importable as 'petls' from its own venv
    # In our venv it may also be installed; prefer the installed one
    import petls as _ref_petls_module

    _ref_petls = _ref_petls_module
except Exception as e:
    _ref_petls = None
    warnings.warn(f"Could not import reference PETLS: {e}")


@pytest.fixture(scope="session")
def reference_petls():
    """Yield the original C++ PETLS package, or skip if unavailable."""
    if _ref_petls is None:
        pytest.skip("Reference PETLS not available")
    return _ref_petls


@pytest.fixture
def small_boundaries():
    """Exact boundary matrices from PETLS tests/core/test_base.py::get_pl()."""
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float32)
    d2 = np.array([[1], [1], [-1]], dtype=np.float32)
    return [d1, d2]


@pytest.fixture
def small_filtrations():
    """Exact filtrations from PETLS tests/core/test_base.py::get_pl()."""
    return [
        [0.0, 1.0, 2.0],  # dim 0 (vertices)
        [3.0, 4.0, 5.0],  # dim 1 (edges)
        [5.0],  # dim 2 (triangle)
    ]


@pytest.fixture
def ref_small_complex(reference_petls, small_boundaries, small_filtrations):
    """Original PETLS Complex on the small test fixture."""
    return reference_petls.Complex(small_boundaries, small_filtrations)


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor | np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    """Assert two tensors/arrays are close within tolerance."""
    if isinstance(expected, np.ndarray):
        expected = torch.from_numpy(expected)
    actual = actual.cpu()
    expected = expected.cpu()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def assert_spectra_close(
    actual: list[float] | torch.Tensor,
    expected: list[float] | np.ndarray,
    atol: float = 1e-4,
) -> None:
    """Assert two spectra (sorted eigenvalue lists) match."""
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy().flatten().tolist()
    if isinstance(expected, np.ndarray):
        expected = expected.flatten().tolist()
    actual = sorted(actual)
    expected = sorted(expected)
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} vs {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) < atol, f"Eigenvalue {i} differs: {a} vs {e} (atol={atol})"
