"""
pytest configuration and shared fixtures.

Provides:
  - reference_petls: imported original PETLS package for ground-truth tests.
  - small_complex_fixtures: exact boundary matrices from PETLS tests/core/test_base.py
  - comparison helpers: assert_tensors_close against reference outputs.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def reference_petls():
    """Yield the installed original PETLS package, or skip if unavailable."""
    try:
        import petls
    except ImportError:
        pytest.skip("Reference PETLS is not installed; run the parity suite with petls")
    return petls


def pytest_collection_modifyitems(config, items):
    """Keep the default suite independent from the optional PETLS reference."""
    try:
        reference_available = importlib.util.find_spec("petls") is not None
    except (ImportError, ModuleNotFoundError):
        reference_available = False
    if not reference_available:
        skip_reference = pytest.mark.skip(
            reason="Reference PETLS is not installed; run the parity suite with petls"
        )
        for item in items:
            if "parity" in item.keywords:
                item.add_marker(skip_reference)


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
