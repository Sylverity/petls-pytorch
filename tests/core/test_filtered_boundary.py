"""
Tests for FilteredBoundaryMatrix — the foundational data structure.

Validates against the original C++ FilteredBoundaryMatrix behavior
by constructing equivalent objects and comparing submatrix extraction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from petls_torch.core.filtered_boundary import FilteredBoundaryMatrix
from tests.conftest import assert_tensors_close


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_construction_from_numpy():
    """Basic construction with numpy dense input."""
    B = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.float32)
    domain_f = [0.0, 1.0, 2.0]
    range_f = [0.0, 1.0]

    fbm = FilteredBoundaryMatrix(
        matrix=torch.from_numpy(B).to_sparse_coo(),
        domain_filtrations=torch.tensor(domain_f, dtype=torch.float64),
        range_filtrations=torch.tensor(range_f, dtype=torch.float64),
    )

    assert fbm.shape == (2, 3)
    assert fbm.num_rows == 2
    assert fbm.num_cols == 3
    assert fbm.matrix._nnz() == 4  # four nonzeros


def test_filtration_sorting_enforced():
    """Unsorted filtrations should raise ValueError."""
    B = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        FilteredBoundaryMatrix(
            matrix=B,
            domain_filtrations=torch.tensor([2.0, 1.0], dtype=torch.float64),
            range_filtrations=torch.tensor([0.0, 1.0], dtype=torch.float64),
        )


# ---------------------------------------------------------------------------
# index_of_filtration tests
# ---------------------------------------------------------------------------


def test_index_of_filtration_basic():
    """Match C++ behavior: largest index where filtration <= a."""
    B = torch.sparse_coo_tensor(
        indices=torch.empty((2, 0), dtype=torch.long),
        values=torch.empty(0),
        size=(4, 5),
    )
    fbm = FilteredBoundaryMatrix(
        matrix=B,
        domain_filtrations=torch.tensor([0.0, 1.0, 3.0, 5.0, 7.0], dtype=torch.float64),
        range_filtrations=torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float64),
    )

    # Domain tests
    assert fbm.index_of_filtration(use_domain=True, a=-1.0) == -1
    assert fbm.index_of_filtration(use_domain=True, a=0.0) == 0
    assert fbm.index_of_filtration(use_domain=True, a=1.0) == 1
    assert fbm.index_of_filtration(use_domain=True, a=2.0) == 1  # between 1 and 3
    assert fbm.index_of_filtration(use_domain=True, a=3.0) == 2
    assert fbm.index_of_filtration(use_domain=True, a=7.0) == 4
    assert fbm.index_of_filtration(use_domain=True, a=100.0) == 4

    # Range tests
    assert fbm.index_of_filtration(use_domain=False, a=-1.0) == -1
    assert fbm.index_of_filtration(use_domain=False, a=0.0) == 0
    assert fbm.index_of_filtration(use_domain=False, a=2.0) == 1
    assert fbm.index_of_filtration(use_domain=False, a=6.0) == 3


# ---------------------------------------------------------------------------
# submatrix_at_filtration tests
# ---------------------------------------------------------------------------


def test_submatrix_at_filtration_exact():
    """
    Extract submatrix and verify shape + values against reference logic.

    We construct a boundary matrix where the top-left block at filtration=1.0
    is easy to predict.
    """
    # 4x3 matrix, domain filts [0,1,2], range filts [0,1,2,3]
    indices = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]])  # row, col
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    B = torch.sparse_coo_tensor(indices=indices, values=values, size=(4, 3))

    fbm = FilteredBoundaryMatrix(
        matrix=B,
        domain_filtrations=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        range_filtrations=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64),
    )

    # At a=1.0: cols 0..1, rows 0..1 -> submatrix is 2x2
    sub = fbm.submatrix_at_filtration(1.0)
    assert sub.shape == (2, 2)

    dense = sub.to_dense()
    expected = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    assert_tensors_close(dense, expected)

    # At a=2.0: cols 0..2, rows 0..2 -> submatrix is 3x3
    sub = fbm.submatrix_at_filtration(2.0)
    assert sub.shape == (3, 3)

    dense = sub.to_dense()
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    assert_tensors_close(dense, expected)


def test_submatrix_at_filtration_empty():
    """Submatrix before any filtrations should be empty."""
    B = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    )
    fbm = FilteredBoundaryMatrix(
        matrix=B,
        domain_filtrations=torch.tensor([1.0, 2.0], dtype=torch.float64),
        range_filtrations=torch.tensor([1.0, 2.0], dtype=torch.float64),
    )

    sub = fbm.submatrix_at_filtration(0.5)
    assert sub.shape == (0, 0)
    assert sub._nnz() == 0


# ---------------------------------------------------------------------------
# Transpose tests
# ---------------------------------------------------------------------------


def test_transpose():
    """Transpose swaps domain/range and filtrations."""
    B = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1]]),
        values=torch.tensor([1.0, 2.0]),
        size=(2, 3),
    )
    fbm = FilteredBoundaryMatrix(
        matrix=B,
        domain_filtrations=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        range_filtrations=torch.tensor([0.0, 1.0], dtype=torch.float64),
    )

    fbm_t = fbm.transpose()
    assert fbm_t.shape == (3, 2)
    assert torch.allclose(fbm_t.domain_filtrations, fbm.range_filtrations)
    assert torch.allclose(fbm_t.range_filtrations, fbm.domain_filtrations)


# ---------------------------------------------------------------------------
# Device tests
# ---------------------------------------------------------------------------


def test_device_placement():
    """All tensors should reside on the configured device."""
    B = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [0]]),
        values=torch.tensor([1.0]),
        size=(2, 2),
    )
    fbm = FilteredBoundaryMatrix(
        matrix=B,
        domain_filtrations=torch.tensor([0.0, 1.0], dtype=torch.float64),
        range_filtrations=torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    assert fbm.matrix.device == fbm.device
    assert fbm.domain_filtrations.device == fbm.device
    assert fbm.range_filtrations.device == fbm.device
