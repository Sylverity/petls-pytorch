"""PETLS-independent tests for core Complex behavior."""

from __future__ import annotations

import pytest
import torch

from petls_pytorch.core.complex import Complex
from tests.core.test_base import get_small_complex

pytestmark = pytest.mark.native


def test_filtration_list_to_spectra_request():
    complex_ = get_small_complex()
    filtrations = [0.0, 3.0, 5.0]
    dimensions = [0, 1, 2]
    requests = complex_.filtration_list_to_spectra_request(filtrations, dimensions)

    expected_len = (len(filtrations) - 1) * len(dimensions) + len(dimensions)
    assert len(requests) == expected_len
    assert requests[0] == (0, 0.0, 3.0)
    assert requests[-1] == (2, 5.0, 5.0)


def test_get_L_beyond_top_dim_is_empty():
    assert get_small_complex().get_L(10, 0.0, 1.0).shape == (0, 0)


def test_eigenvalues_summarize():
    complex_ = get_small_complex()

    betti, least = complex_.eigenvalues_summarize([0.0, 0.0, 0.0, 0.0])
    assert (betti, least) == (4, 0.0)

    betti, least = complex_.eigenvalues_summarize([0.0, 0.0, 1.5])
    assert betti == 2
    assert least == pytest.approx(1.5)

    betti, least = complex_.eigenvalues_summarize([1.0, 2.0, 3.0])
    assert betti == 0
    assert least == pytest.approx(1.0)

    assert complex_.eigenvalues_summarize([]) == (0, 0.0)


def test_small_example_data():
    complex_ = get_small_complex()
    assert complex_.top_dim == 2
    assert complex_.filtered_boundaries[1].shape == (3, 3)
    assert complex_.filtered_boundaries[2].shape == (3, 1)
    assert torch.allclose(
        complex_.filtered_boundaries[1].domain_filtrations.cpu(),
        torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64),
    )
    assert torch.allclose(
        complex_.filtered_boundaries[1].range_filtrations.cpu(),
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"boundaries": []}, {"filtrations": [[0.0]]}],
)
def test_complex_rejects_partial_boundary_data(kwargs):
    with pytest.raises(ValueError, match="provided together"):
        Complex(**kwargs)


def test_spectral_single_query_rejects_missing_filtration_bound():
    complex_ = get_small_complex()
    with pytest.raises(ValueError, match="provided together"):
        complex_.spectra(dim=0, a=0.0)
