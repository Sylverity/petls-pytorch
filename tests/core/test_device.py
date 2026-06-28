"""Tests for explicit device placement."""

from __future__ import annotations

import numpy as np
import torch

import petls_pytorch._config as config
from petls_pytorch.core.complex import Complex
from petls_pytorch.core.filtered_boundary import FilteredBoundaryMatrix
from petls_pytorch.variants.alpha import Alpha
from petls_pytorch.variants.rips import Rips


def _assert_complex_on_device(pl, device: str) -> None:
    expected = torch.device(device)
    assert pl.device == expected
    for fbm in pl.filtered_boundaries:
        assert fbm.matrix.device == expected
        assert fbm.domain_filtrations.device == expected
        assert fbm.range_filtrations.device == expected


def test_complex_explicit_device_overrides_global_default(monkeypatch):
    monkeypatch.setattr(config, "_DEFAULT_DEVICE", torch.device("meta"))

    d1 = np.array([[-1]], dtype=np.float64)
    pl = Complex(boundaries=[d1], filtrations=[[0.0], [1.0]], device=torch.device("cpu"))

    _assert_complex_on_device(pl, "cpu")


def test_alpha_explicit_device_overrides_global_default(monkeypatch):
    monkeypatch.setattr(config, "_DEFAULT_DEVICE", torch.device("meta"))

    pl = Alpha(
        points=[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        max_dim=1,
        device=torch.device("cpu"),
    )

    _assert_complex_on_device(pl, "cpu")


def test_rips_explicit_device_overrides_global_default(monkeypatch):
    monkeypatch.setattr(config, "_DEFAULT_DEVICE", torch.device("meta"))

    pl = Rips(
        distances=[[0.0, 1.0], [1.0, 0.0]],
        max_dim=1,
        threshold=2.0,
        device=torch.device("cpu"),
    )

    _assert_complex_on_device(pl, "cpu")


def test_filtered_boundary_transpose_preserves_device(monkeypatch):
    monkeypatch.setattr(config, "_DEFAULT_DEVICE", torch.device("meta"))

    fbm = FilteredBoundaryMatrix(
        matrix=torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [0]]),
            values=torch.tensor([1.0]),
            size=(1, 1),
        ),
        domain_filtrations=torch.tensor([0.0]),
        range_filtrations=torch.tensor([0.0]),
        device=torch.device("cpu"),
    )

    fbm_t = fbm.transpose()

    assert fbm_t.device == torch.device("cpu")
    assert fbm_t.domain_filtrations.device == torch.device("cpu")
    assert fbm_t.range_filtrations.device == torch.device("cpu")
