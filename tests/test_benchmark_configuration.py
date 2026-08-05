"""Regression tests for object-local benchmark device and dtype settings."""

from __future__ import annotations

import pytest
import torch

from benchmark.datasets import generate_dataset
from benchmark.runner import BenchmarkRunner


@pytest.mark.parametrize("complex_type", ["alpha", "rips"])
def test_generate_dataset_passes_device_and_dtype(complex_type):
    data = generate_dataset(
        name="sphere",
        n_points=12,
        complex_type=complex_type,
        max_dim=1,
        num_filtrations=3,
        package="petls-pytorch",
        device="cpu",
        dtype="float64",
        rips_threshold_quantile=0.25 if complex_type == "rips" else None,
    )

    assert data["complex"].device == torch.device("cpu")
    assert data["complex"].dtype == torch.float64
    assert data["metadata"]["device"] == "cpu"
    assert data["metadata"]["dtype"] == "float64"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_alpha_benchmark_honors_cuda_device():
    data = generate_dataset(
        name="sphere",
        n_points=12,
        complex_type="alpha",
        max_dim=1,
        num_filtrations=3,
        package="petls-pytorch",
        device="cuda",
        dtype="float32",
    )

    assert data["complex"].device.type == "cuda"
    assert data["complex"].dtype == torch.float32


def test_benchmark_dtype_validation(tmp_path):
    with pytest.raises(ValueError, match="dtype"):
        BenchmarkRunner(output_dir=str(tmp_path), dtype="float16")
    with pytest.raises(ValueError, match="dtype"):
        generate_dataset(name="sphere", n_points=4, dtype="float16")


def test_benchmark_csv_schema_records_dtype():
    assert "dtype" in BenchmarkRunner._result_fieldnames()
