"""Regression tests for object-local benchmark device and dtype settings."""

from __future__ import annotations

import sys
from types import SimpleNamespace

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


def test_reference_petls_reports_native_dtype_for_metadata_and_skipped_rows(
    monkeypatch,
    tmp_path,
):
    class FakeBoundary:
        def index_of_filtration(self, use_domain, scale):
            return 0

    class FakeReferenceComplex:
        top_dim = 1
        filtered_boundaries = [FakeBoundary(), FakeBoundary()]

        def __init__(self, **kwargs):
            pass

        def get_all_filtrations(self):
            return [0.0, 1.0]

        def set_eigs_Algorithm(self, algorithm):
            pass

    fake_petls = SimpleNamespace(Alpha=FakeReferenceComplex, Rips=FakeReferenceComplex)
    monkeypatch.setitem(sys.modules, "petls", fake_petls)

    data = generate_dataset(
        name="sphere",
        n_points=4,
        max_dim=0,
        num_filtrations=2,
        package="petls",
        dtype="float64",
    )
    assert data["metadata"]["dtype"] == "native"

    runner = BenchmarkRunner(
        output_dir=str(tmp_path),
        package="petls",
        dtype="float64",
        max_matrix_rows=0,
        verbose=False,
    )
    results = runner.run_trial(
        dataset_name="sphere",
        n_points=4,
        max_dim=0,
        num_filtrations=2,
        dims=[0],
        include_final_request=False,
    )
    assert len(results) == 1
    assert results[0].skipped
    assert results[0].dtype == "native"
