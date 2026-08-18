"""Regression tests for object-local benchmark device and dtype settings."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from benchmark.datasets import generate_dataset
from benchmark.runner import BenchmarkResult, BenchmarkRunner, BenchmarkSuiteResult

pytestmark = pytest.mark.benchmark


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


class _FakeComplex:
    top_dim = 0

    def set_eigs_algorithm(self, algorithm):
        self.algorithm = algorithm


def _fake_dataset(complex_obj):
    return {
        "complex": complex_obj,
        "filtrations": [0.0, 1.0],
        "num_unique_filtrations": 2,
    }


def test_benchmark_records_laplacian_failures_and_reports_them(monkeypatch, tmp_path):
    class FailingLaplacian(_FakeComplex):
        def get_L(self, dim, a, b):
            raise RuntimeError("matrix construction failed")

    monkeypatch.setattr(
        "benchmark.datasets.generate_dataset",
        lambda **kwargs: _fake_dataset(FailingLaplacian()),
    )
    runner = BenchmarkRunner(output_dir=str(tmp_path), verbose=False)
    runner._prepare_backend = lambda package: None

    results = runner.run_trial(
        dataset_name="fake",
        n_points=2,
        max_dim=0,
        dims=[0],
        include_final_request=False,
    )

    assert len(results) == 1
    assert results[0].failed
    assert not results[0].skipped
    assert "matrix construction failed" in results[0].failure_reason
    summary = BenchmarkSuiteResult("failed", results).summary()
    assert summary["num_failed"] == 1
    assert summary["num_completed"] == 0
    assert summary["mean_total_ms"] == 0.0
    BenchmarkSuiteResult("failed", results).print_summary()


def test_benchmark_records_eigensolver_failures(monkeypatch, tmp_path):
    class FailingEigensolver(_FakeComplex):
        def get_L(self, dim, a, b):
            return torch.zeros((2, 2))

        def _solve_eigs(self, matrix):
            raise ValueError("eigensolver failed")

    monkeypatch.setattr(
        "benchmark.datasets.generate_dataset",
        lambda **kwargs: _fake_dataset(FailingEigensolver()),
    )
    runner = BenchmarkRunner(output_dir=str(tmp_path), verbose=False)
    runner._prepare_backend = lambda package: None

    results = runner.run_trial(
        dataset_name="fake",
        n_points=2,
        max_dim=0,
        dims=[0],
        include_final_request=False,
    )

    assert len(results) == 1
    assert results[0].failed
    assert results[0].matrix_rows == 2
    assert "eigensolver failed" in results[0].failure_reason


def test_benchmark_summary_excludes_skipped_and_handles_empty_suite():
    common = dict(
        package="fake",
        dataset="fake",
        n_points=2,
        complex_type="alpha",
        max_dim=0,
        dim=0,
        filtration_a=0.0,
        filtration_b=1.0,
    )
    completed = BenchmarkResult(matrix_rows=3, total_time_ms=100.0, **common)
    skipped = BenchmarkResult(matrix_rows=99, skipped=True, **common)
    summary = BenchmarkSuiteResult("mixed", [completed, skipped]).summary()
    assert summary["num_completed"] == 1
    assert summary["num_skipped"] == 1
    assert summary["mean_total_ms"] == 100.0
    assert summary["max_matrix_rows"] == 99
    assert BenchmarkSuiteResult("empty").summary()["num_trials"] == 0
    BenchmarkSuiteResult("empty").print_summary()
