"""
PETLS Benchmark Suite

Multi-scale benchmarking for Persistent Topological Laplacians.
Stress-tests both matrix construction (Schur complement) and eigendecomposition.
"""

from .datasets import generate_dataset, list_datasets
from .runner import BenchmarkRunner, run_single_benchmark

__all__ = ["generate_dataset", "list_datasets", "BenchmarkRunner", "run_single_benchmark"]
