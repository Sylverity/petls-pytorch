"""
Benchmark runner for PETLS.

Measures wall-clock time for:
  1. Laplacian matrix construction   (get_L)
  2. Full eigendecomposition         (spectra)
  3. End-to-end pipeline             (complex build + spectra)

Supports CPU and GPU backends (GPU via future PyTorch rewrite).
"""

import time
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable
from pathlib import Path
import numpy as np


@dataclass
class BenchmarkResult:
    """Single measurement for one (dataset, n_points, dim, a, b) trial."""

    package: str
    dataset: str
    n_points: int
    complex_type: str
    max_dim: int
    dim: int
    filtration_a: float
    filtration_b: float
    matrix_rows: int
    build_time_ms: float = 0.0  # time to build Laplacian matrix
    eigs_time_ms: float = 0.0  # time for eigendecomposition
    total_time_ms: float = 0.0  # build + eigs
    eigenvalue_count: int = 0
    betti: int = 0
    least_nonzero: float = 0.0
    algorithm: str = "eigvalsh"
    device: str = "cpu"
    seed: int = 42
    config_index: int = 0
    request_index: int = 0
    complex_build_time_ms: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results for a full benchmark run."""

    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_csv(self, path: str):
        if not self.results:
            return
        keys = list(asdict(self.results[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))

    def summary(self) -> Dict:
        if not self.results:
            return {}
        total_times = [r.total_time_ms for r in self.results]
        build_times = [r.build_time_ms for r in self.results]
        eigs_times = [r.eigs_time_ms for r in self.results]
        sizes = [r.matrix_rows for r in self.results]
        completed = [r for r in self.results if not r.skipped]
        skipped = [r for r in self.results if r.skipped]
        config_builds = {}
        for r in self.results:
            key = (r.config_index, r.dataset, r.n_points, r.complex_type, r.max_dim, r.seed)
            config_builds.setdefault(key, r.complex_build_time_ms)

        return {
            "suite_name": self.suite_name,
            "num_trials": len(self.results),
            "num_completed": len(completed),
            "num_skipped": len(skipped),
            "total_time_sec": sum(total_times) / 1000.0,
            "complex_build_time_sec": sum(config_builds.values()) / 1000.0,
            "mean_total_ms": np.mean(total_times),
            "median_total_ms": np.median(total_times),
            "max_total_ms": max(total_times),
            "mean_build_ms": np.mean(build_times),
            "mean_eigs_ms": np.mean(eigs_times),
            "max_matrix_rows": max(sizes),
            "mean_matrix_rows": int(np.mean(sizes)),
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "=" * 60)
        print(f"  Benchmark Suite: {s['suite_name']}")
        print(f"  Trials:          {s['num_trials']}")
        print(f"  Completed:       {s['num_completed']}")
        print(f"  Skipped:         {s['num_skipped']}")
        print(f"  Trial time:      {s['total_time_sec']:.2f} s")
        print(f"  Complex builds:  {s['complex_build_time_sec']:.2f} s")
        print(f"  Mean trial:      {s['mean_total_ms']:.1f} ms")
        print(f"  Median trial:    {s['median_total_ms']:.1f} ms")
        print(f"  Slowest trial:   {s['max_total_ms']:.1f} ms")
        print(f"  Max matrix size: {s['max_matrix_rows']} x {s['max_matrix_rows']}")
        print(f"  Mean matrix size:{s['mean_matrix_rows']} x {s['mean_matrix_rows']}")
        print("=" * 60 + "\n")


class BenchmarkRunner:
    """Runs PETLS benchmarks with configurable intensity."""

    def __init__(
        self,
        output_dir: str = "./benchmark_results",
        algorithm: str = "eigvalsh",
        device: str = "cpu",
        package: str = "petls-pytorch",
        verbose: bool = True,
        max_matrix_rows: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm = algorithm
        self.device = device
        self.package = package.lower()
        self.verbose = verbose
        self.max_matrix_rows = max_matrix_rows

    @staticmethod
    def _result_fieldnames() -> list[str]:
        return list(asdict(BenchmarkResult(package="", dataset="", n_points=0, complex_type="", max_dim=0, dim=0, filtration_a=0.0, filtration_b=0.0, matrix_rows=0)).keys())

    def _print(self, message: str = "") -> None:
        if self.verbose:
            print(message, flush=True)

    def _prepare_backend(self, package: str) -> None:
        """Import and configure package code outside timed benchmark regions."""
        if package == "petls-pytorch":
            import petls_pytorch
            import torch
            from petls_pytorch.core.eigenvalues import solve_eigenvalues

            petls_pytorch.set_device(self.device)
            device = torch.device(self.device)
            dense = torch.eye(4, device=device)
            sparse = dense.to_sparse_coo()
            _ = sparse.to_dense() @ dense
            row_index = torch.arange(600, device=device) % 300
            col_index = torch.arange(600, device=device)
            sparse_boundary = torch.sparse_coo_tensor(
                torch.stack((row_index, col_index)),
                torch.ones(600, device=device),
                size=(300, 600),
                device=device,
            ).coalesce()
            boundary_indices = sparse_boundary.indices()
            boundary_values = sparse_boundary.values()
            boundary_mask = (boundary_indices[0] < 300) & (boundary_indices[1] < 525)
            sparse_submatrix = torch.sparse_coo_tensor(
                boundary_indices[:, boundary_mask],
                boundary_values[boundary_mask],
                size=(300, 525),
                device=device,
            ).coalesce()
            sparse_dense = sparse_submatrix.to_dense()
            _ = sparse_dense @ sparse_dense.T
            singular = torch.eye(256, device=device)
            singular[-1] = 0
            singular[:, -1] = 0
            _ = torch.linalg.pinv(singular, hermitian=True) @ torch.ones(256, 16, device=device)
            _ = torch.linalg.eigvalsh(dense)
            _ = torch.linalg.eigvalsh(torch.eye(300))
            scatter_target = torch.zeros(512, 512, device=device)
            scatter_index = torch.arange(512, device=device)
            scatter_target.index_put_(
                (scatter_index, scatter_index),
                torch.ones(512, device=device),
                accumulate=True,
            )
            if self.device.startswith("cuda"):
                if torch.cuda.is_available():
                    torch.empty(1, device=self.device)
                    _ = solve_eigenvalues(torch.eye(300, device=device), self.algorithm)
                    torch.cuda.synchronize()
        elif package == "petls":
            import petls  # noqa: F401

    def _synchronize(self, package: str) -> None:
        if package == "petls-pytorch" and self.device.startswith("cuda"):
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def _solve_eigs_from_matrix(self, complex_obj, matrix):
        """Time only the eigensolver for an already-built Laplacian matrix."""
        if hasattr(complex_obj, "_solve_eigs"):
            return complex_obj._solve_eigs(matrix)
        return complex_obj.eigs_Algorithm(matrix)

    def _estimate_matrix_rows(self, complex_obj, dim: int, a: float) -> Optional[int]:
        if not hasattr(complex_obj, "filtered_boundaries"):
            return None
        if dim == 0:
            if len(complex_obj.filtered_boundaries) <= 1:
                return 0
            return complex_obj.filtered_boundaries[1].index_of_filtration(False, a) + 1
        if dim > complex_obj.top_dim:
            return 0
        return complex_obj.filtered_boundaries[dim].index_of_filtration(True, a) + 1

    def _write_partial_result(self, path: Path, result: BenchmarkResult) -> None:
        exists = path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._result_fieldnames())
            if not exists:
                writer.writeheader()
            writer.writerow(asdict(result))

    def run_trial(
        self,
        dataset_name: str,
        n_points: int,
        complex_type: str = "alpha",
        max_dim: int = 3,
        num_filtrations: int = 20,
        filtration_mode: str = "quantile",
        seed: int = 42,
        dims: Optional[List[int]] = None,
        filtration_pairs: Optional[List[tuple]] = None,
        package: Optional[str] = None,
        config_index: int = 0,
        max_matrix_rows: Optional[int] = None,
        include_final_request: bool = True,
        compute_matrix_stats: bool = False,
        rips_threshold_quantile: Optional[float] = None,
        on_result: Optional[Callable[[BenchmarkResult], None]] = None,
    ) -> List[BenchmarkResult]:
        """
        Run a single benchmark trial for one dataset at one scale.

        Parameters
        ----------
        dataset_name : str
            Name of dataset (torus, sphere, swiss_roll, klein_bottle).
        n_points : int
            Number of points in the point cloud.
        complex_type : str
            'alpha' or 'rips'.
        max_dim : int
            Maximum simplicial dimension.
        num_filtrations : int
            How many filtration values to sample.
        filtration_mode : str
            'quantile', 'log', or 'early'.
        seed : int
            Random seed.
        dims : list of int, optional
            Which dimensions to benchmark. Defaults to all [0..max_dim].
        filtration_pairs : list of (a,b), optional
            Override the default successive-pair generation.

        Returns
        -------
        List[BenchmarkResult]
        """
        from .datasets import generate_dataset

        package = (self.package if package is None else package).lower()
        self._prepare_backend(package)

        self._print(
            f"\n[Benchmark] #{config_index} {dataset_name} | n={n_points} | {complex_type} | "
            f"max_dim={max_dim} | package={package} | device={self.device}"
        )
        self._print("-" * 60)

        # Build dataset & complex
        t0 = time.perf_counter()
        ds = generate_dataset(
            name=dataset_name,
            n_points=n_points,
            complex_type=complex_type,
            max_dim=max_dim,
            num_filtrations=num_filtrations,
            filtration_mode=filtration_mode,
            seed=seed,
            package=package,
            device=self.device if package == "petls-pytorch" else None,
            compute_matrix_stats=compute_matrix_stats,
            rips_threshold_quantile=rips_threshold_quantile,
        )
        t_build_complex = (time.perf_counter() - t0) * 1000
        self._print(f"  Complex build:       {t_build_complex:.1f} ms")
        self._print(f"  Unique filtrations:  {ds['num_unique_filtrations']}")
        self._print(f"  Sampled filtrations: {len(ds['filtrations'])}")

        complex_obj = ds["complex"]
        complex_obj.set_eigs_Algorithm(self.algorithm)
        filtrations = ds["filtrations"]

        if dims is None:
            dims = list(range(max_dim + 1))

        # Generate (dim, a, b) requests
        if filtration_pairs is None:
            requests = []
            for i in range(len(filtrations) - 1):
                a, b = filtrations[i], filtrations[i + 1]
                for dim in dims:
                    requests.append((dim, a, b))
            if include_final_request:
                for dim in dims:
                    requests.append((dim, filtrations[-1], filtrations[-1]))
        else:
            requests = []
            for a, b in filtration_pairs:
                for dim in dims:
                    requests.append((dim, a, b))

        results: List[BenchmarkResult] = []

        row_cap = max_matrix_rows if max_matrix_rows is not None else self.max_matrix_rows
        for request_index, (dim, a, b) in enumerate(requests, start=1):
            rows_estimate = self._estimate_matrix_rows(complex_obj, dim, a)
            if row_cap is not None and rows_estimate is not None and rows_estimate > row_cap:
                result = BenchmarkResult(
                    package=package,
                    dataset=dataset_name,
                    n_points=n_points,
                    complex_type=complex_type,
                    max_dim=max_dim,
                    dim=dim,
                    filtration_a=round(a, 6),
                    filtration_b=round(b, 6),
                    matrix_rows=rows_estimate,
                    algorithm=self.algorithm,
                    device=self.device,
                    seed=seed,
                    config_index=config_index,
                    request_index=request_index,
                    complex_build_time_ms=t_build_complex,
                    skipped=True,
                    skip_reason=f"matrix_rows>{row_cap}",
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
                self._print(
                    f"  [{request_index:02d}/{len(requests):02d}] dim={dim} "
                    f"a={a:.4f} b={b:.4f} | size~{rows_estimate:5d} | SKIP {result.skip_reason}"
                )
                continue

            # Matrix construction time
            self._synchronize(package)
            t0 = time.perf_counter()
            try:
                L = complex_obj.get_L(dim, a, b)
            except Exception as e:
                self._print(f"    ERROR get_L(dim={dim}, a={a:.4f}, b={b:.4f}): {e}")
                continue
            self._synchronize(package)
            t_build = (time.perf_counter() - t0) * 1000

            rows = int(L.shape[0])
            if row_cap is not None and rows > row_cap:
                result = BenchmarkResult(
                    package=package,
                    dataset=dataset_name,
                    n_points=n_points,
                    complex_type=complex_type,
                    max_dim=max_dim,
                    dim=dim,
                    filtration_a=round(a, 6),
                    filtration_b=round(b, 6),
                    matrix_rows=rows,
                    build_time_ms=t_build,
                    total_time_ms=t_build,
                    algorithm=self.algorithm,
                    device=self.device,
                    seed=seed,
                    config_index=config_index,
                    request_index=request_index,
                    complex_build_time_ms=t_build_complex,
                    skipped=True,
                    skip_reason=f"matrix_rows>{row_cap}",
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
                self._print(
                    f"  [{request_index:02d}/{len(requests):02d}] dim={dim} "
                    f"a={a:.4f} b={b:.4f} | size={rows:5d} | build={t_build:8.1f}ms | "
                    f"SKIP {result.skip_reason}"
                )
                continue

            # Eigenvalue time
            try:
                if rows == 0:
                    eigs = []
                    t_eigs = 0.0
                else:
                    self._synchronize(package)
                    t0 = time.perf_counter()
                    eigs = self._solve_eigs_from_matrix(complex_obj, L)
                    self._synchronize(package)
                    t_eigs = (time.perf_counter() - t0) * 1000
            except Exception as e:
                self._print(f"    ERROR eigs(dim={dim}, a={a:.4f}, b={b:.4f}): {e}")
                continue

            betti, lam = complex_obj.eigenvalues_summarize(eigs)

            result = BenchmarkResult(
                package=package,
                dataset=dataset_name,
                n_points=n_points,
                complex_type=complex_type,
                max_dim=max_dim,
                dim=dim,
                filtration_a=round(a, 6),
                filtration_b=round(b, 6),
                matrix_rows=rows,
                build_time_ms=t_build,
                eigs_time_ms=t_eigs,
                total_time_ms=t_build + t_eigs,
                eigenvalue_count=len(eigs),
                betti=int(betti),
                least_nonzero=float(lam),
                algorithm=self.algorithm,
                device=self.device,
                seed=seed,
                config_index=config_index,
                request_index=request_index,
                complex_build_time_ms=t_build_complex,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)

            self._print(
                f"  [{request_index:02d}/{len(requests):02d}] dim={dim} a={a:.4f} b={b:.4f} | "
                f"size={rows:5d} | build={t_build:8.1f}ms | eigs={t_eigs:8.1f}ms | "
                f"betti={betti}"
            )

        return results

    def run_suite(
        self,
        name: str,
        configs: List[Dict],
    ) -> BenchmarkSuiteResult:
        """
        Run a full suite of benchmark configurations.

        Parameters
        ----------
        name : str
            Suite name (e.g., 'alpha_intensive').
        configs : list of dict
            Each dict is kwargs for run_trial().
        """
        suite = BenchmarkSuiteResult(suite_name=name)
        suite.start_time = time.perf_counter()
        csv_path = self.output_dir / f"{name}.csv"
        summary_path = self.output_dir / f"{name}_summary.json"
        if csv_path.exists():
            csv_path.unlink()

        def on_result(result: BenchmarkResult) -> None:
            suite.results.append(result)
            self._write_partial_result(csv_path, result)
            with open(summary_path, "w") as f:
                json.dump(suite.summary(), f, indent=2)

        for config_index, cfg in enumerate(configs, start=1):
            self.run_trial(config_index=config_index, on_result=on_result, **cfg)

        suite.end_time = time.perf_counter()

        suite.to_csv(str(csv_path))
        with open(summary_path, "w") as f:
            json.dump(suite.summary(), f, indent=2)

        if self.verbose:
            suite.print_summary()

        return suite


def run_single_benchmark(
    dataset: str = "torus",
    n_points: int = 2000,
    complex_type: str = "alpha",
    max_dim: int = 3,
    num_filtrations: int = 20,
    algorithm: str = "eigvalsh",
    package: str = "petls-pytorch",
    device: str = "cpu",
    output_dir: str = "./benchmark_results",
    seed: int = 42,
) -> BenchmarkSuiteResult:
    """Convenience function for a single benchmark run."""
    runner = BenchmarkRunner(
        output_dir=output_dir,
        algorithm=algorithm,
        package=package,
        device=device,
    )
    return runner.run_suite(
        name=f"{package}_{dataset}_{n_points}_{complex_type}",
        configs=[
            {
                "dataset_name": dataset,
                "n_points": n_points,
                "complex_type": complex_type,
                "max_dim": max_dim,
                "num_filtrations": num_filtrations,
                "seed": seed,
            }
        ],
    )
