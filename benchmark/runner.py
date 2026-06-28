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
from typing import List, Dict, Optional
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
    build_time_ms: float = 0.0          # time to build Laplacian matrix
    eigs_time_ms: float = 0.0           # time for eigendecomposition
    total_time_ms: float = 0.0          # build + eigs
    eigenvalue_count: int = 0
    betti: int = 0
    least_nonzero: float = 0.0
    algorithm: str = "eigvalsh"
    device: str = "cpu"
    seed: int = 42


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

        return {
            "suite_name": self.suite_name,
            "num_trials": len(self.results),
            "total_time_sec": sum(total_times) / 1000.0,
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
        print(f"  Total wall time: {s['total_time_sec']:.2f} s")
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
        package: str = "petls_torch",
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm = algorithm
        self.device = device
        self.package = package.lower()
        self.verbose = verbose

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

        if self.verbose:
            print(
                f"\n[Benchmark] {dataset_name} | n={n_points} | {complex_type} | "
                f"max_dim={max_dim} | package={package}"
            )
            print("-" * 60)

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
            device=self.device if package == "petls_torch" else None,
        )
        t_build_complex = (time.perf_counter() - t0) * 1000
        if self.verbose:
            print(f"  Complex build:   {t_build_complex:.1f} ms")
            print(f"  Unique filtrations: {ds['num_unique_filtrations']}")
            print(f"  Sampled filtrations: {len(ds['filtrations'])}")

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
            # Add final (a,a) case
            for dim in dims:
                requests.append((dim, filtrations[-1], filtrations[-1]))
        else:
            requests = []
            for a, b in filtration_pairs:
                for dim in dims:
                    requests.append((dim, a, b))

        results: List[BenchmarkResult] = []

        for dim, a, b in requests:
            # Matrix construction time
            t0 = time.perf_counter()
            try:
                L = complex_obj.get_L(dim, a, b)
            except Exception as e:
                if self.verbose:
                    print(f"    ERROR get_L(dim={dim}, a={a:.4f}, b={b:.4f}): {e}")
                continue
            t_build = (time.perf_counter() - t0) * 1000

            rows = int(L.shape[0])

            # Eigenvalue time
            t0 = time.perf_counter()
            try:
                eigs = complex_obj.spectra(dim, a, b)
            except Exception as e:
                if self.verbose:
                    print(f"    ERROR spectra(dim={dim}, a={a:.4f}, b={b:.4f}): {e}")
                continue
            t_eigs = (time.perf_counter() - t0) * 1000

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
            )
            results.append(result)

            if self.verbose and rows > 100:
                print(
                    f"  dim={dim} a={a:.4f} b={b:.4f} | "
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

        for cfg in configs:
            results = self.run_trial(**cfg)
            suite.results.extend(results)

        suite.end_time = time.perf_counter()

        # Save
        suite.to_csv(str(self.output_dir / f"{name}.csv"))
        with open(self.output_dir / f"{name}_summary.json", "w") as f:
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
    package: str = "petls_torch",
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
