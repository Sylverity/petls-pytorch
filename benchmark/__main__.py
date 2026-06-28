"""
PETLS Benchmark CLI

Usage:
    python -m benchmark --preset standard --package petls_torch
    python -m benchmark --dataset torus --n_points 5000 --complex alpha --max_dim 3
    python -m benchmark --preset quick --package petls --algorithm selfadjoint

Presets:
    quick      : Small datasets for CI/smoke testing (< 1 min)
    standard   : Medium datasets, representative workloads (~ 5 min)
    intensive  : Large datasets, GPU-worthy stress test (~ 30 min)
    extreme    : Push hardware to its limits (> 1 hour)
"""

import argparse
import sys
from pathlib import Path

from .runner import BenchmarkRunner
from .datasets import list_datasets


PRESETS = {
    "quick": {
        "name": "quick_smoke",
        "configs": [
            {
                "dataset_name": "torus",
                "n_points": 500,
                "complex_type": "alpha",
                "max_dim": 2,
                "num_filtrations": 8,
            },
            {
                "dataset_name": "sphere",
                "n_points": 300,
                "complex_type": "alpha",
                "max_dim": 2,
                "num_filtrations": 8,
            },
        ],
    },
    "standard": {
        "name": "standard",
        "configs": [
            {
                "dataset_name": "torus",
                "n_points": 2000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 16,
            },
            {
                "dataset_name": "sphere",
                "n_points": 1500,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 16,
            },
            {
                "dataset_name": "swiss_roll",
                "n_points": 2000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 16,
            },
        ],
    },
    "intensive": {
        "name": "intensive",
        "configs": [
            {
                "dataset_name": "torus",
                "n_points": 5000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 24,
            },
            {
                "dataset_name": "sphere",
                "n_points": 5000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 24,
            },
            {
                "dataset_name": "swiss_roll",
                "n_points": 5000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 24,
            },
            {
                "dataset_name": "klein_bottle",
                "n_points": 3000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 24,
            },
        ],
    },
    "extreme": {
        "name": "extreme",
        "configs": [
            {
                "dataset_name": "torus",
                "n_points": 8000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 32,
                "filtration_mode": "early",
            },
            {
                "dataset_name": "sphere",
                "n_points": 8000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 32,
                "filtration_mode": "early",
            },
            {
                "dataset_name": "swiss_roll",
                "n_points": 8000,
                "complex_type": "alpha",
                "max_dim": 3,
                "num_filtrations": 32,
                "filtration_mode": "early",
            },
            {
                "dataset_name": "torus",
                "n_points": 3000,
                "complex_type": "rips",
                "max_dim": 3,
                "num_filtrations": 20,
                "filtration_mode": "quantile",
            },
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Performance comparison against the reference PETLS implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets: {", ".join(list_datasets())}
Available presets:  {", ".join(PRESETS.keys())}

Examples:
  # Quick smoke test
  python -m benchmark --preset quick --package petls_torch

  # Reference PETLS run
  python -m benchmark --preset standard --package petls --algorithm selfadjoint

  # Intensive GPU stress test
  python -m benchmark --preset intensive --package petls_torch --algorithm eigvalsh --device cuda

  # Single custom run
  python -m benchmark --dataset torus --n_points 5000 --complex alpha --max_dim 3 --package petls_torch
""",
    )
    parser.add_argument(
        "--preset", type=str, choices=list(PRESETS.keys()), help="Run a predefined benchmark preset"
    )
    parser.add_argument(
        "--dataset", type=str, choices=list_datasets(), help="Dataset name (overrides preset)"
    )
    parser.add_argument(
        "--n_points", type=int, default=2000, help="Number of points (default: 2000)"
    )
    parser.add_argument(
        "--complex",
        type=str,
        choices=["alpha", "rips"],
        default="alpha",
        help="Complex type (default: alpha)",
    )
    parser.add_argument(
        "--max_dim", type=int, default=3, help="Maximum simplicial dimension (default: 3)"
    )
    parser.add_argument(
        "--num_filtrations",
        type=int,
        default=20,
        help="Number of sampled filtrations (default: 20)",
    )
    parser.add_argument(
        "--filtration_mode",
        type=str,
        choices=["quantile", "log", "early"],
        default="quantile",
        help="Filtration sampling strategy",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="eigvalsh",
        help="Eigenvalue algorithm (eigvalsh, selfadjoint, spectra, etc.)",
    )
    parser.add_argument(
        "--package",
        type=str,
        choices=["petls", "petls_torch"],
        default="petls_torch",
        help="Backend package to benchmark (default: petls_torch)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./benchmark_results", help="Directory to write results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device label for result tracking (cpu, cuda, etc.)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--list", action="store_true", help="List available datasets and presets")

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for d in list_datasets():
            print(f"  - {d}")
        print("\nAvailable presets:")
        for p, cfg in PRESETS.items():
            print(f"  - {p}: {len(cfg['configs'])} config(s)")
        return 0

    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        algorithm=args.algorithm,
        device=args.device,
        package=args.package,
        verbose=True,
    )

    if args.preset:
        preset = PRESETS[args.preset]
        print(f"\n{'#' * 60}")
        print(f"# Running preset: {args.preset}")
        print(f"# Package:        {args.package}")
        print(f"# Algorithm:      {args.algorithm}")
        print(f"# Device:         {args.device if args.package == 'petls_torch' else 'cpu'}")
        print(f"# Output:         {args.output_dir}")
        print(f"{'#' * 60}\n")
        runner.run_suite(name=f"{args.package}_{preset['name']}", configs=preset["configs"])
        print(f"\nResults written to: {Path(args.output_dir).resolve()}")
        return 0

    # Single custom run
    if args.dataset is None:
        print("Error: must specify --dataset or --preset")
        parser.print_help()
        return 1

    runner.run_suite(
        name=f"{args.package}_{args.dataset}_{args.n_points}_{args.complex}",
        configs=[
            {
                "dataset_name": args.dataset,
                "n_points": args.n_points,
                "complex_type": args.complex,
                "max_dim": args.max_dim,
                "num_filtrations": args.num_filtrations,
                "filtration_mode": args.filtration_mode,
                "seed": args.seed,
            }
        ],
    )
    print(f"\nResults written to: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
