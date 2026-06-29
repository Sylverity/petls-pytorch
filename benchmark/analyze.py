"""
Benchmark results analysis and visualization.

Reads CSV result files and produces:
  - Summary tables (by dataset, by matrix size, by dimension)
  - Scaling plots (time vs matrix size)
  - Build-vs-Eigs breakdown plots
  - Comparison tables (if multiple algorithm backends are present)

Usage:
    python -m benchmark.analyze benchmark_results/*.csv
    python -m benchmark.analyze --dir benchmark_results --plot_dir benchmark_plots
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_csv(path: str) -> List[Dict]:
    """Load benchmark CSV into list of dicts."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def numeric(rows: List[Dict], key: str) -> np.ndarray:
    """Extract numeric column."""
    return np.array([float(r[key]) for r in rows])


def compute_stats(values: np.ndarray) -> Dict:
    """Compute basic statistics."""
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sum": float(np.sum(values)),
    }


def print_table(title: str, headers: List[str], rows: List[List]):
    """Pretty-print a table."""
    print(f"\n{title}")
    print("-" * (sum(len(str(h)) for h in headers) + 4 * len(headers)))
    # Compute column widths
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_str)
    print("-" * len(header_str))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    print()


def analyze_file(path: str, plot_dir: Optional[str] = None) -> Dict:
    """Analyze a single benchmark CSV file."""
    rows = load_csv(path)
    if not rows:
        return {}
    completed_rows = [r for r in rows if str(r.get("skipped", "False")).lower() != "true"]
    skipped_rows = [r for r in rows if str(r.get("skipped", "False")).lower() == "true"]
    timing_rows = completed_rows or rows

    sizes = numeric(timing_rows, "matrix_rows")
    build = numeric(timing_rows, "build_time_ms")
    eigs = numeric(timing_rows, "eigs_time_ms")
    total = numeric(timing_rows, "total_time_ms")
    dims = np.array([int(r["dim"]) for r in timing_rows])

    # Overall stats
    overall = {
        "file": Path(path).name,
        "trials": len(rows),
        "completed": len(completed_rows),
        "skipped": len(skipped_rows),
        "total_time_sec": float(np.sum(total)) / 1000.0,
        "max_matrix": int(np.max(sizes)),
        "mean_matrix": int(np.mean(sizes)),
    }

    print(f"\n{'=' * 60}")
    print(f"  Analysis: {overall['file']}")
    print(f"{'=' * 60}")
    print(f"  Trials:       {overall['trials']}")
    print(f"  Completed:    {overall['completed']}")
    print(f"  Skipped:      {overall['skipped']}")
    print(f"  Total time:   {overall['total_time_sec']:.2f} s")
    print(f"  Max matrix:   {overall['max_matrix']} x {overall['max_matrix']}")
    print(f"  Mean matrix:  {overall['mean_matrix']} x {overall['mean_matrix']}")

    # Stats by dimension
    dim_headers = [
        "Dim",
        "Trials",
        "Max Size",
        "Mean Size",
        "Mean Build (ms)",
        "Mean Eigs (ms)",
        "Mean Total (ms)",
    ]
    dim_rows = []
    for d in sorted(np.unique(dims)):
        mask = dims == d
        dim_rows.append(
            [
                d,
                int(mask.sum()),
                int(np.max(sizes[mask])),
                int(np.mean(sizes[mask])),
                f"{np.mean(build[mask]):.1f}",
                f"{np.mean(eigs[mask]):.1f}",
                f"{np.mean(total[mask]):.1f}",
            ]
        )
    print_table("By Dimension", dim_headers, dim_rows)

    # Size bucket analysis
    buckets = [(0, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 12000), (12000, 999999)]
    bucket_headers = [
        "Size Range",
        "Trials",
        "Mean Build (ms)",
        "Mean Eigs (ms)",
        "Mean Total (ms)",
    ]
    bucket_rows = []
    for lo, hi in buckets:
        mask = (sizes >= lo) & (sizes < hi)
        if not mask.any():
            continue
        bucket_rows.append(
            [
                f"{lo}-{hi}",
                int(mask.sum()),
                f"{np.mean(build[mask]):.1f}",
                f"{np.mean(eigs[mask]):.1f}",
                f"{np.mean(total[mask]):.1f}",
            ]
        )
    print_table("By Matrix Size Bucket", bucket_headers, bucket_rows)

    # Scaling fit: time ~ n^p
    # Fit log(time) = p * log(n) + c for total time
    valid = sizes > 0
    log_n = np.log(sizes[valid])
    log_build = np.log(build[valid] + 1e-3)
    log_eigs = np.log(eigs[valid] + 1e-3)
    log_total = np.log(total[valid] + 1e-3)

    p_build = float(np.polyfit(log_n, log_build, 1)[0]) if len(log_n) > 2 else 0.0
    p_eigs = float(np.polyfit(log_n, log_eigs, 1)[0]) if len(log_n) > 2 else 0.0
    p_total = float(np.polyfit(log_n, log_total, 1)[0]) if len(log_n) > 2 else 0.0

    print("  Scaling exponents (time ~ n^p):")
    print(f"    Build: {p_build:.2f}")
    print(f"    Eigs:  {p_eigs:.2f}")
    print(f"    Total: {p_total:.2f}")
    print(f"{'=' * 60}\n")

    # Plots
    if HAS_MPL and plot_dir:
        plot_dir_path = Path(plot_dir)
        plot_dir_path.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem

        # 1. Time vs Matrix Size
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {0: "tab:blue", 1: "tab:green", 2: "tab:orange", 3: "tab:red"}
        for d in sorted(np.unique(dims)):
            mask = dims == d
            ax.scatter(
                sizes[mask],
                total[mask],
                s=20,
                alpha=0.6,
                color=colors.get(d, "gray"),
                label=f"dim={d}",
            )
        ax.set_xlabel("Matrix size (n)")
        ax.set_ylabel("Total time (ms)")
        ax.set_title(f"{stem} — Total time vs matrix size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir_path / f"{stem}_scaling_total.png", dpi=150)
        plt.close(fig)

        # 2. Build vs Eigs breakdown
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(build, eigs, s=20, alpha=0.6, c=dims, cmap="viridis")
        ax.set_xlabel("Build time (ms)")
        ax.set_ylabel("Eigs time (ms)")
        ax.set_title(f"{stem} — Build vs Eigs time")
        ax.set_xscale("log")
        ax.set_yscale("log")
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Dimension")
        ax.grid(True, ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir_path / f"{stem}_breakdown.png", dpi=150)
        plt.close(fig)

        # 3. Eigs fraction
        fig, ax = plt.subplots(figsize=(8, 4))
        frac = eigs / (total + 1e-6)
        ax.hist(frac, bins=30, color="steelblue", edgecolor="white")
        ax.axvline(np.median(frac), color="red", ls="--", label=f"median={np.median(frac):.2f}")
        ax.set_xlabel("Fraction of time in eigendecomposition")
        ax.set_ylabel("Count")
        ax.set_title(f"{stem} — Eigs time fraction distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir_path / f"{stem}_eigs_fraction.png", dpi=150)
        plt.close(fig)

        print(f"  Plots saved to: {plot_dir_path.resolve()}\n")

    return overall


def main():
    parser = argparse.ArgumentParser(description="Analyze PETLS benchmark results")
    parser.add_argument("files", nargs="*", help="CSV result files to analyze")
    parser.add_argument("--dir", type=str, help="Directory containing CSV files")
    parser.add_argument(
        "--plot_dir", type=str, default="./benchmark_plots", help="Where to save plots"
    )
    args = parser.parse_args()

    files = []
    if args.files:
        files = args.files
    elif args.dir:
        p = Path(args.dir)
        files = sorted(str(f) for f in p.glob("*.csv"))
    else:
        # Default: look for latest benchmark_results directory
        candidates = sorted(Path(".").glob("benchmark_results*"))
        if candidates:
            files = sorted(str(f) for f in candidates[-1].glob("*.csv"))

    if not files:
        print("No benchmark CSV files found.")
        return 1

    summaries = []
    for f in files:
        summaries.append(analyze_file(f, plot_dir=args.plot_dir))

    if len(summaries) > 1:
        print(f"\n{'=' * 60}")
        print("  Multi-file summary")
        print(f"{'=' * 60}")
        for s in summaries:
            print(
                f"  {s['file']:40s} | {s['completed']:4d}/{s['trials']:<4d} done | "
                f"{s['total_time_sec']:8.1f}s | max={s['max_matrix']:6d}"
            )
        print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
