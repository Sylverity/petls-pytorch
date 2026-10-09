"""Repeat matched CPU/CUDA benchmarks and validate against full CPU spectra.

Run from the checkout with ``python -m benchmark.compare --preset standard``.
CSV trials and a JSON comparison (including provenance) stay in benchmark-results/.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import cast

import numpy as np
import torch

from .__main__ import PRESETS
from .runner import BenchmarkResult, BenchmarkRunner, BenchmarkSuiteResult


def validate_results(
    results: list[BenchmarkResult], reference: list[BenchmarkResult], *, rtol=1e-3, atol=1e-7
) -> dict:
    """Check request identity, every leading mode, and available real nullities.

    Residual quality alone cannot certify that an iterative method found the
    *lowest* modes. Compare ordered values and multiplicities independently.
    A correct null-only result is valid; missing or unavailable modes are not.
    """
    expected = {(r.config_index, r.request_index): r for r in reference}
    failures = []
    errors = []
    passed = 0
    seen = set()
    for result in results:
        key = (result.config_index, result.request_index)
        ref = expected.get(key)
        reasons = []
        if key in seen:
            reasons.append("duplicate request")
        seen.add(key)
        if ref is None or ref.failed or ref.skipped:
            reasons.append("missing or unsuccessful reference")
        elif result.failed or result.skipped:
            reasons.append(result.failure_reason or result.skip_reason)
        else:
            identity = (
                "dataset",
                "n_points",
                "complex_type",
                "dim",
                "filtration_a",
                "filtration_b",
                "matrix_rows",
                "seed",
            )
            if any(getattr(result, name) != getattr(ref, name) for name in identity):
                reasons.append("request differs from reference")
            count = 8 if result.algorithm == "partial" else len(ref.positive_eigenvalues)
            target = np.asarray(ref.positive_eigenvalues[:count])
            actual = np.asarray(result.positive_eigenvalues)
            if actual.shape != target.shape:
                reasons.append(f"positive-mode count {len(actual)} != {len(target)}")
            elif len(target):
                errors.extend((np.abs(actual - target) / np.maximum(np.abs(target), atol)).tolist())
                if not np.allclose(actual, target, rtol=rtol, atol=atol):
                    reasons.append("ordered positive eigenvalues differ")
            if any(mode["quality"] == "unavailable" for mode in result.positive_modes):
                reasons.append("unavailable partial mode")
            if result.betti != ref.betti or result.betti_source != ref.betti_source:
                reasons.append("homology differs")
            if (
                result.spectral_nullity is not None
                and result.spectral_nullity != ref.spectral_nullity
            ):
                reasons.append("real nullity differs")
        if reasons:
            failures.append({"config_index": key[0], "request_index": key[1], "reasons": reasons})
        else:
            passed += 1
    for key in expected.keys() - seen:
        failures.append(
            {"config_index": key[0], "request_index": key[1], "reasons": ["missing request"]}
        )
    return {
        "passed": passed,
        "failed": len(failures),
        "max_relative_eigenvalue_error": max(errors, default=0.0),
        "failures": failures,
    }


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.glob("src/**/*.py"), *root.glob("benchmark/*.py"), root / "uv.lock"])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    cpuinfo = Path("/proc/cpuinfo")
    cpu = platform.processor()
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "file_sha256": hashes,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "packages": {
            name: version(name)
            for name in ("petls-pytorch", "torch", "numpy", "scipy", "gudhi", "tadasets")
        },
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def summarize_runs(suites: list[BenchmarkSuiteResult], checks: list[dict]) -> dict:
    summaries = [suite.summary() for suite in suites]
    query_seconds = [s["attempted_time_sec"] for s in summaries]
    pipeline_seconds = [s["attempted_time_sec"] + s["complex_build_time_sec"] for s in summaries]
    return {
        "requests_per_repeat": summaries[0]["num_trials"],
        "median_query_seconds": float(np.median(query_seconds)),
        "min_query_seconds": min(query_seconds),
        "max_query_seconds": max(query_seconds),
        "median_pipeline_seconds": float(np.median(pipeline_seconds)),
        "median_complex_build_seconds": float(
            np.median([s["complex_build_time_sec"] for s in summaries])
        ),
        "median_solve_seconds": float(
            np.median([sum(r.eigs_time_ms for r in suite.results) / 1000 for suite in suites])
        ),
        "failed_requests": sum(s["num_failed"] for s in summaries),
        "skipped_requests": sum(s["num_skipped"] for s in summaries),
        "validated_requests": sum(c["passed"] for c in checks),
        "validation_failures": sum(c["failed"] for c in checks),
        "max_relative_eigenvalue_error": max(c["max_relative_eigenvalue_error"] for c in checks),
        "mode_quality_counts": dict(
            Counter(
                mode["quality"]
                for suite in suites
                for r in suite.results
                for mode in r.positive_modes
            )
        ),
        "solver_counts": dict(
            Counter(r.spectrum_solver for suite in suites for r in suite.results)
        ),
        "eigensolver_device_counts": dict(
            Counter(r.eigensolver_device for suite in suites for r in suite.results)
        ),
        "repeat_summaries": summaries,
        "validation": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="standard")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/comparison"))
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("repeats/threads must be positive and warmups non-negative")
    if "cpu" not in args.devices or len(set(args.devices)) != len(args.devices):
        parser.error(
            "devices must include cpu once for the full-spectrum reference, without duplicates"
        )
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    configs = [
        dict(config, seed=args.seed) for config in cast(list[dict], PRESETS[args.preset]["configs"])
    ]
    variants = [
        (algorithm, device) for algorithm in ("eigvalsh", "partial") for device in args.devices
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": provenance(),
        "preset": args.preset,
        "configs": configs,
        "dtype": "float64",
        "warmup_suites_per_variant": args.warmups,
        "measured_repeats": args.repeats,
        "validation_rtol": 1e-3,
        "validation_atol": 1e-7,
        "timing": "Synchronized complete query wall time including homology and result processing. Pipeline adds each dataset build once. Warmups, generic backend preparation, validation and file IO are excluded. Failed/skipped attempt time is retained.",
    }
    output = args.output_dir / "comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    runs: dict[str, list[BenchmarkSuiteResult]] = {
        f"{algorithm}_{device}": [] for algorithm, device in variants
    }
    for iteration in range(args.warmups + args.repeats):
        warmup = iteration < args.warmups
        repeat = iteration if warmup else iteration - args.warmups
        phase = "warmup" if warmup else "repeat"
        # Rotate order to spread thermal/background drift across configurations.
        offset = iteration % len(variants)
        for algorithm, device in variants[offset:] + variants[:offset]:
            name = f"{algorithm}_{device}"
            label = f"{name}_{phase}_{repeat + 1}"
            print(f"Starting {label}", flush=True)
            started = time.perf_counter()
            runner = BenchmarkRunner(
                output_dir=str(args.output_dir),
                algorithm=algorithm,
                device=device,
                dtype="float64",
                verbose=False,
            )
            suite = runner.run_suite(label, configs)
            print(
                f"Finished {label}: {suite.summary()['attempted_time_sec']:.3f}s query time ({time.perf_counter() - started:.1f}s elapsed)",
                flush=True,
            )
            if not warmup:
                runs[name].append(suite)
                # Preserve structured raw modes, avoiding CSV literal parsing.
                (args.output_dir / f"{label}.json").write_text(
                    json.dumps([asdict(r) for r in suite.results], indent=2) + "\n"
                )
    report["results"] = {}
    for name, suites in runs.items():
        checks = [
            validate_results(suite.results, runs["eigvalsh_cpu"][index].results)
            for index, suite in enumerate(suites)
        ]
        report["results"][name] = summarize_runs(suites, checks)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for name, summary in report["results"].items():
        print(
            f"{name}: median queries {summary['median_query_seconds']:.3f}s, pipeline {summary['median_pipeline_seconds']:.3f}s, validation failures {summary['validation_failures']}"
        )
    print(f"Results: {output.resolve()}")
    return int(any(s["validation_failures"] for s in report["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
