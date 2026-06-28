"""
Timing and profiling utilities.

Mirrors the original PETLS Profile class but is GPU-aware:
- Uses torch.cuda.Event for GPU timing when available.
- Falls back to time.perf_counter on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


class Timer:
    """Simple CPU timer."""

    def __init__(self):
        self._start: float | None = None
        self.duration: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer.stop() called before start()")
        self.duration = time.perf_counter() - self._start
        self._start = None
        return self.duration


class CudaTimer:
    """GPU timer using CUDA events for accurate device-side timing."""

    def __init__(self):
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self.duration: float = 0.0

    def start(self) -> None:
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def stop(self) -> float:
        if self._start_event is None:
            raise RuntimeError("CudaTimer.stop() called before start()")
        self._end_event.record()
        torch.cuda.synchronize()
        self.duration = self._start_event.elapsed_time(self._end_event)  # milliseconds
        self._start_event = None
        self._end_event = None
        return self.duration


@dataclass
class Profile:
    """
    Collects timing and metadata across multiple spectra() calls.

    Fields match the original PETLS Profile CSV output:
      dim, filtration_a, filtration_b,
      duration_all, duration_eigs, duration_L,
      L_rows, betti, lambda
    """

    dims: List[int] = field(default_factory=list)
    filtration_a: List[float] = field(default_factory=list)
    filtration_b: List[float] = field(default_factory=list)
    durations_all: List[float] = field(default_factory=list)
    durations_eigs: List[float] = field(default_factory=list)
    durations_L: List[float] = field(default_factory=list)
    L_rows: List[int] = field(default_factory=list)
    bettis: List[int] = field(default_factory=list)
    lambdas: List[float] = field(default_factory=list)

    _timer_all: Timer = field(default_factory=Timer)
    _timer_eigs: Timer = field(default_factory=Timer)
    _timer_L: Timer = field(default_factory=Timer)

    def start_all(self) -> None:
        self._timer_all.start()

    def start_eigs(self) -> None:
        self._timer_eigs.start()

    def start_L(self) -> None:
        self._timer_L.start()

    def stop_all(self) -> None:
        self.durations_all.append(self._timer_all.stop())

    def stop_eigs(self) -> None:
        self.durations_eigs.append(self._timer_eigs.stop())

    def stop_L(self) -> None:
        self.durations_L.append(self._timer_L.stop())

    def wrap_up(
        self,
        dim: int,
        a: float,
        b: float,
        L_rows: int,
        eigs: torch.Tensor | list[float],
    ) -> None:
        """Record metadata for one completed spectra computation."""
        self.dims.append(dim)
        self.filtration_a.append(a)
        self.filtration_b.append(b)
        self.L_rows.append(L_rows)

        if isinstance(eigs, torch.Tensor):
            eigs = eigs.cpu().numpy()
        else:
            eigs = np.array(eigs)

        tol = 1e-4
        betti = int(np.sum(eigs < tol))
        nonzeros = eigs[eigs > tol]
        least = float(nonzeros.min()) if len(nonzeros) > 0 else 0.0

        self.bettis.append(betti)
        self.lambdas.append(least)

    def to_csv(self, filename: str) -> None:
        """Write profile to CSV."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "dim": self.dims,
                "filtration_a": self.filtration_a,
                "filtration_b": self.filtration_b,
                "duration_all": self.durations_all,
                "duration_eigs": self.durations_eigs,
                "duration_L": self.durations_L,
                "L_rows": self.L_rows,
                "betti": self.bettis,
                "lambda": self.lambdas,
            }
        )
        df.to_csv(filename, index=False)
