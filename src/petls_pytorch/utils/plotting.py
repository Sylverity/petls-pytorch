"""
Plotting utilities for persistence-style summaries.

Mirrors petls.PLutil.summaries and plot_summary.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple


def summaries(
    spectra: List[Tuple[int, float, float, List[float]]],
    func: Callable,
    lower_triangle: float = np.nan,
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """
    Apply a function to eigenvalues for all (dim, a, b, eigs) tuples.

    Returns summary arrays shaped like persistence diagrams.
    """
    all_filtrations = set()
    dims = set()
    for dim, a, b, eigs in spectra:
        all_filtrations.add(a)
        all_filtrations.add(b)
        dims.add(dim)

    all_filtrations = sorted(list(all_filtrations))
    indexed_filtrations = {a: i for i, a in enumerate(all_filtrations)}
    indexed_dims = {dim: i for i, dim in enumerate(sorted(list(dims)))}
    num_filtrations = len(all_filtrations)
    summaries_list = [np.zeros((num_filtrations, num_filtrations)) for _ in range(len(dims))]

    for dim in range(len(indexed_dims)):
        for i in range(num_filtrations):
            for j in range(i + 1, num_filtrations):
                summaries_list[dim][i, j] = lower_triangle

    for dim, a, b, eigs in spectra:
        summaries_list[indexed_dims[dim]][indexed_filtrations[b], indexed_filtrations[a]] += func(
            eigs
        )

    return summaries_list, all_filtrations, sorted(dims)


def plot_summary(ax, summary: np.ndarray, **kwargs):
    """Wrapper for matplotlib imshow with persistence-diagram styling."""
    pos = ax.imshow(summary, origin="lower", **kwargs)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black")
    return pos
