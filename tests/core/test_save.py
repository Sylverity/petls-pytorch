"""Tests for store_L, store_spectra, store_spectra_summary, and time_to_csv."""

from __future__ import annotations

import os

import numpy as np
import pytest

from petls_torch.core.complex import Complex


def get_test_complex():
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=np.float64)
    d2 = np.array([[1], [1], [-1]], dtype=np.float64)
    boundaries = [d1, d2]
    filtrations = [[0, 1, 2], [3, 4, 5], [5]]
    return Complex(boundaries=boundaries, filtrations=filtrations)


def test_store_L():
    pl = get_test_complex()
    prefix = "test_saved_matrix"
    try:
        pl.store_L(0, 0, 1, prefix)
        assert os.path.exists(f"{prefix}.mtx")
    finally:
        if os.path.exists(f"{prefix}.mtx"):
            os.remove(f"{prefix}.mtx")


def test_store_spectra():
    pl = get_test_complex()
    spectra_list = pl.spectra()
    prefix = "test_spectra"
    try:
        pl.store_spectra(spectra_list, prefix)
        for dim in range(pl.top_dim + 1):
            path = f"{prefix}_spectra_{dim}.txt"
            assert os.path.exists(path)
            with open(path) as fh:
                lines = fh.readlines()
            # Each line is a space-separated list of eigenvalues.
            assert all("\t" not in line for line in lines)
    finally:
        for dim in range(pl.top_dim + 1):
            path = f"{prefix}_spectra_{dim}.txt"
            if os.path.exists(path):
                os.remove(path)


def test_store_spectra_summary():
    pl = get_test_complex()
    spectra_list = pl.spectra()
    prefix = "test_summary"
    try:
        pl.store_spectra_summary(spectra_list, prefix)
        path = f"{prefix}_spectra_summary.txt"
        assert os.path.exists(path)
        with open(path) as fh:
            lines = fh.readlines()
        header = lines[0].strip().split("\t")
        expected_header = ["a", "b"]
        for d in range(pl.top_dim + 1):
            expected_header.append(f"betti_{d}")
        for d in range(pl.top_dim + 1):
            expected_header.append(f"lambda_{d}")
        assert header == expected_header
        # One line per unique (a, b) pair.
        unique_pairs = {(float(item[1]), float(item[2])) for item in spectra_list}
        assert len(lines) == len(unique_pairs) + 1
    finally:
        path = f"{prefix}_spectra_summary.txt"
        if os.path.exists(path):
            os.remove(path)


def test_time_to_csv():
    pl = get_test_complex()
    pl.spectra(0, 1, 2)
    pl.spectra(0, 2, 3)
    pl.spectra(1, 2, 3)

    filename = "test_profile.csv"
    try:
        pl.time_to_csv(filename)
        assert os.path.exists(filename)
    finally:
        if os.path.exists(filename):
            os.remove(filename)
