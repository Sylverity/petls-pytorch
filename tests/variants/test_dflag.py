"""Tests for the PyTorch-native dFlag variant.

Tests compare against the original C++ ``petls.dFlag`` implementation.
Note: the original ``petls.dFlag.spectra()`` crashes due to a missing
``profile`` attribute, so we compare ``get_L`` / ``get_up`` / ``get_down``
matrices and their eigenvalues instead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

petls = pytest.importorskip("petls", reason="Reference PETLS not available")
from petls_torch.variants.dflag import dFlag  # noqa: E402

ATOL = 1e-4
RTOL = 1e-3

MWE_PATH = "tests/variants/data/flag/d4-mwe.flag"
CYCLE_PATH = "tests/variants/data/flag/cycle_no_bdy.flag"


class TestdFlagConstruction:
    def test_top_dim_matches_reference_mwe(self):
        ref = petls.dFlag(MWE_PATH, 3)
        test = dFlag(MWE_PATH, 3)
        assert test.top_dim == ref.pl.top_dim

    def test_top_dim_matches_reference_cycle(self):
        ref = petls.dFlag(CYCLE_PATH, 3)
        test = dFlag(CYCLE_PATH, 3)
        assert test.top_dim == ref.pl.top_dim

    def test_filtration_count_matches_mwe(self):
        ref = petls.dFlag(MWE_PATH, 3)
        test = dFlag(MWE_PATH, 3)
        assert test.get_all_filtrations() == pytest.approx(ref.pl.get_all_filtrations(), abs=ATOL)

    def test_filtration_count_matches_cycle(self):
        ref = petls.dFlag(CYCLE_PATH, 3)
        test = dFlag(CYCLE_PATH, 3)
        assert test.get_all_filtrations() == pytest.approx(ref.pl.get_all_filtrations(), abs=ATOL)


class TestdFlagLaplacian:
    def _compare_laplacians(self, ref, test, path):
        filts = ref.pl.get_all_filtrations()
        mismatches = []
        for dim in range(ref.pl.top_dim + 1):
            for i in range(len(filts) - 1):
                a, b = filts[i], filts[i + 1]
                ref_L = ref.get_L(dim, a, b)
                test_L = test.get_L(dim, a, b)

                # Compare shapes first
                if ref_L.shape != test_L.shape:
                    mismatches.append(
                        f"Shape mismatch at ({dim}, {a}, {b}): "
                        f"ref={ref_L.shape} vs test={test_L.shape}"
                    )
                    continue

                if ref_L.shape == (0, 0):
                    continue

                # Compare eigenvalues (invariant to row/column ordering)
                ref_eigs = np.linalg.eigvalsh(ref_L)
                test_eigs = np.linalg.eigvalsh(test_L.cpu().numpy())
                if not np.allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL):
                    mismatches.append(
                        f"Eigenvalue mismatch at ({dim}, {a}, {b}): "
                        f"ref={ref_eigs} vs test={test_eigs}"
                    )

        if mismatches:
            pytest.fail("\n".join(mismatches))

    def test_get_L_eigenvalues_match_mwe(self):
        ref = petls.dFlag(MWE_PATH, 3)
        test = dFlag(MWE_PATH, 3)
        self._compare_laplacians(ref, test, MWE_PATH)

    def test_get_L_eigenvalues_match_cycle(self):
        ref = petls.dFlag(CYCLE_PATH, 3)
        test = dFlag(CYCLE_PATH, 3)
        self._compare_laplacians(ref, test, CYCLE_PATH)

    def test_get_down_eigenvalues_match_mwe(self):
        ref = petls.dFlag(MWE_PATH, 3)
        test = dFlag(MWE_PATH, 3)
        filts = ref.pl.get_all_filtrations()
        for dim in range(ref.pl.top_dim + 1):
            for a in filts:
                ref_down = ref.get_down(dim, a)
                test_down = test.get_down(dim, a)
                if ref_down.shape == (0, 0):
                    continue
                ref_eigs = np.linalg.eigvalsh(ref_down)
                test_eigs = np.linalg.eigvalsh(test_down.cpu().numpy())
                np.testing.assert_allclose(ref_eigs, test_eigs, atol=ATOL, rtol=RTOL)

    def test_L_equals_up_plus_down_mwe(self):
        test = dFlag(MWE_PATH, 3)
        filts = test.get_all_filtrations()
        for dim in range(test.top_dim + 1):
            for i in range(len(filts) - 1):
                a, b = filts[i], filts[i + 1]
                L = test.get_L(dim, a, b)
                up = test.get_up(dim, a, b)
                down = test.get_down(dim, a)
                np.testing.assert_allclose(
                    L.cpu().numpy(),
                    (up + down).cpu().numpy(),
                    atol=ATOL,
                    rtol=RTOL,
                )

    def test_get_up_top_dim_is_zero(self):
        test = dFlag(MWE_PATH, 3)
        filts = test.get_all_filtrations()
        for a, b in [(filts[0], filts[1]), (filts[-2], filts[-1])]:
            up = test.get_up(test.top_dim, a, b)
            assert up.shape[0] == up.shape[1]
            assert torch.allclose(up, torch.zeros_like(up))


class TestdFlagSpectra:
    def test_spectra_mwe_matches_reference(self):
        """The original petls.dFlag crashes on spectra(); we compute our own
        and verify the eigenvalues match get_L eigenvalues from the reference.
        """
        ref = petls.dFlag(MWE_PATH, 3)
        test = dFlag(MWE_PATH, 3)

        filts = ref.pl.get_all_filtrations()
        for dim in range(ref.pl.top_dim + 1):
            for i in range(len(filts) - 1):
                a, b = filts[i], filts[i + 1]
                ref_L = ref.get_L(dim, a, b)
                if ref_L.shape == (0, 0):
                    continue
                ref_eigs = np.linalg.eigvalsh(ref_L)
                test_eigs = test.spectra(dim, a, b)
                np.testing.assert_allclose(ref_eigs, np.array(test_eigs), atol=ATOL, rtol=RTOL)
