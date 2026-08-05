"""Weighted-alpha, topology-inspection, and scaling-safeguard tests."""

from __future__ import annotations

import math

import gudhi
import numpy as np
import pytest
import torch

from petls_pytorch import Alpha, LaplacianSizeError
from petls_pytorch.core.complex import Complex


def _ring(count: int = 12) -> list[list[float]]:
    return [
        [math.cos(2 * math.pi * index / count), math.sin(2 * math.pi * index / count)]
        for index in range(count)
    ]


def _ring_weights(count: int = 12) -> list[float]:
    return [0.10 + 0.01 * (index % 3) for index in range(count)]


def test_weight_validation_and_finite_inputs():
    points = _ring(4)
    with pytest.raises(ValueError, match=r"len\(weights\)"):
        Alpha(points=points, weights=[0.1])
    with pytest.raises(ValueError, match="weights must contain only finite"):
        Alpha(points=points, weights=[0.1, 0.1, float("nan"), 0.1])
    with pytest.raises(ValueError, match="points must contain only finite"):
        Alpha(points=[[0.0, 0.0], [float("inf"), 1.0]])


def test_weighted_ring_has_one_tunnel_and_negative_vertex_births():
    alpha = Alpha(points=_ring(), weights=_ring_weights(), max_dim=2)

    assert alpha.betti_numbers_at(0.0)[1] == 1
    assert min(alpha.simplex_filtrations[0]) < 0.0
    assert alpha.get_all_filtrations()[0] < 0.0
    assert alpha.filtered_boundaries[0].domain_filtrations.tolist() == pytest.approx(
        alpha.simplex_filtrations[0]
    )

    summary = alpha.topology_summary(dimensions=(0, 1), a=0.0, b=0.0)
    assert summary["betti_kind"] == "ordinary"
    assert summary["betti"][1] == 1
    assert summary["betti_source"][1] == "gudhi_persistence"
    assert summary["spectral_nullity"][1] == 1
    assert summary["least_nonzero_eigenvalue"][1] > 0.0
    assert summary["zero_tolerance"][1] >= alpha.zero_atol

    cutoff = Alpha(
        points=_ring(),
        weights=_ring_weights(),
        max_dim=2,
        precision="safe",
        max_alpha_square=0.0,
    )
    assert all(
        filtration <= 0.0
        for values in cutoff.simplex_filtrations
        for filtration in values
    )


def test_filled_tetrahedron_has_no_tunnel():
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    alpha = Alpha(points=points, weights=[0.05, 0.07, 0.09, 0.11], max_dim=3)
    assert alpha.betti_numbers_at(1.0)[1] == 0


def test_weighted_octahedral_shell_has_one_cavity():
    points = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    alpha = Alpha(points=points, weights=[0.8] * len(points), max_dim=3)
    assert alpha.betti_numbers_at(0.0)[2] == 1


def test_unweighted_and_zero_weight_alpha_have_parity():
    points = _ring()
    unweighted = Alpha(points=points, max_dim=2)
    zero_weighted = Alpha(points=points, weights=[0.0] * len(points), max_dim=2)

    for dim in range(unweighted.top_dim + 1):
        left = dict(
            zip(
                unweighted.simplices_by_dimension[dim],
                unweighted.simplex_filtrations[dim],
            )
        )
        right = dict(
            zip(
                zero_weighted.simplices_by_dimension[dim],
                zero_weighted.simplex_filtrations[dim],
            )
        )
        assert left == pytest.approx(right, abs=1e-12)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(unweighted.get_L(1, 0.5, 0.5).numpy()),
        np.linalg.eigvalsh(zero_weighted.get_L(1, 0.5, 0.5).numpy()),
        atol=1e-12,
        rtol=1e-12,
    )


def test_persistence_intervals_and_persistent_betti_queries():
    alpha = Alpha(points=_ring(), weights=_ring_weights(), max_dim=2)
    intervals = alpha.persistence_intervals(dim=1)

    assert intervals.ndim == 2 and intervals.shape[1] == 2
    assert alpha.persistent_betti(1, birth_scale=0.0, death_scale=0.5) == 1
    assert alpha.persistent_betti(1, birth_scale=0.0, death_scale=1.0) == 0
    persistent = alpha.topology_summary(dimensions=(1,), a=0.0, b=0.5)
    assert persistent["betti_kind"] == "persistent"
    assert persistent["betti"][1] == 1


def test_simplex_coordinates_and_harmonic_features_are_traceable():
    labels = [f"point-{index}" for index in range(12)]
    alpha = Alpha(
        points=_ring(),
        weights=_ring_weights(),
        point_labels=labels,
        max_dim=2,
    )

    boundary = alpha.filtered_boundaries[1].matrix.coalesce()
    indices = boundary.indices()
    for edge_index, edge in enumerate(alpha.simplices_by_dimension[1]):
        incident_rows = indices[0, indices[1] == edge_index].tolist()
        incident_vertices = {
            alpha.simplices_by_dimension[0][row][0] for row in incident_rows
        }
        assert incident_vertices == set(edge)

    harmonic = alpha.harmonic_features(dim=1, a=0.0, b=0.0)
    assert harmonic["betti"] == harmonic["spectral_nullity"] == 1
    coefficients = harmonic["features"][0]["simplex_coefficients"]
    assert len(coefficients) == alpha.estimate_laplacian(1, 0.0)["rows"]
    for coefficient in coefficients:
        simplex_index = coefficient["simplex_index"]
        simplex = alpha.simplices_by_dimension[1][simplex_index]
        assert coefficient["simplex"] == list(simplex)
        assert coefficient["labels"] == [labels[vertex] for vertex in simplex]


def test_filtration_merge_tolerance_and_vertex_opt_out():
    boundary = np.array(
        [[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    complex_ = Complex(
        boundaries=[boundary],
        filtrations=[
            [25.0, 25.00000000000006, 25.00000000000011],
            [26.0, 26.00000000000005],
        ],
    )

    assert complex_.get_all_filtrations(merge_tolerance=1e-10) == [25.0, 26.0]
    assert complex_.get_all_filtrations(
        merge_tolerance=1e-10,
        include_vertex_filtrations=False,
    ) == [26.0]
    assert len(complex_.get_all_filtrations(merge_tolerance=0.0)) == 5


def test_dense_guard_and_sparse_spectrum_avoid_dense_laplacian(monkeypatch):
    alpha = Alpha(
        points=_ring(),
        weights=_ring_weights(),
        max_dim=2,
        max_matrix_rows=5,
        on_oversize="homology_only",
    )
    estimate = alpha.estimate_laplacian(dim=1, a=0.0, b=0.0)
    assert estimate["rows"] == 12
    assert estimate["dense_bytes"] == 12 * 12 * 8
    assert estimate["recommended_backend"] == "sparse"
    with pytest.raises(LaplacianSizeError):
        alpha.get_L(1, 0.0, 0.0)

    def fail_if_dense(*args, **kwargs):
        raise AssertionError("dense Laplacian construction was attempted")

    monkeypatch.setattr(alpha, "get_L", fail_if_dense)
    alpha.set_eigs_algorithm("sparse", num_eigenvalues=6)
    sparse_values = alpha.spectra(1, 0.0, 0.0)
    assert len(sparse_values) == 6
    assert abs(sparse_values[0]) < 1e-8

    summary = alpha.topology_summary(
        dimensions=(1,),
        a=-0.03,
        b=0.0,
        on_oversize="homology_only",
    )
    assert summary["betti"][1] == 1
    assert summary["calculation_status"][1] == "homology_only_oversize"
    assert summary["least_nonzero_eigenvalue"][1] is None

    isolated_tree = gudhi.SimplexTree()
    for vertex in range(300):
        isolated_tree.insert([vertex], filtration=0.0)
    disconnected = Complex(
        simplex_tree=isolated_tree,
        dtype=torch.float64,
        max_matrix_rows=5,
    )

    def fail_if_dense_solver(*args, **kwargs):
        raise AssertionError("dense eigendecomposition was attempted")

    monkeypatch.setattr(np.linalg, "eigh", fail_if_dense_solver)
    assert disconnected.ordinary_spectrum(0, 0.0, 10) == [0.0] * 10
    disconnected_summary = disconnected.topology_summary((0,), 0.0, 0.0)
    assert disconnected_summary["betti"][0] == 300
    assert disconnected_summary["calculation_status"][0] == "sparse_null_modes_only"
    assert disconnected_summary["least_nonzero_eigenvalue"][0] is None


def test_object_specific_dtype_device_and_small_weighted_parity():
    points = _ring()
    weights = _ring_weights()
    alpha32 = Alpha(points=points, weights=weights, max_dim=2, dtype=torch.float32)
    alpha64 = Alpha(points=points, weights=weights, max_dim=2, dtype=torch.float64)

    assert alpha32.device.type == alpha64.device.type == "cpu"
    assert alpha32.dtype == torch.float32
    assert alpha64.dtype == torch.float64
    np.testing.assert_allclose(
        alpha32.get_L(1, 0.0, 0.0).numpy(),
        alpha64.get_L(1, 0.0, 0.0).numpy(),
        atol=2e-6,
        rtol=2e-6,
    )

    automatic = Alpha(points=points, weights=weights, max_dim=1, device="auto")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert automatic.device.type == expected
    if torch.cuda.is_available():
        cuda = Alpha(points=points, weights=weights, max_dim=2, device="cuda")
        np.testing.assert_allclose(
            cuda.get_L(1, 0.0, 0.0).cpu().numpy(),
            alpha64.get_L(1, 0.0, 0.0).numpy(),
            atol=1e-10,
            rtol=1e-10,
        )
