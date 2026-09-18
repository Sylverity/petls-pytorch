"""Regression contracts for spectra, chain replacement, and allocation guards."""

import gudhi
import numpy as np
import pytest
import torch

from petls_pytorch import Complex
from petls_pytorch.core.complex import LaplacianSizeError

pytestmark = pytest.mark.native


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dummy_basis", [False, True])
def test_positive_spectrum_preserves_small_modes_and_multiplicity(device, dtype, dummy_basis):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    # Two disconnected weighted edges: spectrum {0, 0, 1e-5, 1e-5}.
    boundary = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]) * np.sqrt(5e-6)
    complex_ = Complex(
        boundaries=[boundary], filtrations=[[0] * 4, [0, 0]], device=device, dtype=dtype
    )
    complex_.set_eigs_algorithm("sparse", num_eigenvalues=1)
    positive = complex_.nonzero_spectra(0, 0, 0, use_dummy_harmonic_basis=dummy_basis)
    np.testing.assert_allclose(positive, [1e-5, 1e-5], rtol=1e-5)


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_relative_zero_test_is_invariant_under_uniform_boundary_scaling(scale):
    complex_ = Complex(
        boundaries=[np.diag([1.0, 2.0]) * scale],
        filtrations=[[0, 0], [0, 0]],
        dtype=torch.float64,
        zero_atol=0,
        zero_rtol=0.3,
    )
    full = complex_.spectra(0, 0, 0)
    assert complex_.eigenvalues_summarize(full)[0] == 1
    np.testing.assert_allclose(complex_.nonzero_spectra(0, 0, 0), [4 * scale**2])


def test_positive_spectrum_honors_absolute_tolerance():
    complex_ = Complex(
        boundaries=[np.diag(np.sqrt([1e-5, 1e-3]))],
        filtrations=[[0, 0], [0, 0]],
        zero_atol=1e-4,
        zero_rtol=0,
        dtype=torch.float64,
    )
    np.testing.assert_allclose(complex_.nonzero_spectra(0, 0, 0), [1e-3])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_harmonic_basis_cannot_change_positive_spectrum(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    complex_ = Complex(
        boundaries=[np.array([[-1.0], [1.0]])],
        filtrations=[[0, 0], [0]],
        dtype=torch.float64,
        device=device,
    )
    for basis in (
        [1.0, 1.0],
        [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        np.empty((2, 0)),
        np.zeros((2, 1)),
        torch.ones((2, 1), dtype=torch.float64, device=device),
    ):
        assert complex_.nonzero_spectra(0, 0, 0, PH_basis=basis) == pytest.approx([2.0])
    # Even a tiny non-harmonic column must not be hidden by its scaling.
    with pytest.raises(ValueError, match="harmonic"):
        complex_.nonzero_spectra(0, 0, 0, PH_basis=[1e-30, -1e-30])
    for basis in ([1.0], [np.nan, 0.0], [1j, 1j]):
        with pytest.raises(ValueError, match="PH_basis"):
            complex_.nonzero_spectra(0, 0, 0, PH_basis=basis)


def test_boundary_replacement_discards_old_homology_and_coordinates():
    tree = gudhi.SimplexTree()
    tree.insert([0, 1], filtration=0)
    complex_ = Complex(simplex_tree=tree)
    assert complex_.betti_numbers_at(0)[0] == 1
    complex_.spectra(0, 0, 0)
    complex_.point_labels = ["left", "right"]
    complex_.simplex_labels_by_dimension = [[("left",), ("right",)], [("left", "right")]]

    complex_.set_boundaries_filtrations([], [[0, 0, 1]])
    assert complex_.simplex_tree is None
    assert not complex_._persistence_computed
    assert complex_.simplices_by_dimension == []
    assert complex_.simplex_to_index == []
    assert complex_.point_labels is None
    assert complex_.simplex_labels_by_dimension is None
    assert complex_.simplex_filtrations == [[0, 0, 1]]
    assert complex_.profile.dims == []
    summary = complex_.topology_summary((0,), 1, 1)
    assert summary["betti"] == {0: 3}
    assert summary["spectral_nullity"] == {0: 3}
    assert summary["betti_source"] == {0: "laplacian_nullity"}
    with pytest.raises(RuntimeError, match="SimplexTree"):
        complex_.persistence_intervals(0)
    with pytest.raises(RuntimeError, match="Simplex mappings"):
        complex_.harmonic_features(0, 1)


def test_invalid_replacement_leaves_original_complex_usable():
    tree = gudhi.SimplexTree()
    tree.insert([0, 1], filtration=0)
    complex_ = Complex(simplex_tree=tree)
    original_intervals = complex_.persistence_intervals(0)
    original_matrix = complex_.get_L(0, 0, 0)
    with pytest.raises(ValueError, match="shape"):
        complex_.set_boundaries_filtrations([np.ones((4, 1))], [[0, 0], [0]])
    torch.testing.assert_close(complex_.get_L(0, 0, 0), original_matrix)
    np.testing.assert_array_equal(complex_.persistence_intervals(0), original_intervals)


def test_source_tree_mutation_does_not_change_constructed_complex():
    tree = gudhi.SimplexTree()
    tree.insert([0, 1], filtration=0)
    complex_ = Complex(simplex_tree=tree)
    tree.insert([2], filtration=0)
    assert complex_.betti_numbers_at(0)[0] == 1
    assert complex_.get_L(0, 0, 0).shape == (2, 2)


@pytest.mark.parametrize("part", ["up", "down"])
def test_guard_counts_rectangular_boundaries_before_any_dense_allocation(monkeypatch, part):
    boundary = np.zeros((100, 2))
    boundary[0, 0], boundary[1, 0], boundary[2, 1], boundary[3, 1] = -1, 1, -1, 1
    if part == "up":
        boundary = boundary.T
    complex_ = Complex(
        boundaries=[boundary],
        filtrations=[[0] * boundary.shape[0], [0] * boundary.shape[1]],
        dtype=torch.float64,
        max_matrix_bytes=64,
    )
    dim = 0 if part == "up" else 1
    estimate = complex_.estimate_laplacian(dim, 0)
    assert estimate["dense_bytes"] == 32
    assert estimate["boundary_dense_bytes"] == 1600
    assert estimate["peak_dense_bytes"] == 1600
    assert not estimate["within_dense_limits"]

    def fail_if_dense(*args, **kwargs):
        pytest.fail("Dense allocation happened before the guard")

    monkeypatch.setattr(torch.Tensor, "to_dense", fail_if_dense)
    with pytest.raises(LaplacianSizeError):
        complex_.get_L(dim, 0, 0)
    with pytest.raises(LaplacianSizeError):
        if part == "up":
            complex_.get_up(dim, 0, 0)
        else:
            complex_.get_down(dim, 0)
    # A sparse calculation still gives the correct nonzero eigenvalues.
    np.testing.assert_allclose(complex_.ordinary_spectrum(dim, 0, 2), [2, 2])


def test_empty_source_requires_no_large_future_allocation():
    complex_ = Complex(
        boundaries=[np.array([[-1.0], [1.0]])],
        filtrations=[[1, 1], [1]],
        max_matrix_bytes=1,
    )
    assert complex_.estimate_laplacian(0, 0, 1)["peak_dense_bytes"] == 0
    assert complex_.get_L(0, 0, 1).shape == (0, 0)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_batch_cardinality_and_export_are_stable(tmp_path, count):
    complex_ = Complex(boundaries=[], filtrations=[[0, 0]])
    batch = complex_.spectra(request_list=[(0, 0, 0)] * count)
    assert batch == [(0, 0.0, 0.0, [0.0, 0.0])] * count
    prefix = str(tmp_path / "result")
    complex_.store_spectra(batch, prefix)
    files = list(tmp_path.iterdir())
    assert len(files) == (1 if count else 0)
    if count:
        assert files[0].read_text().splitlines() == ["0.0 0.0"] * count
    assert complex_.spectra(0, 0, 0) == [0.0, 0.0]
    assert complex_.spectra() == [(0, 0.0, 0.0, [0.0, 0.0])]


def test_vertex_only_query_rejects_reversed_filtration_interval():
    complex_ = Complex(boundaries=[], filtrations=[[0]])
    with pytest.raises(ValueError, match="greater than or equal"):
        complex_.spectra(0, 1, 0)


@pytest.mark.parametrize(
    ("dimension", "a", "b", "expected"),
    [(0, 0, 0, [0, 0, 0]), (0, 0, 1, [0, 3, 3]), (1, 1, 1, [0, 3, 3]), (1, 1, 2, [3, 3, 3])],
)
def test_triangle_hodge_spectrum_is_independent_of_simplex_orientation(dimension, a, b, expected):
    boundary_1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=float)
    boundary_2 = np.array([[1], [1], [-1]], dtype=float)
    np.testing.assert_array_equal(boundary_1 @ boundary_2, np.zeros((3, 1)))
    # Reverse and permute edge coordinates; both adjacent boundaries transform.
    for change_of_basis in (np.eye(3), np.eye(3)[:, [2, 0, 1]] @ np.diag([-1, 1, -1])):
        complex_ = Complex(
            boundaries=[boundary_1 @ change_of_basis, change_of_basis.T @ boundary_2],
            filtrations=[[0, 0, 0], [1, 1, 1], [2]],
            dtype=torch.float64,
        )
        actual = complex_.spectra(dimension, a, b)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        positive = [value for value in expected if value > 0]
        np.testing.assert_allclose(complex_.nonzero_spectra(dimension, a, b), positive, atol=1e-12)
        assert complex_.topology_summary([dimension], a, b)["betti"][dimension] == expected.count(0)
