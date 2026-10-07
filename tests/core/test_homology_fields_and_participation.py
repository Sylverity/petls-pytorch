"""Real Hodge spectra and projector localization on genuine simplicial complexes."""

import gudhi
import numpy as np
import pytest
import torch

from petls_pytorch import Complex
from petls_pytorch.core.complex import LaplacianSizeError

pytestmark = pytest.mark.native
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")),
]


def torsion_tree():
    """Triangulate the degree-11 mapping cone; attach its cone at time one.

    One stellar subdivision breaks the first positive eigenvalue's multiplicity.
    At time zero the mapping cylinder retracts to S^1. At time one H_1 is Z/11.
    """
    tree = gudhi.SimplexTree()
    for i in range(33):
        top, next_top = 3 + i, 3 + (i + 1) % 33
        bottom, next_bottom = i % 3, (i + 1) % 3
        tree.insert([top, bottom, next_bottom], filtration=0)
        tree.insert([top, next_top, next_bottom], filtration=0)
        tree.insert([36, top, next_top], filtration=1)
    tree.remove_maximal_simplex([36, 3, 4])
    for edge in ([36, 3], [36, 4], [3, 4]):
        tree.insert([37, *edge], filtration=1)
    return tree


def two_cycles(**kwargs):
    tree = gudhi.SimplexTree()
    for edge in ([0, 1], [1, 2], [0, 2], [0, 3], [3, 4], [0, 4]):
        tree.insert(edge, filtration=0)
    tree.insert([0, 1, 2], filtration=1)
    return Complex(simplex_tree=tree, dtype=torch.float64, **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("a,b", [(1, 1), (0, 1)])
def test_torsion_does_not_skip_real_positive_modes(device, a, b):
    tree = torsion_tree()
    tree.persistence(homology_coeff_field=2, persistence_dim_max=True)
    assert tree.betti_numbers() == [1, 0, 0]
    complex_ = Complex(simplex_tree=tree, dtype=torch.float64, device=device)
    assert complex_.persistent_betti(1, a, b) == 1  # F_11, independently correct.
    b1 = complex_.filtered_boundaries[1].matrix.to_dense()
    b2 = complex_.filtered_boundaries[2].matrix.to_dense()
    assert torch.count_nonzero(b1 @ b2) == 0
    matrix = complex_.get_L(1, a, b)
    # Independent NumPy generalized Schur complement, including singular D.
    down = complex_.filtered_boundaries[1].submatrix_at_filtration(a).to_dense().cpu().numpy()
    up = complex_.filtered_boundaries[2].submatrix_at_filtration(b).to_dense().cpu().numpy()
    gram = up @ up.T
    rows = matrix.shape[0]
    cross = gram[rows:, :rows]
    expected = down.T @ down + gram[:rows, :rows]
    if cross.shape[0]:
        expected -= cross.T @ np.linalg.pinv(gram[rows:, rows:], hermitian=True) @ cross
    np.testing.assert_allclose(matrix.cpu().numpy(), expected, atol=1e-11)
    reference = np.linalg.eigvalsh(expected)
    assert reference[0] > 1e-3
    if a == b:
        assert reference[1] > reference[0] + 1e-5

    for count in (8, 24, 128):
        result = complex_.positive_spectrum(1, a, b, positive_modes=count)
        assert result["homology_coeff_field"] == 11
        assert result["spectral_coeff_field"] == "real"
        assert result["betti"] == 1
        assert result["least_nonzero_eigenvalue"] == pytest.approx(reference[0], rel=1e-6)
        assert all(mode["quality"] != "unavailable" for mode in result["positive_modes"])
        np.testing.assert_allclose(
            [mode["value"] for mode in result["positive_modes"]], reference[:count], rtol=1e-6
        )
        if count == 8:
            assert result["nullspace_handling"] == "range_restriction"
            assert result["spectral_nullity"] is None
            assert result["homology_spectral_agreement"] is None
        else:
            assert result["spectral_nullity"] == 0
            assert result["homology_spectral_agreement"] is False

    for mode in ("partial", "full"):
        summary = complex_.topology_summary((1,), a, b, spectral_mode=mode, positive_modes=24)
        assert summary["betti"][1] == 1
        assert summary["spectral_nullity"][1] == 0
        assert summary["homology_spectral_agreement"][1] is False
        assert summary["least_nonzero_eigenvalue"][1] == pytest.approx(reference[0])
        assert summary["spectrum_certified"][1]


@pytest.mark.parametrize("device", DEVICES)
def test_torsion_does_not_cap_real_rank_or_invent_harmonic_features(device):
    complex_ = Complex(simplex_tree=torsion_tree(), dtype=torch.float64, device=device)
    result = complex_.positive_spectrum(2, 1, positive_modes=128)
    assert result["betti"] == 1
    assert result["spectral_nullity"] == 0
    assert len(result["positive_modes"]) == 101  # All real positive modes, not 101 - 1.
    for dim in (1, 2):
        features = complex_.harmonic_features(dim, 1)
        assert features["betti"] == 1
        assert features["spectral_nullity"] == 0
        assert features["features"] == []
        assert features["features_complete"]
        participation = complex_.harmonic_participation(dim, 1)
        assert participation["participation_complete"]
        assert participation["homology_spectral_agreement"] is False
        assert participation["total_participation"] == 0
        assert all(s["participation"] == 0 for s in participation["simplex_participation"])


def test_sparse_torsion_localization_and_full_summary_are_not_invalidated_by_field():
    complex_ = Complex(simplex_tree=torsion_tree(), dtype=torch.float64, eigs_algorithm="sparse")
    result = complex_.harmonic_participation(1, 1)
    assert result["participation_complete"]
    assert result["total_participation"] == 0
    assert result["homology_spectral_agreement"] is False
    summary = complex_.topology_summary((1,), 1, 1, spectral_mode="full")
    assert summary["least_nonzero_eigenvalue"][1] == pytest.approx(0.01725544720792108)
    assert summary["homology_spectral_agreement"][1] is False


@pytest.mark.parametrize("device", DEVICES)
def test_participation_is_invariant_under_harmonic_basis_rotation(device, monkeypatch):
    complex_ = two_cycles(device=device)
    before = complex_.harmonic_participation(1, 0)
    values, vectors = complex_.eigenpairs(1, 0, 0)
    indices = np.flatnonzero(np.abs(values) <= complex_._zero_tolerance(values))
    assert len(indices) == 2
    rotated = vectors.clone()
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=vectors.dtype, device=vectors.device)
    rotated[:, indices] = vectors[:, indices] @ rotation
    assert not torch.allclose(rotated[:, indices], vectors[:, indices])
    monkeypatch.setattr(complex_, "eigenpairs", lambda *args: (values, rotated))
    after = complex_.harmonic_participation(1, 0)
    np.testing.assert_allclose(
        [item["participation"] for item in before["simplex_participation"]],
        [item["participation"] for item in after["simplex_participation"]],
        atol=1e-12,
    )
    # Each triangle boundary has normalized squared coefficients 1/3.
    assert after["participation_complete"]
    assert after["participation_scope"] == "full_harmonic_space"
    assert after["total_participation"] == pytest.approx(2)
    assert all(
        item["participation"] == pytest.approx(1 / 3) for item in after["simplex_participation"]
    )
    points = {item["vertex"]: item["participation"] for item in after["point_participation"]}
    assert points == pytest.approx({0: 2 / 3, 1: 1 / 3, 2: 1 / 3, 3: 1 / 3, 4: 1 / 3})


@pytest.mark.parametrize("device", DEVICES)
def test_persistent_participation_localizes_surviving_cycle_and_preserves_labels(device):
    complex_ = two_cycles(device=device)
    labels = [f"molecule-{i}" for i in range(5)]
    complex_.point_labels = labels
    complex_.simplex_labels_by_dimension = [
        [tuple(labels[v] for v in simplex) for simplex in simplices]
        for simplices in complex_.simplices_by_dimension
    ]
    result = complex_.harmonic_participation(1, 0, 1)
    assert result["participation_complete"]
    assert result["total_participation"] == pytest.approx(1)
    for item in result["simplex_participation"]:
        expected = 1 / 3 if set(item["simplex"]) <= {0, 3, 4} else 0
        assert item["participation"] == pytest.approx(expected, abs=1e-12)
        assert item["labels"] == [labels[v] for v in item["simplex"]]
    assert sum(item["participation"] for item in result["point_participation"]) == pytest.approx(1)
    assert all(item["label"] == labels[item["vertex"]] for item in result["point_participation"])


def test_truncated_participation_does_not_claim_full_basis_invariance():
    complex_ = two_cycles()
    result = complex_.harmonic_participation(1, 0, max_features=1)
    assert not result["participation_complete"]
    assert result["participation_scope"] == "computed_subspace"
    assert result["spectral_nullity"] == 2
    assert result["returned_features"] == 1
    assert result["total_participation"] == pytest.approx(1)
    assert not complex_.harmonic_features(1, 0, max_features=1)["features_complete"]

    tree = gudhi.SimplexTree()
    for vertex in range(30):
        tree.insert([vertex], filtration=0)
    oversized = Complex(simplex_tree=tree, max_matrix_rows=5)
    result = oversized.harmonic_participation(0, 0)
    assert not result["participation_complete"]
    assert result["calculation_status"] == "truncated_for_scale"
    assert result["total_participation"] == pytest.approx(10)
    assert result["spectral_nullity"] == 30
    with pytest.raises(LaplacianSizeError):
        oversized.harmonic_participation(0, 0, 1)


def test_participation_empty_sublevel_and_argument_validation():
    complex_ = two_cycles()
    empty = complex_.harmonic_participation(1, -1)
    assert empty["participation_complete"]
    assert empty["simplex_participation"] == []
    assert empty["point_participation"] == []
    for cap in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_features"):
            complex_.harmonic_participation(1, 0, max_features=cap)
    with pytest.raises(ValueError, match="greater"):
        complex_.harmonic_participation(1, 1, 0)


@pytest.mark.parametrize("device", DEVICES)
def test_float32_participation_preserves_projector_trace(device):
    tree = gudhi.SimplexTree()
    tree.insert([0, 1], filtration=0)
    tree.insert([2], filtration=0)
    complex_ = Complex(simplex_tree=tree, dtype=torch.float32, device=device)
    result = complex_.harmonic_participation(0, 0)
    assert result["participation_complete"]
    assert result["total_participation"] == pytest.approx(2, abs=1e-6)
    np.testing.assert_allclose(
        [r["participation"] for r in result["simplex_participation"]],
        [0.5, 0.5, 1],
        atol=1e-6,
    )


def test_zero_operator_and_insufficient_real_rank_are_handled_without_modular_rank():
    tree = gudhi.SimplexTree()
    for vertex in range(100):
        tree.insert([vertex], filtration=0)
    tree.insert([0, 1], filtration=0)
    complex_ = Complex(simplex_tree=tree, dtype=torch.float64)
    result = complex_.positive_spectrum(0, 0)
    assert result["least_nonzero_eigenvalue"] == pytest.approx(2)
    assert len(result["positive_modes"]) == 1
    assert result["spectral_nullity"] is None
    zero = complex_.positive_spectrum(0, -1)
    assert zero["calculation_status"] == "null_modes_only"
    assert zero["positive_modes"] == []
