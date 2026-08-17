"""Tests for sheaf_simplex_tree and PersistentSheafLaplacian."""

from __future__ import annotations

from math import sqrt

import gudhi
import numpy as np

from petls_pytorch.variants.sheaf import sheaf_simplex_tree, PersistentSheafLaplacian


def my_restriction(simplex, coface, sst):
    if len(simplex) == 1:
        if simplex == [coface[0]]:
            sibling = [coface[1]]
        else:
            sibling = [coface[0]]

        coords_simplex = sst.extra_data[tuple(simplex)][0:3]
        coords_sibling = sst.extra_data[tuple(sibling)][0:3]
        distance = sqrt(
            (coords_simplex[0] - coords_sibling[0]) ** 2
            + (coords_simplex[1] - coords_sibling[2]) ** 2
            + (coords_simplex[2] - coords_sibling[1]) ** 2
        )
        return sst.extra_data[tuple(sibling)][3] / distance
    elif len(simplex) == 2:
        coeff = 1.0
        for sibling, _ in sst.st.get_boundaries(coface):
            if list(sibling) == simplex:
                opposite_vertex = coface[sst.coface_index(simplex, coface)]
                coeff = coeff * sst.extra_data[tuple([opposite_vertex])][3]
            else:
                coeff = coeff / sst.st.filtration(sibling)
        return coeff
    return 1


def get_sst(points, charges):
    st = gudhi.RipsComplex(points=points, max_edge_length=6).create_simplex_tree(max_dimension=3)

    extra_data = {
        tuple([0]): [*points[0], charges[0]],
        tuple([1]): [*points[1], charges[1]],
        tuple([2]): [*points[2], charges[2]],
        tuple([0, 1]): 1,
        tuple([0, 2]): 1,
        tuple([1, 2]): 1,
        tuple([0, 1, 2]): 0,
    }

    sst = sheaf_simplex_tree(st, extra_data, my_restriction)
    return sst


def test_sheaf_simplex_tree():
    points = [[0, 0, 0], [3, 0, 0], [0, 4, 0]]
    as_np = [np.array(x) for x in points]
    dists = [
        np.linalg.norm(as_np[0] - as_np[1]),
        np.linalg.norm(as_np[0] - as_np[2]),
        np.linalg.norm(as_np[1] - as_np[2]),
    ]
    charges = [2, 7, 11]
    expected_cbdys = [
        np.array(
            [
                [-charges[1] / dists[0], charges[0] / dists[0], 0],
                [-charges[2] / dists[1], 0, charges[0] / dists[1]],
                [0, -charges[2] / dists[2], charges[1] / dists[2]],
            ]
        ),
        np.array(
            [
                [
                    charges[2] / (dists[1] * dists[2]),
                    -charges[1] / (dists[0] * dists[2]),
                    charges[0] / (dists[0] * dists[1]),
                ]
            ]
        ),
    ]
    expected_filtrations = [[0, 0, 0], [3, 4, 5], [5]]
    sst = get_sst(points, charges)
    coboundaries, filtrations = sst.apply_restriction_function()

    for i in range(len(coboundaries)):
        np.testing.assert_allclose(coboundaries[i], expected_cbdys[i], rtol=1e-4)
    for i in range(len(filtrations)):
        np.testing.assert_allclose(np.array(filtrations[i]), np.array(expected_filtrations[i]))


def test_persistent_sheaf_laplacian():
    points = [[0, 0, 0], [3, 0, 0], [0, 4, 0]]
    as_np = [np.array(x) for x in points]
    dists = [
        np.linalg.norm(as_np[0] - as_np[1]),
        np.linalg.norm(as_np[0] - as_np[2]),
        np.linalg.norm(as_np[1] - as_np[2]),
    ]
    charges = [2, 7, 11]
    q0, q1, q2 = charges
    d01, d02, d12 = dists

    cbdy0 = np.array(
        [
            [-q1 / d01, q0 / d01, 0],
            [-q2 / d02, 0, q0 / d02],
            [0, -q2 / d12, q1 / d12],
        ]
    )
    cbdy1 = np.array([[q2 / (d02 * d12), -q1 / (d01 * d12), q0 / (d01 * d02)]])

    bdy1 = cbdy0.T
    bdy2 = cbdy1.T

    expected_L0 = bdy1 @ bdy1.T
    expected_L1 = bdy1.T @ bdy1 + bdy2 @ bdy2.T
    expected_L2 = bdy2.T @ bdy2

    sst = get_sst(points, charges)
    psl = PersistentSheafLaplacian(sst)

    np.testing.assert_allclose(psl.get_L(0, 5, 5).cpu().numpy(), expected_L0, rtol=1e-4)
    np.testing.assert_allclose(psl.get_L(1, 5, 5).cpu().numpy(), expected_L1, rtol=1e-4)
    np.testing.assert_allclose(psl.get_L(2, 5, 5).cpu().numpy(), expected_L2, rtol=1e-4)


def test_sheaf_cochains_preserve_isolated_and_zero_restriction_simplices():
    """Cochain coordinates follow all simplex lists, not only nonzero maps."""

    def make_edge_isolated_tree(reverse_insertion: bool = False):
        st = gudhi.SimplexTree()
        entries = [
            ([0], 0.0),
            ([1], 0.0),
            ([2], 0.0),
            ([0, 1], 1.0),
        ]
        if reverse_insertion:
            entries.reverse()
        for simplex, filtration in entries:
            st.insert(simplex, filtration=filtration)
        return st

    def unit_restriction(simplex, coface, sst):
        return 1.0

    def zero_restriction(simplex, coface, sst):
        return 0.0

    def assert_cochain_shapes(sst, coboundaries, filtrations):
        assert len(filtrations) == sst.complex_dim + 1
        assert len(coboundaries) == sst.complex_dim
        for dim, coboundary in enumerate(coboundaries):
            assert coboundary.shape == (
                len(filtrations[dim + 1]),
                len(filtrations[dim]),
            )

    sst = sheaf_simplex_tree(make_edge_isolated_tree(), {}, unit_restriction)
    coboundaries, filtrations = sst.apply_restriction_function()
    assert_cochain_shapes(sst, coboundaries, filtrations)
    assert coboundaries[0].shape == (1, 3)

    laplacian = PersistentSheafLaplacian(sst).get_L(0, 1.0, 1.0).cpu().numpy()
    assert laplacian.shape == (3, 3)
    np.testing.assert_allclose(laplacian @ np.array([0.0, 0.0, 1.0]), 0.0)

    zero_sst = sheaf_simplex_tree(make_edge_isolated_tree(), {}, zero_restriction)
    zero_coboundaries, zero_filtrations = zero_sst.apply_restriction_function()
    assert_cochain_shapes(zero_sst, zero_coboundaries, zero_filtrations)
    assert not np.any(zero_coboundaries[0])
    np.testing.assert_allclose(
        PersistentSheafLaplacian(zero_sst).get_L(0, 1.0, 1.0).cpu().numpy(),
        np.zeros((3, 3)),
    )

    vertex_tree = gudhi.SimplexTree()
    vertex_tree.insert([0], filtration=2.0)
    vertex_tree.insert([1], filtration=0.0)
    vertex_tree.insert([2], filtration=1.0)
    vertex_sst = sheaf_simplex_tree(vertex_tree, {}, zero_restriction)
    vertex_coboundaries, vertex_filtrations = vertex_sst.apply_restriction_function()
    assert_cochain_shapes(vertex_sst, vertex_coboundaries, vertex_filtrations)
    assert vertex_coboundaries == []
    assert PersistentSheafLaplacian(vertex_sst).get_L(0, 2.0, 2.0).shape == (3, 3)

    repeated = sheaf_simplex_tree(
        make_edge_isolated_tree(reverse_insertion=True), {}, unit_restriction
    )
    repeated_coboundaries, repeated_filtrations = repeated.apply_restriction_function()
    assert sst.simplices_by_dimension == repeated.simplices_by_dimension
    assert filtrations == repeated_filtrations
    for first, second in zip(coboundaries, repeated_coboundaries):
        np.testing.assert_array_equal(first, second)
