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
