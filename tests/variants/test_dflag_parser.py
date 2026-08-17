"""Tests for dFlag .flag parsing and directed simplex enumeration."""

from __future__ import annotations

import numpy as np
import pytest

from petls_pytorch.variants.dflag import (
    _FlagData,
    _enumerate_directed_simplices,
    _read_flag_file,
    dFlag,
)


def _bruteforce_directed_simplices(graph, max_dim: int):
    n = len(graph.vertex_weights)
    simplices: list[list[tuple[int, ...]]] = [[] for _ in range(max_dim + 1)]
    filtrations: list[list[float]] = [[] for _ in range(max_dim + 1)]

    for v in range(n):
        simplices[0].append((v,))
        filtrations[0].append(float(np.float32(graph.vertex_weights[v])))

    for dim in range(1, max_dim + 1):
        for code in range(n ** (dim + 1)):
            tup_list = []
            tmp = code
            for _ in range(dim + 1):
                tup_list.append(tmp % n)
                tmp //= n
            tup = tuple(tup_list)
            if len(set(tup)) != dim + 1:
                continue

            max_weight = max(float(np.float32(graph.vertex_weights[v])) for v in tup)
            for i in range(dim + 1):
                for j in range(i + 1, dim + 1):
                    if not graph.edge_present[tup[i], tup[j]]:
                        break
                    weight = graph.edge_weights[tup[i], tup[j]]
                    max_weight = max(max_weight, float(np.float32(weight)))
                else:
                    continue
                break
            else:
                simplices[dim].append(tup)
                filtrations[dim].append(max_weight)

        indexed = list(enumerate(zip(simplices[dim], filtrations[dim])))
        indexed.sort(key=lambda x: (x[1][1], x[0]))
        simplices[dim] = [s for _, (s, _) in indexed]
        filtrations[dim] = [f for _, (_, f) in indexed]

    indexed = list(enumerate(zip(simplices[0], filtrations[0])))
    indexed.sort(key=lambda x: (x[1][1], x[0]))
    simplices[0] = [s for _, (s, _) in indexed]
    filtrations[0] = [f for _, (_, f) in indexed]
    return simplices, filtrations


def test_read_flag_file_parses_weighted_graph(tmp_path):
    path = tmp_path / "graph.flag"
    path.write_text(
        """
        # vertex weights
        dim 0
        0.5 0.25 0.75

        dim 1
        0 1 1.5
        1 2 2.5 # comments are ignored
        """,
    )

    graph = _read_flag_file(str(path))

    np.testing.assert_allclose(graph.vertex_weights, [0.5, 0.25, 0.75])
    np.testing.assert_allclose(
        graph.edge_weights,
        [[0.0, 1.5, 0.0], [0.0, 0.0, 2.5], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_array_equal(
        graph.edge_present,
        [[False, True, False], [False, False, True], [False, False, False]],
    )


def test_read_flag_file_allows_vertex_only_graph(tmp_path):
    path = tmp_path / "vertices.flag"
    path.write_text("dim0\n1 2 3\n")

    graph = _read_flag_file(str(path))

    np.testing.assert_allclose(graph.vertex_weights, [1.0, 2.0, 3.0])
    assert not graph.edge_present.any()


def test_read_flag_file_rejects_bad_edge(tmp_path):
    path = tmp_path / "bad.flag"
    path.write_text("dim 0\n0 0\n dim 1\n0 3 1.0\n")

    with pytest.raises(ValueError, match="out of range"):
        _read_flag_file(str(path))


def test_clique_expansion_matches_bruteforce_enumeration():
    edge_weights = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [4.0, 0.0, 3.0, 1.5],
            [0.0, 2.5, 0.0, 1.0],
            [1.0, 0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    graph = _FlagData(
        vertex_weights=np.diag(edge_weights),
        edge_weights=edge_weights,
        edge_present=(edge_weights > 0) & ~np.eye(edge_weights.shape[0], dtype=bool),
    )

    expected_simplices, expected_filtrations = _bruteforce_directed_simplices(graph, 3)
    simplices, filtrations = _enumerate_directed_simplices(graph, 3)

    assert simplices == expected_simplices
    assert filtrations == expected_filtrations


def test_dflag_preserves_edge_presence_and_face_monotone_filtrations(tmp_path):
    """Every parsed edge contributes, and every coface dominates its faces."""
    sparse_path = tmp_path / "signed_edges.flag"
    sparse_path.write_text(
        """dim 0
        2.0 0.0 0.0
        dim 1
        0 1 0.0
        1 2 -1.0
        """
    )
    sparse = _read_flag_file(str(sparse_path))
    assert sparse.edge_present[0, 1]
    assert sparse.edge_present[1, 2]
    assert not sparse.edge_present[0, 2]
    assert sparse.edge_weights[0, 2] == 0.0

    complex_ = dFlag(str(sparse_path), max_dim=1)
    edge_filtrations = dict(
        zip(complex_.simplices_by_dimension[1], complex_.simplex_filtrations[1])
    )
    assert (0, 1) in edge_filtrations
    assert (1, 2) in edge_filtrations
    assert (0, 2) not in edge_filtrations
    assert edge_filtrations[(0, 1)] == pytest.approx(2.0)
    assert edge_filtrations[(1, 2)] == pytest.approx(0.0)

    clique_path = tmp_path / "negative_clique.flag"
    clique_path.write_text(
        """dim 0
        0.0 1.0 3.0 0.0
        dim 1
        0 1 -5.0
        0 2 -4.0
        0 3 -3.0
        1 2 -2.0
        1 3 -1.0
        2 3 -6.0
        """
    )
    clique = dFlag(str(clique_path), max_dim=3)
    filtration_by_simplex = {
        simplex: value
        for dim, simplices in enumerate(clique.simplices_by_dimension)
        for simplex, value in zip(simplices, clique.simplex_filtrations[dim])
    }
    for simplex, value in filtration_by_simplex.items():
        if len(simplex) == 1:
            continue
        for index in range(len(simplex)):
            face = simplex[:index] + simplex[index + 1 :]
            assert filtration_by_simplex[face] <= value

    for name, contents, match in (
        (
            "duplicate.flag",
            "dim 0\n0 0\ndim 1\n0 1 1\n0 1 2\n",
            "Duplicate",
        ),
        ("nan-edge.flag", "dim 0\n0 0\ndim 1\n0 1 nan\n", "finite"),
        ("inf-edge.flag", "dim 0\n0 0\ndim 1\n0 1 inf\n", "finite"),
        ("nan-vertex.flag", "dim 0\nnan 0\n", "Vertex weights"),
    ):
        invalid_path = tmp_path / name
        invalid_path.write_text(contents)
        with pytest.raises(ValueError, match=match):
            _read_flag_file(str(invalid_path))
