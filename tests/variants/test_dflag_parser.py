"""Tests for dFlag .flag parsing and directed simplex enumeration."""

from __future__ import annotations

import numpy as np
import pytest

from petls_pytorch.variants.dflag import _enumerate_directed_simplices, _read_flag_file


def _bruteforce_directed_simplices(adj: np.ndarray, max_dim: int):
    n = adj.shape[0]
    simplices = [[] for _ in range(max_dim + 1)]
    filtrations = [[] for _ in range(max_dim + 1)]

    for v in range(n):
        simplices[0].append((v,))
        filtrations[0].append(float(np.float32(adj[v, v])))

    for dim in range(1, max_dim + 1):
        for code in range(n ** (dim + 1)):
            tup = []
            tmp = code
            for _ in range(dim + 1):
                tup.append(tmp % n)
                tmp //= n
            tup = tuple(tup)
            if len(set(tup)) != dim + 1:
                continue

            max_weight = 0.0
            for i in range(dim + 1):
                for j in range(i + 1, dim + 1):
                    weight = adj[tup[i], tup[j]]
                    if weight <= 0 or weight == np.inf:
                        break
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

    adj = _read_flag_file(str(path))

    expected = np.array(
        [
            [0.5, 1.5, 0.0],
            [0.0, 0.25, 2.5],
            [0.0, 0.0, 0.75],
        ]
    )
    np.testing.assert_allclose(adj, expected)


def test_read_flag_file_allows_vertex_only_graph(tmp_path):
    path = tmp_path / "vertices.flag"
    path.write_text("dim0\n1 2 3\n")

    adj = _read_flag_file(str(path))

    np.testing.assert_allclose(adj, np.diag([1.0, 2.0, 3.0]))


def test_read_flag_file_rejects_bad_edge(tmp_path):
    path = tmp_path / "bad.flag"
    path.write_text("dim 0\n0 0\n dim 1\n0 3 1.0\n")

    with pytest.raises(ValueError, match="out of range"):
        _read_flag_file(str(path))


def test_clique_expansion_matches_bruteforce_enumeration():
    adj = np.array(
        [
            [0.0, 1.0, 2.0, 0.0],
            [4.0, 0.0, 3.0, 1.5],
            [0.0, 2.5, 0.0, 1.0],
            [1.0, 0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected_simplices, expected_filtrations = _bruteforce_directed_simplices(adj, 3)
    simplices, filtrations = _enumerate_directed_simplices(adj, 3)

    assert simplices == expected_simplices
    assert filtrations == expected_filtrations
