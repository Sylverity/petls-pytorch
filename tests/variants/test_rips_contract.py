"""Public Rips dimension contract tests."""

from __future__ import annotations

import gudhi
import numpy as np
import torch

from petls_pytorch import Rips


def _betti_at(simplex_tree, dim: int, scale: float) -> int:
    simplex_tree.persistence(min_persistence=-1.0, persistence_dim_max=True)
    intervals = simplex_tree.persistence_intervals_in_dimension(dim)
    return int(np.sum((intervals[:, 0] <= scale) & (intervals[:, 1] > scale)))


def test_rips_public_dimension_keeps_support_boundary_internal():
    """Public dimensions stop at max_dim while its up term remains available."""
    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    scale = 2.0

    for max_dim in (0, 1, 2):
        rips = Rips(points=points, max_dim=max_dim, threshold=scale)

        assert rips.top_dim == max_dim
        assert rips._boundary_top_dim >= max_dim

        spectra = rips.spectra()
        assert {item[0] for item in spectra} == set(range(max_dim + 1))
        assert rips.spectra(max_dim + 1, scale, scale) == []
        assert set(rips.topology_summary(a=scale, b=scale)["betti"]) == set(range(max_dim + 1))

        laplacian = rips.get_L(max_dim, scale, scale)
        down = rips.get_down(max_dim, scale)
        up = rips.get_up(max_dim, scale, scale)
        assert torch.count_nonzero(up) > 0
        torch.testing.assert_close(laplacian, down + up)

        reference = gudhi.RipsComplex(
            points=points,
            max_edge_length=scale,
        ).create_simplex_tree(max_dimension=max_dim + 1)
        assert rips.betti_numbers_at(scale)[max_dim] == _betti_at(
            reference,
            max_dim,
            scale,
        )
