"""Tests for Complex construction from a Gudhi SimplexTree."""

from __future__ import annotations

import gudhi

from petls_pytorch.core.complex import Complex


def test_simplex_tree_constructor():
    st = gudhi.SimplexTree()
    st.insert([0, 1])
    st.insert([0, 1, 2], filtration=4.0)

    pl = Complex(simplex_tree=st)
    assert pl.top_dim == 2

    spectra = pl.spectra()
    assert isinstance(spectra, list)
    # Should have entries for dims 0, 1, 2
    dims = [s[0] for s in spectra]
    assert 0 in dims
    assert 1 in dims
    assert 2 in dims


def test_simplex_tree_filtrations():
    st = gudhi.SimplexTree()
    st.insert([0], filtration=0.0)
    st.insert([1], filtration=1.0)
    st.insert([2], filtration=2.0)
    st.insert([0, 1], filtration=3.0)
    st.insert([0, 2], filtration=4.0)
    st.insert([1, 2], filtration=5.0)
    st.insert([0, 1, 2], filtration=6.0)

    pl = Complex(simplex_tree=st)
    filts = pl.get_all_filtrations()
    # get_all_filtrations returns domain filtrations only (edges, triangles, etc.)
    # plus the dummy d_0 domain [0.0]. Vertex filtrations are range filtrations
    # of d_1 and are NOT included.
    assert filts == [0.0, 3.0, 4.0, 5.0, 6.0]
