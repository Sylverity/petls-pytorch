"""
PETLS — GPU-Native Persistent Topological Laplacians in PyTorch.

This module exposes the supported PyTorch API without compatibility aliases or
third-party namespace re-exports.
"""

from petls_pytorch.core.complex import Complex, LaplacianSizeError
from petls_pytorch.core.profile import Profile, Timer

from petls_pytorch.variants.alpha import Alpha
from petls_pytorch.variants.dflag import dFlag
from petls_pytorch.variants.rips import Rips
from petls_pytorch.variants.sheaf import sheaf_simplex_tree, PersistentSheafLaplacian
from petls_pytorch.utils.plotting import summaries, plot_summary
from petls_pytorch.utils.simplex_tree import simplex_tree_boundaries_filtrations

__version__ = "1.1.1"


__all__ = [
    "Complex",
    "LaplacianSizeError",
    "Profile",
    "Timer",
    "Alpha",
    "dFlag",
    "Rips",
    "sheaf_simplex_tree",
    "PersistentSheafLaplacian",
    "summaries",
    "plot_summary",
    "simplex_tree_boundaries_filtrations",
]
