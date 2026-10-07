"""Read-only reproduction: F_11 homology versus real Hodge spectrum.
Run from project: PYTHONPATH=src .venv/bin/python docs/research/openai-math-2026-10-06/torsion11_reproduction.py
All simplices have filtration zero. This is a triangulated mapping cone of
S^1 -> S^1 of degree 11: a 33-edge top cycle maps around a 3-edge bottom cycle
11 times, then the top cycle is coned off.
"""

import json
import numpy as np
import torch
import gudhi
from petls_pytorch.core.complex import Complex

torch.set_num_threads(2)
st = gudhi.SimplexTree()
for i in range(33):
    ti, tj = 3 + i, 3 + (i + 1) % 33
    bi, bj = i % 3, (i + 1) % 3
    st.insert([ti, bi, bj], filtration=0.0)
    st.insert([ti, tj, bj], filtration=0.0)
    st.insert([36, ti, tj], filtration=0.0)
# Stellar subdivision preserves homeomorphism type and breaks spectral symmetry.
st.remove_maximal_simplex([36, 3, 4])
for edge in ([36, 3], [36, 4], [3, 4]):
    st.insert([37, *edge], filtration=0.0)
counts = [sum(len(s) == q + 1 for s, _ in st.get_simplices()) for q in range(3)]
result = {"simplex_counts": counts, "euler_characteristic": counts[0] - counts[1] + counts[2]}
for field in (2, 11):
    copy = st.copy()
    copy.persistence(homology_coeff_field=field, min_persistence=-1.0, persistence_dim_max=True)
    result[f"betti_F{field}"] = copy.betti_numbers()
complex_ = Complex(simplex_tree=st, dtype=torch.float64, device="cpu")
b1 = complex_.filtered_boundaries[1].matrix.to_dense()
b2 = complex_.filtered_boundaries[2].matrix.to_dense()
result["boundary_squared_max_abs"] = float((b1 @ b2).abs().max())
L = complex_.get_L(1, 0.0, 0.0)
values = torch.linalg.eigvalsh(L).numpy()
result["real_H1_nullity_at_1e-9"] = int((np.abs(values) <= 1e-9).sum())
result["full_H1_first_5_eigenvalues"] = values[:5].tolist()
result["library_F11_betti_H1"] = complex_.persistent_betti(1, 0.0, 0.0)
for count in (8, 24, 128):
    partial = complex_.positive_spectrum(1, 0.0, 0.0, positive_modes=count)
    result[f"partial_{count}"] = {
        key: partial[key]
        for key in (
            "betti",
            "spectral_nullity",
            "least_nonzero_eigenvalue",
            "spectrum_solver",
            "calculation_status",
            "requested_positive_modes",
            "working_block_size",
        )
    }
    result[f"partial_{count}"]["first_3_modes"] = partial["positive_modes"][:3]
print(json.dumps(result, indent=2))
