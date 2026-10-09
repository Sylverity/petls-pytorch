"""Exercise a wheel installed in an isolated target directory."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_packages", type=Path)
    args = parser.parse_args()
    target = args.site_packages.resolve()
    sys.path.insert(0, str(target))
    import petls_pytorch

    if not Path(petls_pytorch.__file__).resolve().is_relative_to(target):
        raise RuntimeError("Smoke test imported the checkout instead of the installed wheel")
    alpha = petls_pytorch.Alpha(
        points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        weights=[0.20, 0.18, 0.20, 0.18],
        max_dim=2,
        max_alpha_square=1.0,
        dtype=torch.float64,
        device="cpu",
    )
    summary = alpha.topology_summary((0, 1), a=0.1, b=0.1)
    assert summary["betti"] == {0: 1, 1: 1}
    assert summary["homology_coeff_field"] == 11
    modes = alpha.positive_spectrum(1, 0.1)
    np.testing.assert_allclose([mode["value"] for mode in modes["positive_modes"]], [2, 2, 4])
    assert all(mode["quality"] == "certified" for mode in modes["positive_modes"])
    participation = alpha.harmonic_participation(1, 0.1)
    assert participation["participation_complete"]
    np.testing.assert_allclose(participation["total_participation"], 1)
    np.testing.assert_allclose(
        [p["participation"] for p in participation["point_participation"]], [0.25] * 4
    )
    print(
        f"Installed wheel {petls_pytorch.__version__} passed spectrum and participation smoke tests."
    )


if __name__ == "__main__":
    main()
