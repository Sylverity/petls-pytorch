"""
Benchmark dataset generation.

Creates reproducible synthetic geometric datasets at multiple scales:
  - torus      : classic TDA benchmark, nontrivial homology H0, H1, H2
  - sphere     : 3-sphere embedded in R^4 (or 2-sphere in R^3)
  - swiss_roll : nonlinear manifold, tests non-uniform sampling
  - klein_bottle: non-orientable surface, interesting topology

Each dataset produces point clouds that are fed into Alpha or Rips complexes.
"""

import numpy as np
import tadasets
from typing import Dict, List, Optional
import json
from pathlib import Path

DATASET_REGISTRY: Dict[str, dict] = {
    "torus": {
        "generator": "torus",
        "params": {"c": 3, "a": 1, "noise": 0.05},
        "description": "Torus in R^3, H0=Z, H1=Z^2, H2=Z",
    },
    "sphere": {
        "generator": "sphere",
        "params": {"r": 1.0, "noise": 0.02},
        "description": "2-sphere in R^3, H0=Z, H1=0, H2=Z",
    },
    "swiss_roll": {
        "generator": "swiss_roll",
        "params": {"noise": 0.05},
        "description": "Swiss roll in R^3, contractible",
    },
    "klein_bottle": {
        "generator": "klein_bottle",
        "params": {"noise": 0.03},
        "description": "Klein bottle in R^4 (projected), H0=Z, H1=Z+Z/2Z, H2=0",
    },
}


def _klein_bottle(n: int, noise: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate n points on a Klein bottle embedded in R^4.
    Parametrization: (u,v) -> ((r+cos(v))cos(u), (r+cos(v))sin(u), sin(v)cos(u/2), sin(v)sin(u/2))
    with r=2, then projected to R^3 for Alpha complex compatibility.
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    r = 2.0
    x = (r + np.cos(v)) * np.cos(u)
    y = (r + np.cos(v)) * np.sin(u)
    z = np.sin(v) * np.cos(u / 2)
    pts = np.column_stack([x, y, z])
    if noise > 0:
        pts += rng.normal(0, noise, pts.shape)
    return pts


def generate_point_cloud(name: str, n: int, seed: int = 42) -> np.ndarray:
    """Generate a point cloud of n points for the named dataset."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")

    entry = DATASET_REGISTRY[name]
    gen = entry["generator"]
    params = dict(entry["params"])
    params["seed"] = seed

    if gen == "torus":
        params["n"] = n
        return tadasets.torus(**params)
    elif gen == "sphere":
        params["n"] = n
        return tadasets.sphere(**params)
    elif gen == "swiss_roll":
        params["n"] = n
        return tadasets.swiss_roll(**params)
    elif gen == "klein_bottle":
        return _klein_bottle(n, noise=params.get("noise", 0.0), seed=seed)
    else:
        raise ValueError(f"Unknown generator '{gen}'")


def choose_filtrations(
    filtrations: List[float], num_samples: int, mode: str = "quantile"
) -> List[float]:
    """
    Select a sparse set of filtration values from the full list.

    Parameters
    ----------
    filtrations : list of float
        Sorted list of all unique filtration values.
    num_samples : int
        How many values to pick.
    mode : str
        'quantile'  -> evenly spaced quantiles
        'log'       -> logarithmically spaced (good when range spans orders of magnitude)
        'early'     -> densely sample early filtrations where complex grows fastest
    """
    if len(filtrations) <= num_samples:
        return filtrations

    arr = np.array(filtrations)
    if mode == "quantile":
        idx = np.linspace(0, len(arr) - 1, num_samples, dtype=int)
        return arr[idx].tolist()
    elif mode == "log":
        # Log spacing over index space
        idx = np.unique(np.geomspace(1, len(arr), num_samples).astype(int) - 1)
        return arr[idx].tolist()
    elif mode == "early":
        # Densely sample first half, sparsely second half
        n_early = num_samples // 2
        n_late = num_samples - n_early
        idx_early = np.linspace(0, len(arr) // 2, n_early, dtype=int)
        idx_late = np.linspace(len(arr) // 2, len(arr) - 1, n_late, dtype=int)
        idx = np.unique(np.concatenate([idx_early, idx_late]))
        return arr[idx].tolist()
    else:
        raise ValueError(f"Unknown mode '{mode}'")


def generate_dataset(
    name: str,
    n_points: int,
    complex_type: str = "alpha",
    max_dim: int = 3,
    num_filtrations: int = 20,
    filtration_mode: str = "quantile",
    seed: int = 42,
    cache_dir: Optional[str] = None,
    package: str = "petls_pytorch",
    device: Optional[str] = None,
) -> dict:
    """
    Generate a complete benchmark dataset: point cloud -> complex -> sampled filtrations.

    Returns a dict with:
        points          : np.ndarray, shape (n_points, 3)
        complex_type    : 'alpha' or 'rips'
        package         : 'petls' or 'petls_pytorch'
        max_dim         : int
        filtrations     : List[float] (sampled subset)
        all_filtrations : List[float] (full set)
        metadata        : dict
    """
    package = package.lower()
    if package == "petls":
        import petls

        Alpha = petls.Alpha
        Rips = petls.Rips
    elif package == "petls_pytorch":
        import petls_pytorch

        if device is not None:
            petls_pytorch.set_device(device)
        Alpha = petls_pytorch.Alpha
        Rips = petls_pytorch.Rips
    else:
        raise ValueError(f"package must be 'petls' or 'petls_pytorch', got {package}")

    points = generate_point_cloud(name, n_points, seed=seed)

    if complex_type.lower() == "alpha":
        complex_obj = Alpha(points=points.tolist(), max_dim=max_dim)
    elif complex_type.lower() == "rips":
        # For Rips we need a threshold; set it to cover the diameter
        from scipy.spatial.distance import pdist

        diam = pdist(points).max() * 1.1
        complex_obj = Rips(points=points.tolist(), max_dim=max_dim, threshold=float(diam))
    else:
        raise ValueError(f"complex_type must be 'alpha' or 'rips', got {complex_type}")

    all_filts = complex_obj.get_all_filtrations()
    sampled_filts = choose_filtrations(all_filts, num_filtrations, mode=filtration_mode)

    # Compute matrix size statistics at sampled filtrations
    matrix_stats = []
    for f in sampled_filts:
        for dim in range(max_dim + 1):
            try:
                L = complex_obj.get_L(dim, f, f)
                matrix_stats.append(
                    {
                        "filtration": float(f),
                        "dim": dim,
                        "rows": int(L.shape[0]),
                    }
                )
            except Exception:
                matrix_stats.append(
                    {
                        "filtration": float(f),
                        "dim": dim,
                        "rows": 0,
                    }
                )

    result = {
        "name": name,
        "n_points": n_points,
        "complex_type": complex_type,
        "package": package,
        "max_dim": max_dim,
        "points": points,
        "complex": complex_obj,
        "filtrations": sampled_filts,
        "all_filtrations": all_filts,
        "num_unique_filtrations": len(all_filts),
        "matrix_stats": matrix_stats,
        "metadata": {
            "seed": seed,
            "filtration_mode": filtration_mode,
            "package": package,
            "device": device,
            **DATASET_REGISTRY[name]["params"],
        },
    }

    if cache_dir:
        p = Path(cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in result.items() if k not in ("points", "complex", "matrix_stats")}
        meta["matrix_stats"] = matrix_stats
        with open(p / f"{name}_{n_points}_{complex_type}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        np.save(p / f"{name}_{n_points}_points.npy", points)

    return result


def list_datasets() -> List[str]:
    """Return available dataset names."""
    return list(DATASET_REGISTRY.keys())
