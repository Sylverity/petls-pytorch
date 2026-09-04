"""Render the static PETLS-PyTorch topology hero used by the README."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

import petls_pytorch


FILTRATION_SCALE = 0.30
MAJOR_RADIUS = 2.60
MINOR_RADIUS = 1.00
CYAN = "#27e6d1"
MAGENTA = "#ff3dbb"


def torus_points(n_major: int = 30, n_minor: int = 15) -> np.ndarray:
    """Sample a deterministic torus point cloud."""
    major_angles = np.linspace(0.0, 2 * np.pi, n_major, endpoint=False)
    minor_angles = np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False)
    return np.asarray(
        [
            [
                (MAJOR_RADIUS + MINOR_RADIUS * np.cos(v)) * np.cos(u),
                (MAJOR_RADIUS + MINOR_RADIUS * np.cos(v)) * np.sin(u),
                MINOR_RADIUS * np.sin(v),
            ]
            for u in major_angles
            for v in minor_angles
        ]
    )


def gf2_rank(matrix: np.ndarray) -> int:
    """Return matrix rank over the two-element field."""
    reduced = np.asarray(matrix, dtype=np.uint8).copy()
    pivot_row = 0
    for column in range(reduced.shape[1]):
        candidates = np.flatnonzero(reduced[pivot_row:, column])
        if candidates.size == 0:
            continue
        selected = pivot_row + int(candidates[0])
        reduced[[pivot_row, selected]] = reduced[[selected, pivot_row]]
        other_rows = np.flatnonzero(reduced[:, column])
        other_rows = other_rows[other_rows != pivot_row]
        reduced[other_rows] ^= reduced[pivot_row]
        pivot_row += 1
        if pivot_row == reduced.shape[0]:
            break
    return pivot_row


def render(output: Path) -> None:
    n_major, n_minor = 30, 15
    points = torus_points(n_major, n_minor)
    major_angles = np.repeat(np.linspace(0.0, 2 * np.pi, n_major, endpoint=False), n_minor)
    minor_angles = np.tile(np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False), n_major)
    surface_normals = np.column_stack(
        [
            np.cos(major_angles) * np.cos(minor_angles),
            np.sin(major_angles) * np.cos(minor_angles),
            np.sin(minor_angles),
        ]
    )
    lifted_points = points + 0.075 * surface_normals
    alpha = petls_pytorch.Alpha(
        points=points,
        max_dim=3,
        max_alpha_square=0.50,
        dtype="float64",
    )
    betti = alpha.betti_numbers_at(FILTRATION_SCALE)
    if [betti[dimension] for dimension in (0, 1, 2)] != [1, 2, 1]:
        raise RuntimeError(f"unexpected torus topology at hero scale: {betti}")

    triangles = np.asarray(alpha.simplices_by_dimension[2], dtype=int)
    triangle_filtrations = np.asarray(alpha.simplex_filtrations[2])
    visible = triangle_filtrations <= FILTRATION_SCALE
    triangles = triangles[visible]
    triangle_filtrations = triangle_filtrations[visible]

    lower, upper = np.quantile(triangle_filtrations, [0.03, 0.97])
    color_position = np.clip((triangle_filtrations - lower) / (upper - lower), 0.0, 1.0)
    face_colors = colormaps["turbo"](0.05 + 0.45 * color_position)
    face_colors[:, 3] = 0.92

    fig = plt.figure(figsize=(12, 7.4), facecolor="none")
    axis = fig.add_subplot(111, projection="3d", computed_zorder=False)
    axis.set_facecolor((0, 0, 0, 0))
    surface = Poly3DCollection(
        points[triangles],
        facecolors=face_colors,
        edgecolors=(1.0, 1.0, 1.0, 0.48),
        linewidths=0.32,
        antialiased=True,
        zorder=1,
    )
    axis.add_collection3d(surface)

    edge_filtrations = {
        tuple(sorted(map(int, edge))): float(filtration)
        for edge, filtration in zip(alpha.simplices_by_dimension[1], alpha.simplex_filtrations[1])
    }
    fixed_minor = 5
    major_loop = [
        (
            major * n_minor + fixed_minor,
            ((major + 1) % n_major) * n_minor + fixed_minor,
        )
        for major in range(n_major)
    ]
    fixed_major = 26
    minor_loop = [
        (
            fixed_major * n_minor + minor,
            fixed_major * n_minor + (minor + 1) % n_minor,
        )
        for minor in range(n_minor)
    ]
    for loop in (major_loop, minor_loop):
        if any(edge_filtrations[tuple(sorted(edge))] > FILTRATION_SCALE for edge in loop):
            raise RuntimeError("representative cycle is not present at the hero scale")

    active_edges = sorted(
        edge for edge, filtration in edge_filtrations.items() if filtration <= FILTRATION_SCALE
    )
    edge_index = {edge: index for index, edge in enumerate(active_edges)}
    boundary_2 = np.zeros((len(active_edges), len(triangles)), dtype=np.uint8)
    for triangle_index, triangle in enumerate(triangles):
        for first, second in ((0, 1), (0, 2), (1, 2)):
            boundary_2[
                edge_index[tuple(sorted((int(triangle[first]), int(triangle[second]))))],
                triangle_index,
            ] = 1
    cycle_vectors = np.zeros((len(active_edges), 2), dtype=np.uint8)
    for cycle_index, loop in enumerate((major_loop, minor_loop)):
        for edge in loop:
            cycle_vectors[edge_index[tuple(sorted(edge))], cycle_index] = 1
    boundary_rank = gf2_rank(boundary_2)
    if gf2_rank(np.column_stack([boundary_2, cycle_vectors])) != boundary_rank + 2:
        raise RuntimeError("highlighted cycles do not form an independent homology basis")

    for loop, color in ((major_loop, MAGENTA), (minor_loop, CYAN)):
        segments = lifted_points[np.asarray(loop)]
        axis.add_collection3d(
            Line3DCollection(
                segments,
                colors="#17233f",
                linewidths=7.0,
                alpha=0.72,
                zorder=10,
            )
        )
        axis.add_collection3d(
            Line3DCollection(
                segments,
                colors=color,
                linewidths=4.8,
                alpha=1.0,
                zorder=11,
            )
        )

    axis.set_xlim(-3.85, 3.85)
    axis.set_ylim(-3.85, 3.85)
    axis.set_zlim(-1.25, 1.25)
    axis.set_box_aspect((1.65, 1.65, 0.62))
    axis.set_proj_type("persp", focal_length=0.86)
    axis.view_init(elev=25, azim=-43, roll=-3)
    axis.set_axis_off()
    fig.subplots_adjust(left=-0.06, right=1.06, bottom=-0.16, top=1.12)

    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, transparent=True, pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    raw = Image.open(buffer).convert("RGBA")
    bounds = raw.getchannel("A").getbbox()
    if bounds is None:
        raise RuntimeError("hero render was empty")
    cropped = raw.crop(bounds)
    canvas = Image.new("RGBA", (1600, 1000), (255, 255, 255, 0))
    cropped.thumbnail((1500, 920), Image.Resampling.LANCZOS)
    offset = ((canvas.width - cropped.width) // 2, (canvas.height - cropped.height) // 2)
    canvas.alpha_composite(cropped, offset)
    canvas.save(output, optimize=True)


def main() -> None:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "docs/assets/petls-topology-hero.png",
    )
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
