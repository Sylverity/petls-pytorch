"""Render the README crystal-topology showcase with PETLS-PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gemmi
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import petls_pytorch


BACKGROUND = "#06101d"
PANEL = "#0b1b2d"
TEXT = "#edf5ff"
MUTED = "#91a7c0"
GOLD = "#ffbd45"
CYAN = "#34ddd0"
MAGENTA = "#ff4fce"
VIOLET = "#8978ff"
COVALENT_RADIUS = {"C": 0.76, "N": 0.71, "O": 0.66}
MINIMUM_RENDERER_VERSION = (0, 8, 4)
HARMONIC_SCALE = 13.50


@dataclass(frozen=True)
class Molecule:
    kind: str
    elements: np.ndarray
    coordinates: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.coordinates, axis=0)

    @property
    def radius_of_gyration(self) -> float:
        centered = self.coordinates - self.center
        return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def cif_number(value: str) -> float:
    return float(re.sub(r"\([^)]*\)$", "", value))


def molecule_kind(label: str, element: str) -> str:
    number = int(re.search(r"\d+", label).group())
    if element == "N" or (element == "C" and number >= 8) or (element == "O" and number >= 5):
        return "theobromine"
    return "dhba"


def cartesian(cell: gemmi.UnitCell, fractional: np.ndarray) -> np.ndarray:
    positions = []
    for xyz in fractional:
        position = cell.orthogonalize(gemmi.Fractional(*xyz))
        positions.append([position.x, position.y, position.z])
    return np.asarray(positions)


def unwrap_molecule(
    cell: gemmi.UnitCell,
    labels: np.ndarray,
    fractional: np.ndarray,
    bond_pairs: list[tuple[str, str]],
) -> np.ndarray:
    """Unwrap one molecule by following its published covalent-bond graph."""
    indices = {label: index for index, label in enumerate(labels)}
    neighbors: list[list[int]] = [[] for _ in labels]
    for first, second in bond_pairs:
        if first in indices and second in indices:
            left, right = indices[first], indices[second]
            neighbors[left].append(right)
            neighbors[right].append(left)
    unwrapped = np.full_like(fractional, np.nan)
    unwrapped[0] = fractional[0]
    pending = [0]
    shifts = np.asarray(
        [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    while pending:
        source = pending.pop()
        for target in neighbors[source]:
            if np.all(np.isfinite(unwrapped[target])):
                continue
            candidates = fractional[target] + shifts
            distances = np.linalg.norm(cartesian(cell, candidates - unwrapped[source]), axis=1)
            unwrapped[target] = candidates[np.argmin(distances)]
            pending.append(target)
    if not np.all(np.isfinite(unwrapped)):
        raise ValueError("published bond graph did not connect every heavy atom")
    return unwrapped


def build_supercell(path: Path, repeats: tuple[int, int, int]) -> tuple[list[Molecule], np.ndarray]:
    """Expand the asymmetric unit and keep every molecule geometrically whole."""
    block = gemmi.cif.read_file(str(path)).sole_block()
    cell = gemmi.UnitCell(
        *[
            cif_number(block.find_value(tag))
            for tag in (
                "_cell_length_a",
                "_cell_length_b",
                "_cell_length_c",
                "_cell_angle_alpha",
                "_cell_angle_beta",
                "_cell_angle_gamma",
            )
        ]
    )
    labels = np.asarray(list(block.find_values("_atom_site_label")))
    elements = np.asarray(list(block.find_values("_atom_site_type_symbol")))
    fractional = np.column_stack(
        [
            [cif_number(value) for value in block.find_values(tag)]
            for tag in (
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            )
        ]
    )
    heavy = elements != "H"
    labels, elements, fractional = labels[heavy], elements[heavy], fractional[heavy]
    kinds = np.asarray([molecule_kind(label, element) for label, element in zip(labels, elements)])
    bond_pairs = list(
        zip(
            block.find_values("_geom_bond_atom_site_label_1"),
            block.find_values("_geom_bond_atom_site_label_2"),
        )
    )
    operations = [
        gemmi.Op(value.strip("'\""))
        for value in block.find_values("_space_group_symop_operation_xyz")
    ]
    unit_cell_molecules = []
    for operation in operations:
        transformed = np.asarray([operation.apply_to_xyz(xyz.tolist()) for xyz in fractional])
        for kind in ("theobromine", "dhba"):
            selected = kinds == kind
            molecule_fractional = unwrap_molecule(
                cell, labels[selected], transformed[selected], bond_pairs
            )
            molecule_fractional -= np.floor(np.mean(molecule_fractional, axis=0))
            unit_cell_molecules.append(
                Molecule(kind, elements[selected], cartesian(cell, molecule_fractional))
            )
    molecules = []
    for i in range(repeats[0]):
        for j in range(repeats[1]):
            for k in range(repeats[2]):
                shift = cartesian(cell, np.asarray([[i, j, k]], dtype=float))[0]
                molecules.extend(
                    Molecule(m.kind, m.elements, m.coordinates + shift) for m in unit_cell_molecules
                )
    vectors = cartesian(
        cell,
        np.asarray(
            [[repeats[0], 0, 0], [0, repeats[1], 0], [0, 0, repeats[2]]],
            dtype=float,
        ),
    )
    return molecules, vectors


def flatten_crystal(
    molecules: list[Molecule],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    elements, coordinates, bonds = [], [], []
    offset = 0
    for molecule in molecules:
        elements.extend(molecule.elements.tolist())
        coordinates.extend(molecule.coordinates.tolist())
        for first in range(len(molecule.coordinates)):
            for second in range(first + 1, len(molecule.coordinates)):
                cutoff = 1.25 * (
                    COVALENT_RADIUS[molecule.elements[first]]
                    + COVALENT_RADIUS[molecule.elements[second]]
                )
                distance = np.linalg.norm(
                    molecule.coordinates[first] - molecule.coordinates[second]
                )
                if 0.5 < distance <= cutoff:
                    bonds.append((offset + first, offset + second))
        offset += len(molecule.coordinates)
    return np.asarray(elements), np.asarray(coordinates), bonds


def betti_curve(intervals: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if intervals.size == 0:
        return np.zeros_like(scales, dtype=int)
    return np.sum(
        (intervals[:, 0, None] <= scales) & (intervals[:, 1, None] > scales),
        axis=0,
    )


def choose_harmonic_feature(
    alpha: petls_pytorch.Alpha, points: np.ndarray, scale: float
) -> list[dict[str, object]]:
    """Choose a central harmonic 2-cycle with concentrated simplex support."""
    result = alpha.harmonic_features(dim=2, a=scale, coefficient_atol=0.015, max_features=24)
    cloud_center = np.mean(points, axis=0)
    cloud_radius = np.linalg.norm(np.ptp(points, axis=0))
    ranked = []
    for feature in result["features"]:
        coefficients = feature["simplex_coefficients"]
        values = np.asarray([float(item["coefficient"]) for item in coefficients])
        strengths = values**2
        triangle_centers = np.asarray(
            [np.mean(points[item["simplex"]], axis=0) for item in coefficients]
        )
        feature_center = np.average(triangle_centers, axis=0, weights=strengths)
        inverse_participation = np.sum(strengths**2) / np.sum(strengths) ** 2
        centrality = np.exp(-3 * np.linalg.norm(feature_center - cloud_center) / cloud_radius)
        ranked.append((inverse_participation * centrality, coefficients))
    return max(ranked, key=lambda item: item[0])[1] if ranked else []


def prepare_topology(molecules: list[Molecule]):
    points = np.asarray([molecule.center for molecule in molecules])
    weights = np.asarray([molecule.radius_of_gyration**2 for molecule in molecules])
    alpha = petls_pytorch.Alpha(
        points=points,
        weights=weights,
        point_labels=[f"{m.kind}:{i}" for i, m in enumerate(molecules)],
        max_dim=3,
        max_alpha_square=18.0,
        dtype="float64",
    )
    return points, weights, alpha


def rotation_matrix(angle: float) -> np.ndarray:
    tilt = np.deg2rad(-12.0 + 4.0 * np.sin(angle))
    rotate_z = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rotate_x = np.asarray(
        [
            [1, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt), np.cos(tilt)],
        ]
    )
    return rotate_x @ rotate_z


def write_renderer_trajectory(
    path: Path,
    molecules: list[Molecule],
    points: np.ndarray,
    alpha: petls_pytorch.Alpha,
    scales: np.ndarray,
    harmonic_triangles: list[tuple[int, int, int]],
) -> None:
    """Write a V3000 SDF whose dynamic bonds are the Alpha filtration."""
    elements, atomic_coordinates, molecular_bonds = flatten_crystal(molecules)
    harmonic_vertices = sorted({vertex for triangle in harmonic_triangles for vertex in triangle})
    harmonic_index = {vertex: index for index, vertex in enumerate(harmonic_vertices)}
    harmonic_edges = {
        tuple(sorted((harmonic_index[triangle[first]], harmonic_index[triangle[second]])))
        for triangle in harmonic_triangles
        for first in range(3)
        for second in range(first + 1, 3)
    }
    all_elements = np.concatenate(
        [elements, np.full(len(points), "He"), np.full(len(harmonic_vertices), "Ne")]
    )
    all_coordinates = np.concatenate([atomic_coordinates, points, points[harmonic_vertices]])
    origin = np.mean(all_coordinates, axis=0)
    center_offset = len(elements)
    harmonic_offset = center_offset + len(points)
    edge_simplices = np.asarray(alpha.simplices_by_dimension[1], dtype=int)
    edge_filtrations = np.asarray(alpha.simplex_filtrations[1])
    with path.open("w", encoding="utf-8") as handle:
        for frame, scale in enumerate(scales):
            phase = 2 * np.pi * frame / len(scales)
            rotation = rotation_matrix(0.32 * np.sin(phase))
            rotated = (all_coordinates - origin) @ rotation.T
            active_edges = edge_simplices[edge_filtrations <= scale]
            scaffold_bonds = [
                (center_offset + int(first), center_offset + int(second))
                for first, second in active_edges
            ]
            feature_bonds = (
                [
                    (harmonic_offset + first, harmonic_offset + second)
                    for first, second in harmonic_edges
                ]
                if abs(scale - HARMONIC_SCALE) <= 1.0
                else []
            )
            bonds = molecular_bonds + scaffold_bonds + feature_bonds
            handle.write(f"PETLS crystal filtration {frame + 1}\n")
            handle.write("  PETLS-PyTorch topology showcase\n\n")
            handle.write("  0  0  0     0  0            999 V3000\n")
            handle.write("M  V30 BEGIN CTAB\n")
            handle.write(f"M  V30 COUNTS {len(rotated)} {len(bonds)} 0 0 0\n")
            handle.write("M  V30 BEGIN ATOM\n")
            for index, (element, xyz) in enumerate(zip(all_elements, rotated), start=1):
                handle.write(f"M  V30 {index} {element} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} 0\n")
            handle.write("M  V30 END ATOM\nM  V30 BEGIN BOND\n")
            for index, (first, second) in enumerate(bonds, start=1):
                handle.write(f"M  V30 {index} 1 {first + 1} {second + 1}\n")
            handle.write("M  V30 END BOND\nM  V30 END CTAB\nM  END\n")
            handle.write(f">  <ALPHA_SQUARE>\n{scale:.8f}\n\n$$$$\n")


def locate_renderer(explicit: Path | None) -> Path:
    candidate = explicit or (
        Path(os.environ["MOLECULE_RENDERER_BIN"])
        if os.environ.get("MOLECULE_RENDERER_BIN")
        else None
    )
    if candidate is None or not candidate.is_file():
        raise FileNotFoundError(
            "A compatible renderer executable is required. "
            "Pass --renderer or set MOLECULE_RENDERER_BIN."
        )
    version_text = subprocess.check_output([str(candidate), "--version"], text=True).strip()
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_text)
    if not match or tuple(map(int, match.groups())) < MINIMUM_RENDERER_VERSION:
        minimum = ".".join(map(str, MINIMUM_RENDERER_VERSION))
        raise RuntimeError(
            f"Renderer version {minimum}+ is required; {candidate} reported {version_text!r}."
        )
    return candidate


def renderer_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if Path("/mnt/wslg").is_dir():
        environment.setdefault("DISPLAY", ":0")
        environment.setdefault("WAYLAND_DISPLAY", "wayland-0")
        if Path("/mnt/wslg/runtime-dir/wayland-0").exists():
            environment.setdefault("XDG_RUNTIME_DIR", "/mnt/wslg/runtime-dir")
        environment.setdefault("GALLIUM_DRIVER", "d3d12")
        if Path("/usr/lib/wsl/lib/nvidia-smi").exists():
            environment.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
        current = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"/usr/lib/wsl/lib:{current}" if current else "/usr/lib/wsl/lib"
        )
    return environment


def render_molecular_frames(
    binary: Path,
    workspace: Path,
    molecules: list[Molecule],
    points: np.ndarray,
    alpha: petls_pytorch.Alpha,
    scales: np.ndarray,
    harmonic_triangles: list[tuple[int, int, int]],
    fps: int,
) -> list[Path]:
    trajectory_path = workspace / "crystal-filtration.sdf"
    frames_path = workspace / "renderer-frames"
    job_path = workspace / "renderer-job.json"
    write_renderer_trajectory(trajectory_path, molecules, points, alpha, scales, harmonic_triangles)
    job = {
        "version": 1,
        "topology": {"path": trajectory_path.name, "bond_perception_tolerance": 1.3},
        "trajectory": {
            "path": trajectory_path.name,
            "coordinate_space": "source",
            "bond_transitions": "instant",
            "transition_frames": 2,
        },
        "frames": {"start": 0, "end": len(scales) - 1, "stride": 1},
        "layers": [
            {
                "type": "molecule",
                "name": "experimental crystal",
                "selection": "not element He and not element Ne",
                "color": None,
                "opacity": 0.92,
                "render_profile": "cylview",
                "hydrogen_visibility": "hidden",
            },
            {
                "type": "molecule",
                "name": "Alpha scaffold",
                "selection": "element He",
                "color": VIOLET,
                "opacity": 0.28,
                "render_profile": "ball-stick",
                "hydrogen_visibility": "hidden",
            },
            {
                "type": "molecule",
                "name": "harmonic void support",
                "selection": "element Ne",
                "color": MAGENTA,
                "opacity": 1.0,
                "render_profile": "ball-stick",
                "hydrogen_visibility": "hidden",
            },
        ],
        "camera": {"policy": "fit_topology", "margin": 0.04, "frame_range": True},
        "render": {
            "width": 720,
            "height": 607,
            "background": BACKGROUND,
            "transparent": False,
            "antialias": True,
            "improved_shadows": True,
            "ambient_occlusion": True,
            "depth_aware_outline": True,
            "tone_mapping": "aces",
        },
        "output": {
            "path": frames_path.name,
            "format": "png_sequence",
            "fps": float(fps),
            "resume": False,
            "retain_png_frames": False,
            "png_directory": None,
        },
        "encoder": {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "crf": 18,
            "preset": "medium",
            "ffmpeg_path": None,
        },
        "timeline_overlay": {
            "frame_number": False,
            "simulation_time": False,
            "position": "bottom_right",
            "color": TEXT,
            "font_size": 18,
            "simulation_time_precision": 2,
        },
        "reproducibility": {
            "created_at": "2026-09-02T00:00:00Z",
            "created_by": "PETLS-PyTorch README demo",
            "source_sha256": {
                trajectory_path.name: hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
            },
        },
    }
    job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [str(binary), "render", str(job_path), "--wait", "--progress=jsonl"],
        check=True,
        cwd=workspace,
        env=renderer_environment(),
    )
    rendered = sorted(frames_path.glob("frame_*.png"))
    if len(rendered) != len(scales):
        raise RuntimeError(f"Renderer wrote {len(rendered)} frames; expected {len(scales)}.")
    return rendered


def render(
    input_path: Path,
    output_path: Path,
    poster_path: Path,
    renderer_binary: Path,
    frames: int,
    fps: int,
) -> None:
    molecules, _ = build_supercell(input_path, repeats=(4, 5, 1))
    points, _, alpha = prepare_topology(molecules)
    # Begin after every weighted vertex has entered, so the story starts with
    # 160 separate molecular components rather than an empty complex.
    curve_scales = np.linspace(-4.80, 15.65, 220)
    betti = {
        dimension: betti_curve(alpha.persistence_intervals(dimension), curve_scales)
        for dimension in (0, 1, 2)
    }
    spectrum_scales = np.linspace(-4.80, 15.65, 42)
    spectral_gaps = {1: [], 2: []}
    for scale in spectrum_scales:
        summary = alpha.topology_summary(
            dimensions=(1, 2), a=float(scale), b=float(scale), smallest_eigenvalues=2
        )
        for dimension in (1, 2):
            gap = summary["least_nonzero_eigenvalue"][dimension]
            spectral_gaps[dimension].append(np.nan if not gap else gap)
    spectral_gaps = {dimension: np.asarray(values) for dimension, values in spectral_gaps.items()}
    harmonic = choose_harmonic_feature(alpha, points, HARMONIC_SCALE)
    harmonic_triangles = [
        tuple(int(vertex) for vertex in item["simplex"])
        for item in harmonic
        if abs(float(item["coefficient"])) >= 0.08
    ]
    half = max(frames // 2, 2)
    scale_path = np.concatenate(
        [
            np.linspace(curve_scales[0], curve_scales[-1], half),
            np.linspace(curve_scales[-1], curve_scales[0], frames - half),
        ]
    )
    # Guarantee that even short smoke renders include the selected harmonic feature.
    scale_path[np.argmin(np.abs(scale_path - HARMONIC_SCALE))] = HARMONIC_SCALE

    with tempfile.TemporaryDirectory(prefix="petls-renderer-") as temporary:
        molecular_frames = render_molecular_frames(
            renderer_binary,
            Path(temporary),
            molecules,
            points,
            alpha,
            scale_path,
            harmonic_triangles,
            fps,
        )
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        fig = plt.figure(figsize=(12, 6.75), facecolor=BACKGROUND)
        grid = fig.add_gridspec(2, 5, width_ratios=[1.35, 1.35, 1.35, 1, 1], hspace=0.26)
        crystal_ax = fig.add_subplot(grid[:, :3])
        betti_ax = fig.add_subplot(grid[0, 3:])
        spectrum_ax = fig.add_subplot(grid[1, 3:])
        for axis in (crystal_ax, betti_ax, spectrum_ax):
            axis.set_facecolor(BACKGROUND if axis is crystal_ax else PANEL)
        for axis in (betti_ax, spectrum_ax):
            axis.tick_params(colors=MUTED, labelsize=8)
            axis.grid(color="#29405c", alpha=0.35, linewidth=0.7)
            for spine in axis.spines.values():
                spine.set_color("#29405c")

        crystal_image = crystal_ax.imshow(
            np.asarray(Image.open(molecular_frames[0]).convert("RGB"))
        )
        crystal_ax.set_axis_off()
        for dimension, color, name in (
            (0, GOLD, "components"),
            (1, MAGENTA, "tunnels"),
            (2, CYAN, "voids"),
        ):
            betti_ax.plot(
                curve_scales,
                betti[dimension],
                color=color,
                linewidth=2,
                label=rf"{name}  $\beta_{dimension}$",
            )
        betti_ax.set_xlim(curve_scales[0], curve_scales[-1])
        betti_ax.set_ylim(-4, max(max(values) for values in betti.values()) * 1.06)
        betti_ax.set_title("What exists across scale?", color=TEXT, loc="left", fontsize=12, pad=10)
        betti_ax.set_ylabel("number of features", color=MUTED)
        betti_ax.legend(frameon=False, labelcolor=TEXT, loc="upper center", ncol=1, fontsize=8)

        for dimension, color in ((1, MAGENTA), (2, CYAN)):
            valid = np.isfinite(spectral_gaps[dimension])
            spectrum_ax.plot(
                spectrum_scales[valid],
                spectral_gaps[dimension][valid],
                color=color,
                linewidth=2,
                label=rf"$\lambda_{dimension}^+$",
            )
        spectrum_ax.set_xlim(curve_scales[0], curve_scales[-1])
        spectrum_ax.set_ylim(bottom=0)
        spectrum_ax.set_title(
            "How are features coupled?", color=TEXT, loc="left", fontsize=12, pad=10
        )
        spectrum_ax.set_xlabel(r"filtration scale  $\alpha^2$  (Å²)  →", color=MUTED)
        spectrum_ax.set_ylabel("smallest positive eigenvalue", color=MUTED)
        spectrum_ax.legend(frameon=False, labelcolor=TEXT, loc="upper right", ncol=2)
        betti_cursor = betti_ax.axvline(curve_scales[0], color=TEXT, linewidth=1, alpha=0.75)
        spectrum_cursor = spectrum_ax.axvline(curve_scales[0], color=TEXT, linewidth=1, alpha=0.75)
        scale_label = crystal_ax.text(
            0.02,
            0.96,
            "",
            transform=crystal_ax.transAxes,
            color=TEXT,
            fontsize=10.5,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": PANEL,
                "edgecolor": "#29405c",
            },
        )
        void_label = crystal_ax.text(
            0.02,
            0.06,
            "Localized void signature: strongest harmonic 2-cycle support",
            transform=crystal_ax.transAxes,
            color=MAGENTA,
            fontsize=11,
            alpha=0.35,
        )
        fig.text(
            0.035,
            0.965,
            "Topology inside a molecular crystal",
            color=TEXT,
            fontsize=19,
            weight="bold",
            va="top",
        )
        fig.text(
            0.035,
            0.925,
            "160 molecules  •  experimental X-ray structure  •  PETLS-PyTorch",
            color=MUTED,
            fontsize=10.5,
            va="top",
        )
        fig.text(
            0.79,
            0.02,
            "Finite 4×5×1 supercell • molecular centers weighted by heavy-atom Rg²",
            color=MUTED,
            fontsize=8,
            ha="center",
        )

        def update(frame: int):
            scale = float(scale_path[frame])
            crystal_image.set_data(np.asarray(Image.open(molecular_frames[frame]).convert("RGB")))
            current_betti = {
                dimension: int(
                    betti_curve(alpha.persistence_intervals(dimension), np.asarray([scale]))[0]
                )
                for dimension in (0, 1, 2)
            }
            betti_cursor.set_xdata([scale, scale])
            spectrum_cursor.set_xdata([scale, scale])
            scale_label.set_text(
                rf"scale  $\alpha^2$ = {scale:5.2f} Å²"
                "\n"
                rf"components $\beta_0$ = {current_betti[0]}  •  "
                rf"tunnels $\beta_1$ = {current_betti[1]}  •  "
                rf"voids $\beta_2$ = {current_betti[2]}"
            )
            void_alpha = float(np.exp(-(((scale - HARMONIC_SCALE) / 0.9) ** 2)))
            void_label.set_alpha(void_alpha)
            return (
                crystal_image,
                betti_cursor,
                spectrum_cursor,
                scale_label,
                void_label,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        poster_frame = int(np.argmin(np.abs(scale_path - HARMONIC_SCALE)))
        update(poster_frame)
        fig.savefig(poster_path, dpi=150, facecolor=BACKGROUND, bbox_inches="tight")
        movie = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
        movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=90)
        plt.close(fig)


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=example_dir / "7235246.cif")
    parser.add_argument(
        "--output",
        type=Path,
        default=example_dir.parent.parent / "docs/assets/theobromine-crystal-topology.gif",
    )
    parser.add_argument(
        "--poster",
        type=Path,
        default=example_dir.parent.parent / "docs/assets/theobromine-crystal-topology.png",
    )
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--frames", type=int, default=46)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    render(
        args.input,
        args.output,
        args.poster,
        locate_renderer(args.renderer),
        args.frames,
        args.fps,
    )


if __name__ == "__main__":
    main()
