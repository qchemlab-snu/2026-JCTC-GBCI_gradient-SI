"""SI figure: SA-CASSCF energies evaluated on the CASCI S1 NEB geometries.

Both methods are evaluated on the *same* geometries -- the full CASCI NEB path,
S0 geometry -> planar -> TICT -- so the curves differ only by electronic-structure
method, not by structure. That is the point of the figure: it isolates what
SA-CASSCF says about a path that CASCI produced.

The path is stored as two NEB segments that share their junction geometry
(the planar structure); the duplicate junction image is dropped when they are
joined, and a tick marks where the two segments meet.

Panel (a) shows S0 and S1 for both methods, each referenced to its own S0 at the
planar end so the two methods can be overlaid. Panel (b) shows the S1 - S0 gap,
which is the quantity that is directly comparable between methods without any
reference choice.

Encoding: colour = method, line style = state. Only two hues are used, and the
pair is checked for colour-vision deficiency separation (OKLab dE >= 8 under
simulated protanopia/deuteranopia/tritanopia), so the figure survives greyscale
and CVD readers; state is carried by line style, never by colour alone.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PP_DIR = SCRIPT_DIR.parent

# The two CASCI NEB segments that make up the path, in order. Both methods are
# read from the same image directory, so the geometries are identical by
# construction.
# Two reaction paths, each built from two NEB segments that share their junction
# geometry (the planar structure). SA-CASSCF is evaluated on both paths, so the
# panels differ only in which method produced the geometries.
PATHS = {
    "CASCI": {
        "label": "CASCI geometries",
        "segments": [
            {
                "name": "ground_state_geometry_to_planar",
                "image_dir": SCRIPT_DIR / "pp_s1_pes_image_outputs" / "casci_ground_to_planar",
                "pattern": "pp_s1_pes_casci_ground_to_planar_*.xyz",
                "native_npz": (
                    SCRIPT_DIR / "pp_s1_pes_image_outputs" / "casci_ground_to_planar"
                    / "pp_s0_s1_pes_casci_ground_to_planar_single_point_energies.npz"
                ),
                "casscf_csv": (
                    PP_DIR / "casscf" / "pp_casscf_S1_pes_ground_to_planar_outputs"
                    / "pp_casscf_S1_pes_ground_to_planar_summary.csv"
                ),
            },
            {
                "name": "planar_to_TICT",
                "image_dir": PP_DIR / "casci" / "pp_casci_S1_neb_outputs",
                "pattern": "pp_casci_S1_neb_*.xyz",
                "native_npz": (
                    PP_DIR / "casci" / "pp_casci_S1_neb_outputs"
                    / "pp_casci_S0_S1_neb_single_point_energies.npz"
                ),
                "casscf_csv": (
                    PP_DIR / "casscf" / "pp_casscf_S1_neb_image_outputs"
                    / "pp_casscf_S1_neb_images_summary.csv"
                ),
            },
        ],
    },
    "GBCI": {
        "label": "GBCI geometries",
        "segments": [
            {
                "name": "ground_state_geometry_to_planar",
                "image_dir": SCRIPT_DIR / "pp_s1_pes_image_outputs" / "gbci_ground_to_planar",
                "pattern": "pp_s1_pes_gbci_ground_to_planar_*.xyz",
                "native_npz": (
                    SCRIPT_DIR / "pp_s1_pes_image_outputs" / "gbci_ground_to_planar"
                    / "pp_s0_s1_pes_gbci_ground_to_planar_single_point_energies.npz"
                ),
                "casscf_csv": (
                    PP_DIR / "casscf" / "pp_casscf_S1_gbci_ground_to_planar_outputs"
                    / "pp_casscf_S1_gbci_ground_to_planar_summary.csv"
                ),
            },
            {
                "name": "planar_to_TICT",
                "image_dir": PP_DIR / "gbci" / "pp_gbci_S1_planar_tict_neb_outputs",
                "pattern": "pp_gbci_S1_planar_tict_neb_*.xyz",
                "native_npz": (
                    PP_DIR / "gbci" / "pp_gbci_S1_planar_tict_neb_outputs"
                    / "pp_gbci_S0_S1_planar_tict_neb_single_point_energies.npz"
                ),
                "casscf_csv": (
                    PP_DIR / "casscf" / "pp_casscf_S1_gbci_path_outputs"
                    / "pp_casscf_S1_gbci_path_summary.csv"
                ),
            },
        ],
    },
}

OUTPUT_PREFIX = SCRIPT_DIR / "pp_si_casscf_on_casci_neb_figure"

# colour = method. Checked for CVD separation: minimum OKLab dE x100 across
# protan/deutan/tritan is 24.6, normal vision 35.7 (floors are 8 and 15).
# CASCI keeps the blue it has in the main PES figure.
METHOD_STYLE = {
    "CASCI": {"color": "#1f77b4", "label": "CASCI"},
    "GBCI": {"color": "#d62728", "label": "GBCI"},
    "CASSCF": {"color": "#ff7f0e", "label": "SA-CASSCF"},
}
# line style = state, so identity never rests on colour alone.
STATE_STYLE = {
    "S0": {"linestyle": "--", "linewidth": 1.7, "marker": "s", "markersize": 4.5},
    "S1": {"linestyle": "-", "linewidth": 2.2, "marker": "o", "markersize": 5.5},
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"

KEY_POINT_LABELS = ("S$_0$", "planar", "TICT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the SI figure comparing CASCI and SA-CASSCF energies on the "
            "full CASCI NEB path (S0 geometry -> planar -> TICT)."
        )
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUT_PREFIX,
        help="Output path without extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to save, e.g. png pdf svg.",
    )
    parser.add_argument(
        "--states",
        choices=("s1", "both"),
        default="s1",
        help=(
            "Which states panel (a) shows. 's1' (default) drops the S0 curves so "
            "the y-range is not dominated by the S0/S1 separation, which is what "
            "makes the small CASCI S1 barrier visible."
        ),
    )
    parser.add_argument(
        "--x-axis",
        choices=("reaction_coordinate", "image_index"),
        default="reaction_coordinate",
        help="Horizontal axis quantity.",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure window.")
    return parser.parse_args()


def read_xyz_positions(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].split()[0])
    return np.asarray(
        [[float(value) for value in line.split()[1:4]] for line in lines[2 : 2 + n_atoms]],
        dtype=float,
    )


def reaction_coordinate(image_paths: list[Path]) -> np.ndarray:
    """Normalised cumulative RMS displacement, as used by the main PES figure."""
    positions = [read_xyz_positions(path) for path in image_paths]
    distances = [0.0]
    for previous, current in zip(positions, positions[1:]):
        distances.append(float(np.sqrt(np.mean((current - previous) ** 2))))
    path_lengths = np.cumsum(np.asarray(distances, dtype=float))
    total = float(path_lengths[-1])
    return path_lengths if total == 0.0 else path_lengths / total


def load_casci_segment(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing CASCI energies: {npz_path}")
    with np.load(npz_path) as data:
        for key in ("s0_energies_ev", "s1_energies_ev"):
            if key not in data:
                raise KeyError(f"{npz_path} does not contain {key!r}.")
        return {
            "S0": np.asarray(data["s0_energies_ev"], dtype=float),
            "S1": np.asarray(data["s1_energies_ev"], dtype=float),
        }


def load_casscf_segment(csv_path: Path) -> dict[str, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing SA-CASSCF energies: {csv_path}. "
            "Run pp_casscf_S1_neb_images.py first."
        )
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise ValueError(f"{csv_path} has no rows.")

    not_converged = [
        row["image_index"]
        for row in rows
        if row.get("casscf_converged") != "True" or row.get("rhf_converged") != "True"
    ]
    if not_converged:
        print(f"WARNING: SA-CASSCF images not converged: {not_converged}")

    return {
        "S0": np.asarray([float(row["s0_energy_ev"]) for row in rows], dtype=float),
        "S1": np.asarray([float(row["s1_energy_ev"]) for row in rows], dtype=float),
    }


def load_path(path_key: str) -> tuple[list[Path], dict[str, dict[str, np.ndarray]], list[int]]:
    """Join the segments into one path, dropping each shared junction image.

    Returns the image paths, {method: {state: energies}}, and the indices of the
    key points (S0 geometry, planar, TICT) in the joined path.
    """
    segments = PATHS[path_key]["segments"]
    image_paths: list[Path] = []
    per_method: dict[str, dict[str, list[np.ndarray]]] = {
        path_key: {"S0": [], "S1": []},
        "CASSCF": {"S0": [], "S1": []},
    }
    key_indices = [0]

    for position, segment in enumerate(segments):
        image_dir = Path(segment["image_dir"])
        paths = sorted(image_dir.glob(segment["pattern"]))
        if not paths:
            raise FileNotFoundError(
                f"No images matching {segment['pattern']!r} in {image_dir}."
            )
        energies = {
            path_key: load_casci_segment(Path(segment["native_npz"])),
            "CASSCF": load_casscf_segment(Path(segment["casscf_csv"])),
        }
        for method, states in energies.items():
            for state, values in states.items():
                if len(values) != len(paths):
                    raise ValueError(
                        f"{segment['name']}: {method} {state} has {len(values)} "
                        f"energies for {len(paths)} images."
                    )

        # Segments share their junction geometry; keep only one copy of it.
        start = 0 if position == 0 else 1
        if position > 0:
            previous = read_xyz_positions(image_paths[-1])
            junction = read_xyz_positions(paths[0])
            offset = float(np.abs(previous - junction).max())
            if offset > 1.0e-6:
                raise ValueError(
                    f"{segment['name']} does not start at the previous segment's "
                    f"last geometry (max|d| = {offset:.2e} Angstrom)."
                )

        image_paths.extend(paths[start:])
        for method, states in energies.items():
            for state, values in states.items():
                per_method[method][state].append(values[start:])
        key_indices.append(len(image_paths) - 1)

    combined = {
        method: {
            state: np.concatenate(chunks)
            for state, chunks in states.items()
        }
        for method, states in per_method.items()
    }
    return image_paths, combined, key_indices


def style_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.8, labelsize=8.5)
    ax.grid(False)


def plot_energies(
    ax: Any,
    x: np.ndarray,
    energies: dict[str, dict[str, np.ndarray]],
    states_shown: str = "s1",
) -> None:
    plotted = ("S1",) if states_shown == "s1" else ("S0", "S1")
    for method, states in energies.items():
        color = METHOD_STYLE[method]["color"]
        reference = float(states["S0"][0])
        for state in plotted:
            style = STATE_STYLE[state]
            ax.plot(
                x,
                states[state] - reference,
                color=color,
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{METHOD_STYLE[method]['label']} {state}",
            )

    if states_shown == "s1":
        ax.set_ylabel("Relative S$_1$ energy (eV)", fontsize=9.5, color=INK)
    else:
        ax.set_ylabel("Relative energy (eV)", fontsize=9.5, color=INK)
        ax.legend(fontsize=7.4, frameon=False, loc="center left", labelcolor=INK,
                  ncol=1, handlelength=2.6, borderaxespad=0.3)


def label_curves_directly(
    ax: Any,
    x: np.ndarray,
    energies: dict[str, dict[str, np.ndarray]],
) -> None:
    """Label each S1 curve in place, where the curves are furthest apart.

    Labelling at a fixed end collides whenever the curves converge there, which
    they do on the GBCI path at TICT.
    """
    curves = {
        method: states["S1"] - float(states["S0"][0])
        for method, states in energies.items()
    }
    (first, second) = list(curves)
    separation = np.abs(curves[first] - curves[second])
    # Stay off the very edges so the text has room.
    interior = slice(1, len(x) - 1)
    index = int(np.argmax(separation[interior])) + 1

    y_low, y_high = ax.get_ylim()
    height = max(y_high - y_low, 1.0e-9)
    upper = first if curves[first][index] >= curves[second][index] else second
    for method, curve in curves.items():
        above = method == upper
        # Near the axis floor there is no room beneath the curve; go above it.
        if not above and (curve[index] - y_low) / height < 0.16:
            above = True
        ax.annotate(
            METHOD_STYLE[method]["label"],
            xy=(x[index], curve[index]),
            xytext=(0, 11 if above else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if above else "top",
            fontsize=7.8,
            color=INK,
        )


def annotate_barrier(
    ax: Any,
    x: np.ndarray,
    energies: dict[str, dict[str, np.ndarray]],
    key_indices: list[int],
    method: str = "CASCI",
) -> None:
    """Mark the small post-planar maximum on the S1 curve, if there is one."""
    states = energies[method]
    curve = states["S1"] - float(states["S0"][0])
    planar = key_indices[1]
    tail = curve[planar:]
    offset = int(np.argmax(tail))
    height = float(tail[offset] - tail[0])
    if offset == 0 or height < 1.0e-3:
        return

    index = planar + offset
    ax.annotate(
        f"{height * 1000:.0f} meV",
        xy=(x[index], curve[index]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color=MUTED,
        arrowprops={
            "arrowstyle": "-",
            "color": MUTED,
            "linewidth": 0.7,
            "shrinkA": 1.0,
            "shrinkB": 3.0,
        },
    )


def plot_gap(ax: Any, x: np.ndarray, energies: dict[str, dict[str, np.ndarray]]) -> None:
    for method, states in energies.items():
        gap = states["S1"] - states["S0"]
        ax.plot(
            x,
            gap,
            color=METHOD_STYLE[method]["color"],
            linestyle="-",
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=METHOD_STYLE[method]["label"],
        )
        # Direct label at the right end instead of relying on the legend alone.
        ax.annotate(
            METHOD_STYLE[method]["label"],
            xy=(x[-1], gap[-1]),
            xytext=(-4, 10),
            textcoords="offset points",
            va="bottom",
            ha="right",
            fontsize=7.8,
            color=INK,
        )
    ax.set_ylabel("S$_1$ - S$_0$ gap (eV)", fontsize=9.5, color=INK)


def annotate_key_points(ax: Any, x: np.ndarray, key_indices: list[int]) -> None:
    """Name the stationary points under the axis and mark the segment junction."""
    for position, (index, label) in enumerate(zip(key_indices, KEY_POINT_LABELS)):
        # Interior key points are where two NEB segments meet; a faint rule makes
        # the join visible without competing with the data.
        if 0 < position < len(key_indices) - 1:
            ax.axvline(
                x[index],
                color=MUTED,
                linewidth=0.7,
                linestyle=":",
                zorder=0,
            )
        ha = "center"
        if position == 0:
            ha = "left"
        elif position == len(key_indices) - 1:
            ha = "right"
        ax.annotate(
            label,
            xy=(x[index], 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -20),
            textcoords="offset points",
            ha=ha,
            va="top",
            fontsize=7.8,
            color=MUTED,
            annotation_clip=False,
        )


def main() -> None:
    args = parse_args()

    panels = {}
    for path_key in PATHS:
        try:
            panels[path_key] = load_path(path_key)
        except (FileNotFoundError, KeyError, ValueError) as error:
            print(f"WARNING: skipping the {path_key} panel: {error}")
    if not panels:
        raise SystemExit("No path had complete energies; nothing to plot.")

    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.6 * len(panels), 3.3), squeeze=False
    )
    axes = list(axes[0])

    # One shared y-range makes the two paths directly comparable by eye.
    all_y = []
    for path_key, (paths_, energies, _) in panels.items():
        for method, states in energies.items():
            all_y.append(states["S1"] - float(states["S0"][0]))
    y_min = float(min(values.min() for values in all_y))
    y_max = float(max(values.max() for values in all_y))
    span = max(y_max - y_min, 0.1)

    for ax, tag, (path_key, (image_paths, energies, key_indices)) in zip(
        axes, ("(a)", "(b)"), panels.items()
    ):
        n_images = len(image_paths)
        if args.x_axis == "image_index":
            x = np.arange(n_images, dtype=float)
            x_label = "Image index"
            x_lim = (-0.3, n_images - 0.7)
        else:
            x = reaction_coordinate(image_paths)
            x_label = "Reaction coordinate"
            x_lim = (-0.03, 1.03)

        plot_energies(ax, x, energies, states_shown=args.states)
        annotate_barrier(ax, x, energies, key_indices, method=path_key)
        style_axes(ax)
        ax.set_xlim(*x_lim)
        ax.set_ylim(y_min - 0.05 * span, y_max + 0.12 * span)
        if args.states == "s1":
            label_curves_directly(ax, x, energies)
        ax.set_xlabel(x_label, fontsize=9.5, color=INK, labelpad=16)
        annotate_key_points(ax, x, key_indices)
        ax.text(
            -0.02,
            1.04,
            f"{tag} {PATHS[path_key]['label']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )

    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.22, top=0.90, wspace=0.10
    )

    output_prefix = args.output_prefix.expanduser().resolve()
    for fmt in args.formats:
        output_path = output_prefix.with_suffix(f".{fmt.lstrip('.')}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved figure: {output_path}")

    for path_key, (image_paths, energies, key_indices) in panels.items():
        native = energies[path_key]["S1"] - float(energies[path_key]["S0"][0])
        casscf = energies["CASSCF"]["S1"] - float(energies["CASSCF"]["S0"][0])
        print(f"\n===== {PATHS[path_key]['label']}: S1 relative energies (eV) =====")
        print(f"images: {len(image_paths)}  key points at {key_indices}")
        print(f"{'image':>5s} {path_key:>9s} {'SA-CASSCF':>10s}")
        for index, (a, b) in enumerate(zip(native, casscf)):
            tag = ""
            if index in key_indices:
                tag = "  <- " + KEY_POINT_LABELS[key_indices.index(index)]
            print(f"{index:5d} {a:9.4f} {b:10.4f}{tag}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
