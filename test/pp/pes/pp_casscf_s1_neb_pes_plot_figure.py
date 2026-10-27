"""Standalone figure: the SA-CASSCF S1 NEB path from the S0 geometry to TICT.

This stands on its own rather than being overlaid on the CASCI/GBCI panels. The
SA-CASSCF path does not pass through a planar S1 minimum -- SA-CASSCF has none,
an optimization started from the planar structure relaxes all the way to TICT --
so its reaction coordinate is not the CASCI/GBCI one and the two cannot share an
axis beyond their endpoints.

Both states are drawn. S1 carries the forces that produced the band; S0 comes
from the same state-averaged calculation at each image, so the closing gap is
read off directly. Energies are referenced to S0 at the first image, which makes
the left-hand gap the vertical excitation energy.

Structures are attached at the two endpoints only, since those are the only
images that are stationary points (the first on S0, the last on S1).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

PROFILE_CSV = (
    REPO_ROOT
    / "test"
    / "pp"
    / "casscf"
    / "pp_casscf_S1_fc_tict_neb_outputs"
    / "pp_casscf_S1_fc_tict_neb_profile.csv"
)
OUTPUT_PREFIX = SCRIPT_DIR / "pp_casscf_s1_neb_pes_figure"

# Only S1 is drawn, so the single curve takes plain black rather than the
# orange SA-CASSCF slot used where several methods share a panel.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
CURVE_STYLE = {
    "color": INK,
    "linestyle": "-",
    "linewidth": 2.2,
    "marker": "o",
    "markersize": 5.0,
}

# Rendered from the SA-CASSCF NEB endpoints themselves, but in the frames of
# the published CASCI structures (see align_neb_endpoints_to_reference.py) and
# with the same camera, so they compose like the existing figures.
ENDPOINT_IMAGES = {
    "start": SCRIPT_DIR / "pp_casscf_neb_start_structure.png",
    "end": SCRIPT_DIR / "pp_casscf_neb_end_structure.png",
}
ENDPOINT_LABELS = {"start": "S$_0$ geometry", "end": "TICT"}

# Each structure sits in the clear space immediately next to its own endpoint:
# the start point is top left with room to its right along the top, the end
# point is at the right with room underneath. Neither box crosses the curve.
INSET_BOXES = {"start": (0.130, 0.620, 0.295, 0.300),
               "end": (0.665, 0.035, 0.315, 0.300)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the standalone SA-CASSCF S1 NEB PES figure for PP."
    )
    parser.add_argument("--input", type=Path, default=PROFILE_CSV, help="Profile CSV.")
    parser.add_argument(
        "--output-prefix", type=Path, default=OUTPUT_PREFIX, help="Output path stem."
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument(
        "--no-structures", action="store_true", help="Skip the endpoint structures."
    )
    return parser.parse_args()


def load_profile(csv_path: Path) -> dict[str, np.ndarray]:
    resolved = csv_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing NEB profile: {resolved}. Run "
            "pp_casscf_S1_fc_tict_neb_profile_export.py first."
        )
    with resolved.open() as fh:
        rows = list(csv.DictReader(fh))

    s = np.asarray([float(r["path_length_angstrom"]) for r in rows], dtype=float)
    span = s[-1] - s[0]
    if span <= 0:
        raise ValueError(f"{resolved} has a non-increasing path length.")

    return {
        "x": (s - s[0]) / span,
        "s0": np.asarray([float(r["s0_relative_to_s0_geometry_ev"]) for r in rows]),
        "s1": np.asarray([float(r["s1_relative_to_s0_geometry_ev"]) for r in rows]),
        "gap": np.asarray([float(r["s1_minus_s0_ev"]) for r in rows]),
        "twist": np.asarray([float(r["twist_deg"]) for r in rows]),
        "path_length": s,
    }


def read_cropped_structure_image(path: Path) -> np.ndarray:
    image = mpimg.imread(path)
    rgb = image[..., :3]
    nonwhite = np.any(rgb < 0.98, axis=2)
    if not np.any(nonwhite):
        return image
    y_idx, x_idx = np.where(nonwhite)
    pad = 16
    y0 = max(int(y_idx.min()) - pad, 0)
    y1 = min(int(y_idx.max()) + pad + 1, image.shape[0])
    x0 = max(int(x_idx.min()) - pad, 0)
    x1 = min(int(x_idx.max()) + pad + 1, image.shape[1])
    return image[y0:y1, x0:x1]


def add_structure_insets(ax: Any) -> None:
    for key, box in INSET_BOXES.items():
        path = ENDPOINT_IMAGES[key]
        inset = ax.inset_axes(box)
        inset.axis("off")
        if key == "end":
            inset.text(0.5, -0.04, ENDPOINT_LABELS[key],
                       transform=inset.transAxes, ha="center", va="top",
                       fontsize=9.0, color=INK)
        else:
            inset.set_title(
                ENDPOINT_LABELS[key], fontsize=9.0, color=INK, pad=1.5
            )
        if not path.exists():
            inset.text(
                0.5, 0.5, f"Missing\n{path.name}", transform=inset.transAxes,
                ha="center", va="center", fontsize=7.5, color=MUTED,
            )
            continue
        inset.imshow(read_cropped_structure_image(path))


def main() -> None:
    args = parse_args()
    data = load_profile(args.input)
    x = data["x"]

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.135, top=0.95)

    ax.plot(
        x, data["s1"],
        color=CURVE_STYLE["color"],
        linestyle=CURVE_STYLE["linestyle"],
        linewidth=CURVE_STYLE["linewidth"],
        marker=CURVE_STYLE["marker"],
        markersize=CURVE_STYLE["markersize"],
        markerfacecolor="white",
        markeredgewidth=1.3,
        markeredgecolor=CURVE_STYLE["color"],
        zorder=3,
    )

    # The two numbers the figure exists to show: the vertical gap at the start
    # and the gap once the path reaches TICT.
    drop = data["s1"][-1] - data["s1"][0]
    for idx, ha, offset in ((0, "left", (2, 12)), (len(x) - 1, "right", (16, 10))):
        ax.annotate(
            f"{data['s1'][idx]:.3f} eV",
            xy=(x[idx], data["s1"][idx]), xytext=offset,
            textcoords="offset points", ha=ha,
            va="bottom" if offset[1] > 0 else "top",
            fontsize=9.5, color=INK, zorder=8,
        )

    if not args.no_structures:
        add_structure_insets(ax)

    ax.set_xlabel("Reaction Coordinate")
    ax.set_ylabel("Relative Energy (eV)")
    ax.set_xlim(-0.06, 1.06)
    y_all = data["s1"]
    span = float(y_all.max() - y_all.min())
    # Room below the curve on the left and above it on the right for the insets.
    ax.set_ylim(float(y_all.min()) - 0.75 * span, float(y_all.max()) + 0.38 * span)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    mid = len(x) // 2
    ax.annotate("S$_1$", xy=(x[mid], data["s1"][mid]), xytext=(0, 10),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=11.0, color=INK, zorder=8)

    print(
        f"SA-CASSCF NEB PES: {len(x)} images, path {data['path_length'][-1]:.3f} A, "
        f"twist {data['twist'][0]:.1f} -> {data['twist'][-1]:.1f} deg"
    )
    print(
        f"  S1 {data['s1'][0]:.3f} -> {data['s1'][-1]:.3f} eV ({drop:+.3f}); "
        f"gap {data['gap'][0]:.3f} -> {data['gap'][-1]:.3f} eV (S0 not plotted)"
    )

    for fmt in args.formats:
        out = args.output_prefix.with_suffix(f".{fmt.lstrip('.')}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"Saved figure: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
