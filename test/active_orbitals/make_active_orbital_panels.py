"""Tile the rendered active orbitals of each system into one Supporting-Information panel.

Step three, after render_active_orbitals.py (cubes) and make_active_orbital_images.sh
(per-orbital PNGs).  One panel per molecule: orbitals left to right in active-window
order, views top to bottom, every tile labelled with its RHF MO number and orbital
energy.  Labels are read from the manifest the renderer wrote, not typed in, so a
caption cannot drift away from the orbital above it.

Jmol pads every render with the same generous margin.  The tiles are cropped to the
union of all non-white boxes in a panel rather than to each tile's own box, so the
molecule stays the same size across the panel and the orbitals remain visually
comparable.

Usage::

    python make_active_orbital_panels.py                     # every rendered system
    python make_active_orbital_panels.py --systems so2_8e5o
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

WHITE_THRESHOLD = 245  # a pixel above this in every channel counts as background
CROP_PAD = 12          # pixels of white kept around the union box
VIEW_LABELS = {"front": "in-plane view", "side": "edge-on view"}
TILE_INCHES = 2.3

# Label size is set per inch of figure width, not in absolute points.  A panel is
# as wide as it has orbitals -- two for pp, five for SO2 -- but every panel ends up
# at the same column width on the page, so a fixed point size comes out large on
# the narrow panels and small on the wide ones.  Scaling with the figure width
# cancels that: at 1.15 pt/in a four-orbital panel gets the familiar ~10.5 pt.
LABEL_PT_PER_INCH = 1.15


def content_box(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of everything that is not background white."""
    grey = image.convert("L").point(lambda v: 0 if v >= WHITE_THRESHOLD else 255)
    return grey.getbbox()


def union_box(boxes, size: tuple[int, int]) -> tuple[int, int, int, int]:
    present = [b for b in boxes if b is not None]
    if not present:
        return (0, 0, *size)
    left = max(min(b[0] for b in present) - CROP_PAD, 0)
    upper = max(min(b[1] for b in present) - CROP_PAD, 0)
    right = min(max(b[2] for b in present) + CROP_PAD, size[0])
    lower = min(max(b[3] for b in present) + CROP_PAD, size[1])
    return (left, upper, right, lower)


def panel(name: str, fig_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    manifest = json.loads((fig_dir / name / f"{name}_manifest.json").read_text())
    image_dir = fig_dir / name / "orbitals"
    views = manifest["views"]
    orbitals = manifest["orbitals"]

    tiles = {}
    for view in views:
        for orbital in orbitals:
            path = image_dir / f"{name}_mo{orbital['mo']:02d}_{view}.png"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing; run make_active_orbital_images.sh {name}")
            tiles[(view, orbital["mo"])] = Image.open(path).convert("RGB")

    any_tile = next(iter(tiles.values()))
    box = union_box([content_box(t) for t in tiles.values()], any_tile.size)
    width, height = box[2] - box[0], box[3] - box[1]

    ncols, nrows = len(orbitals), len(views)
    fig_width = TILE_INCHES * ncols
    label_size = LABEL_PT_PER_INCH * fig_width
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(fig_width, TILE_INCHES * height / width * nrows),
    )
    for row, view in enumerate(views):
        for col, orbital in enumerate(orbitals):
            ax = axes[row][col]
            ax.imshow(tiles[(view, orbital["mo"])].crop(box))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                occupancy = "occ." if orbital["occ"] > 0.5 else "virt."
                ax.set_title(f"MO {orbital['mo']} ({occupancy})\n"
                             f"{orbital['energy']:.3f} $E_\\mathrm{{h}}$",
                             fontsize=label_size, pad=0.6 * label_size)
            if col == 0 and len(views) > 1:
                ax.set_ylabel(VIEW_LABELS.get(view, view), fontsize=label_size)

    # No panel title: the system, active space and basis belong in the SI caption,
    # and a title repeated above every panel only costs figure height.
    fig.tight_layout()

    written = []
    for suffix in formats:
        out = fig_dir / name / f"{name}_active_orbitals.{suffix}"
        fig.savefig(out, dpi=dpi)
        written.append(out)
        print(f"wrote {out}")
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="",
                        help="comma-separated systems; default is every rendered one")
    parser.add_argument("--fig-dir", type=Path,
                        default=SCRIPT_DIR / "reference" / "active_orbital_figures")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    names = args.systems.replace(",", " ").split()
    if not names:
        names = sorted(p.parent.name for p in args.fig_dir.glob("*/*_manifest.json"))
    if not names:
        raise SystemExit(f"no manifests under {args.fig_dir}; "
                         "run render_active_orbitals.py first")
    for name in names:
        panel(name, args.fig_dir, args.formats, args.dpi)


if __name__ == "__main__":
    main()
