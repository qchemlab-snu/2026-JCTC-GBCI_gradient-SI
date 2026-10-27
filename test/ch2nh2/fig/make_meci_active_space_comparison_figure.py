"""Draw the CH2NH2 cc-pVTZ S0/S1 MECI comparison across methods and active spaces.

A 2 x 3 grid: CASCI, GBCI (groupocc) and SA-CASSCF across the columns -- the same
column order as the single-row figure -- (8e,5o) on the
top row and (2e,2o) on the bottom.  It answers one question visually -- does shrinking
the active space to the HOMO/LUMO pair move the intersection? -- which the three-panel
make_geometry_comparison_figure_ccpvtz.py cannot show because it covers one active
space only.

Panels are drawn by that script's ``draw_panel``, so the C-N caliper, the
pyramidalization arc and the C/N center detection are shared rather than copied; this
file only supplies the panel table and the two-row layout.

Usage::

    python ../analysis/ch2nh2_meci_geometry_descriptors.py   # descriptors + aligned xyz
    bash ../analysis/make_meci_jmol_images.sh                # renders
    python make_meci_active_space_comparison_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from make_geometry_comparison_figure import (  # noqa: E402
    ANGLE_LABEL_X_OFFSET,
    nonwhite_bbox,
    ANGLE_VERTEX_OFFSET,
    MOLECULE_ROTATION_OFFSET_DEG,
    load_font,
    save_figure,
)
from make_geometry_comparison_figure_ccpvtz import (  # noqa: E402
    PANEL_SIZE,
    draw_panel,
    load_descriptors,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis"
# The 215-degree renders, not the pipeline default of 145: 215 is the viewing angle
# of the published single-row cc-pVTZ figure, fitted by a full 360-degree render
# scan scored on silhouette overlap (see CCPVTZ_FIGURE_SPIN_ABOUT_CN_DEG in
# ch2nh2_meci_geometry_descriptors.py).  Regenerate with
#     python ch2nh2_meci_geometry_descriptors.py --spin-about-cn-deg 215 \
#         --aligned-dir outputs/ch2nh2_ccpvtz_meci_images_spin215
#     IMAGE_DIR=outputs/ch2nh2_ccpvtz_meci_images_spin215 bash make_meci_jmol_images.sh
IMAGE_DIR = ANALYSIS_DIR / "outputs" / "ch2nh2_ccpvtz_meci_images_spin215"
DEFAULT_DESCRIPTORS_JSON = (
    ANALYSIS_DIR
    / "outputs"
    / "ch2nh2_ccpvtz_meci_descriptors"
    / "ch2nh2_ccpvtz_meci_descriptors_descriptors.json"
)
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "ch2nh2_meci_active_space_comparison"

NCOL = 3
ACTIVE_SPACE_LABEL = {"5o8e": "(8e,5o)", "2o2e": "(2e,2o)"}
# The angle label sits well to the left of the carbon, so the leftmost column needs a
# margin that the single-row figure does not: there the label still lands inside the
# panel because its renders are drawn smaller.
MARGIN_LEFT = 36

# Row-major, three columns per row.  The panel title is built from "method" and, only
# when both active spaces are drawn, the active space -- a one-row figure has it in
# the caption already, so repeating it on every panel is noise.
# "label" keys the descriptors JSON, so the (2e,2o)
# entries carry the suffixed labels added to ch2nh2_meci_geometry_descriptors.py.
# "angle_label_offset" nudges only that panel's angle text: a small pyramidalization
# angle puts the label on top of the dashed C-N extension, so the two SA-CASSCF panels
# and the (2e,2o) GBCI panel need it moved clear.
PANEL_DATA = [
    {
        "active_space": "5o8e",
        "method": "CASCI",
        "label": "CASCI",
        "angle_label_offset": (0.0, 0.0),
        "image": IMAGE_DIR / "casci_5o8e_ccpvtz_s0s1_meci.png",
        "rotation_offset_deg": 0.0,
    },
    {
        "active_space": "5o8e",
        "method": "GBCI",
        "label": "GBCI (groupocc)",
        "angle_label_offset": (0.0, 0.0),
        "image": IMAGE_DIR / "gbci_groupocc_5o8e_ccpvtz_s0s1_meci.png",
        "rotation_offset_deg": 0.0,
    },
    {
        "active_space": "5o8e",
        "method": "SA-CASSCF",
        "label": "CASSCF",
        "angle_label_offset": (-10.0, 20.0),
        "image": IMAGE_DIR / "casscf_5o8e_ccpvtz_s0s1_meci.png",
        "rotation_offset_deg": 0.0,
    },
    {
        "active_space": "2o2e",
        "method": "CASCI",
        "label": "CASCI 2o2e",
        "angle_label_offset": (0.0, 0.0),
        "image": IMAGE_DIR / "casci_2o2e_ccpvtz_s0s1_meci.png",
        "rotation_offset_deg": 0.0,
    },
    {
        "active_space": "2o2e",
        "method": "GBCI",
        "label": "GBCI 2o2e (groupocc)",
        "angle_label_offset": (-6.0, 10.0),
        "image": IMAGE_DIR / "gbci_groupocc_2o2e_ccpvtz_s0s1_meci.png",
        "rotation_offset_deg": 0.0,
    },
    {
        "active_space": "2o2e",
        "method": "SA-CASSCF",
        "label": "CASSCF 2o2e",
        "angle_label_offset": (-10.0, 22.0),
        "image": IMAGE_DIR / "casscf_2o2e_ccpvtz_s0s1_meci.png",
        "rotation_offset_deg": 0.0,
    },
]

# The single-row figure's renders are 500 px wide and it keeps a 78 px white border
# around the molecule, so the molecule fills a fixed fraction of the panel.  These
# renders are 900 px wide, so a fixed 78 px would leave the molecule much larger.
# Scaling the border by the molecule's own width reproduces the original composition.
CROP_MARGIN_RATIO = 78.0 / 271.0


def set_crop_margins(panels: list[dict]) -> None:
    """Give each panel the crop margin that matches the reference composition."""
    for panel in panels:
        with Image.open(panel["image"]) as image:
            left, _, right, _ = nonwhite_bbox(image.convert("RGB"))
        panel["crop_margin"] = int(round(CROP_MARGIN_RATIO * (right - left)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--descriptors-json", type=Path, default=DEFAULT_DESCRIPTORS_JSON)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument(
        "--active-space",
        choices=("all", "5o8e", "2o2e"),
        default="all",
        help=(
            "Which rows to draw.  A single active space gives one row of three "
            "panels, the layout of the published single-row figure, and its name "
            "is suffixed so it does not overwrite the two-row figure."
        ),
    )
    parser.add_argument(
        "--rotation-offset-deg", type=float, default=MOLECULE_ROTATION_OFFSET_DEG
    )
    parser.add_argument("--angle-label-dx", type=float, default=ANGLE_LABEL_X_OFFSET)
    parser.add_argument("--angle-label-y-from-cn-bond", type=float, default=0.0)
    parser.add_argument("--angle-vertex-dx", type=float, default=ANGLE_VERTEX_OFFSET[0])
    parser.add_argument("--angle-vertex-dy", type=float, default=ANGLE_VERTEX_OFFSET[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    descriptors = load_descriptors(args.descriptors_json)

    panels = [
        dict(panel)
        for panel in PANEL_DATA
        if args.active_space in ("all", panel["active_space"])
    ]
    show_space = args.active_space == "all"
    for index, panel in enumerate(panels):
        letter = chr(ord("a") + index)
        suffix = f" {ACTIVE_SPACE_LABEL[panel['active_space']]}" if show_space else ""
        panel["title"] = f"({letter}) {panel['method']}{suffix}"

    output_prefix = args.output_prefix
    if args.active_space != "all" and output_prefix == DEFAULT_OUTPUT_PREFIX:
        output_prefix = output_prefix.with_name(
            f"{output_prefix.name}_{args.active_space}"
        )

    missing = [p["image"] for p in panels if not p["image"].exists()]
    if missing:
        raise SystemExit(
            "missing panel image(s) -- run make_meci_jmol_images.sh first:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    set_crop_margins(panels)

    panel_w, panel_h = PANEL_SIZE
    nrow = (len(panels) + NCOL - 1) // NCOL
    canvas = Image.new(
        "RGB", (MARGIN_LEFT + panel_w * NCOL, panel_h * nrow), "white"
    )
    fonts = {
        "small": load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20),
        "bold": load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24),
    }

    for index, panel in enumerate(panels):
        row, col = divmod(index, NCOL)
        draw_panel(
            canvas,
            panel,
            descriptors,
            index,
            fonts,
            args,
            origin=(MARGIN_LEFT + col * panel_w, row * panel_h),
        )

    save_figure(canvas, output_prefix, args.formats)
    for fmt in args.formats:
        print(f"wrote {output_prefix.with_suffix('.' + fmt.lower())}")


if __name__ == "__main__":
    main()
