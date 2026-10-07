"""Draw the CH2NH2 cc-pVTZ S0/S1 MECI geometry comparison figure.

The cc-pVTZ counterpart of make_geometry_comparison_figure.py: three panels comparing
CASCI, GBCI and SA-CASSCF at the 5o8e active space.  Layout, guide lines and labelling are
taken from that script -- the shared drawing helpers are imported rather than copied.

Two things differ from the original:

* The bond length and pyramidalization angle printed on each panel are read from the
  descriptors JSON written by ../analysis/ch2nh2_meci_geometry_descriptors.py
  instead of being typed into the panel table, so a label cannot drift away from the
  geometry it describes.
* The C and N centers are found by eroding the bond sticks away and taking the centroid
  of what is left of each sphere, rather than the centroid of the whole colored region.
  The original centroid includes the bond sticks, which drags the C and N marks off their
  atoms and under-reports the C-N tilt by several degrees (1.97 instead of 7.59 on the
  CASCI panel), so the "aligned" panels came out visibly crooked.  The same centers also
  anchor the bond caliper, which now starts on the carbon instead of inside it.

* align_cn_horizontal is redefined here with the opposite rotation sign.  The original
  rotates by -angle, which doubles the C-N tilt instead of removing it.

Usage::

    python ../analysis/ch2nh2_meci_geometry_descriptors.py   # produces the JSON
    python make_geometry_comparison_figure_ccpvtz.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from make_geometry_comparison_figure import (  # noqa: E402
    ANGLE_ARC_RADIUS,
    ANGLE_EXTENSION_LENGTH,
    ANGLE_LABEL_RADIUS,
    ANGLE_LABEL_X_OFFSET,
    ANGLE_SLANTED_RAY_LENGTH,
    ANGLE_VERTEX_LEFT_FROM_C,
    ANGLE_VERTEX_OFFSET,
    ANGSTROM,
    BOND_LABEL_Y_OFFSET,
    DEGREE,
    MOLECULE_ROTATION_OFFSET_DEG,
    arc_points,
    dashed_line,
    draw_centered_text_label,
    draw_text_label,
    load_font,
    nonwhite_bbox,
    rotate_vector,
    save_figure,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CCPVTZ_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "ch2nh2_geometry_comparison_ccpvtz"
DEFAULT_DESCRIPTORS_JSON = (
    CCPVTZ_DIR
    / "analysis"
    / "outputs"
    / "ch2nh2_ccpvtz_meci_descriptors"
    / "ch2nh2_ccpvtz_meci_descriptors_descriptors.json"
)

# One entry per panel.  "label" must match the label used in the descriptors JSON; the
# bond length and angle come from there.  "image" is the Jmol render of that run, which
# lives in the run's own output directory.  "angle_label_offset" nudges just that panel's
# angle text in pixels -- a small angle puts the label right on top of the dashed C-N
# extension, so the SA-CASSCF panel needs it moved clear.  "label" stays "CASSCF"
# because that is the key the descriptors JSON uses.
PANEL_DATA = [
    {
        "title": "(a) CASCI",
        "label": "CASCI",
        "angle_label_offset": (0.0, 0.0),
        "image": (
            CCPVTZ_DIR
            / "casci/outputs/ch2nh2_casci_S0S1_meci_5o8e_ccpvtz/casci_ccpvtz_meci.png"
        ),
        "rotation_offset_deg": 0.0,
    },
    {
        "title": "(b) GBCI",
        "label": "GBCI (groupocc)",
        "angle_label_offset": (0.0, 0.0),
        "image": (
            CCPVTZ_DIR
            / "gbci/outputs/ch2nh2_gbci_S0S1_meci_5o8e_groupocc_ccpvtz/gbci_ccpvtz_meci.png"
        ),
        "rotation_offset_deg": 0.0,
    },
    {
        "title": "(c) SA-CASSCF",
        "label": "CASSCF",
        "angle_label_offset": (-10.0, 20.0),
        "image": (
            CCPVTZ_DIR
            / "casscf/outputs/ch2nh2_casscf_S0S1_meci_5o8e_ccpvtz/casscf_ccpvtz_meci.png"
        ),
        "rotation_offset_deg": 0.0,
    },
]

# The C-N alignment is iterated: one pass leaves about a degree because the atom centers
# are re-measured on a resampled image.  Two passes reach a few hundredths of a degree, and
# a third only adds resampling noise, so stop on either the tolerance or the pass limit.
MAX_ALIGNMENT_PASSES = 3
ALIGNMENT_TOLERANCE_DEG = 0.25

PANEL_SIZE = (430, 360)
MOLECULE_MAX_SIZE = (360, 275)
MOLECULE_PASTE_Y = 58


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def _core_center(mask: np.ndarray, erosion_radius: int) -> tuple[float, float] | None:
    """Centroid of a sphere after the thin bond sticks have been eroded away."""
    eroded = ndimage.binary_erosion(mask, _disk(erosion_radius))
    labels, count = ndimage.label(eroded)
    if count == 0:
        return None
    sizes = ndimage.sum(eroded, labels, range(1, count + 1))
    core = labels == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(core)
    return float(xs.mean()), float(ys.mean())


def atom_centers(image: Image.Image) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return the C and N sphere centers of a rendered panel.

    Replaces the colour-centroid version in the original script.  Jmol draws the bonds much
    thinner than the atoms, so eroding by a fraction of the sphere radius removes the sticks
    and leaves one blob per atom whose centroid is the atom center.  The erosion radius is
    derived from the nitrogen sphere so this keeps working after the panel is scaled down.
    """
    arr = np.asarray(image.convert("RGB"), dtype=int)
    red, green, blue_ch = arr[..., 0], arr[..., 1], arr[..., 2]
    mean = (red + green + blue_ch) / 3.0
    saturation = arr.max(axis=2) - arr.min(axis=2)

    nitrogen = (blue_ch > 110) & (red < 110) & (green < 150) & ((blue_ch - red) > 40)
    carbon = (mean > 45) & (mean < 185) & (saturation < 60) & ~nitrogen
    if not nitrogen.any() or not carbon.any():
        return None

    sphere_radius = ndimage.distance_transform_edt(nitrogen).max()
    erosion_radius = max(3, int(round(0.45 * sphere_radius)))

    c_center = _core_center(carbon, erosion_radius)
    n_center = _core_center(nitrogen, erosion_radius)
    if c_center is None or n_center is None:
        return None
    return c_center, n_center


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the CH2NH2 cc-pVTZ CASCI/GBCI/SA-CASSCF geometry comparison figure."
    )
    parser.add_argument(
        "--descriptors-json",
        type=Path,
        default=DEFAULT_DESCRIPTORS_JSON,
        help="Descriptors JSON written by ch2nh2_meci_geometry_descriptors.py.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output formats to save, e.g. png pdf.",
    )
    parser.add_argument(
        "--rotation-offset-deg",
        type=float,
        default=MOLECULE_ROTATION_OFFSET_DEG,
        help="Extra molecule rotation in degrees after C-N alignment.",
    )
    parser.add_argument(
        "--angle-label-dx",
        type=float,
        default=ANGLE_LABEL_X_OFFSET,
        help="Horizontal offset of the angle label from the angle vertex, in pixels.",
    )
    parser.add_argument(
        "--angle-label-y-from-cn-bond",
        type=float,
        default=0.0,
        help="Vertical offset of the angle label from the angle-guide vertex, in pixels.",
    )
    parser.add_argument(
        "--angle-vertex-dx",
        type=float,
        default=ANGLE_VERTEX_OFFSET[0],
        help="Horizontal offset of the angle vertex from the detected carbon center.",
    )
    parser.add_argument(
        "--angle-vertex-dy",
        type=float,
        default=ANGLE_VERTEX_OFFSET[1],
        help="Vertical offset of the angle vertex from the detected C-N bond height.",
    )
    return parser.parse_args()


def load_descriptors(path: Path) -> dict[str, dict]:
    """Map each geometry label to its record in the descriptors JSON."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run ch2nh2_meci_geometry_descriptors.py first"
        )
    payload = json.loads(path.read_text())
    return {record["label"]: record for record in payload["geometries"]}


def panel_labels(panel: dict, descriptors: dict[str, dict]) -> tuple[str, str, float]:
    """Return the bond text, angle text and angle value for one panel."""
    try:
        record = descriptors[panel["label"]]
    except KeyError:
        raise SystemExit(
            f"no entry labelled {panel['label']!r} in the descriptors JSON"
        ) from None
    bond = record["cn_bond_angstrom"]
    angle = record["pyramidalization_at_C_deg"]
    return f"{bond:.3f} {ANGSTROM}", f"{angle:.1f}{DEGREE}", angle


def align_cn_horizontal(image: Image.Image, rotation_offset_deg: float) -> Image.Image:
    """Rotate the panel so the C-N bond is horizontal.

    Differs from the original helper in two ways.  The sign is flipped: PIL rotates
    counter-clockwise on screen, which is clockwise in the y-down coordinates the tilt is
    measured in, so removing a tilt of +angle needs rotate(+angle); the original passes
    -angle, which roughly doubles the tilt.  And the correction is iterated, because
    measuring the centers on a rotated image leaves about a degree behind -- two passes
    reach a few hundredths of a degree.  Each pass re-rotates the original by the running
    total rather than rotating an already-rotated image, so the panel is resampled once.
    """
    total_deg = 0.0
    current = image
    for _ in range(MAX_ALIGNMENT_PASSES):
        centers = atom_centers(current)
        if centers is None:
            return current
        (c_x, c_y), (n_x, n_y) = centers
        tilt = math.degrees(math.atan2(n_y - c_y, n_x - c_x))
        if abs(tilt) < ALIGNMENT_TOLERANCE_DEG:
            break
        total_deg += tilt
        current = image.rotate(
            total_deg + rotation_offset_deg,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor="white",
        )
    return current


def prepare_panel_image(
    path: Path, max_size: tuple[int, int], rotation_offset_deg: float
) -> Image.Image:
    image = align_cn_horizontal(Image.open(path).convert("RGB"), rotation_offset_deg)
    left, top, right, bottom = nonwhite_bbox(image)
    margin = 78
    cropped = image.crop(
        (
            max(left - margin, 0),
            max(top - margin, 0),
            min(right + margin, image.width),
            min(bottom + margin, image.height),
        )
    )
    cropped.thumbnail(max_size, Image.Resampling.LANCZOS)
    return cropped


def draw_panel(
    canvas: Image.Image,
    panel: dict,
    descriptors: dict[str, dict],
    panel_index: int,
    fonts: dict[str, object],
    args: argparse.Namespace,
) -> None:
    panel_width, _ = PANEL_SIZE
    x0 = panel_index * panel_width
    bond_text, angle_text, angle_value = panel_labels(panel, descriptors)

    molecule = prepare_panel_image(
        panel["image"],
        max_size=MOLECULE_MAX_SIZE,
        rotation_offset_deg=args.rotation_offset_deg + panel["rotation_offset_deg"],
    )
    paste_x = x0 + (panel_width - molecule.width) // 2
    paste_y = MOLECULE_PASTE_Y
    canvas.paste(molecule, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)
    draw.text((x0 + 18, 16), panel["title"], fill=(0, 0, 0), font=fonts["bold"])

    centers = atom_centers(molecule)
    if centers is None:
        c_x = paste_x + molecule.width * 0.38
        c_y = paste_y + molecule.height * 0.54
        n_x = paste_x + molecule.width * 0.64
        n_y = paste_y + molecule.height * 0.54
    else:
        (local_c_x, local_c_y), (local_n_x, local_n_y) = centers
        c_x = paste_x + local_c_x
        c_y = paste_y + local_c_y
        n_x = paste_x + local_n_x
        n_y = paste_y + local_n_y

    guide_color = (80, 80, 80)

    # C-N bond length: a dashed caliper above the bond.
    y_line = min(c_y, n_y) - 34
    dashed_line(draw, (c_x + 8, y_line), (n_x - 8, y_line), fill=guide_color)
    draw.line((c_x + 8, y_line - 5, c_x + 8, y_line + 5), fill=guide_color, width=2)
    draw.line((n_x - 8, y_line - 5, n_x - 8, y_line + 5), fill=guide_color, width=2)
    bond_width = draw.textlength(bond_text, font=fonts["small"])
    draw_text_label(
        draw,
        ((c_x + n_x - bond_width) / 2, y_line + BOND_LABEL_Y_OFFSET),
        bond_text,
        fonts["small"],
    )

    # Pyramidalization: the C-N extension plus a ray opened by the measured angle.
    cn_extension = (-1.0, 0.0)
    angle_y = (c_y + n_y) / 2.0
    angle_center = (
        c_x - ANGLE_VERTEX_LEFT_FROM_C + args.angle_vertex_dx,
        angle_y + args.angle_vertex_dy,
    )
    extension_start = (c_x - 2.0, angle_center[1])
    extension_end = (angle_center[0] - ANGLE_EXTENSION_LENGTH, angle_center[1])
    slanted_vector = rotate_vector(cn_extension, -angle_value)
    slanted_end = (
        angle_center[0] + slanted_vector[0] * ANGLE_SLANTED_RAY_LENGTH,
        angle_center[1] + slanted_vector[1] * ANGLE_SLANTED_RAY_LENGTH,
    )
    arc_start = math.degrees(math.atan2(cn_extension[1], cn_extension[0]))
    arc_end = math.degrees(math.atan2(slanted_vector[1], slanted_vector[0]))
    label_vector = rotate_vector(cn_extension, -angle_value / 2.0)
    label_dx, label_dy = panel.get("angle_label_offset", (0.0, 0.0))
    label_center = (
        angle_center[0]
        + label_vector[0] * ANGLE_LABEL_RADIUS
        + args.angle_label_dx
        + label_dx,
        angle_center[1]
        + label_vector[1] * ANGLE_LABEL_RADIUS
        + args.angle_label_y_from_cn_bond
        + label_dy,
    )

    dashed_line(draw, extension_start, extension_end, fill=guide_color, dash=6, gap=5)
    dashed_line(draw, angle_center, slanted_end, fill=guide_color, dash=6, gap=5)
    draw.line(
        arc_points(angle_center, ANGLE_ARC_RADIUS, arc_start, arc_end, steps=20),
        fill=guide_color,
        width=2,
    )
    draw_centered_text_label(draw, label_center, angle_text, fonts["small"])


def main() -> None:
    args = parse_args()
    descriptors = load_descriptors(args.descriptors_json)

    missing = [p["image"] for p in PANEL_DATA if not p["image"].exists()]
    if missing:
        raise SystemExit(
            "missing panel image(s):\n  " + "\n  ".join(str(p) for p in missing)
        )

    canvas = Image.new(
        "RGB", (PANEL_SIZE[0] * len(PANEL_DATA), PANEL_SIZE[1]), "white"
    )
    fonts = {
        "small": load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20),
        "bold": load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24),
    }

    for index, panel in enumerate(PANEL_DATA):
        draw_panel(canvas, panel, descriptors, index, fonts, args)

    save_figure(canvas, args.output_prefix, args.formats)


if __name__ == "__main__":
    main()
