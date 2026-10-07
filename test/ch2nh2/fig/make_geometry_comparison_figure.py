from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = SCRIPT_DIR / "ch2nh2_geometry_comparison"

ANGSTROM = "\u00c5"
DEGREE = "\u00b0"

PANEL_DATA = [
    {
        "title": "(a) Reference",
        "image": "xms_s0s1_meci.png",
        "bond": f"1.412 {ANGSTROM}",
        "angle": f"31.3{DEGREE}",
        "rotation_offset_deg": 0.0,
    },
    {
        "title": "(b) CASCI",
        "image": "casci_s0s1_meci.png",
        "bond": f"1.308 {ANGSTROM}",
        "angle": f"66.2{DEGREE}",
        "rotation_offset_deg": 10.0,
    },
    {
        "title": "(c) GBCI",
        "image": "gbci_s0s1_meci.png",
        "bond": f"1.458 {ANGSTROM}",
        "angle": f"44.2{DEGREE}",
        "rotation_offset_deg": 0.0,
    },
]

MOLECULE_ROTATION_OFFSET_DEG = 0.0
ANGLE_VERTEX_LEFT_FROM_C = 58.0
ANGLE_EXTENSION_LENGTH = 96.0
ANGLE_SLANTED_RAY_LENGTH = 74.0
ANGLE_ARC_RADIUS = 34.0
ANGLE_LABEL_RADIUS = 58.0
ANGLE_VERTEX_OFFSET = (0.0, 0.0)
ANGLE_LABEL_X_OFFSET = -4.0
ANGLE_LABEL_Y_FROM_ANGLE_VERTEX = 0.0
BOND_LABEL_Y_OFFSET = -30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the CH2NH2 XMS/CASCI/GBCI geometry comparison figure."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing the source geometry PNG files.",
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
        help=(
            "Extra molecule rotation in degrees after C-N alignment. "
            "Positive values rotate counterclockwise."
        ),
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
        default=ANGLE_LABEL_Y_FROM_ANGLE_VERTEX,
        help=(
            "Vertical offset of the angle label from the angle-guide vertex, in pixels."
        ),
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


def load_font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def atom_centers(image: Image.Image) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return approximate C and N centers from the rendered molecule image."""
    arr = np.asarray(image.convert("RGB"), dtype=int)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    yy, xx = np.indices(red.shape)

    nitrogen_mask = (blue > 120) & (red < 90) & (green < 135) & ((blue - red) > 50)
    y_n, x_n = np.where(nitrogen_mask)
    if len(x_n) == 0:
        return None
    n_center = (float(x_n.mean()), float(y_n.mean()))

    mean = (red + green + blue) / 3
    saturation = (
        np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    )
    carbon_mask = (
        (mean > 55)
        & (mean < 175)
        & (saturation < 50)
        & (xx < n_center[0] - 18)
        & (xx > n_center[0] - 190)
        & (yy > n_center[1] - 100)
        & (yy < n_center[1] + 100)
    )
    y_c, x_c = np.where(carbon_mask)
    if len(x_c) == 0:
        return None

    weights = (185 - mean[carbon_mask]) ** 2
    c_center = (
        float((x_c * weights).sum() / weights.sum()),
        float((y_c * weights).sum() / weights.sum()),
    )
    return c_center, n_center


def align_cn_horizontal(image: Image.Image, rotation_offset_deg: float) -> Image.Image:
    centers = atom_centers(image)
    if centers is None:
        return image

    (c_x, c_y), (n_x, n_y) = centers
    angle = math.degrees(math.atan2(n_y - c_y, n_x - c_x))
    return image.rotate(
        -angle + rotation_offset_deg,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor="white",
    )


def nonwhite_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    mask = np.min(arr, axis=2) < 245
    y_idx, x_idx = np.where(mask)
    if len(x_idx) == 0:
        return 0, 0, image.width, image.height
    return int(x_idx.min()), int(y_idx.min()), int(x_idx.max() + 1), int(y_idx.max() + 1)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int] = (80, 80, 80),
    width: int = 2,
    dash: int = 8,
    gap: int = 6,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return

    ux = (x2 - x1) / length
    uy = (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end_pos = min(pos + dash, length)
        draw.line(
            (
                x1 + ux * pos,
                y1 + uy * pos,
                x1 + ux * end_pos,
                y1 + uy * end_pos,
            ),
            fill=fill,
            width=width,
        )
        pos += dash + gap


def unit_vector(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return 1.0, 0.0
    return dx / length, dy / length


def rotate_vector(vector: tuple[float, float], degrees: float) -> tuple[float, float]:
    theta = math.radians(degrees)
    x, y = vector
    return x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)


def arc_points(
    center: tuple[float, float],
    radius: float,
    start_deg: float,
    end_deg: float,
    *,
    steps: int = 24,
) -> list[tuple[float, float]]:
    points = []
    for idx in range(steps + 1):
        theta = math.radians(start_deg + (end_deg - start_deg) * idx / steps)
        points.append(
            (
                center[0] + radius * math.cos(theta),
                center[1] + radius * math.sin(theta),
            )
        )
    return points


def angle_value_degrees(text: str) -> float:
    return float(text.replace(DEGREE, "").strip())


def draw_text_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 1
    draw.rectangle(
        (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
        fill="white",
    )
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def draw_centered_text_label(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw_text_label(draw, (center[0] - width / 2, center[1] - height / 2), text, font)


def prepare_panel_image(
    path: Path,
    max_size: tuple[int, int],
    rotation_offset_deg: float,
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
    panel: dict[str, str],
    image_dir: Path,
    panel_index: int,
    panel_size: tuple[int, int],
    fonts: dict[str, ImageFont.ImageFont],
    args: argparse.Namespace,
) -> None:
    panel_width, _ = panel_size
    x0 = panel_index * panel_width
    source_path = image_dir / panel["image"]
    molecule = prepare_panel_image(
        source_path,
        max_size=(360, 275),
        rotation_offset_deg=args.rotation_offset_deg + panel["rotation_offset_deg"],
    )

    paste_x = x0 + (panel_width - molecule.width) // 2
    paste_y = 58
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
    y_line = min(c_y, n_y) - 34
    dashed_line(draw, (c_x + 8, y_line), (n_x - 8, y_line), fill=guide_color)
    draw.line((c_x + 8, y_line - 5, c_x + 8, y_line + 5), fill=guide_color, width=2)
    draw.line((n_x - 8, y_line - 5, n_x - 8, y_line + 5), fill=guide_color, width=2)
    bond_width = draw.textlength(panel["bond"], font=fonts["small"])
    draw_text_label(
        draw,
        ((c_x + n_x - bond_width) / 2, y_line + BOND_LABEL_Y_OFFSET),
        panel["bond"],
        fonts["small"],
    )

    angle_value = angle_value_degrees(panel["angle"])
    cn_extension = (-1.0, 0.0)
    angle_y = (c_y + n_y) / 2.0
    angle_center = (
        c_x - ANGLE_VERTEX_LEFT_FROM_C + args.angle_vertex_dx,
        angle_y + args.angle_vertex_dy,
    )
    extension_start = (
        c_x - 2.0,
        angle_center[1],
    )
    extension_end = (
        angle_center[0] - ANGLE_EXTENSION_LENGTH,
        angle_center[1],
    )
    slanted_vector = rotate_vector(cn_extension, -angle_value)
    slanted_end = (
        angle_center[0] + slanted_vector[0] * ANGLE_SLANTED_RAY_LENGTH,
        angle_center[1] + slanted_vector[1] * ANGLE_SLANTED_RAY_LENGTH,
    )
    arc_start = math.degrees(math.atan2(cn_extension[1], cn_extension[0]))
    arc_end = math.degrees(math.atan2(slanted_vector[1], slanted_vector[0]))
    label_vector = rotate_vector(cn_extension, -angle_value / 2.0)
    label_center = (
        angle_center[0] + label_vector[0] * ANGLE_LABEL_RADIUS + args.angle_label_dx,
        angle_center[1]
        + label_vector[1] * ANGLE_LABEL_RADIUS
        + args.angle_label_y_from_cn_bond,
    )

    dashed_line(
        draw,
        extension_start,
        extension_end,
        fill=guide_color,
        dash=6,
        gap=5,
    )
    dashed_line(
        draw,
        angle_center,
        slanted_end,
        fill=guide_color,
        dash=6,
        gap=5,
    )
    draw.line(
        arc_points(
            angle_center,
            ANGLE_ARC_RADIUS,
            arc_start,
            arc_end,
            steps=20,
        ),
        fill=guide_color,
        width=2,
    )
    draw_centered_text_label(
        draw,
        label_center,
        panel["angle"],
        fonts["small"],
    )


def save_figure(canvas: Image.Image, output_prefix: Path, formats: list[str]) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        output_path = output_prefix.with_suffix(f".{fmt.lower()}")
        if fmt.lower() == "pdf":
            canvas.save(output_path, resolution=300)
        else:
            canvas.save(output_path, dpi=(300, 300))
        print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    panel_size = (430, 360)
    canvas = Image.new("RGB", (panel_size[0] * len(PANEL_DATA), panel_size[1]), "white")
    fonts = {
        "small": load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20),
        "bold": load_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24),
    }

    for index, panel in enumerate(PANEL_DATA):
        draw_panel(canvas, panel, args.image_dir, index, panel_size, fonts, args)

    save_figure(canvas, args.output_prefix, args.formats)


if __name__ == "__main__":
    main()
