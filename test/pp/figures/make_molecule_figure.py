"""Structure figure of 4-(1H-pyrrol-1-yl)pyridine (pp), in the 3HF style.

Drawn from the CASCI S0-optimized geometry that
``test/active_orbitals/export_active_space_moldens.py`` uses, projected onto the best-fit plane
of the heavy atoms. Labels follow the scheme used in the manuscript: the pyrrole
ring is N1/C1-C4 and the pyridine ring C5-C9/N2, with the two rings joined at
N1-C5.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.molecule_figure import FIG_WIDTH, draw_molecule, plane_projection

# pp is wide and short, so the shared style leaves it small in a canvas sized
# for 3HF.  Enlarge the drawing inside the same 4.6 in width rather than the
# canvas: the figure is placed at a fixed column width, so only the molecule's
# share of it is visible as a size.
MARKER_SCALE = 430 / 340
FONT_SCALE = 10.5 / 9.0
MARGINS = 0.03
PAD_Y, PAD_X = 0.7, 0.9

OUTPUT_STEM = SCRIPT_DIR / "pp_molecule_figure"
OUTPUT_LABELS = SCRIPT_DIR / "pp_atom_labels.json"

# Heavy-atom labels, keyed by index in the geometry block. Derived from the
# connectivity, not guessed: N1 is the pyrrole nitrogen that also carries C5,
# and the ring order N1-C1-C2-C3-C4 / C5-C6-C7-N2-C8-C9 follows the manuscript
# drawing. Hydrogens are numbered H1..H8 in input order.
HEAVY_LABELS = {
    6: "N1", 1: "C1", 0: "C2", 2: "C3", 3: "C4",
    9: "C5", 10: "C6", 11: "C7", 13: "N2", 15: "C8", 16: "C9",
}
# Pyrrole to the right of nothing in particular: pin the pyridine to the right
# of the pyrrole nitrogen and C1 above centre, matching the manuscript drawing.
ANCHORS = {"x_from_to": (6, 9), "y_above": 1}


def geometry():
    """Symbols and coordinates of the pp benchmark geometry."""
    spec = importlib.util.spec_from_file_location(
        "_export",
        REPO_ROOT / "test" / "active_orbitals" / "export_active_space_moldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    symbols, coords = [], []
    for line in module.PP_ATOMS.strip().splitlines():
        fields = line.split()
        symbols.append(fields[0])
        coords.append([float(x) for x in fields[1:]])
    return symbols, np.array(coords)


def atom_labels(symbols):
    labels = dict(HEAVY_LABELS)
    missing = [i for i, s in enumerate(symbols) if s != "H" and i not in labels]
    if missing:
        raise RuntimeError(f"no label for heavy atoms {missing}")
    count = 0
    for i, symbol in enumerate(symbols):
        if symbol == "H":
            count += 1
            labels[i] = f"H{count}"
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", default="pdf", choices=("pdf", "svg", "png"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    symbols, coords = geometry()
    labels = atom_labels(symbols)
    xy, _, rms = plane_projection(symbols, coords, ANCHORS)
    print(f"heavy-atom plane: rms out-of-plane deviation {rms:.4f} A")

    span = xy.max(axis=0) - xy.min(axis=0)
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_WIDTH * (span[1] + PAD_Y) / (span[0] + PAD_X)))
    draw_molecule(ax, symbols, coords, labels, anchors=ANCHORS,
                  marker_scale=MARKER_SCALE, font_scale=FONT_SCALE,
                  margins=MARGINS)
    fig.tight_layout()

    stem = args.output or OUTPUT_STEM
    out = stem.with_suffix("." + args.format)
    fig.savefig(out, bbox_inches="tight", transparent=False)
    plt.close(fig)

    OUTPUT_LABELS.write_text(json.dumps(
        {str(i): labels[i] for i in range(len(symbols))}, indent=2))
    print(f"Saved {out} and {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()
