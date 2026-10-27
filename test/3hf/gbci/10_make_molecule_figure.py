"""Figure of 3-hydroxyflavone to accompany the gradient table.

Projects the B3LYP-optimized geometry onto its own molecular plane (the heavy
atoms are planar to 0.005 A, so this is the real geometry rather than a redrawn
sketch) and labels the atoms.

One figure comes out of this: the plain structure that accompanies the
analytic/numerical comparison.  It carries no highlighting -- every atom is
displaced in the finite difference now, so marking five of them would be
misleading, and with nothing marked the legend goes too.

Drawing is shared with the other structure figures through
``utils/molecule_figure.py``; only what is specific to this molecule lives here.
"""

from pathlib import Path
import argparse
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

GEOMETRY = SCRIPT_DIR / ".." / "dft" / "outputs" / "3hf_dft_S0_opt.xyz"
OUTPUT_PLAIN = SCRIPT_DIR / "outputs" / "10_molecule_figure"
OUTPUT_LABELS = SCRIPT_DIR / "outputs" / "10_atom_labels.json"

# Standard numbering for 2-phenyl-3-hydroxy-4H-chromen-4-one: O1/C2/C3/C4 are
# the pyranone ring, C4a-C8a the fused benzo ring, primed labels the 2-phenyl
# (B) ring, O4 the keto oxygen and O3 the hydroxy oxygen. Hydrogens are simply
# numbered H1..H10 in input order, so the transferring proton is H1.
HEAVY_LABELS = {
    0: "O4", 1: "C4", 2: "C3", 3: "O3", 4: "C2",
    5: "C1'", 6: "C2'", 7: "C3'", 8: "C4'", 9: "C5'", 10: "C6'",
    11: "O1", 12: "C8a", 13: "C8", 14: "C7", 15: "C6", 16: "C5", 17: "C4a",
}


def read_xyz(path):
    symbols, coords = [], []
    for line in path.read_text().splitlines()[2:]:
        fields = line.split()
        if len(fields) == 4:
            symbols.append(fields[0])
            coords.append([float(x) for x in fields[1:]])
    return symbols, np.array(coords)


def atom_labels(symbols):
    labels = dict(HEAVY_LABELS)
    missing = [i for i, s in enumerate(symbols) if s != "H" and i not in labels]
    if missing:
        raise RuntimeError(f"no chemical label for heavy atoms {missing}")
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

    symbols, coords = read_xyz(GEOMETRY)
    labels = atom_labels(symbols)
    xy, _, rms = plane_projection(symbols, coords)
    print(f"heavy-atom plane: rms out-of-plane deviation {rms:.4f} A")

    span = xy.max(axis=0) - xy.min(axis=0)
    # 1.2 rather than the 2.4 the legend used to need below the structure.
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_WIDTH * (span[1] + 1.2) / (span[0] + 1.6)))
    draw_molecule(ax, symbols, coords, labels)

    fig.tight_layout()

    stem = args.output or OUTPUT_PLAIN
    out = stem.with_suffix("." + args.format)
    fig.savefig(out, bbox_inches="tight", transparent=False)
    plt.close(fig)

    OUTPUT_LABELS.write_text(json.dumps(
        {str(i): labels[i] for i in range(len(symbols))}, indent=2))
    print(f"Saved {out} and {OUTPUT_LABELS}")


if __name__ == "__main__":
    main()
