"""One-panel figure of the analytic vs numerical agreement over the whole gradient.

The point is not to let the reader recover numbers -- Table S for that -- but
to show at a glance that the agreement holds for every atom, not only the five
the main-text table lists.

Plotting the difference against the analytic component, rather than numerical
against analytic, is what makes that visible: on an x = y plot all 84 points
would sit on the line and a 1e-7 disagreement would be invisible, so the plot
would say nothing.  Here the ordinate is the difference alone.

Both axes are absolute values and the ordinate is logarithmic, which is what
lets a difference of 1e-9 and one of 2.5e-7 appear in the same panel; on a
linear axis the small ones pile up on zero.  The cost is that the sign is
gone, but the signs carry no message -- they split 42 to 42.

"MAX ABS" rather than "MAX": the largest difference in absolute value happens
to be the largest positive one as well (+2.499e-7, against -1.856e-7 the other
way), so a bare "MAX" would be indistinguishable from the positive extreme and
would read as if the negative half had been left out.  It also matches the
wording of the tables, which say "maximum absolute difference".
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

from utils.molecule_figure import FIG_WIDTH

RESULT_JSON = (SCRIPT_DIR / "outputs"
               / "01_gbci_s1_grad_check_631g_all_h1e4.json")
OUTPUT_STEM = SCRIPT_DIR / "outputs" / "14_agreement_figure"
# Written by 10_make_molecule_figure.py; its labels start with the element
# symbol (O4, C8a, H1), which is all the colouring needs.
LABELS_JSON = SCRIPT_DIR / "outputs" / "10_atom_labels.json"

COLOR = {"H": "#7a7a7a", "C": "#303030", "O": "#c0392b"}
# The floor is the smallest difference itself, so the lowest point sits on the
# frame and is drawn half outside it.  A round 1e-10 would leave an empty
# half-decade below it; a round 1e-9 would drop two points off the figure
# altogether, which in a figure about agreement would be the two best ones.
YTOP = 1e-5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--format", default="pdf", choices=("pdf", "svg", "png"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = json.loads(RESULT_JSON.read_text())
    analytic = np.asarray(result["analytic_gradient"])
    records = sorted(result["displacements"], key=lambda r: (r["atom"], r["axis"]))
    if len(records) != 84:
        raise RuntimeError(f"expected 84 components, got {len(records)}")
    if not LABELS_JSON.exists():
        raise RuntimeError(f"run 10_make_molecule_figure.py first to write "
                           f"{LABELS_JSON}")
    labels = {int(k): v for k, v in json.loads(LABELS_JSON.read_text()).items()}

    g = np.abs([analytic[r["atom"]][r["axis"]] for r in records])
    d = np.abs([analytic[r["atom"]][r["axis"]] - r["numerical"]
                for r in records])
    elements = [labels[r["atom"]][0] for r in records]

    bound = float(d.max())
    rms = float(np.sqrt((d ** 2).mean()))
    if d.max() > YTOP:
        raise RuntimeError(f"largest difference {d.max():g} is above the axis "
                           f"top {YTOP:g}")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH * 0.62))
    ax.set_yscale("log")
    ax.set_ylim(float(d.min()), YTOP)
    ax.grid(axis="y", which="major", color="#d8d8d8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    for element in ("C", "O", "H"):
        sel = [i for i, e in enumerate(elements) if e == element]
        if sel:
            ax.scatter(g[sel], d[sel], s=18, facecolor="none", linewidth=0.9,
                       edgecolor=COLOR[element], label=element, zorder=3)

    # The two summary numbers as lines, so each one is also a level the points
    # can be read against: dashed for the bound, dotted for the bulk.
    GAP = 1.3
    for value, style, name, offset in ((bound, (0, (5, 3)), "MAX ABS", GAP),
                                       (rms, (0, (1, 2)), "RMS", 1 / GAP)):
        ax.axhline(value, color="#303030", lw=0.8, ls=style, zorder=2)
        ax.text(0.98, value * offset,
                rf"{name} $= {value / 1e-7:.3f} \times 10^{{-7}}$",
                transform=ax.get_yaxis_transform(), fontsize=8.5,
                ha="right", va="bottom" if offset > 1 else "top",
                color="black", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    ax.set_xlabel(r"|Analytic gradient| (hartree/bohr)", fontsize=9)
    # Both axes name the quantity the same way.  "component" is dropped from
    # both rather than added to both: spelled out, the ordinate runs past the
    # height of the axis and is clipped, and the caption already says that
    # each point is one Cartesian component.
    ax.set_ylabel(r"|Analytic $-$ Numerical gradient|" "\n"
                  r"(hartree/bohr)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.3,
              borderaxespad=0.4, labelspacing=0.35)

    fig.tight_layout()
    stem = args.output or OUTPUT_STEM
    out = stem.with_suffix("." + args.format)
    fig.savefig(out, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"largest difference {bound:.3e}, rms {rms:.3e} hartree/bohr over "
          f"{len(records)} components; smallest {d.min():.3e}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
