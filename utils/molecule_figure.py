"""Flat structure figures of a (near-)planar molecule, drawn from real coordinates.

The molecule is projected onto the best-fit plane of its heavy atoms rather than
redrawn as a sketch, so bond lengths and angles in the figure are the computed
ones. Callers supply the atom labels and may mark a subset of atoms, add a
dashed contact, and attach per-atom vectors drawn as arrows.

Sizes default to a single journal column with relatively large type, so the
figure needs no further scaling in a manuscript.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05,
       "Cl": 1.02}
COLOR = {"C": "#303030", "O": "#c0392b", "N": "#1b5e9c", "H": "#707070"}
HIGHLIGHT = "#1f6feb"

FIG_WIDTH = 4.6                     # inches
FONT_ATOM = 9.0
FONT_ATOM_MARKED = 10.0
FONT_LEGEND = 9.0
FONT_ANNOTATION = 9.5
# Markers scale with the type: the circle has to hold its label, and the widest
# are three characters ("C8a", "C1'").
MARKER_HEAVY = 340
MARKER_H = 250


def plane_projection(symbols, coords, anchors=None):
    """2D coordinates in the best-fit plane of the heavy atoms.

    Returns ``(xy, (e1, e2), rms)``, where ``rms`` is the out-of-plane spread of
    the heavy atoms: the projection is only faithful while that is small.

    The singular vectors are fixed only up to sign, and the signs are whatever
    LAPACK returns, so ``anchors`` pins them to the chemistry instead:

        "x_from_to": (i, j)  atom j is drawn to the right of atom i
        "y_above":   k       atom k is drawn above the centre
    """
    symbols = list(symbols)
    heavy = np.array([s != "H" for s in symbols])
    centred = np.asarray(coords) - np.asarray(coords)[heavy].mean(axis=0)
    _, _, vt = np.linalg.svd(centred[heavy])
    e1, e2, normal = vt[0], vt[1], vt[2]
    rms = float(np.sqrt(((centred[heavy] @ normal) ** 2).mean()))

    if anchors:
        xy = np.column_stack([centred @ e1, centred @ e2])
        if "x_from_to" in anchors:
            i, j = anchors["x_from_to"]
            if xy[j, 0] < xy[i, 0]:
                e1 = -e1
        if "y_above" in anchors:
            if (centred[anchors["y_above"]] @ e2) < 0.0:
                e2 = -e2

    return np.column_stack([centred @ e1, centred @ e2]), (e1, e2), rms


def bond_list(symbols, coords, scale=1.3):
    """Index pairs within ``scale`` times the sum of covalent radii."""
    out = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            r = float(np.linalg.norm(np.asarray(coords[i]) - np.asarray(coords[j])))
            if r < scale * (COV.get(symbols[i], 0.8) + COV.get(symbols[j], 0.8)):
                out.append((i, j))
    return out


def draw_molecule(ax, symbols, coords, labels, *, anchors=None, highlight=(),
                  dashed=None, dashed_label=None, arrows=None, arrow_scale=1.0,
                  legend_entries=(), fig_width=FIG_WIDTH,
                  marker_scale=1.0, font_scale=1.0, margins=0.06):
    """Draw the molecule on ``ax``. Returns the 2D coordinates and plane basis.

    ``dashed`` is an ``(i, j)`` pair drawn as a dashed contact, labelled with
    ``dashed_label`` at the centroid of the ring it closes when one is given.
    ``arrows`` maps an atom index to a 3-vector, drawn from that atom at
    ``arrow_scale`` Angstrom per unit.

    ``marker_scale``, ``font_scale`` and ``margins`` enlarge the drawing inside
    a canvas of unchanged size, for a molecule whose span leaves the figure
    emptier than the others.  The defaults reproduce the shared style exactly.
    """
    symbols = list(symbols)
    xy, (e1, e2), _ = plane_projection(symbols, coords, anchors)
    highlight = set(highlight)

    for i, j in bond_list(symbols, coords):
        ax.plot(*zip(xy[i], xy[j]), color="#404040", lw=1.4,
                solid_capstyle="round", zorder=1)

    if dashed is not None:
        a, b = dashed
        ax.plot(*zip(xy[a], xy[b]), color=HIGHLIGHT, lw=1.3, ls=(0, (4, 3)),
                alpha=0.9, zorder=1)

    if arrows:
        for i, vector in arrows.items():
            vector = np.asarray(vector, dtype=float) * arrow_scale
            tip = (xy[i, 0] + vector @ e1, xy[i, 1] + vector @ e2)
            ax.annotate("", xy=tip, xytext=(xy[i, 0], xy[i, 1]), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT,
                                        lw=1.6, shrinkA=8, shrinkB=0))
            # Annotations do not drive autoscaling; pin the tip so it stays in
            # view instead of being clipped at the figure edge.
            ax.scatter(*tip, s=1, alpha=0.0, zorder=0)

    for i, symbol in enumerate(symbols):
        marked = i in highlight
        ax.scatter(*xy[i],
                   s=(MARKER_HEAVY if symbol != "H" else MARKER_H) * marker_scale,
                   facecolor="white",
                   edgecolor=HIGHLIGHT if marked else "#b8b8b8",
                   linewidth=1.5 if marked else 0.7, zorder=2)
        ax.text(*xy[i], labels[i], ha="center", va="center", zorder=3,
                fontsize=(FONT_ATOM_MARKED if marked else FONT_ATOM)
                         * font_scale,
                color=COLOR.get(symbol, "#303030"),
                fontweight="bold" if marked else "normal")

    if legend_entries:
        ax.legend(handles=list(legend_entries), loc="upper center",
                  frameon=False, fontsize=FONT_LEGEND,
                  bbox_to_anchor=(0.5, -0.01), handletextpad=0.8,
                  labelspacing=0.5)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(margins)
    return xy, (e1, e2)


def highlight_handle(label):
    return Line2D([], [], marker="o", ls="", markerfacecolor="white",
                  markeredgecolor=HIGHLIGHT, markeredgewidth=1.5,
                  markersize=7, label=label)


def dashed_handle(label):
    return Line2D([], [], color=HIGHLIGHT, ls=(0, (4, 3)), lw=1.3, label=label)


def arrow_handle(label):
    return Line2D([], [], color=HIGHLIGHT, lw=1.6, label=label)
