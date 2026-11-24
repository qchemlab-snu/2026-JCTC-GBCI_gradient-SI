"""Figure for the CH2NH2+ MECI path scan written by ch2nh2_meci_pes_smoothness.py.

Panels along the CASCI -> GBCI -> SA-CASSCF path, all sharing the arc-length
coordinate with zero at the GBCI MECI.  ``--panels`` selects which to draw:

``diabatic``
    The default single panel: the same surfaces as ``energy``, but each labelled
    by its leading configuration, so no curve changes character at a seam and
    the lower curve is S0 by inspection.  Replaces the ``energy`` + ``charge``
    pair.
``energy``
    The default single panel: adiabatic S0 and S1 energies of the three methods
    on one absolute scale, so the selective stabilization of the open-shell
    state by GBCI is visible.  Each curve runs in a pale casing colored by the
    configuration it carries, which is the ``charge`` panel folded in: the
    casing switches exactly where that method's states exchange character.
``charge``
    The Loewdin NH2 charge of each state, which exchanges between S0 and S1 at
    every seam -- the crossing is a charge-transfer one.
``gap``
    The S1-S0 gap, which touches zero once per method, at that method's own MECI.
``groups``
    Weight of each bath group (equivalently, of each configuration -- this
    active space puts one in each group) in S0 and S1, for all three methods.
    Every method is a clean one-group state that exchanges S0 and S1 at its own
    seam, and the three seams are the only places anything moves.
``npop``
    Loewdin population on nitrogen *within the active space* of each GBCI
    configuration: the pure configuration's density in its own relaxed bath,
    minus that bath's core, so only the two active electrons are counted and
    the ``0 2`` pattern neither eigenstate ever leads with is included too.
    Colors match the bath groups of the ``groups`` panel.
``curvature``
    The second derivative of the mean energy, the smoothness diagnostic.  On a
    piecewise-linear path the interior waypoint is a corner (the legs meet at 62
    degrees) and every method spikes there by the same geometric factor, so that
    point is taken off scale and quoted separately; on the circular-arc path the
    curvature of the path is constant and nothing is excluded.

Usage::

    python3 ch2nh2_meci_pes_smoothness_plot_figure.py
    python3 ch2nh2_meci_pes_smoothness_plot_figure.py --panels all
    python3 ch2nh2_meci_pes_smoothness_plot_figure.py --path casci_gbci_casscf
    python3 ch2nh2_meci_pes_smoothness_plot_figure.py \
        --panels groups npop --suffix _groups \
        --title "CH$_2$NH$_2^+$ configuration character along the MECI path"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_STEM = "ch2nh2_ccpvtz_meci_pes_smoothness"
DEFAULT_PATH = "casci_gbci_casscf_smooth"
DEFAULT_PANELS = ("energy",)
# The figure goes under a caption in the manuscript, so it carries no title
# of its own; pass --title to put one back for a standalone copy.
DEFAULT_TITLE = ""
# The path interpolates between the three optimized MECI structures; it is a
# probe of the surfaces, not a minimum-energy reaction path.  The unit is kept
# because the coordinate is a real arc length in Cartesian space.
# The path interpolates whole structures, so its arc length mixes bond
# lengths with angles; it is left unitless rather than labelled in Angstrom.
DEFAULT_XLABEL = "Reaction coordinate"
ALL_PANELS = ("diabatic", "energy", "gap", "charge", "curvature", "groups",
              "weights_s0", "weights_s1",
              "npop")

HARTREE2EV = 27.211386245988
MOLECULAR_CHARGE = 1

# Atom 0 of every structure is the nitrogen.
N_ATOM = 0
N_CHARGE = 7

# The two configurations that cross at the seam.  Resolving the states by
# configuration instead of by energy removes the adiabatic relabelling, so a
# curve that is continuous here is continuous in the method itself.
DIABATS = ("2 0", "1 1")
# The occupation strings "2 0" / "1 1" count electrons in the two active
# orbitals; spelled out, those are the pi and pi* occupations.
CONFIG_TEXT = {"2 0": "closed shell (2 0)",
               "1 1": "open shell (1 1)",
               "0 2": "doubly excited (0 2)"}
CONFIG_STYLE = {"2 0": {"linestyle": "-", "linewidth": 2.0},
                "1 1": {"linestyle": "--", "linewidth": 1.8}}
# The two adiabatic states, drawn the same way as in the energy panel.
STATE_TEXT = ("S$_0$", "S$_1$")
STATE_STYLE = ({"linestyle": "-", "linewidth": 2.0},
               {"linestyle": "--", "linewidth": 1.6})
# Light fills for the character band above the curves.  Deliberately not the
# group colors: in the energy and charge panels the saturated colors already
# mean a method, so the band stays pale and says which configuration it is in
# words.
# Pale enough to sit under a colored curve.  Teal and amber rather than
# green and blue: the method colors are blue, red and black, and a blue band
# under the blue CASCI curve read as part of the curve.
CHARACTER_FILL = {"2 0": "#b9ded5", "1 1": "#f7d7ae", "0 2": "#dccfe8"}

# Colors for the GBCI bath groups, keyed by the occupation they collect.
# The saturated version of CHARACTER_FILL, so a configuration keeps its hue
# across the main and SI figures.  Neither is red or blue, which the seam lines
# and the method curves already use.
GROUP_COLOR = {"2 0": "#2a9d8f", "1 1": "#e8872f", "0 2": "#9467bd"}
GROUP_FALLBACK = ["#d62728", "#2a9d8f", "#9467bd", "#8c6d31"]

# Same method colors as the PP PES figures.
METHOD_STYLE = {
    "CASCI": {"color": "#1f77b4", "marker": "o"},
    "GBCI": {"color": "#d62728", "marker": "s"},
    "SA-CASSCF": {"color": "#1a1a1a", "marker": "^"},
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help="path name used by the scan script")
    parser.add_argument("--panels", nargs="+", default=list(DEFAULT_PANELS),
                        help=f"any of {ALL_PANELS}, or 'all'")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--suffix", default="",
                        help="appended to the output file stem")
    parser.add_argument("--title", default=DEFAULT_TITLE,
                        help="figure title; pass an empty string for none")
    parser.add_argument("--width", type=float, default=None,
                        help="total figure width in inches; set it to the "
                             "width the figure occupies in the manuscript so "
                             "LaTeX does not scale it (and shrink the text)")
    parser.add_argument("--xlabel", default=DEFAULT_XLABEL,
                        help="shared x-axis label")
    parser.add_argument("--zero", choices=("common", "meci"), default="common",
                        help="energy zero: one common zero for all methods "
                             "(default), or each method's own MECI")
    parser.add_argument("--no-flip", dest="flip", action="store_false",
                        default=True,
                        help="keep the scan's own direction of the path "
                             "(CASCI end on the left) instead of the plotted "
                             "one")
    return parser.parse_args()


def load_scan(path_name):
    run_name = f"{RUN_STEM}_{path_name}"
    json_path = SCRIPT_DIR / "outputs" / run_name / f"{run_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing scan result {json_path}. Run "
            f"ch2nh2_meci_pes_smoothness.py --path {path_name} first."
        )
    return run_name, json.loads(json_path.read_text())


def nh2_charge(state):
    """Loewdin charge on NH2.

    The scan stores the CH2 fragment charge, and the two fragments between them
    hold every atom, so the NH2 charge is the molecular charge minus it.  Newer
    scans store ``q_NH2`` directly and it is used when present.
    """
    if "q_NH2" in state:
        return float(state["q_NH2"])
    return MOLECULAR_CHARGE - float(state["q_CH2"])


def n_population(state):
    """Loewdin population on nitrogen, in electrons."""
    if "q_N" in state:
        return N_CHARGE - float(state["q_N"])
    if "q_atoms" in state:
        return N_CHARGE - float(state["q_atoms"][N_ATOM])
    return float("nan")


def diabatic_series(rows, conf, getter):
    """``getter`` of the state whose leading configuration is ``conf``."""
    out = []
    for row in rows:
        hit = [st for st in row.get("states", []) if st.get("top_conf") == conf]
        out.append(getter(hit[0]) if hit else float("nan"))
    return np.asarray(out, dtype=float)


def group_weight_series(rows, label, igroup, rank):
    """Weight of bath group ``igroup`` (occupation ``label``) in state ``rank``.

    GBCI records the group weights themselves; CASCI and SA-CASSCF have no
    baths, so the weight of the configuration the group holds is used, which is
    the same number whenever a group holds one configuration.  Configuration
    lists are truncated below the scan's tolerance, so a missing entry is a
    weight below that tolerance and is read as zero.
    """
    out = []
    for row in rows:
        states = row.get("states")
        if not states or rank >= len(states):
            out.append(np.nan)
            continue
        state = states[rank]
        if "group_weights" in state:
            out.append(float(state["group_weights"][igroup]))
            continue
        out.append(float(next(
            (c["weight"] for c in state.get("configurations", [])
             if " ".join(str(int(x)) for x in c["occ"]) == label),
            0.0,
        )))
    return np.asarray(out, dtype=float)


def group_labels(scan):
    """Occupation label of every GBCI bath group, or ``None`` if not recorded."""
    for row in scan.get("GBCI", []):
        if "group_labels" in row:
            return list(row["group_labels"])
    return None


def state_series(rows, getter, rank):
    return np.asarray(
        [getter(row["states"][rank]) if "states" in row else np.nan
         for row in rows],
        dtype=float,
    )


def draw_waypoints(ax, waypoints, by_method=True):
    for wp in waypoints:
        # Each seam carries its method's color so a label can be matched to its
        # own line by color alone; the staggered label rows are too close
        # together to tell apart otherwise.
        style = METHOD_STYLE.get(wp["label"]) if by_method else None
        ax.axvline(wp["s"], color=style["color"] if style else MUTED,
                   linestyle=":", linewidth=1.1 if style else 0.9,
                   alpha=0.85 if style else 1.0, zorder=0)


def panel_diabatic(ax, ctx):
    """Energies resolved by configuration instead of by state.

    This is the energy panel and the charge panel in one.  Labelling the two
    surfaces by their leading configuration rather than by their energy order
    removes the swap at every seam, so each curve carries one character from
    end to end: the line style says which configuration it is, the color says
    which method, and S0 is simply whichever curve is lower at that point.  The
    seams are then ordinary crossings, marked with a dot, and no second panel
    is needed to say which state is which.
    """
    s, scan, methods = ctx["s"], ctx["scan"], ctx["methods"]
    e_ref = min(
        float(np.nanmin([row.get("E0", np.nan) for row in scan[m]]))
        for m in methods
    )
    # The CASCI and GBCI closed-shell curves lie on top of each other -- the
    # grouped bath buys the closed-shell configuration nothing -- so the
    # methods are drawn as nested bands, widest first, and a hidden curve shows
    # as a halo around the one covering it.
    for rank, method in enumerate(methods):
        color = METHOD_STYLE[method]["color"]
        rows = scan[method]
        widen = 1.0 + 0.55 * (len(methods) - 1 - rank)
        for conf in DIABATS:
            e = diabatic_series(rows, conf, lambda st: float(st["energy"]))
            style = dict(CONFIG_STYLE[conf])
            style["linewidth"] *= widen
            ax.plot(s, (e - e_ref) * HARTREE2EV, color=color, **style)

    # The crossing of a method's two curves is that method's MECI.
    for wp in ctx["waypoints"]:
        method = wp["label"]
        if method not in scan:
            continue
        idx = int(np.argmin(np.abs(s - wp["s"])))
        row = scan[method][idx]
        y = (0.5 * (row["E0"] + row["E1"]) - e_ref) * HARTREE2EV
        ax.plot([wp["s"]], [y], marker="o", markersize=5.0,
                color=METHOD_STYLE[method]["color"], markeredgecolor="white",
                markeredgewidth=1.0, linestyle="none", zorder=5)

    ax.set_ylabel("Energy (eV)")
    ax.margins(y=0.12)
    handles = [
        Line2D([], [], color=METHOD_STYLE[m]["color"], linewidth=2.0, label=m)
        for m in methods
    ] + [
        Line2D([], [], color=MUTED, label=CONFIG_TEXT[conf],
               **CONFIG_STYLE[conf])
        for conf in DIABATS
    ] + [
        Line2D([], [], color=MUTED, marker="o", markersize=5.0,
               linestyle="none", label="MECI of that method")
    ]
    ax.legend(handles=handles, fontsize=8.2, frameon=False, loc="upper right",
              ncol=2, columnspacing=1.2, handlelength=1.8)
    return "surfaces labelled by configuration, not by energy order"


def draw_character_casing(ax, s, y, chars):
    """Pale casing under a curve, colored by the configuration it carries.

    This is the charge panel folded into the energy panel: instead of a second
    axes showing that the states exchange character at each seam, the curve
    itself changes casing there.  Drawn behind the curves so the method colors
    stay the foreground, and the runs overlap by one point so the casing has no
    gap at a switch.
    """
    start = 0
    for k in range(1, len(chars) + 1):
        if k < len(chars) and chars[k] == chars[start]:
            continue
        stop = min(k + 1, len(chars))
        ax.plot(s[start:stop], y[start:stop],
                color=CHARACTER_FILL.get(chars[start], "#eeeeee"),
                linewidth=6.5, solid_capstyle="butt", zorder=0)
        start = k


def label_seams(ax, ctx, headroom=0.20, by_method=True):
    """Name the dotted lines: which method has its MECI there.

    Put in a strip above the curves, with the two close seams staggered so the
    labels do not run into each other.
    """
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + headroom * (high - low))
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for k, wp in enumerate(ctx["waypoints"]):
        method = wp["label"]
        if method not in ctx["scan"]:
            continue
        # A centered label straddles its own line and reaches over the next
        # one, so each label instead starts at its seam and runs away from it:
        # the end that touches a dotted line names that line.  Near the right
        # edge it runs the other way so it stays on the axes.
        x0, x1 = ax.get_xlim()
        frac = (wp["s"] - x0) / (x1 - x0)
        ha = "right" if frac > 0.60 else "left"
        pad = 3.0 if ha == "left" else -3.0
        ax.annotate(f"{method} MECI", xy=(wp["s"], 0.985 - 0.075 * (k % 2)),
                    xycoords=trans, xytext=(pad, 0), textcoords="offset points",
                    ha=ha, va="top", fontsize=8.0,
                    color=METHOD_STYLE[method]["color"] if by_method else INK,
                    bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                              edgecolor="none"))


def energy_references(ctx):
    """Energy zero per method; see ``--zero``.

    ``common`` puts every method on one absolute scale, which is what shows
    that GBCI stabilizes the open-shell state and CASCI does not.  ``meci``
    instead sets each method's own MECI to zero, which lines the three seams up
    at the same height and compares the *shape* of the surfaces around them.
    """
    s, scan, methods = ctx["s"], ctx["scan"], ctx["methods"]
    if ctx.get("zero") != "meci":
        e_ref = min(
            float(np.nanmin([row.get("E0", np.nan) for row in scan[m]]))
            for m in methods
        )
        return {method: e_ref for method in methods}

    refs = {}
    for wp in ctx["waypoints"]:
        method = wp["label"]
        if method not in scan:
            continue
        idx = int(np.argmin(np.abs(s - wp["s"])))
        row = scan[method][idx]
        refs[method] = 0.5 * (row["E0"] + row["E1"])
    missing = [m for m in methods if m not in refs]
    if missing:
        raise SystemExit(
            f"--zero meci needs an MECI waypoint for {', '.join(missing)}"
        )
    return refs


def panel_energy(ax, ctx):
    s, scan, methods = ctx["s"], ctx["scan"], ctx["methods"]
    refs = energy_references(ctx)
    # Curves of different methods run on top of each other wherever they agree
    # -- the CASCI S0 and GBCI S1 closed-shell branches are the same surface --
    # so each carries its method's marker, offset so the markers interleave
    # instead of stacking.
    for imethod, method in enumerate(methods):
        color = METHOD_STYLE[method]["color"]
        marker = METHOD_STYLE[method]["marker"]
        rows = scan[method]
        for rank in (0, 1):
            e = ((np.asarray([row[f"E{rank}"] for row in rows]) - refs[method])
                 * HARTREE2EV)
            chars = [row["states"][rank]["top_conf"] if "states" in row else None
                     for row in rows]
            draw_character_casing(ax, s, e, chars)
            ax.plot(s, e, color=color, label=f"{method} S$_{rank}$",
                    marker=marker, markersize=3.8, markerfacecolor="white",
                    markeredgewidth=0.9, markevery=(imethod + 3 * rank, 6),
                    **STATE_STYLE[rank])
    ax.set_ylabel("Energy relative to that method's MECI (eV)"
                  if ctx.get("zero") == "meci" else "Energy (eV)")
    ax.margins(y=0.12)
    if ctx.get("zero") == "meci":
        ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=":", zorder=0)

    # Extra headroom beyond the seam labels' own: the character legend sits in
    # the upper-left corner, and without it the S1 curves run under its keys.
    label_seams(ax, ctx, headroom=0.44)

    handles = [
        Line2D([], [], color=CHARACTER_FILL[conf], linewidth=6.5,
               label=CONFIG_TEXT[conf])
        for conf in DIABATS
    ]
    # The seam lines run straight through this corner, so the legend gets an
    # opaque (but borderless) backing instead of showing them behind the keys.
    leg = ax.legend(handles=handles, fontsize=8.2, loc="upper left",
                    bbox_to_anchor=(0.0, 0.86), ncol=1, handlelength=1.8,
                    frameon=True, facecolor="white", edgecolor="none",
                    framealpha=1.0, borderpad=0.5)
    leg.set_zorder(5)
    # A legend drawn later on the same axes would replace this one, so it is
    # registered as a plain artist instead.
    ax.add_artist(leg)
    return ""


def panel_charge(ax, ctx):
    s, scan, methods = ctx["s"], ctx["scan"], ctx["methods"]
    # No character casing here: the charge itself is what tells the two
    # configurations apart, so a band saying the same thing only adds color.
    for imethod, method in enumerate(methods):
        color = METHOD_STYLE[method]["color"]
        marker = METHOD_STYLE[method]["marker"]
        rows = scan[method]
        for rank in (0, 1):
            q = state_series(rows, nh2_charge, rank)
            ax.plot(s, q, color=color, label=f"{method} S$_{rank}$",
                    marker=marker, markersize=3.8, markerfacecolor="white",
                    markeredgewidth=0.9, markevery=(imethod + 3 * rank, 6),
                    **STATE_STYLE[rank])
    ax.set_ylabel("Löwdin charge on NH$_2$ (e)")
    ax.margins(y=0.14)
    label_seams(ax, ctx, headroom=0.16)
    # Only the panel letter goes above the axes; a sentence there would run
    # into the figure legend, and the caption says it better anyway.
    return ""


def panel_groups(ax, ctx):
    """Weight of every bath group in S0 and S1, method by method.

    The question is whether a state's grouped bath changes discontinuously
    along the path.  It does not: in every method each adiabatic state sits
    with weight ~1 in a single group, and the only thing that happens anywhere
    on the path is that S0 and S1 exchange groups at that method's own seam --
    which is the adiabatic labelling, not the method.  CASCI and SA-CASSCF have
    no baths, so their configuration weights are plotted instead; for this
    active space a group holds exactly one configuration, so the two are the
    same number.
    """
    s, scan, methods = ctx["s"], ctx["scan"], ctx["methods"]
    labels = group_labels(scan)
    if labels is None:
        raise SystemExit(
            "this scan predates group labels; rerun ch2nh2_meci_pes_smoothness.py"
        )

    for method in methods:
        rows = scan[method]
        marker = METHOD_STYLE[method]["marker"]
        for ig, label in enumerate(labels):
            color = GROUP_COLOR.get(
                label, GROUP_FALLBACK[ig % len(GROUP_FALLBACK)]
            )
            for rank in (0, 1):
                w = group_weight_series(rows, label, ig, rank)
                style = dict(STATE_STYLE[rank])
                ax.plot(s, w, color=color, marker=marker, markersize=3.2,
                        markevery=(ig + 2 * rank, 6), markerfacecolor="white",
                        markeredgewidth=0.8, **style)

    ax.set_ylim(-0.06, 1.12)
    ax.set_ylabel("Weight in the state")
    # Every curve sits at 0 or 1 except where a method's states exchange, so
    # the band between them is free for naming which seam is whose.
    for wp in ctx["waypoints"]:
        if wp["label"] not in scan:
            continue
        ax.annotate(wp["label"], xy=(wp["s"], 0.5), xytext=(-4, 0),
                    textcoords="offset points", rotation=90, fontsize=7.6,
                    ha="right", va="center",
                    color=METHOD_STYLE[wp["label"]]["color"],
                    bbox=dict(boxstyle="square,pad=0.08", facecolor="white",
                              edgecolor="none"))
    handles = [
        Line2D([], [], color=GROUP_COLOR.get(
            label, GROUP_FALLBACK[ig % len(GROUP_FALLBACK)]),
            linewidth=2.0, label=f"group {label}")
        for ig, label in enumerate(labels)
    ] + [
        Line2D([], [], color=MUTED, linewidth=1.0,
               marker=METHOD_STYLE[m]["marker"], markersize=3.6,
               markerfacecolor="white", markeredgewidth=0.8, label=m)
        for m in methods
    ]
    # One column, parked in the empty block between the first and second seam,
    # so it clears both the curves and the seam labels.
    ax.legend(handles=handles, fontsize=7.6, frameon=False,
              loc="center left", bbox_to_anchor=(0.40, 0.5),
              ncol=1, columnspacing=1.0, handlelength=1.8)
    return "S$_0$/S$_1$ exchange groups only at each seam"


def panel_npop(ax, ctx):
    """Active-space Loewdin population on N of every GBCI configuration.

    Not of the eigenstates: each curve is the density of one pure configuration
    in the bath GBCI relaxed for it, so the pattern the eigenstates never lead
    with ("0 2") has a curve as well, and nothing jumps where the adiabatic
    labels swap.
    """
    s, scan = ctx["s"], ctx["scan"]
    rows = scan.get("GBCI")
    if not rows:
        raise SystemExit("the 'npop' panel needs a scan that includes GBCI")
    if not any("pop_act_N" in cs
               for row in rows for cs in row.get("configuration_states", [])):
        raise SystemExit(
            "this scan has no per-configuration densities; rerun "
            "ch2nh2_meci_pes_smoothness.py"
        )

    labels = []
    for row in rows:
        for cs in row.get("configuration_states", []):
            if cs["occ"] not in labels:
                labels.append(cs["occ"])

    handles = []
    for ig, label in enumerate(labels):
        color = GROUP_COLOR.get(label, GROUP_FALLBACK[ig % len(GROUP_FALLBACK)])
        y = np.asarray([
            next((float(cs["pop_act_N"])
                  for cs in row.get("configuration_states", [])
                  if cs["occ"] == label and "pop_act_N" in cs), np.nan)
            for row in rows
        ], dtype=float)
        ax.plot(s, y, color=color, linewidth=2.0)
        handles.append(Line2D([], [], color=color, linewidth=2.0,
                              label=f"config {label}"))
    ax.set_ylabel("Active-space Löwdin population on N (e)")
    ax.margins(y=0.16)
    # The three curves span the panel, so the legend and the seam labels each
    # get a strip of their own above them rather than sitting on a curve.
    label_seams(ax, ctx, headroom=0.42)
    ax.legend(handles=handles, fontsize=8.0, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 0.88),
              ncol=len(handles), columnspacing=1.0, handlelength=1.6)
    return "active-space N population per configuration"


def panel_gap(ax, ctx):
    s, scan, methods = ctx["s"], ctx["scan"], ctx["methods"]
    for method in methods:
        style = METHOD_STYLE[method]
        gap = np.asarray([row["gap_eV"] for row in scan[method]])
        ax.plot(s, gap, color=style["color"], linewidth=2.0,
                marker=style["marker"], markersize=3.6, markerfacecolor="white",
                markeredgewidth=0.9, label=method)
    # Placed for the plotted direction of the path (closed-shell end first),
    # where the three minima fall at the right, the middle and the left.
    offsets = [(30, 24), (12, 120), (-16, 86)]
    for k, wp in enumerate(ctx["waypoints"]):
        method = wp["label"]
        if method not in scan:
            continue
        idx = int(np.argmin(np.abs(s - wp["s"])))
        gap = scan[method][idx]["gap_eV"]
        ax.annotate(
            f"{method}\n{gap * 1000:.1f} meV",
            xy=(wp["s"], gap), xytext=offsets[k % len(offsets)],
            textcoords="offset points", fontsize=8.0, ha="center",
            color=METHOD_STYLE[method]["color"],
            arrowprops=dict(arrowstyle="-", color=METHOD_STYLE[method]["color"],
                            linewidth=0.8, shrinkA=1.0, shrinkB=2.0),
        )
    ax.set_ylabel("S$_1$ - S$_0$ gap (eV)")
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.06)
    ax.legend(fontsize=8.4, frameon=False, loc="upper right")
    return "each seam sits at its own MECI"


def panel_curvature(ax, ctx):
    s, methods, payload = ctx["s"], ctx["methods"], ctx["payload"]
    junction_s = ctx["junction_s"]
    on_scale = []
    junction_values = []
    for method in methods:
        style = METHOD_STYLE[method]
        curv = np.asarray(
            [np.nan if x is None else x
             for x in payload["mean_energy_second_difference"][method]],
            dtype=float,
        )
        mask = np.ones(curv.shape, dtype=bool)
        for js in junction_s:
            mask[np.argmin(np.abs(s - js))] = False
        ax.plot(s[mask], curv[mask], color=style["color"], linewidth=2.0,
                marker=style["marker"], markersize=3.6, markerfacecolor="white",
                markeredgewidth=0.9, label=method)
        on_scale.extend(curv[mask][np.isfinite(curv[mask])].tolist())
        for value in curv[~mask]:
            if np.isfinite(value):
                junction_values.append((method, float(value)))

    if on_scale:
        low, high = float(np.min(on_scale)), float(np.max(on_scale))
        pad = 0.25 * (high - low)
        bottom_pad = 2.2 * pad if junction_values else 0.7 * pad
        ax.set_ylim(low - bottom_pad, high + pad)
    if junction_values:
        # The corner values are off scale, so they are stated as numbers; the
        # rest of the explanation belongs in the caption, not in the axes.
        note = "off scale:  " + ",  ".join(
            f"{method} {value:.1f}" for method, value in junction_values
        )
        ax.annotate(note, xy=(0.5, 0.035), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=7.8, color=MUTED)
    ax.set_ylabel("d$^2\\langle E\\rangle$/ds$^2$ (E$_h$/$\\AA^2$)")
    ax.legend(fontsize=8.4, frameon=False, loc="upper left")
    return "smoothness of the mean energy"


# energy and charge both draw S0 solid and S1 dashed in the method colors, so
# one legend for the figure says it for every panel that uses that encoding.
STATE_RESOLVED_PANELS = ("energy", "charge")


# groups draws S0 solid and S1 dashed in colors that mean a bath group, so the
# solid/dashed convention goes in the figure-wide legend while the colors --
# which mean a group there and a configuration in npop -- stay in the panels.
GROUP_PANELS = ("groups",)

# Where a single-panel figure puts its method legend inside the axes, by panel.
# Only the energy panel: its upper right corner is genuinely empty, while the
# charge panel's two levels leave no band tall enough for a three-row legend,
# so that one keeps the legend above the axes.
INSIDE_LEGEND = {
    "energy": ("upper right", (1.0, 0.88)),
}



# The weight panels encode the configuration in the color and the method in the
# marker, so the figure legend names the methods and the panel names the colors.
METHOD_PANELS = ("weights_s0", "weights_s1")


def method_legend_handles(methods):
    # Interleaved method/configuration keys: matplotlib fills a legend column
    # by column, so in three columns this puts the methods on the first row and
    # the configurations on the second.
    keys = [
        # Drawn in grey: in these panels color already means configuration,
        # and a colored key here would claim the method owns that color.
        Line2D([], [], color=MUTED, linewidth=1.4,
               marker=METHOD_STYLE[m]["marker"], markersize=6.4,
               markerfacecolor="white", markeredgecolor=MUTED,
               markeredgewidth=0.9, label=m)
        for m in methods
    ] + [
        # The second legend row: here color is the configuration, so the keys
        # are plain colored lines with no marker.
        Line2D([], [], color=GROUP_COLOR[conf], linewidth=2.6,
               label=CONFIG_TEXT[conf])
        for conf in CONF_ORDER
    ]
    half = len(keys) // 2
    return [h for pair in zip(keys[:half], keys[half:]) for h in pair]


def group_legend_handles():
    return [
        Line2D([], [], color=MUTED, label=STATE_TEXT[rank], **STATE_STYLE[rank])
        for rank in (0, 1)
    ]


def state_legend_handles(methods, markers=False):
    """Method x state proxies; ``markers`` matches the energy panel's markers."""
    handles = []
    for method in methods:
        color = METHOD_STYLE[method]["color"]
        extra = dict(marker=METHOD_STYLE[method]["marker"], markersize=3.8,
                     markerfacecolor="white", markeredgewidth=0.9) if markers else {}
        for rank in (0, 1):
            handles.append(Line2D([], [], color=color,
                                  label=f"{method} S$_{rank}$",
                                  **STATE_STYLE[rank], **extra))
    return handles


# Every spatial occupation of the (2e,2o) active space, in the order they are
# named in the text.
CONF_ORDER = ("2 0", "1 1", "0 2")


def conf_weight_series(rows, conf, rank):
    """Weight of configuration ``conf`` in state ``rank``, for any method.

    Unlike ``group_weight_series`` this never falls back on GBCI's group
    weights: the quantity plotted is the CI weight of one spatial occupation,
    which every method reports the same way.  Configuration lists are truncated
    below the scan's tolerance, so a missing entry is read as zero.
    """
    out = []
    for row in rows:
        states = row.get("states")
        if not states or rank >= len(states):
            out.append(np.nan)
            continue
        out.append(float(next(
            (c["weight"] for c in states[rank].get("configurations", [])
             if " ".join(str(int(x)) for x in c["occ"]) == conf),
            0.0,
        )))
    return np.asarray(out, dtype=float)


def panel_conf_weights(ax, ctx, rank):
    """Configuration weights of one adiabatic state, for all three methods.

    Color is the configuration and the marker is the method, so three curves
    per method sit on top of each other at 0 and 1 and separate only where that
    method's states exchange character.
    """
    s, scan, methods = ctx["s"], ctx["scan"], ctx["methods"]
    for imethod, method in enumerate(methods):
        rows = scan[method]
        marker = METHOD_STYLE[method]["marker"]
        for iconf, conf in enumerate(CONF_ORDER):
            ax.plot(s, conf_weight_series(rows, conf, rank),
                    color=GROUP_COLOR[conf], linewidth=1.8, marker=marker,
                    # Larger than elsewhere: the marker is the only thing that
                    # tells the three coincident curves apart here.
                    markersize=6.4, markerfacecolor="white",
                    markeredgewidth=1.1,
                    # Nine curves, nine distinct phases in a period of nine:
                    # no two markers land on the same point.
                    markevery=(3 * iconf + imethod, 9))
    ax.set_ylim(-0.06, 1.10)
    ax.set_ylabel(f"Weight in S$_{rank}$")
    label_seams(ax, ctx, headroom=0.16)
    # The configuration keys go in the figure legend together with the method
    # keys: the curves run along 0 and 1 across the whole panel, so any legend
    # inside the axes sits on top of them.
    return ""


def panel_weights_s0(ax, ctx):
    return panel_conf_weights(ax, ctx, 0)


def panel_weights_s1(ax, ctx):
    return panel_conf_weights(ax, ctx, 1)


PANEL_DRAW = {
    "weights_s0": panel_weights_s0,
    "weights_s1": panel_weights_s1,
    "energy": panel_energy,
    "gap": panel_gap,
    "charge": panel_charge,
    "curvature": panel_curvature,
    "diabatic": panel_diabatic,
    "groups": panel_groups,
    "npop": panel_npop,
}


def main():
    args = parse_args()
    panels = list(ALL_PANELS) if "all" in args.panels else list(args.panels)
    for name in panels:
        if name not in PANEL_DRAW:
            raise SystemExit(f"unknown panel {name!r}; choose from {ALL_PANELS}")

    run_name, payload = load_scan(args.path)
    s = np.asarray(payload["s_values"], dtype=float)
    scan = payload["scan"]
    waypoints = payload["waypoints"]
    methods = [m for m in ("CASCI", "GBCI", "SA-CASSCF") if m in scan]

    # Only a piecewise-linear path has corners; on the constant-curvature arc
    # every point is equivalent, so nothing is taken off scale.
    # The direction of the probe path is arbitrary -- it is a line through
    # three MECI structures, not a reaction -- and the scan runs it from the
    # CASCI end, which leaves the closed-shell branch on the right.  Plotting
    # against -s turns it around so the closed-shell side is read first, with
    # the axis still increasing to the right.
    if args.flip:
        s = -s
        waypoints = [dict(wp, s=-float(wp["s"])) for wp in waypoints]

    is_linear = payload.get("mode", "linear") == "linear"
    ctx = {
        "s": s, "scan": scan, "methods": methods, "waypoints": waypoints,
        "payload": payload, "zero": args.zero,
        "junction_s": [wp["s"] for wp in waypoints[1:-1]] if is_linear else [],
    }

    # Up to three panels stay on one row; beyond that the figure would be too
    # wide for a page, so it wraps into a two-column grid.
    ncols = len(panels) if len(panels) <= 3 else 2
    nrows = int(np.ceil(len(panels) / ncols))
    # Font sizes are absolute, so the figure must be authored at the size it is
    # printed at: scaling a 10 in figure down to a 7 in column shrinks every
    # label with it.  --width fixes the total width and keeps the panel aspect.
    panel_w = 5.1 if args.width is None else args.width / ncols
    panel_h = panel_w * (3.6 / 5.1)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows + 0.3),
                             sharex=True, squeeze=False)
    # A one-column figure is too narrow for the state legend on a single row,
    # so it wraps and the panel gives up the extra height.
    # A single panel has room for the state legend inside the axes, which
    # keeps the whole figure height for the curves.
    # On a column-width figure the legend would cover the curves, so it only
    # moves inside the axes when the panel is wide enough to spare the room.
    inside_legend = (len(panels) == 1 and panel_w >= 4.5
                     and panels[0] in INSIDE_LEGEND)
    # Six keys fit on one row only on a wide figure; on a narrow one they wrap,
    # and the panels give up the height the extra row needs.
    legend_ncol = 6 if (ncols >= 2 and panel_w >= 4.2) else 3
    legend_rows = 0 if inside_legend else int(np.ceil(6 / legend_ncol))
    top = (0.845 - 0.062 * legend_rows) if nrows == 1 else 0.880
    # The left margin is a fraction, but what has to fit there -- tick labels
    # plus the y-label -- is an absolute 0.62 in, so a narrow figure needs a
    # larger fraction or the y-label falls off the canvas.
    left = max(0.115 if ncols == 1 else 0.085, 0.62 / (panel_w * ncols))
    fig.subplots_adjust(left=left, right=0.985,
                        bottom=0.145 if nrows == 1 else 0.085,
                        top=top, wspace=0.22, hspace=0.16)
    flat_axes = axes.ravel()

    for k, name in enumerate(panels):
        ax = flat_axes[k]
        title = PANEL_DRAW[name](ax, ctx)
        # A single-panel figure has nothing to label (a) against, and a panel
        # that returns no title gets none.
        prefix = "" if len(panels) == 1 else f"({chr(ord('a') + k)}) "
        # A single panel under a figure title needs no title of its own.
        if len(panels) == 1 and args.title:
            title = ""
        if title or prefix:
            ax.set_title(f"{prefix}{title}".rstrip(), fontsize=10.0,
                         loc="left", color=INK, pad=3.0)
    for ax in flat_axes[len(panels):]:
        ax.set_visible(False)
    # Nothing titled the panels, so the strip the titles would have used goes
    # back to the axes.
    if not any(a.get_title() for a in flat_axes[:len(panels)]):
        fig.subplots_adjust(top=top + 0.050)
    # Nothing is written above the axes at all, so the top strip goes back too.
    # The strip above the axes is free only when nothing else uses it: a
    # wrapped legend already fills it.
    if not args.title:
        # A wrapped legend needs the strip, but not all of it: on a one-panel
        # figure the two rows still leave a wide gap above the axes.
        fig.subplots_adjust(top=fig.subplotpars.top
                            + (0.030 if legend_rows < 2 else 0.045))

    for k, ax in enumerate(flat_axes[:len(panels)]):
        draw_waypoints(ax, waypoints)
        ax.tick_params(labelsize=9.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if k >= len(panels) - ncols:
            ax.set_xlabel(args.xlabel)

    # The path construction (circular arc vs piecewise linear) and the active
    # space are recorded in the scan's JSON and in the caption, not in the
    # figure's own title.
    if args.title:
        fig.suptitle(args.title, fontsize=11.5, color=INK, y=0.985, va="top")

    handles = []
    if any(name in STATE_RESOLVED_PANELS for name in panels):
        handles.extend(state_legend_handles(
            methods, markers="energy" in panels))
    # The energy/charge legend already spells out solid S0 / dashed S1, so the
    # groups panel only adds its own when those panels are not in the figure.
    if not handles and any(name in GROUP_PANELS for name in panels):
        handles.extend(group_legend_handles())
    if not handles and any(name in METHOD_PANELS for name in panels):
        handles.extend(method_legend_handles(methods))
    if handles and inside_legend:
        # matplotlib fills a legend column by column, so interleaving the
        # handles puts the S0 curves in one column and the S1 curves in the
        # other, one method per row.
        handles = handles[0::2] + handles[1::2]
        leg = flat_axes[0].legend(
            handles=handles, labels=[h.get_label() for h in handles],
            # Where the panel leaves room: the energy panel has an empty
            # upper right (the character legend's headroom), the charge panel
            # has an empty band between its two charge levels.
            loc=INSIDE_LEGEND[panels[0]][0],
            bbox_to_anchor=INSIDE_LEGEND[panels[0]][1],
            ncol=2, fontsize=8.4, columnspacing=1.2,
            handlelength=2.0, frameon=True, facecolor="white",
            edgecolor="none", framealpha=1.0, borderpad=0.5)
        leg.set_zorder(5)
    elif handles:
        fig.legend(handles=handles, labels=[h.get_label() for h in handles],
                   loc="upper center",
                   # With no figure title the legend takes that strip, which
                   # keeps it clear of the panel letters.
                   bbox_to_anchor=(0.5, 0.995 if not args.title else 0.945),
                   ncol=min(len(handles), legend_ncol), frameon=False,
                   fontsize=8.4, columnspacing=1.4, handlelength=2.0)

    out_dir = SCRIPT_DIR / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        out_path = out_dir / f"{run_name}_figure{args.suffix}.{fmt}"
        fig.savefig(out_path, dpi=220 if fmt == "png" else None)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
