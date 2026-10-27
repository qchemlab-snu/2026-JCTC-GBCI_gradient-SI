"""LaTeX table formatting for gradient and geometry results.

The manuscript tables list one row per Cartesian component with digits grouped
in threes (``0.003\\,461\\,36``).  :func:`gradient_table` builds that table from
an analytic gradient and, when a finite-difference gradient is supplied, adds
the numerical column together with the max/rms difference rows.
"""

from __future__ import annotations

import numpy as np

COMPONENTS = ("x", "y", "z")


def format_number(value: float, decimals: int = 8) -> str:
    """``-0.014962454`` -> ``$-0.014\\,962\\,45$`` (digits grouped in threes)."""
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    groups = [frac[i:i + 3] for i in range(0, len(frac), 3)]
    return f"${sign}{whole}." + r"\," .join(groups) + "$"


def gradient_table(
    labels: list[str],
    analytic: np.ndarray,
    atoms: list[int],
    numerical: np.ndarray | None = None,
    torque: float | None = None,
    decimals: int = 8,
) -> str:
    """LaTeX tabular of the gradient of ``atoms`` (0-based indices).

    ``numerical`` may contain NaN for components that were not differenced;
    those rows print an em dash and are left out of the difference statistics.
    """
    two_column = numerical is not None
    spec = "llrr" if two_column else "llr"

    lines = [r"\begin{tabular}{%s}" % spec, r"    \hline"]
    header = (r"    \multirow{2}{*}{Atom} & \multirow{2}{*}{Comp.} &" "\n"
              r"    \multicolumn{1}{c}{Analytic Gradient}")
    if two_column:
        header += " &\n" + r"    \multicolumn{1}{c}{Numerical Gradient} \\"
    else:
        header += r" \\"
    lines.append(header)
    units = r"    & & \multicolumn{1}{c}{(hartree/bohr)}"
    if two_column:
        units += r" & \multicolumn{1}{c}{(hartree/bohr)} \\"
    else:
        units += r" \\"
    lines.append(units)
    lines.append(r"    \hline")

    for position, atom in enumerate(atoms):
        last_atom = position == len(atoms) - 1
        width = max(len(labels[a]) for a in atoms)
        for axis, comp in enumerate(COMPONENTS):
            # The atom label sits on the x row only, as in the manuscript tables.
            head = "    " + (labels[atom].ljust(width) if axis == 0
                             else " " * width)
            cells = [format_number(float(analytic[atom, axis]), decimals)]
            if two_column:
                value = float(numerical[atom, axis])
                cells.append("---" if np.isnan(value)
                             else format_number(value, decimals))
            row = f"{head} & ${comp}$ & " + " & ".join(cells) + r" \\"
            if axis == 2 and not last_atom:
                row += "[2pt]"
            lines.append(row)

    lines.append(r"    \hline")
    if two_column:
        mask = ~np.isnan(numerical[atoms])
        diff = (analytic[atoms] - numerical[atoms])[mask]
        lines.append(r"    \multicolumn{3}{l}{Maximum absolute difference} & $%s$ \\"
                     % latex_exponent(float(np.max(np.abs(diff)))))
        lines.append(r"    \multicolumn{3}{l}{Root-mean-square difference} & $%s$ \\"
                     % latex_exponent(float(np.sqrt(np.mean(diff ** 2)))))
    if torque is not None:
        span = 3 if two_column else 2
        lines.append(
            r"    \multicolumn{%d}{l}{Net torque $|\sum_A \mathbf{R}_A \times "
            r"\mathbf{g}_A|$ (hartree)} & $%s$ \\" % (span, latex_exponent(torque)))
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    return "\n".join(lines)


def latex_exponent(value: float, digits: int = 3) -> str:
    """``6.047e-11`` -> ``6.047 \\times 10^{-11}``."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def geometry_longtable(
    groups: list[tuple[str, list[str], "np.ndarray"]],
    caption: str,
    label: str,
    header: str = "Method",
    decimals: int = 8,
) -> str:
    """Supporting-information longtable of Cartesian coordinates.

    ``groups`` is one ``(heading, symbols, coords)`` triple per structure; each
    becomes a bold spanning row followed by one row per atom.  The column layout
    matches the geometry tables already in
    ``test/supporting_information_geometries.tex``.
    """
    lines = [
        r"\begin{longtable}{llrrr}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\hline",
        rf"{header} & Atom & $x$ & $y$ & $z$ \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        rf"{header} & Atom & $x$ & $y$ & $z$ \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
    ]
    for heading, symbols, coords in groups:
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{heading}}}}}\\")
        for symbol, xyz in zip(symbols, coords):
            cells = " & ".join(f"{float(v):{decimals + 3}.{decimals}f}" for v in xyz)
            lines.append(rf" & {symbol} & {cells} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"
