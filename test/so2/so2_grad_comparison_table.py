"""Tabulate the analytic and numerical SO2 GBCI gradients from a sweep JSON.

This is a reporting script, not a calculation: it reads a JSON written earlier by
``gbci_so2_grad_fd_sweep.py`` and re-presents it as a table.  Nothing is
recomputed, which matters here because the analytic gradient on this system
costs 5 s at CAS(8e,5o) and 2587 s at CAS(12e,9o).

SO2 is a distorted C1 triatomic, so all nine Cartesian components are nonzero
and the table is indexed by (step, atom, axis) rather than by atom alone as in
the diatomic Li-halide tables.

Usage
-----
    python3 so2_grad_comparison_table.py
    python3 so2_grad_comparison_table.py --input outputs/cas8e5o_fd_sweep_fixed_grouping.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

DEFAULT_INPUT = SCRIPT_DIR / "outputs" / "cas8e5o_fd_sweep.json"
AXES = ("x", "y", "z")

COLUMNS = [
    "active_space", "step_bohr", "atom_index", "atom_symbol", "axis",
    "analytic_hartree_per_bohr", "numerical_hartree_per_bohr",
    "analytic_minus_numerical_hartree_per_bohr",
    "max_abs_error_hartree_per_bohr", "rms_error_hartree_per_bohr",
    "best_step",
]


def active_space_label(calculation: dict[str, Any]) -> str:
    """e.g. 'CAS(8e,5o)' from the stored ncas/nelecas."""
    return f"CAS({sum(calculation['nelecas'])}e,{calculation['ncas']}o)"


def usable_entries(sweep: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sweep entries that produced a gradient, newest-largest step first."""
    return sorted(
        (entry for entry in sweep if "error" not in entry),
        key=lambda entry: -entry["step_bohr"],
    )


def best_entry(sweep: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The step with the smallest maximum error, or None if every step failed."""
    usable = usable_entries(sweep)
    if not usable:
        return None
    return min(usable, key=lambda entry: entry["max_abs_error_hartree_per_bohr"])


def classify_regime(slope: float) -> str:
    """Name the finite-difference regime a log-log slope corresponds to."""
    if slope > 1.0:
        return "truncation, step too large"
    if slope < -0.5:
        return "energy noise, step too small"
    return "near the optimum"


def scaling_rows(sweep: list[dict[str, Any]], fd_order: int) -> list[dict[str, Any]]:
    """Local log-log slope of the error between neighbouring steps.

    Truncation scales as h^fd_order, noise as 1/h, so the sign of the slope says
    which side of the optimum a step sits on.
    """
    usable = sorted(usable_entries(sweep), key=lambda entry: entry["step_bohr"])
    rows = []
    for lower, upper in zip(usable, usable[1:]):
        slope = float(
            np.log(
                upper["max_abs_error_hartree_per_bohr"]
                / lower["max_abs_error_hartree_per_bohr"]
            )
            / np.log(upper["step_bohr"] / lower["step_bohr"])
        )
        rows.append(
            {
                "step_from_bohr": lower["step_bohr"],
                "step_to_bohr": upper["step_bohr"],
                "log_log_slope": slope,
                "expected_truncation_slope": fd_order,
                "regime": classify_regime(slope),
            }
        )
    return rows


def table_rows(result: dict[str, Any]) -> list[list[object]]:
    """One row per (step, atom, axis), carrying both gradients side by side."""
    calculation = result["calculation"]
    label = active_space_label(calculation)
    symbols = [row[0] for row in result["molecule"]["geometry_angstrom"]]
    analytic = np.asarray(result["analytic_gradient_hartree_per_bohr"])
    best = best_entry(result["sweep"])
    best_step = best["step_bohr"] if best else None

    rows: list[list[object]] = []
    for entry in usable_entries(result["sweep"]):
        numerical = np.asarray(entry["numerical_gradient_hartree_per_bohr"])
        difference = np.asarray(entry["difference_hartree_per_bohr"])
        for atom, symbol in enumerate(symbols):
            for axis, axis_name in enumerate(AXES):
                # A coordinate left out of the sweep is stored as NaN.
                if np.isnan(numerical[atom, axis]):
                    continue
                rows.append(
                    [
                        label, entry["step_bohr"], atom, symbol, axis_name,
                        float(analytic[atom, axis]),
                        float(numerical[atom, axis]),
                        float(difference[atom, axis]),
                        entry["max_abs_error_hartree_per_bohr"],
                        entry["rms_error_hartree_per_bohr"],
                        "yes" if entry["step_bohr"] == best_step else "",
                    ]
                )
    return rows


# ============================================================
# Reporting
# ============================================================

def print_report(result: dict[str, Any]) -> None:
    """Print the step summary, the scaling analysis, and the full comparison."""
    calculation = result["calculation"]
    diagnostics = result["diagnostics"]
    symbols = [row[0] for row in result["molecule"]["geometry_angstrom"]]
    analytic = np.asarray(result["analytic_gradient_hartree_per_bohr"])
    sweep = result["sweep"]

    print("=" * 78)
    print(f"{result['molecule']['name']}  {active_space_label(calculation)}  "
          f"basis={result['molecule']['basis']}  "
          f"MO {calculation['active_mo_indices']} (ncore={calculation['ncore']})")
    print(f"  E(GBCI) = {result['reference_energy_hartree']:.10f} Eh   "
          f"net force = {diagnostics['analytic_net_force_hartree_per_bohr']:.2e}   "
          f"torque = {diagnostics['analytic_torque_hartree']:.2e} Eh")
    print(f"  order-{calculation['fd_order']} central difference, "
          f"fasscf {calculation['fasscf_conv_tol']:.0e}/"
          f"{calculation['fasscf_conv_tol_grad']:.0e}, "
          f"gbci {calculation['gbci_conv_tol']:.0e}")
    print(f"  source: {Path(result['script']).name}, {result['timestamp_utc']}")
    print("=" * 78)

    for entry in sweep:
        if "error" in entry:
            print(f"  h = {entry['step_bohr']:.2e} skipped: {entry['error'][:90]}")

    usable = usable_entries(sweep)
    if not usable:
        print("  no usable steps in this sweep")
        return

    print(f"\n{'h (bohr)':>12}{'max |diff|':>16}{'rms':>16}")
    for entry in usable:
        print(f"{entry['step_bohr']:>12.2e}"
              f"{entry['max_abs_error_hartree_per_bohr']:>16.3e}"
              f"{entry['rms_error_hartree_per_bohr']:>16.3e}")

    rows = scaling_rows(sweep, calculation["fd_order"])
    if rows:
        print(f"\n{'h range (bohr)':>26}{'log-log slope':>16}   "
              f"regime (truncation would be {calculation['fd_order']:.0f})")
        for row in rows:
            print(f"{row['step_from_bohr']:>11.1e} -> {row['step_to_bohr']:<11.1e}"
                  f"{row['log_log_slope']:>14.2f}   {row['regime']}")

    best = best_entry(sweep)
    print(f"\n  best: h = {best['step_bohr']:.2e} bohr, "
          f"max |diff| = {best['max_abs_error_hartree_per_bohr']:.3e} Eh/bohr")

    numerical = np.asarray(best["numerical_gradient_hartree_per_bohr"])
    difference = np.asarray(best["difference_hartree_per_bohr"])
    print(f"\nAll nine components at the best step, h = {best['step_bohr']:.2e} bohr")
    print(f"{'atom':>6}{'axis':>6}{'analytic':>24}{'numerical':>24}{'diff':>14}")
    for atom, symbol in enumerate(symbols):
        for axis, axis_name in enumerate(AXES):
            if np.isnan(numerical[atom, axis]):
                continue
            print(f"{symbol + str(atom):>6}{axis_name:>6}"
                  f"{analytic[atom, axis]:>24.12e}{numerical[atom, axis]:>24.12e}"
                  f"{difference[atom, axis]:>14.2e}")


def write_csv(result: dict[str, Any], path: Path) -> None:
    """One row per (step, atom, axis)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in table_rows(result):
            writer.writerow(["" if value is None else value for value in row])


def write_xlsx(result: dict[str, Any], path: Path) -> None:
    """Workbook with the full comparison, a per-step summary, and the settings."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        print("openpyxl is not installed; skipping the .xlsx output")
        return

    science = "0.000E+00"
    bold = Font(bold=True)

    def add_header(sheet, header: list[str]) -> None:
        sheet.append(header)
        for cell in sheet[sheet.max_row]:
            cell.font = bold

    def autosize(sheet) -> None:
        for column in sheet.columns:
            width = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column
            )
            sheet.column_dimensions[column[0].column_letter].width = min(width + 2, 48)

    calculation = result["calculation"]
    diagnostics = result["diagnostics"]
    sweep = result["sweep"]

    workbook = Workbook()

    gradients = workbook.active
    gradients.title = "gradients"
    add_header(gradients, COLUMNS)
    for row in table_rows(result):
        gradients.append(row)
        for column in ("B", "F", "G", "H", "I", "J"):
            gradients[f"{column}{gradients.max_row}"].number_format = science
    gradients.freeze_panes = "A2"
    autosize(gradients)

    summary = workbook.create_sheet("by_step")
    add_header(summary, ["step_bohr", "max_abs_error_Eh_per_bohr",
                         "rms_error_Eh_per_bohr", "best_step", "status"])
    best = best_entry(sweep)
    best_step = best["step_bohr"] if best else None
    for entry in sorted(sweep, key=lambda item: -item["step_bohr"]):
        if "error" in entry:
            summary.append([entry["step_bohr"], None, None, "", entry["error"]])
        else:
            summary.append([
                entry["step_bohr"],
                entry["max_abs_error_hartree_per_bohr"],
                entry["rms_error_hartree_per_bohr"],
                "yes" if entry["step_bohr"] == best_step else "",
                "ok",
            ])
        for column in ("A", "B", "C"):
            summary[f"{column}{summary.max_row}"].number_format = science
    summary.append([])
    add_header(summary, ["h from (bohr)", "h to (bohr)", "log-log slope",
                         "truncation slope", "regime"])
    for row in scaling_rows(sweep, calculation["fd_order"]):
        summary.append([row["step_from_bohr"], row["step_to_bohr"],
                        row["log_log_slope"], row["expected_truncation_slope"],
                        row["regime"]])
        for column in ("A", "B"):
            summary[f"{column}{summary.max_row}"].number_format = science
        summary[f"C{summary.max_row}"].number_format = "0.00"
    autosize(summary)

    settings = workbook.create_sheet("settings")
    add_header(settings, ["setting", "value"])
    geometry = result["molecule"]["geometry_angstrom"]
    for label, value in (
        ("molecule", result["molecule"]["name"]),
        ("basis", result["molecule"]["basis"]),
        ("charge", result["molecule"]["charge"]),
        ("spin", result["molecule"]["spin"]),
        ("geometry (Angstrom)", "; ".join(
            f"{row[0]} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}" for row in geometry)),
        ("active space", active_space_label(calculation)),
        ("ncore", calculation["ncore"]),
        ("active MO indices", str(calculation["active_mo_indices"])),
        ("group_a", str(calculation["group_a"])),
        ("n bath groups", calculation["n_bath_groups"]),
        ("fd order", calculation["fd_order"]),
        ("scf_conv_tol", calculation["scf_conv_tol"]),
        ("scf_conv_tol_grad", calculation["scf_conv_tol_grad"]),
        ("fasscf_conv_tol", calculation["fasscf_conv_tol"]),
        ("fasscf_conv_tol_grad", calculation["fasscf_conv_tol_grad"]),
        ("gbci_conv_tol", calculation["gbci_conv_tol"]),
        ("reference energy (Eh)", result["reference_energy_hartree"]),
        ("analytic net force (Eh/bohr)",
         diagnostics["analytic_net_force_hartree_per_bohr"]),
        ("analytic torque (Eh)", diagnostics["analytic_torque_hartree"]),
        ("source script", result["script"]),
        ("source timestamp (UTC)", result["timestamp_utc"]),
        ("pyscf version", result["pyscf_version"]),
        ("gbci gradient module", result["gbci_gradient_module"]),
    ):
        settings.append([label, value])
    autosize(settings)

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tabulate analytic vs numerical SO2 GBCI gradients from a sweep JSON "
            "written earlier by gbci_so2_grad_fd_sweep.py.  Nothing is recomputed."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Sweep JSON to read.")
    parser.add_argument("--csv-output", type=Path, default=None,
                        help="Defaults to <input stem>_table.csv next to the input.")
    parser.add_argument("--xlsx-output", type=Path, default=None,
                        help="Defaults to <input stem>_table.xlsx next to the input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"No such sweep JSON: {args.input}")
    result = json.loads(args.input.read_text(encoding="utf-8"))

    csv_path = args.csv_output or args.input.with_name(f"{args.input.stem}_table.csv")
    xlsx_path = args.xlsx_output or args.input.with_name(f"{args.input.stem}_table.xlsx")

    print_report(result)
    write_csv(result, csv_path)
    write_xlsx(result, xlsx_path)
    print(f"\nSaved table to    {csv_path}")
    print(f"Saved workbook to {xlsx_path}")


if __name__ == "__main__":
    main()
