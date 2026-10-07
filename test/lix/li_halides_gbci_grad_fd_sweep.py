"""Sweep finite-difference step sizes for the LiH/LiF/LiCl GBCI gradients.

The analytic gradient does not depend on the step, so it is evaluated once per
molecule and reused; each step size then costs only the four displaced energies.
Everything else -- the molecules, the sigma-only ``act_list``, the tolerances,
and the objective-stability guards -- is imported from
``li_halides_gbci_grad_check.py`` so the two scripts cannot drift apart.

The sweep answers a question the single-step check cannot: whether a
disagreement is truncation error (falls as h^2, so the gradient is fine and the
step is too large) or noise in the displaced energies (rises as 1/h, so the step
is too small for the energy's smoothness).  A clean h^2 region bracketing the
minimum is the evidence that the analytic gradient is right.

Usage
-----
    OMP_NUM_THREADS=1 python3 li_halides_gbci_grad_fd_sweep.py
    OMP_NUM_THREADS=1 python3 li_halides_gbci_grad_fd_sweep.py --molecule LiCl
    OMP_NUM_THREADS=1 python3 li_halides_gbci_grad_fd_sweep.py --steps 2e-3,1e-3

Run single-threaded: threaded BLAS moves the energy by ~1e-9 Eh between runs and
a central difference amplifies that by 1/(2h).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
from pyscf import lib


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]

# Load the check script by path for the same reason it loads utils by path:
# putting REPO_ROOT on sys.path makes pyscf treat the repository's own pyscf/
# tree as a plugin, shadowing pyscf-forge's fixed pyscf.grad.gbci.  Importing it
# also runs its module-resolution guard, so a stale gbci gradient fails here too.
_spec = importlib.util.spec_from_file_location(
    "_li_halides_grad_check", SCRIPT_DIR / "li_halides_gbci_grad_check.py")
_check = importlib.util.module_from_spec(_spec)
# dataclasses resolves field types through sys.modules[cls.__module__], so the
# module has to be registered before its body runs.
sys.modules[_spec.name] = _check
_spec.loader.exec_module(_check)

MOLECULES = _check.MOLECULES
TILT_DEGREES = _check.TILT_DEGREES


# ============================================================
# User settings
# ============================================================

STEPS_BOHR = (1.0e-3, 5.0e-4, 1.0e-4, 5.0e-5, 1.0e-5)

DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "li_halides_gbci_grad_fd_sweep.json"
DEFAULT_CSV_OUTPUT = SCRIPT_DIR / "outputs" / "li_halides_gbci_grad_fd_sweep.csv"
DEFAULT_XLSX_OUTPUT = SCRIPT_DIR / "outputs" / "li_halides_gbci_grad_fd_sweep.xlsx"


def sweep_molecule(spec: Any, args: argparse.Namespace) -> dict[str, object]:
    """Evaluate the analytic gradient once, then finite-difference it at each step."""
    mol = _check.build_molecule(
        spec, _check.reference_coords(spec), unit="Angstrom",
        basis=args.basis, verbose=args.verbose,
    )
    coords_bohr = mol.atom_coords(unit="Bohr")
    gbci = _check.run_gbci(spec, mol, args)
    reference_groups = _check.normalized_groups(gbci._gbci_intermediates["group"])
    reference_active = _check.active_block(gbci).copy()
    reference_energy = _check.scalar_root_energy(gbci.e_tot)

    analytic = _check.analytic_gradient(gbci)
    analytic_z = analytic[:, 2]
    net_force, torque = _check.gradient_invariance(mol, analytic)

    print(f"\n{'=' * 72}")
    print(
        f"{spec.name}  CAS({sum(spec.nelecas)}e,{spec.ncas}o)  "
        f"act_list={list(spec.act_list)} (1-based)  basis={args.basis}"
    )
    print(f"  E(GBCI) = {reference_energy:.10f} Eh   "
          f"net force = {net_force:.2e}   torque = {torque:.2e}")
    print(f"{'=' * 72}")
    print(f"{'h (bohr)':>12}{'max |diff|':>16}{'rms':>16}{'min overlap':>14}")

    entries: list[dict[str, object]] = []
    for step in args.steps:
        step_args = Namespace(**vars(args))
        step_args.step_bohr = step
        try:
            numerical, records = _check.numerical_z_gradient(
                spec, coords_bohr, reference_groups, reference_active, step_args
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            print(f"{step:>12.3e}   skipped: {error}")
            entries.append({"step_bohr": step, "error": str(error)})
            continue

        difference = analytic_z - numerical
        max_abs = float(np.max(np.abs(difference)))
        rms = float(np.sqrt(np.mean(difference ** 2)))
        min_overlap = float(records[0]["min_active_overlap"])
        print(f"{step:>12.3e}{max_abs:>16.3e}{rms:>16.3e}{min_overlap:>14.6f}")
        entries.append(
            {
                "step_bohr": step,
                "numerical_z_gradient_hartree_per_bohr": numerical.tolist(),
                "analytic_z_minus_numerical_hartree_per_bohr": difference.tolist(),
                "max_abs_error_hartree_per_bohr": max_abs,
                "rms_error_hartree_per_bohr": rms,
                "min_active_overlap": min_overlap,
                "displaced_calculations": records,
            }
        )

    report_scaling(entries)
    report_gradients(entries, analytic_z, [symbol for symbol in ("Li", spec.partner)])

    return {
        "molecule": {
            "name": spec.name,
            "geometry_angstrom": [
                [symbol, *map(float, coord)]
                for symbol, coord in zip(("Li", spec.partner),
                                         _check.reference_coords(spec))
            ],
            "bond_length_angstrom": spec.bond_length,
            "charge": 0,
            "spin": 0,
            "basis": args.basis,
            "unit": "Angstrom",
        },
        "calculation": {
            "method": "RHF / GBCI",
            "ncas": spec.ncas,
            "nelecas": list(spec.nelecas),
            "act_list": list(spec.act_list),
            "act_base": _check.ACT_BASE,
            "act_list_character": spec.act_comment,
            "group_a": spec.group_a,
            "lowdin_threshold": spec.group_a.get("threshold"),
            "reference_groups": reference_groups,
            "steps_bohr": list(args.steps),
            "finite_difference": "two-point central difference along z",
            "scf_conv_tol": args.scf_conv_tol,
            "scf_conv_tol_grad": args.scf_conv_tol_grad,
            "fasscf_conv_tol": args.fasscf_conv_tol,
            "fasscf_conv_tol_grad": args.fasscf_conv_tol_grad,
            "gbci_conv_tol": args.gbci_conv_tol,
            "max_cycle": args.max_cycle,
        },
        "reference_energy_hartree": reference_energy,
        "analytic_gradient_hartree_per_bohr": analytic.tolist(),
        "diagnostics": {
            "net_force_norm_hartree_per_bohr": net_force,
            "torque_hartree": torque,
        },
        "sweep": entries,
        "scaling": scaling_rows(entries),
        "best_step": best_entry(entries),
    }


def classify_regime(slope: float) -> str:
    """Name the finite-difference regime a log-log slope corresponds to."""
    if slope > 1.0:
        return "truncation (h^2), step too large"
    if slope < -0.5:
        return "energy noise (1/h), step too small"
    return "near the optimum"


def scaling_rows(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    """Local log-log slope of the error between neighbouring steps.

    Truncation error scales as h^2 (slope +2 as h grows); energy noise scales as
    1/h (slope -1).  The sign of the slope says which side of the optimum a step
    sits on, which is the thing a bare error column cannot tell you.
    """
    usable = [entry for entry in entries if "error" not in entry]
    if len(usable) < 2:
        return []
    ordered = sorted(usable, key=lambda entry: entry["step_bohr"])
    rows = []
    for lower, upper in zip(ordered, ordered[1:]):
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
                "regime": classify_regime(slope),
            }
        )
    return rows


def best_entry(entries: list[dict[str, object]]) -> dict[str, object] | None:
    """The step with the smallest maximum error, or None if every step failed."""
    usable = [entry for entry in entries if "error" not in entry]
    if not usable:
        return None
    return min(usable, key=lambda entry: entry["max_abs_error_hartree_per_bohr"])


def report_scaling(entries: list[dict[str, object]]) -> None:
    """Print the slope table and the best step."""
    rows = scaling_rows(entries)
    if not rows:
        return
    print(f"\n{'h range (bohr)':>26}{'log-log slope':>16}   regime")
    for row in rows:
        print(
            f"{row['step_from_bohr']:>11.1e} -> {row['step_to_bohr']:<11.1e}"
            f"{row['log_log_slope']:>14.2f}   {row['regime']}"
        )
    best = best_entry(entries)
    print(f"  best: h = {best['step_bohr']:.1e} bohr, "
          f"max |diff| = {best['max_abs_error_hartree_per_bohr']:.3e} Eh/bohr")


def report_gradients(
    entries: list[dict[str, object]], analytic_z: np.ndarray, symbols: list[str]
) -> None:
    """Print the analytic and numerical gradient itself at every step.

    The error columns above say how well the two agree; this says what they are.
    The analytic value is one number per atom for the whole sweep, so a numerical
    column drifting away from it down the table is the sweep's actual content.
    """
    usable = [entry for entry in entries if "error" not in entry]
    if not usable:
        return
    best_step = best_entry(entries)["step_bohr"]

    print(f"\n{'h (bohr)':>12}", end="")
    for symbol in symbols:
        print(f"{'numerical z ' + symbol:>24}{'diff':>13}", end="")
    print()
    print(f"{'analytic':>12}", end="")
    for index, symbol in enumerate(symbols):
        print(f"{analytic_z[index]:>24.12e}{'':>13}", end="")
    print()
    for entry in sorted(usable, key=lambda item: -item["step_bohr"]):
        marker = " *" if entry["step_bohr"] == best_step else "  "
        print(f"{entry['step_bohr']:>10.1e}{marker}", end="")
        numerical = entry["numerical_z_gradient_hartree_per_bohr"]
        difference = entry["analytic_z_minus_numerical_hartree_per_bohr"]
        for index in range(len(symbols)):
            print(f"{numerical[index]:>24.12e}{difference[index]:>13.2e}", end="")
        print()
    print("  (* best step; diff = analytic - numerical)")


GRADIENT_COLUMNS = [
    "molecule", "ncas", "nelecas", "act_list", "step_bohr",
    "atom_index", "atom_symbol",
    "analytic_z_hartree_per_bohr", "numerical_z_hartree_per_bohr",
    "analytic_minus_numerical_hartree_per_bohr",
    "max_abs_error_hartree_per_bohr", "rms_error_hartree_per_bohr",
    "min_active_overlap", "best_step", "error",
]


def gradient_rows(results: list[dict[str, object]]) -> list[list[object]]:
    """One row per (molecule, step, atom), carrying both gradients side by side.

    Long rather than wide because the partner atom differs between molecules, so
    a wide layout would need per-molecule column names and could not be pivoted
    or plotted as one table.
    """
    rows: list[list[object]] = []
    for result in results:
        calculation = result["calculation"]
        name = result["molecule"]["name"]
        act_list = ",".join(str(index) for index in calculation["act_list"])
        symbols = [row[0] for row in result["molecule"]["geometry_angstrom"]]
        analytic_z = np.asarray(result["analytic_gradient_hartree_per_bohr"])[:, 2]
        best = result["best_step"]
        best_step = best["step_bohr"] if best else None

        for entry in result["sweep"]:
            numerical = entry.get("numerical_z_gradient_hartree_per_bohr")
            difference = entry.get("analytic_z_minus_numerical_hartree_per_bohr")
            for index, symbol in enumerate(symbols):
                rows.append(
                    [
                        name, calculation["ncas"], sum(calculation["nelecas"]),
                        act_list, entry["step_bohr"], index, symbol,
                        float(analytic_z[index]),
                        numerical[index] if numerical else None,
                        difference[index] if difference else None,
                        entry.get("max_abs_error_hartree_per_bohr"),
                        entry.get("rms_error_hartree_per_bohr"),
                        entry.get("min_active_overlap"),
                        "yes" if entry["step_bohr"] == best_step else "",
                        entry.get("error", ""),
                    ]
                )
    return rows


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    """One row per (molecule, step, atom) for quick plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(GRADIENT_COLUMNS)
        for row in gradient_rows(results):
            writer.writerow(["" if value is None else value for value in row])


def write_xlsx(results: list[dict[str, object]], path: Path) -> None:
    """Write the sweep as a workbook: one summary sheet plus one sheet per molecule.

    openpyxl is imported here rather than at module scope so the sweep still runs
    and still writes JSON and CSV on an environment without it.
    """
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
            sheet.column_dimensions[column[0].column_letter].width = min(width + 2, 46)

    workbook = Workbook()
    summary = workbook.active
    summary.title = "sweep"
    add_header(summary, GRADIENT_COLUMNS)
    # Columns carrying Eh/bohr or a step size; see GRADIENT_COLUMNS for the order.
    numeric_columns = ("E", "H", "I", "J", "K", "L")
    for row in gradient_rows(results):
        summary.append(row)
        for column in numeric_columns:
            summary[f"{column}{summary.max_row}"].number_format = science
    summary.freeze_panes = "A2"
    autosize(summary)

    for result in results:
        name = result["molecule"]["name"]
        calculation = result["calculation"]
        sheet = workbook.create_sheet(name)
        analytic = np.asarray(result["analytic_gradient_hartree_per_bohr"])
        symbols = [row[0] for row in result["molecule"]["geometry_angstrom"]]

        add_header(sheet, ["setting", "value"])
        for label, value in (
            ("basis", result["molecule"]["basis"]),
            ("bond length (Angstrom)", result["molecule"]["bond_length_angstrom"]),
            ("ncas", calculation["ncas"]),
            ("nelecas", sum(calculation["nelecas"])),
            ("act_list (1-based)", str(calculation["act_list"])),
            ("act_list character", calculation["act_list_character"]),
            ("reference energy (Eh)", result["reference_energy_hartree"]),
            ("net force norm (Eh/bohr)", result["diagnostics"]["net_force_norm_hartree_per_bohr"]),
            ("torque (Eh)", result["diagnostics"]["torque_hartree"]),
        ):
            sheet.append([label, value])
        sheet.append([])

        add_header(sheet, ["atom", "analytic z (Eh/bohr)"])
        for index, symbol in enumerate(symbols):
            sheet.append([symbol, float(analytic[index, 2])])
            sheet[f"B{sheet.max_row}"].number_format = science
        sheet.append([])

        # Per atom: analytic, numerical, and their difference side by side.  The
        # analytic column repeats down the table on purpose -- it does not depend
        # on the step, and seeing it next to the drifting numerical column is the
        # point of the sweep.
        gradient_header = ["step_bohr"]
        for symbol in symbols:
            gradient_header += [
                f"analytic z {symbol} (Eh/bohr)",
                f"numerical z {symbol} (Eh/bohr)",
                f"diff {symbol} (Eh/bohr)",
            ]
        gradient_header += [
            "max_abs_error_Eh_per_bohr", "rms_error_Eh_per_bohr",
            "min_active_overlap", "best_step",
        ]
        add_header(sheet, gradient_header)
        # step_bohr, the 3 gradient columns per atom, then max_abs and rms; the
        # overlap and the best-step flag are not Eh/bohr quantities.
        last_science_column = 1 + 3 * len(symbols) + 2
        best = result["best_step"]
        best_step = best["step_bohr"] if best else None
        for entry in result["sweep"]:
            numerical = entry.get("numerical_z_gradient_hartree_per_bohr")
            difference = entry.get("analytic_z_minus_numerical_hartree_per_bohr")
            row: list[object] = [entry["step_bohr"]]
            for index in range(len(symbols)):
                row += [
                    float(analytic[index, 2]),
                    numerical[index] if numerical else None,
                    difference[index] if difference else None,
                ]
            row += [
                entry.get("max_abs_error_hartree_per_bohr"),
                entry.get("rms_error_hartree_per_bohr"),
                entry.get("min_active_overlap"),
                "yes" if entry["step_bohr"] == best_step else "",
            ]
            sheet.append(row)
            for index in range(1, last_science_column + 1):
                sheet.cell(row=sheet.max_row, column=index).number_format = science
        sheet.append([])

        add_header(sheet, ["h from (bohr)", "h to (bohr)", "log-log slope", "regime"])
        for row in result["scaling"]:
            sheet.append(
                [row["step_from_bohr"], row["step_to_bohr"],
                 row["log_log_slope"], row["regime"]]
            )
            for column in ("A", "B"):
                sheet[f"{column}{sheet.max_row}"].number_format = science
            sheet[f"C{sheet.max_row}"].number_format = "0.00"
        autosize(sheet)

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def step_list(value: str) -> tuple[float, ...]:
    steps = tuple(float(item) for item in value.split(",") if item.strip())
    if not steps or any(step <= 0.0 for step in steps):
        raise argparse.ArgumentTypeError("steps must be positive and comma-separated")
    return steps


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the finite-difference step size for the LiH/LiF/LiCl GBCI "
            "gradient check on sigma-only active spaces."
        )
    )
    parser.add_argument("--steps", type=step_list, default=STEPS_BOHR,
                        help="Comma-separated steps in bohr.")
    parser.add_argument("--basis", default=_check.BASIS)
    parser.add_argument("--active-overlap-thresh", type=positive_float, default=0.9)
    # Tolerances are shared with the SO2 sweep
    # (test/so2/gbci_so2_grad_fd_sweep.py) except for the two energy
    # thresholds, which stay at 1e-12 here.  SO2's 1e-14 is below what these
    # molecules can resolve in double precision -- LiCl sits at -467 Eh, so
    # 1e-14 is a relative 2e-17 -- and an unreachable threshold makes the SCF
    # and the Davidson stop on numerical noise instead of converging.  Measured:
    # raising both to 1e-14 multiplied the energy noise by 10-20x (LiCl dE
    # 2e-13 -> 3.6e-12) and the finite-difference floor with it.  The orbital
    # gradient is stated rather than left to pyscf's sqrt(conv_tol) = 1e-6,
    # because that is the criterion that actually decides when the SCF stops.
    parser.add_argument("--scf-conv-tol", type=positive_float, default=1.0e-12)
    parser.add_argument("--scf-conv-tol-grad", type=positive_float, default=1.0e-9)
    parser.add_argument("--fasscf-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--fasscf-conv-tol-grad", type=positive_float, default=1.0e-12)
    parser.add_argument("--gbci-conv-tol", type=positive_float, default=1.0e-12)
    parser.add_argument("--max-cycle", type=int, default=400)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--molecule", action="append", default=None,
                        help="Restrict to one molecule; repeatable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--xlsx-output", type=Path, default=DEFAULT_XLSX_OUTPUT)
    return parser.parse_args()


def main() -> None:
    lib.num_threads(1)

    args = parse_args()
    selected = MOLECULES
    if args.molecule:
        wanted = {name.lower() for name in args.molecule}
        selected = [spec for spec in MOLECULES if spec.name.lower() in wanted]
        if not selected:
            raise SystemExit(f"No molecule matched {args.molecule}")

    results = [sweep_molecule(spec, args) for spec in selected]

    payload = {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pyscf_version": pyscf.__version__,
        "modules": {
            "gbci": str(Path(_check.gbci_module.__file__).resolve()),
            "gbci_gradient": str(Path(_check.gbci_grad.__file__).resolve()),
        },
        "num_threads": lib.num_threads(),
        "steps_bohr": list(args.steps),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(results, args.csv_output)
    write_xlsx(results, args.xlsx_output)
    print(f"\nSaved results to {args.output}")
    print(f"Saved table to    {args.csv_output}")
    print(f"Saved workbook to {args.xlsx_output}")


if __name__ == "__main__":
    main()
