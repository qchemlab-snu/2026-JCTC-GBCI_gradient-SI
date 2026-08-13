"""Sweep central finite-difference steps for LiX SFNOCI gradients.

The molecule definitions and SFNOCI energy/gradient routines are reused from
``compare_sfnoci_grad_li_halides.py``.  For each selected molecule, the
analytic gradient is evaluated once and the bond-direction numerical gradient
is evaluated at every requested finite-difference step.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pyscf import lib

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_sfnoci_grad_li_halides import (
    MOLECULES,
    MoleculeSpec,
    run_sfnoci_energy,
    run_sfnoci_gradient,
)


DEFAULT_STEPS_ANGSTROM = (5.0e-3, 1.0e-3, 5.0e-4, 1.0e-4, 5.0e-5, 1.0e-5, 5.0e-6, 1.0e-6, 5.0e-7, 1.0e-7)
DEFAULT_JSON_OUTPUT = SCRIPT_DIR / "outputs" / "li_halide_sfnoci_fd_step_sweep.json"
DEFAULT_CSV_OUTPUT = SCRIPT_DIR / "outputs" / "li_halide_sfnoci_fd_step_sweep.csv"


def numerical_bond_gradient(
    spec: MoleculeSpec,
    step_angstrom: float,
    *,
    scf_conv_tol: float,
    fci_conv_tol: float,
    verbose: int,
) -> dict[str, float]:
    """Return a central-difference derivative for the partner-atom z coordinate."""
    energy_minus = run_sfnoci_energy(
        spec,
        spec.bond_length - step_angstrom,
        scf_conv_tol=scf_conv_tol,
        fci_conv_tol=fci_conv_tol,
        verbose=verbose,
    )
    energy_plus = run_sfnoci_energy(
        spec,
        spec.bond_length + step_angstrom,
        scf_conv_tol=scf_conv_tol,
        fci_conv_tol=fci_conv_tol,
        verbose=verbose,
    )
    step_bohr = step_angstrom / lib.param.BOHR
    numerical_gradient = (energy_plus - energy_minus) / (2.0 * step_bohr)
    return {
        "step_angstrom": float(step_angstrom),
        "step_bohr": float(step_bohr),
        "energy_minus_hartree": float(energy_minus),
        "energy_plus_hartree": float(energy_plus),
        "numerical_z_atom1_hartree_per_bohr": float(numerical_gradient),
    }


def run_step_sweep(
    spec: MoleculeSpec,
    steps_angstrom: tuple[float, ...],
    *,
    scf_conv_tol: float,
    fci_conv_tol: float,
    verbose: int,
) -> dict[str, object]:
    """Evaluate the analytic reference and all finite-difference steps."""
    analytic_gradient = run_sfnoci_gradient(
        spec,
        scf_conv_tol=scf_conv_tol,
        fci_conv_tol=fci_conv_tol,
        verbose=verbose,
        show_svd_details=False,
        svd_log_rows=None,
    )
    analytic_z = float(analytic_gradient[1, 2])

    finite_difference_results = []
    for step_angstrom in steps_angstrom:
        result = numerical_bond_gradient(
            spec,
            step_angstrom,
            scf_conv_tol=scf_conv_tol,
            fci_conv_tol=fci_conv_tol,
            verbose=verbose,
        )
        difference = result["numerical_z_atom1_hartree_per_bohr"] - analytic_z
        result["difference_hartree_per_bohr"] = float(difference)
        result["abs_difference_hartree_per_bohr"] = float(abs(difference))
        finite_difference_results.append(result)

    return {
        "molecule": asdict(spec),
        "analytic_gradient_hartree_per_bohr": analytic_gradient.tolist(),
        "analytic_z_atom1_hartree_per_bohr": analytic_z,
        "finite_difference_results": finite_difference_results,
    }


def print_molecule_results(result: dict[str, object]) -> None:
    """Print one compact finite-difference convergence table."""
    molecule = result["molecule"]
    analytic_z = result["analytic_z_atom1_hartree_per_bohr"]
    print(f"\n=== {molecule['name']} ===")
    print(f"Analytic partner-atom z gradient: {analytic_z:+.12e} Eh/Bohr")
    print(
        " step (Angstrom)       numerical (Eh/Bohr)"
        "       numerical - analytic       abs(error)"
    )
    for row in result["finite_difference_results"]:
        print(
            f" {row['step_angstrom']:15.8e}"
            f"  {row['numerical_z_atom1_hartree_per_bohr']:+22.12e}"
            f"  {row['difference_hartree_per_bohr']:+22.12e}"
            f"  {row['abs_difference_hartree_per_bohr']:15.8e}"
        )


def save_csv(results: list[dict[str, object]], output_path: Path) -> None:
    """Save one row per molecule and finite-difference step."""
    fieldnames = [
        "molecule",
        "partner",
        "basis",
        "ncas",
        "nelecas",
        "bond_length_angstrom",
        "step_angstrom",
        "step_bohr",
        "energy_minus_hartree",
        "energy_plus_hartree",
        "analytic_z_atom1_hartree_per_bohr",
        "numerical_z_atom1_hartree_per_bohr",
        "difference_hartree_per_bohr",
        "abs_difference_hartree_per_bohr",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            molecule = result["molecule"]
            for row in result["finite_difference_results"]:
                writer.writerow(
                    {
                        "molecule": molecule["name"],
                        "partner": molecule["partner"],
                        "basis": molecule["basis"],
                        "ncas": molecule["ncas"],
                        "nelecas": json.dumps(molecule["nelecas"]),
                        "bond_length_angstrom": molecule["bond_length"],
                        "analytic_z_atom1_hartree_per_bohr": (
                            result["analytic_z_atom1_hartree_per_bohr"]
                        ),
                        **row,
                    }
                )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("finite-difference steps must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    molecule_names = [spec.name for spec in MOLECULES]
    parser = argparse.ArgumentParser(
        description=(
            "Sweep central finite-difference steps for the LiH, LiF, and LiCl "
            "SFNOCI bond-direction gradients."
        )
    )
    parser.add_argument(
        "--steps",
        type=positive_float,
        nargs="+",
        default=DEFAULT_STEPS_ANGSTROM,
        metavar="ANGSTROM",
        help=(
            "Finite-difference steps in Angstrom "
            "(default: 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)."
        ),
    )
    parser.add_argument(
        "--molecules",
        nargs="+",
        choices=molecule_names,
        default=molecule_names,
        help="Molecules to calculate (default: all).",
    )
    parser.add_argument("--scf-conv-tol", type=positive_float, default=1.0e-12)
    parser.add_argument("--fci-conv-tol", type=positive_float, default=1.0e-10)
    parser.add_argument("--verbose", type=int, default=4, help="PySCF verbosity level.")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_names = set(args.molecules)
    selected_specs = [spec for spec in MOLECULES if spec.name in selected_names]
    steps_angstrom = tuple(dict.fromkeys(float(step) for step in args.steps))

    results = []
    for spec in selected_specs:
        result = run_step_sweep(
            spec,
            steps_angstrom,
            scf_conv_tol=args.scf_conv_tol,
            fci_conv_tol=args.fci_conv_tol,
            verbose=args.verbose,
        )
        results.append(result)
        print_molecule_results(result)

    payload = {
        "script": Path(__file__).name,
        "method": "SFNOCI central finite difference along the Li-X bond",
        "step_unit": "Angstrom",
        "gradient_unit": "Hartree/Bohr",
        "scf_conv_tol": args.scf_conv_tol,
        "fci_conv_tol": args.fci_conv_tol,
        "results": results,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(results, args.csv_output)
    print(f"\nSaved JSON results to {args.json_output}")
    print(f"Saved CSV results to {args.csv_output}")


if __name__ == "__main__":
    main()
