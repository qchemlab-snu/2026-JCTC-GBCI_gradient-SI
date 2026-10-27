"""Shared S1 SA-CASSCF frequency driver for PP planar geometries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
from pyscf import gto


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pp_casscf_opt as casscf_config  # noqa: E402
from utils.hessian import (  # noqa: E402
    imaginary_mode_to_displaced_mols,
    numerical_hessian_from_scanner_factory,
)


TARGET_STATE = "S1"
TARGET_ROOT = 1
GRADIENT_RMS_THRESHOLD = 3.0e-4
GRADIENT_MAX_THRESHOLD = 4.5e-4
DEFAULT_HESSIAN_STEP = 1.0e-3
DEFAULT_IMAG_THRESHOLD_CM = 30.0
DEFAULT_IMAG_AMPLITUDE = 0.05


def parse_args(default_output_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a supplied PP planar geometry is an S1 SA-CASSCF "
            "saddle point using the S1 gradient and a numerical Hessian."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for structured results and restartable Hessian gradients.",
    )
    parser.add_argument(
        "--hessian-step",
        type=float,
        default=DEFAULT_HESSIAN_STEP,
        help="Central finite-difference displacement in Bohr.",
    )
    parser.add_argument(
        "--imag-threshold-cm",
        type=float,
        default=DEFAULT_IMAG_THRESHOLD_CM,
        help="A mode below minus this value is counted as imaginary.",
    )
    parser.add_argument(
        "--gradient-only",
        action="store_true",
        help="Evaluate only the S1 energy and gradient, without the Hessian.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=casscf_config.VERBOSE,
        help="PySCF verbosity.",
    )
    return parser.parse_args()


def build_molecule(atoms: str, verbose: int) -> gto.Mole:
    return gto.M(
        atom=atoms,
        basis=casscf_config.BASIS,
        charge=casscf_config.CHARGE,
        spin=casscf_config.SPIN,
        unit=casscf_config.UNIT,
        verbose=verbose,
    )


def write_xyz(mol: gto.Mole, path: Path) -> None:
    path.write_text(mol.tostring(format="xyz") + "\n")


def save_json(path: Path, data: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    with path.open("w") as handle:
        json.dump(convert(data), handle, indent=2)


def cache_key(
    source_label: str,
    mol: gto.Mole,
    target_energy: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(source_label.encode())
    digest.update(casscf_config.METHOD.encode())
    digest.update(casscf_config.BASIS.encode())
    digest.update(np.asarray([TARGET_ROOT, casscf_config.NCAS]).tobytes())
    digest.update(np.asarray(casscf_config.NELECAS).tobytes())
    digest.update(np.asarray(casscf_config.STATE_WEIGHTS).tobytes())
    digest.update(np.asarray(mol.atom_coords(unit="Bohr")).tobytes())
    digest.update(f"{target_energy:.12f}".encode())
    return digest.hexdigest()


def print_frequencies(freqs_cm: np.ndarray) -> None:
    print("\nFrequencies / cm^-1:")
    for index, frequency in enumerate(freqs_cm):
        if frequency < 0.0:
            print(f"mode {index:3d}: i{abs(frequency):.2f}")
        else:
            print(f"mode {index:3d}:  {frequency:.2f}")


def write_frequency_table(
    path: Path,
    freqs_cm: np.ndarray,
    imag_threshold_cm: float,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "mode",
                "frequency_cm_inv",
                "is_significant_imaginary",
            ),
        )
        writer.writeheader()
        for index, frequency in enumerate(freqs_cm):
            writer.writerow(
                {
                    "mode": index,
                    "frequency_cm_inv": f"{float(frequency):.12f}",
                    "is_significant_imaginary": bool(
                        frequency < -abs(imag_threshold_cm)
                    ),
                }
            )


def saddle_classification(stationary: bool, n_imaginary: int) -> str:
    if not stationary:
        return "not_stationary_saddle_classification_not_valid"
    if n_imaginary == 0:
        return "minimum_not_saddle"
    if n_imaginary == 1:
        return "first_order_saddle"
    return "higher_order_saddle"


def run(
    *,
    atoms: str,
    source_label: str,
    source_file: str,
    run_name: str,
) -> None:
    args = parse_args(SCRIPT_DIR / "outputs" / run_name)
    if args.hessian_step <= 0.0:
        raise ValueError("--hessian-step must be positive.")
    if args.imag_threshold_cm <= 0.0:
        raise ValueError("--imag-threshold-cm must be positive.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mol = build_molecule(atoms, verbose=args.verbose)
    input_xyz = output_dir / f"{run_name}_input.xyz"
    write_xyz(mol, input_xyz)

    print(f"===== PP {TARGET_STATE} SA-CASSCF planar saddle check =====")
    print(f"geometry source = {source_label} ({source_file})")
    print(f"basis = {casscf_config.BASIS}")
    print(
        "active space = "
        f"({casscf_config.NCAS}o, {sum(casscf_config.NELECAS)}e), "
        f"MOs {casscf_config.ACTIVE_ORBITALS_1BASED} (1-based)"
    )
    print(f"state averaging = {casscf_config.STATE_WEIGHTS} for S0/S1")
    print(f"frequency target = root {TARGET_ROOT} ({TARGET_STATE})")
    print(f"output directory = {output_dir}")

    reference_scanner = casscf_config.build_optimizer(
        target_root=TARGET_ROOT,
        verbose=True,
    )
    target_energy, gradient = reference_scanner(mol)
    gradient = np.asarray(gradient, dtype=float)
    if gradient.shape != (mol.natm, 3):
        raise ValueError(
            f"S1 gradient has shape {gradient.shape}; expected {(mol.natm, 3)}."
        )
    reference = reference_scanner.get_reference()
    root_energies = np.asarray(reference["e_states"], dtype=float)

    gradient_norm = float(np.linalg.norm(gradient))
    gradient_rms = float(np.sqrt(np.mean(gradient**2)))
    gradient_max = float(np.max(np.abs(gradient)))
    stationary = bool(
        gradient_rms <= GRADIENT_RMS_THRESHOLD
        and gradient_max <= GRADIENT_MAX_THRESHOLD
    )
    gradient_path = output_dir / f"{run_name}_S1_gradient.npz"
    np.savez(
        gradient_path,
        gradient=gradient,
        energy_hartree=float(target_energy),
        root_energies_hartree=root_energies,
        gradient_norm=gradient_norm,
        gradient_rms=gradient_rms,
        gradient_max=gradient_max,
    )

    print("\n===== S1 gradient at the supplied planar geometry =====")
    print(f"S1 energy / Hartree = {float(target_energy):.12f}")
    print(f"gradient norm / Eh Bohr^-1 = {gradient_norm:.8e}")
    print(f"gradient RMS  / Eh Bohr^-1 = {gradient_rms:.8e}")
    print(f"gradient max  / Eh Bohr^-1 = {gradient_max:.8e}")
    print(f"CASSCF S1 stationary by configured thresholds = {stationary}")

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(sys.argv[0]).name,
        "driver": Path(__file__).name,
        "run_name": run_name,
        "molecule": casscf_config.MOLECULE_NAME,
        "method": casscf_config.METHOD,
        "state": TARGET_STATE,
        "target_root": TARGET_ROOT,
        "geometry_source_label": source_label,
        "geometry_source_file": source_file,
        "geometry_angstrom": atoms,
        "basis": casscf_config.BASIS,
        "charge": casscf_config.CHARGE,
        "spin": casscf_config.SPIN,
        "unit": casscf_config.UNIT,
        "pyscf_version": pyscf.__version__,
        "active_space": {
            "ncas": casscf_config.NCAS,
            "nelecas": list(casscf_config.NELECAS),
            "active_orbitals_1based": casscf_config.ACTIVE_ORBITALS_1BASED,
        },
        "state_average": {
            "states": list(casscf_config.STATE_LABELS),
            "weights": casscf_config.STATE_WEIGHTS,
        },
        "energies_hartree": {
            "S0": float(root_energies[0]),
            "S1": float(root_energies[1]),
        },
        "gradient": {
            "norm_eh_per_bohr": gradient_norm,
            "rms_eh_per_bohr": gradient_rms,
            "max_eh_per_bohr": gradient_max,
            "rms_threshold": GRADIENT_RMS_THRESHOLD,
            "max_threshold": GRADIENT_MAX_THRESHOLD,
            "is_stationary": stationary,
            "npz": str(gradient_path),
        },
        "frequency_check": {
            "completed": False,
        },
    }
    summary_path = output_dir / f"{run_name}_summary.json"
    save_json(summary_path, summary)

    if args.gradient_only:
        print("Gradient-only run complete; numerical Hessian was skipped.")
        return

    reference_mol = reference["mol"].copy()
    reference_mo = np.array(reference["mo_coeff"], copy=True)
    reference_ci = [np.array(value, copy=True) for value in reference["ci"]]
    reference_energies = np.asarray(reference["e_states"], dtype=float)

    def scanner_factory():
        scanner = casscf_config.build_optimizer(
            target_root=TARGET_ROOT,
            verbose=False,
        )
        scanner.set_reference(
            mol=reference_mol,
            mo_coeff=reference_mo,
            ci=reference_ci,
            e_states=reference_energies,
        )
        return scanner

    displacement_dir = output_dir / f"{run_name}_hessian_displacements"
    hess4, hess2 = numerical_hessian_from_scanner_factory(
        scanner_factory=scanner_factory,
        mol=reference_mol,
        step=args.hessian_step,
        cache_dir=displacement_dir,
        cache_key=cache_key(source_label, reference_mol, target_energy),
    )
    hessian_path = output_dir / f"{run_name}_S1_hessian.npz"
    np.savez(
        hessian_path,
        hess4=hess4,
        hess2=hess2,
        step_bohr=float(args.hessian_step),
    )

    modes_path = output_dir / f"{run_name}_S1_modes.npz"
    (
        freqs_cm,
        _,
        imag_freq_cm,
        _,
        mol_plus,
        mol_minus,
    ) = imaginary_mode_to_displaced_mols(
        mol=reference_mol,
        hess2=hess2,
        imag_threshold_cm=args.imag_threshold_cm,
        amplitude=DEFAULT_IMAG_AMPLITUDE,
        amplitude_unit="Angstrom",
        save_path=modes_path,
    )
    print_frequencies(freqs_cm)

    frequency_csv = output_dir / f"{run_name}_S1_frequencies.csv"
    write_frequency_table(
        frequency_csv,
        freqs_cm,
        args.imag_threshold_cm,
    )
    imaginary_indices = np.where(
        freqs_cm < -abs(args.imag_threshold_cm)
    )[0]
    n_imaginary = int(imaginary_indices.size)
    classification = saddle_classification(stationary, n_imaginary)

    plus_path = None
    minus_path = None
    if mol_plus is not None:
        plus_path = output_dir / f"{run_name}_S1_imag_plus.xyz"
        minus_path = output_dir / f"{run_name}_S1_imag_minus.xyz"
        write_xyz(mol_plus, plus_path)
        write_xyz(mol_minus, minus_path)

    print("\n===== S1 saddle-point verdict =====")
    print(
        f"significant imaginary modes (< -{args.imag_threshold_cm:g} cm^-1) "
        f"= {n_imaginary}"
    )
    print(f"classification = {classification}")
    if not stationary:
        print(
            "The supplied geometry is not stationary on the CASSCF S1 "
            "surface, so its saddle order is not formally defined."
        )

    summary["frequency_check"] = {
        "completed": True,
        "hessian_step_bohr": float(args.hessian_step),
        "imag_threshold_cm": float(args.imag_threshold_cm),
        "imaginary_mode_indices": imaginary_indices,
        "imaginary_frequencies_cm": freqs_cm[imaginary_indices],
        "number_of_significant_imaginary_modes": n_imaginary,
        "most_negative_frequency_cm": (
            None if imag_freq_cm is None else float(imag_freq_cm)
        ),
        "classification": classification,
        "is_saddle_point": bool(stationary and n_imaginary >= 1),
        "is_first_order_saddle": bool(stationary and n_imaginary == 1),
        "hessian_npz": str(hessian_path),
        "modes_npz": str(modes_path),
        "frequencies_csv": str(frequency_csv),
        "displacement_cache_dir": str(displacement_dir),
        "imag_plus_xyz": None if plus_path is None else str(plus_path),
        "imag_minus_xyz": None if minus_path is None else str(minus_path),
    }
    save_json(summary_path, summary)
    print(f"Saved summary: {summary_path}")

