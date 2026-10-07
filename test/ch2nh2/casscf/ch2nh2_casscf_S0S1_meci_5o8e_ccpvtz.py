from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
from pyscf import gto
from pyscf.geomopt.addons import as_pyscf_method
from pyscf.geomopt.geometric_solver import optimize
from pyscf.tools import molden


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hessian import (  # noqa: E402
    imaginary_mode_to_displaced_mols,
    numerical_hessian_from_scanner_factory,
)
from utils.meci_optimizer import (  # noqa: E402
    CASSCFActiveRootTrackedPairScanner,
    HARTREE2EV,
    PenaltyFunctionMECIScanner,
)


RUN_NAME = "ch2nh2_casscf_S0S1_meci_5o8e_ccpvtz"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / RUN_NAME

# Converged cc-pVDZ CASCI (6e,4o) S0/S1 MECI geometry -- the same start for the
# CASCI, CASSCF and GBCI (8e,5o) runs.  Not an (8e,5o) seam (enlarging the
# active space moves the crossing), but close enough that the penalty starts in
# its driving regime, unlike the S0 minimum where the gap is 9.3 eV and the
# penalty is saturated by a factor of 100.
START_GEOMETRY_SOURCE = (
    "../../casci/outputs/ch2nh2_casci_S0S1_meci_4o6e/"
    "ch2nh2_casci_S0S1_meci_4o6e_optimized.xyz"
)

ATOMS = """N        0.63439558      -0.04344093      -0.02563145
C       -0.65655473      -0.22409714      -0.13753472
H        1.08629733       0.74030711       0.46229476
H        1.31478226      -0.68652595      -0.43193619
H       -1.12703216      -0.18816588       0.86559213
H       -1.12453928       0.69440579      -0.54474553
"""

BASIS = "cc-pVTZ"
CHARGE = 1
SPIN = 0
UNIT = "Angstrom"
VERBOSE = 4

NCAS = 5
NELECAS = (4, 4)
# (8e,5o): pi(C), pi(N), sigma(CN), pi(N), pi(C) at the start geometry.  The
# (6e,4o) runs used [5,7,8,9] and the tracker kept swapping MO7 (sigma(CN)) out
# for MO6, leaving an all-pi space; including 5-9 removes that choice.  Note
# there is no sigma*(CN) here -- MO10/11 are H-dominated diffuse sigma*.
ACTIVE_ORBITALS_1BASED = [5, 6, 7, 8, 9]
NROOTS = 2
TARGET_ROOTS = (0, 1)
STATE_WEIGHTS = [0.5, 0.5]

SCF_CASSCF_CONV_TOL = 1.0e-10
SCF_CASSCF_MAX_CYCLE = 200

SIGMA = 1.0
ALPHA = 1.0e-4
MAX_STEPS = 500

GEOM_PARAMS = {
    "trust": 0.01,
    "tmax": 0.03,
    "convergence_energy": 1.0e-6,
    "convergence_grms": 3.0e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}

GAP_TARGET_EV = 1.0e-3
HESSIAN_STEP = 1.0e-3
IMAG_THRESHOLD_CM = 30.0
IMAG_AMPLITUDE = 0.05
IMAG_AMPLITUDE_UNIT = "Angstrom"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize the CH2NH2+ S0/S1 MECI with equal-weight "
            "SA2-CASSCF(6e,4o) and a penalty function."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "optimize", "frequency"),
        default="all",
        help="Run optimization, penalty-Hessian frequency analysis, or both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for structured calculation outputs.",
    )
    parser.add_argument(
        "--maxsteps",
        type=int,
        default=MAX_STEPS,
        help="Maximum number of geomeTRIC optimization steps.",
    )
    parser.add_argument(
        "--hessian-step",
        type=float,
        default=HESSIAN_STEP,
        help="Central-difference Hessian step in Bohr.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=VERBOSE,
        help="PySCF verbosity.",
    )
    return parser.parse_args()


def build_molecule(
    atoms: Any = ATOMS,
    verbose: int = VERBOSE,
) -> gto.Mole:
    return gto.M(
        atom=atoms,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=verbose,
    )


def build_pair_scanner(
    act_list_1based=ACTIVE_ORBITALS_1BASED,
    verbose: bool = True,
) -> CASSCFActiveRootTrackedPairScanner:
    return CASSCFActiveRootTrackedPairScanner(
        ncas=NCAS,
        nelecas=NELECAS,
        act_list=act_list_1based,
        act_base=1,
        nroots=NROOTS,
        target_roots=TARGET_ROOTS,
        state_weights=STATE_WEIGHTS,
        conv_tol=SCF_CASSCF_CONV_TOL,
        max_cycle=SCF_CASSCF_MAX_CYCLE,
        root_tracking=False,
        active_tracking=True,
        pair_order="energy",
        reuse_ci_guess=True,
        project_previous_orbitals=True,
        strict_convergence=True,
        verbose=verbose,
    )


def build_penalty_scanner(
    pair_scanner,
    verbose: bool = True,
) -> PenaltyFunctionMECIScanner:
    return PenaltyFunctionMECIScanner(
        pair_scanner,
        sigma=SIGMA,
        alpha=ALPHA,
        verbose=verbose,
        print_level=4,
    )


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): to_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_serializable(data), indent=2, sort_keys=True) + "\n"
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_xyz(mol: gto.Mole, path: Path) -> None:
    path.write_text(mol.tostring(format="xyz") + "\n")


def base_metadata(maxsteps: int) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "molecule": "CH2NH2+",
        "method": "SA2-CASSCF penalty MECI",
        "states": ["S0", "S1"],
        "start_geometry_source": START_GEOMETRY_SOURCE,
        "start_geometry_angstrom": ATOMS,
        "basis": BASIS,
        "charge": CHARGE,
        "spin": SPIN,
        "unit": UNIT,
        "pyscf_version": pyscf.__version__,
        "active_space": {
            "ncas": NCAS,
            "nelecas": list(NELECAS),
            "active_orbitals_1based": ACTIVE_ORBITALS_1BASED,
        },
        "state_average": {
            "nroots": NROOTS,
            "target_roots": list(TARGET_ROOTS),
            "weights": STATE_WEIGHTS,
        },
        "root_tracking": False,
        "project_previous_orbitals": True,
        "reuse_ci_guess": True,
        "strict_convergence": True,
        "conv_tol": SCF_CASSCF_CONV_TOL,
        "max_cycle": SCF_CASSCF_MAX_CYCLE,
        "penalty": {
            "sigma_hartree": SIGMA,
            "alpha_hartree_squared": ALPHA,
        },
        "geometry_optimization": {
            "maxsteps": int(maxsteps),
            "parameters": GEOM_PARAMS,
            "gap_target_ev": GAP_TARGET_EV,
        },
        "frequency_check": {
            "completed": False,
            "objective": "SA2-CASSCF S0/S1 penalty function",
        },
    }


def save_reference_checkpoint(
    path: Path,
    reference: dict[str, Any],
) -> None:
    mol = reference["mol"]
    ci_list = reference["ci"]
    np.savez(
        path,
        atom_charges=np.asarray(mol.atom_charges(), dtype=int),
        coords_angstrom=np.asarray(
            mol.atom_coords(unit="Angstrom"),
            dtype=float,
        ),
        mo_coeff=np.asarray(reference["mo_coeff"]),
        ci_root_0=np.asarray(ci_list[0]),
        ci_root_1=np.asarray(ci_list[1]),
        e_states=np.asarray(reference["e_states"], dtype=float),
        selected_act_0based=np.asarray(
            reference["selected_act_0based"],
            dtype=int,
        ),
        ncas=NCAS,
        nelecas=np.asarray(NELECAS, dtype=int),
        state_weights=np.asarray(STATE_WEIGHTS, dtype=float),
    )


def load_reference_checkpoint(
    path: Path,
    verbose: int,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference checkpoint not found: {path}. "
            "Run --stage optimize first."
        )

    with np.load(path, allow_pickle=False) as saved:
        atom_charges = np.asarray(saved["atom_charges"], dtype=int)
        coords_angstrom = np.asarray(
            saved["coords_angstrom"],
            dtype=float,
        )
        mo_coeff = np.array(saved["mo_coeff"], copy=True)
        ci = [
            np.array(saved["ci_root_0"], copy=True),
            np.array(saved["ci_root_1"], copy=True),
        ]
        e_states = np.asarray(saved["e_states"], dtype=float)
        selected_act_0based = np.asarray(
            saved["selected_act_0based"],
            dtype=int,
        )
        saved_ncas = int(saved["ncas"])
        saved_nelecas = tuple(
            int(value)
            for value in np.asarray(saved["nelecas"], dtype=int)
        )
        saved_weights = np.asarray(
            saved["state_weights"],
            dtype=float,
        )

    if saved_ncas != NCAS or saved_nelecas != NELECAS:
        raise ValueError("Checkpoint active space does not match this script.")
    if not np.allclose(
        saved_weights,
        STATE_WEIGHTS,
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "Checkpoint state-average weights do not match this script."
        )

    atom_spec = [
        (int(charge), tuple(coords))
        for charge, coords in zip(atom_charges, coords_angstrom)
    ]
    return {
        "mol": build_molecule(atom_spec, verbose=verbose),
        "mo_coeff": mo_coeff,
        "ci": ci,
        "e_states": e_states,
        "selected_act_0based": selected_act_0based,
    }


def reference_cache_key(reference: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).name.encode())
    digest.update(BASIS.encode())
    digest.update(np.asarray(STATE_WEIGHTS, dtype=float).tobytes())
    digest.update(np.asarray([NCAS, *NELECAS], dtype=int).tobytes())
    digest.update(
        np.asarray(
            reference["mol"].atom_coords(unit="Bohr"),
            dtype=float,
        ).tobytes()
    )
    digest.update(np.asarray(reference["mo_coeff"]).tobytes())
    for ci_vector in reference["ci"]:
        digest.update(np.asarray(ci_vector).tobytes())
    digest.update(np.asarray(reference["e_states"], dtype=float).tobytes())
    return digest.hexdigest()


def print_settings(output_dir: Path) -> None:
    print("===== CH2NH2+ SA2-CASSCF S0/S1 penalty MECI search =====")
    print(f"start_geometry_source = {START_GEOMETRY_SOURCE}")
    print(f"basis = {BASIS}")
    print(f"charge = {CHARGE}")
    print(f"spin = {SPIN}")
    print(f"active orbitals, 1-based = {ACTIVE_ORBITALS_1BASED}")
    print(f"ncas = {NCAS}")
    print(f"nelecas = {NELECAS}")
    print(f"nroots = {NROOTS}")
    print(f"target_roots = {TARGET_ROOTS}")
    print(f"state_weights = {STATE_WEIGHTS}")
    print(f"sigma = {SIGMA}")
    print(f"alpha = {ALPHA}")
    print(f"output_dir = {output_dir}")


def run_optimization(
    output_dir: Path,
    maxsteps: int,
    verbose: int,
) -> dict[str, Any]:
    mol = build_molecule(verbose=verbose)
    pair_scanner = build_pair_scanner(verbose=True)
    penalty_scanner = build_penalty_scanner(
        pair_scanner,
        verbose=True,
    )
    fake_method = as_pyscf_method(mol, penalty_scanner)

    print_settings(output_dir)
    mol_opt = optimize(
        fake_method,
        maxsteps=maxsteps,
        **GEOM_PARAMS,
    )

    print("\n===== Final SA2-CASSCF S0/S1 pair evaluation =====")
    final_pair = pair_scanner(mol_opt)
    final_objective, final_penalty_gradient, penalty_info = (
        penalty_scanner.objective_from_pair_result(final_pair)
    )
    reference = pair_scanner.get_reference()

    optimized_xyz = output_dir / f"{RUN_NAME}_optimized.xyz"
    write_xyz(mol_opt, optimized_xyz)
    print("\n===== Final S0/S1 MECI geometry =====")
    print(mol_opt.tostring(format="xyz"))
    print(f"Saved optimized geometry: {optimized_xyz}")

    final_pair_path = output_dir / f"{RUN_NAME}_final_pair.npz"
    np.savez(
        final_pair_path,
        energies_hartree=np.asarray(final_pair.energies, dtype=float),
        gradients_hartree_per_bohr=np.asarray(
            final_pair.gradients,
            dtype=float,
        ),
        roots=np.asarray(final_pair.roots, dtype=int),
        penalty_objective_hartree=float(final_objective),
        penalty_gradient_hartree_per_bohr=np.asarray(
            final_penalty_gradient,
            dtype=float,
        ),
        selected_act_0based=np.asarray(
            final_pair.diagnostics["selected_act"],
            dtype=int,
        ),
        selected_act_1based=np.asarray(
            final_pair.diagnostics["selected_act"],
            dtype=int,
        )
        + 1,
    )

    history_path = output_dir / f"{RUN_NAME}_penalty_history.json"
    write_json(
        history_path,
        {"evaluations": penalty_scanner.history},
    )
    pair_history_path = output_dir / f"{RUN_NAME}_pair_history.json"
    write_json(
        pair_history_path,
        {"evaluations": pair_scanner.history},
    )

    reference_path = output_dir / f"{RUN_NAME}_reference.npz"
    save_reference_checkpoint(reference_path, reference)

    orbitals_path = output_dir / f"{RUN_NAME}_orbitals.molden"
    molden.from_mcscf(pair_scanner.last_mc, str(orbitals_path))

    energies = np.asarray(final_pair.energies, dtype=float)
    gap_hartree = float(energies[1] - energies[0])
    gap_ev = float(gap_hartree * HARTREE2EV)
    penalty_gradient = np.asarray(
        final_penalty_gradient,
        dtype=float,
    )
    penalty_gradient_rms = float(
        np.sqrt(np.mean(penalty_gradient**2))
    )
    penalty_gradient_max = float(
        np.max(np.abs(penalty_gradient))
    )

    summary = base_metadata(maxsteps=maxsteps)
    summary.update(
        {
            "optimized_xyz": str(optimized_xyz),
            "reference_checkpoint": str(reference_path),
            "orbitals_molden": str(orbitals_path),
            "final_pair_npz": str(final_pair_path),
            "penalty_history_json": str(history_path),
            "pair_history_json": str(pair_history_path),
            "final_roots": list(final_pair.roots),
            "final_energies_hartree": energies.tolist(),
            "final_gap_hartree": gap_hartree,
            "final_gap_ev": gap_ev,
            "final_penalty_objective_hartree": float(final_objective),
            "final_penalty_gradient_rms_hartree_per_bohr": (
                penalty_gradient_rms
            ),
            "final_penalty_gradient_max_hartree_per_bohr": (
                penalty_gradient_max
            ),
            "final_penalty_info": penalty_info,
            "final_active_orbitals_1based": (
                np.asarray(
                    final_pair.diagnostics["selected_act"],
                    dtype=int,
                )
                + 1
            ).tolist(),
            "electronic_structure_converged": bool(
                pair_scanner.converged
            ),
            "acceptance": {
                "gap_target_ev": GAP_TARGET_EV,
                "gap_target_met": abs(gap_ev) <= GAP_TARGET_EV,
                "gradient_rms_target_hartree_per_bohr": (
                    GEOM_PARAMS["convergence_grms"]
                ),
                "gradient_rms_target_met": (
                    penalty_gradient_rms
                    <= GEOM_PARAMS["convergence_grms"]
                ),
                "gradient_max_target_hartree_per_bohr": (
                    GEOM_PARAMS["convergence_gmax"]
                ),
                "gradient_max_target_met": (
                    penalty_gradient_max
                    <= GEOM_PARAMS["convergence_gmax"]
                ),
            },
        }
    )
    write_json(output_dir / f"{RUN_NAME}_summary.json", summary)

    print(f"Final S0/S1 gap: {gap_ev:+.8f} eV")
    print(
        "Final penalty gradient: "
        f"RMS={penalty_gradient_rms:.6e}, "
        f"max={penalty_gradient_max:.6e} Eh/Bohr"
    )
    return reference


def print_frequencies(freqs_cm: np.ndarray) -> None:
    print("Penalty-objective frequencies / cm^-1:")
    for index, frequency in enumerate(freqs_cm):
        if frequency < 0.0:
            print(f"mode {index:3d}: i{abs(frequency):.2f}")
        else:
            print(f"mode {index:3d}:  {frequency:.2f}")


def run_frequency_check(
    output_dir: Path,
    hessian_step: float,
    verbose: int,
    reference: dict[str, Any] | None = None,
) -> None:
    reference_path = output_dir / f"{RUN_NAME}_reference.npz"
    if reference is None:
        reference = load_reference_checkpoint(
            reference_path,
            verbose=verbose,
        )

    reference_mol = reference["mol"].copy()
    reference_mo = np.array(reference["mo_coeff"], copy=True)
    reference_ci = [
        np.array(ci_vector, copy=True)
        for ci_vector in reference["ci"]
    ]
    reference_energies = np.asarray(
        reference["e_states"],
        dtype=float,
    )
    act_list_1based = (
        np.asarray(reference["selected_act_0based"], dtype=int) + 1
    ).tolist()

    def scanner_factory():
        pair_scanner = build_pair_scanner(
            act_list_1based=act_list_1based,
            verbose=False,
        )
        pair_scanner.set_reference(
            mol=reference_mol,
            mo_coeff=reference_mo,
            ci=reference_ci,
            e_states=reference_energies,
        )
        return build_penalty_scanner(
            pair_scanner,
            verbose=False,
        )

    print("\n===== SA2-CASSCF penalty-objective Hessian check =====")
    print(f"Finite-difference step = {hessian_step} Bohr")

    cache_dir = output_dir / f"{RUN_NAME}_hessian_displacements"
    hess4, hess2 = numerical_hessian_from_scanner_factory(
        scanner_factory=scanner_factory,
        mol=reference_mol,
        step=hessian_step,
        cache_dir=cache_dir,
        cache_key=reference_cache_key(reference),
    )

    hessian_path = output_dir / f"{RUN_NAME}_penalty_hessian.npz"
    np.savez(
        hessian_path,
        hess4=hess4,
        hess2=hess2,
        step_bohr=float(hessian_step),
    )

    modes_path = output_dir / f"{RUN_NAME}_penalty_modes.npz"
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
        imag_threshold_cm=IMAG_THRESHOLD_CM,
        amplitude=IMAG_AMPLITUDE,
        amplitude_unit=IMAG_AMPLITUDE_UNIT,
        save_path=modes_path,
    )
    print_frequencies(freqs_cm)

    plus_path = None
    minus_path = None
    if mol_plus is None:
        print(
            "No imaginary penalty-objective frequency above the "
            "configured threshold was found."
        )
    else:
        plus_path = output_dir / f"{RUN_NAME}_imag_plus.xyz"
        minus_path = output_dir / f"{RUN_NAME}_imag_minus.xyz"
        write_xyz(mol_plus, plus_path)
        write_xyz(mol_minus, minus_path)
        print(
            "Imaginary penalty-objective frequency found: "
            f"i{abs(float(imag_freq_cm)):.2f} cm^-1"
        )
        print(f"Saved + displacement: {plus_path}")
        print(f"Saved - displacement: {minus_path}")

    summary_path = output_dir / f"{RUN_NAME}_summary.json"
    summary = load_json(summary_path)
    if not summary:
        summary = base_metadata(maxsteps=MAX_STEPS)
    summary["frequency_check"] = {
        "completed": True,
        "objective": "SA2-CASSCF S0/S1 penalty function",
        "hessian_step_bohr": float(hessian_step),
        "imag_threshold_cm": IMAG_THRESHOLD_CM,
        "imag_amplitude": IMAG_AMPLITUDE,
        "imag_amplitude_unit": IMAG_AMPLITUDE_UNIT,
        "has_imaginary_frequency": mol_plus is not None,
        "imag_frequency_cm": (
            None if imag_freq_cm is None else float(imag_freq_cm)
        ),
        "hessian_npz": str(hessian_path),
        "modes_npz": str(modes_path),
        "displacement_cache_dir": str(cache_dir),
        "imag_plus_xyz": None if plus_path is None else str(plus_path),
        "imag_minus_xyz": None if minus_path is None else str(minus_path),
        "interpretation": (
            "These are curvature diagnostics of the penalty objective, "
            "not ordinary single-surface vibrational frequencies."
        ),
    }
    write_json(summary_path, summary)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_mol = build_molecule(verbose=args.verbose)
    write_xyz(input_mol, output_dir / f"{RUN_NAME}_input.xyz")
    write_json(
        output_dir / f"{RUN_NAME}_input.json",
        base_metadata(maxsteps=args.maxsteps),
    )

    reference = None
    if args.stage in ("all", "optimize"):
        reference = run_optimization(
            output_dir=output_dir,
            maxsteps=args.maxsteps,
            verbose=args.verbose,
        )

    if args.stage in ("all", "frequency"):
        run_frequency_check(
            output_dir=output_dir,
            hessian_step=args.hessian_step,
            verbose=args.verbose,
            reference=reference,
        )


if __name__ == "__main__":
    main()
