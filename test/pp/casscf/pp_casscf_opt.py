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
from pyscf.geomopt.addons import as_pyscf_method
from pyscf.geomopt.geometric_solver import optimize
from pyscf.tools import molden


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.active_space_tracking_optimizer import (  # noqa: E402
    CASSCF_Active_Root_Tracking_Optimizer,
)
from utils.hessian import (  # noqa: E402
    imaginary_mode_to_displaced_mols,
    numerical_hessian_from_scanner_factory,
)


MOLECULE_NAME = "pp"
STATE_LABELS = ("S0", "S1")

BASIS = "ccpvdz"
CHARGE = 0
SPIN = 0
UNIT = "Angstrom"
VERBOSE = 4

NCAS = 2
NELECAS = (1, 1)
ACTIVE_ORBITALS_1BASED = [38, 39]
# Per-state CASSCF settings.
#
# S0 is optimized with a plain ground-state CASSCF. The equal-weight S0/S1
# average left the S0 optimization unable to converge: the gradient norm grew
# monotonically while the energy stalled, and SA-CASSCF finally failed outright
# (see pp_casscf_S0_all.out). Optimizing root 0 on its own removes the excited
# root from the orbital optimization entirely.
#
# S1 keeps the equal-weight average, which is what keeps the excited root well
# defined; that path is unchanged.
STATE_SETTINGS = {
    "S0": {
        "method": "CASSCF",
        "nroots": 1,
        "state_average": False,
        "state_weights": None,
    },
    "S1": {
        "method": "SA-CASSCF",
        "nroots": 2,
        "state_average": True,
        "state_weights": [0.5, 0.5],
    },
}

# State-averaged defaults, kept under their original names for the S1
# saddle-check scripts that import this module.
METHOD = STATE_SETTINGS["S1"]["method"]
NROOTS = STATE_SETTINGS["S1"]["nroots"]
STATE_WEIGHTS = STATE_SETTINGS["S1"]["state_weights"]

# Loosened from 1e-10: the tight tolerance left SA-CASSCF unconverged at some
# geometries (see pp_casscf_S0_all.out). This value is passed to both the RHF
# reference and the CASSCF solver by the tracking optimizer.
SCF_CASSCF_CONV_TOL = 1e-6
SCF_CASSCF_MAX_CYCLE = 200

GEOM_PARAMS = {
    "trust": 0.02,
    "tmax": 0.03,
    # Kept a decade looser than SCF_CASSCF_CONV_TOL so the geometry convergence
    # test is not decided by wavefunction noise.
    "convergence_energy": 1e-5,
    "convergence_grms": 3e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}

HESSIAN_STEP = 1e-3
IMAG_THRESHOLD_CM = 30.0
IMAG_AMPLITUDE = 0.05
IMAG_AMPLITUDE_UNIT = "Angstrom"

# Final geometry in pp_casci_S0_im1.out.  This structure is a CASCI S0
# minimum with no imaginary frequency.  Both the S0 and S1 CASSCF
# optimizations intentionally start from this same geometry.
INITIAL_GEOMETRY_SOURCE = "test/pp/casci/pp_casci_S0_im1.out"
INITIAL_ATOMS = """C          -3.73143993        0.65188355       -0.21325648
C          -2.51817307        1.24685434       -0.32807972
C          -3.50289652       -0.69490677        0.21335648
C          -2.16127315       -0.85635017        0.32814287
H          -4.68168724        1.11637761       -0.41896743
H          -4.24671897       -1.44686974        0.41907899
N          -1.54245055        0.33052230       -0.00002689
H          -1.59175012       -1.70747153        0.65557748
H          -2.26130062        2.23819524       -0.65554408
C          -0.16244599        0.56467722       -0.00003783
C           0.74312415       -0.46184653       -0.25010711
C           2.09644794       -0.16766994       -0.23075471
H           0.41102441       -1.46229433       -0.48010942
N           2.59760033        1.03308530       -0.00002661
H           2.81355702       -0.95592807       -0.42551776
C           1.72840944        2.00130767        0.23070888
C           0.35380603        1.83246734        0.25004055
H           2.14530813        2.98201506        0.42549135
H          -0.28975431        2.66733240        0.48006407"""


def parse_args(default_state: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize PP S0 with ground-state-only CASSCF or S1 with "
            "equal-weight S0/S1 SA-CASSCF, and "
            "optionally run a restartable numerical frequency check."
        )
    )
    parser.add_argument(
        "--state",
        choices=STATE_LABELS,
        default=default_state,
        required=default_state is None,
        help="Target adiabatic state. S0 and S1 are always state-averaged.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "optimize", "frequency"),
        default="all",
        help="Run optimization, frequency analysis, or both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/pp_casscf_<state>.",
    )
    parser.add_argument(
        "--maxsteps",
        type=int,
        default=500,
        help="Maximum geomeTRIC optimization steps.",
    )
    parser.add_argument(
        "--hessian-step",
        type=float,
        default=HESSIAN_STEP,
        help="Central-difference Hessian displacement in Bohr.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=VERBOSE,
        help="PySCF verbosity.",
    )
    return parser.parse_args()


def build_molecule(atoms: Any = INITIAL_ATOMS, verbose: int = VERBOSE) -> gto.Mole:
    return gto.M(
        atom=atoms,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=verbose,
    )


def state_settings(
    state_label: str | None = None,
    target_root: int | None = None,
) -> dict[str, Any]:
    """Settings for a state, addressed by label or by its adiabatic root index."""
    if state_label is None:
        if target_root is None:
            raise ValueError("Give either state_label or target_root.")
        state_label = STATE_LABELS[int(target_root)]
    return STATE_SETTINGS[state_label]


def build_optimizer(
    target_root: int,
    verbose: bool = True,
    state_label: str | None = None,
) -> CASSCF_Active_Root_Tracking_Optimizer:
    settings = state_settings(state_label=state_label, target_root=target_root)
    return CASSCF_Active_Root_Tracking_Optimizer(
        ncas=NCAS,
        nelecas=NELECAS,
        act_list=ACTIVE_ORBITALS_1BASED,
        nroots=settings["nroots"],
        target_root=target_root,
        conv_tol=SCF_CASSCF_CONV_TOL,
        max_cycle=SCF_CASSCF_MAX_CYCLE,
        act_base=1,
        root_tracking=False,
        state_average=settings["state_average"],
        state_weights=settings["state_weights"],
        project_previous_orbitals=True,
        reuse_ci_guess=True,
        strict_convergence=True,
        verbose=verbose,
    )


def write_xyz(mol: gto.Mole, path: Path) -> None:
    path.write_text(mol.tostring(format="xyz") + "\n")


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(to_serializable(data), handle, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def save_history(
    history: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    save_json(json_path, {"evaluations": history})

    fieldnames = [
        "step",
        "target_root",
        "target_energy_hartree",
        "s0_energy_hartree",
        "s1_energy_hartree",
        "s1_minus_s0_hartree",
        "scf_converged",
        "casscf_converged",
        "selected_active_orbitals_1based",
        "active_min_singular_value",
        "projected_active_min_singular_value",
        "active_selection_gap",
        "natural_occupations",
        "gradient_norm",
        "gradient_rms",
        "gradient_max",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            energies = record["root_energies_hartree"]
            row = {
                "step": record["step"],
                "target_root": record["target_root"],
                "target_energy_hartree": record["target_energy_hartree"],
                "s0_energy_hartree": energies[0],
                # None for a ground-state-only CASSCF, which has a single root.
                "s1_energy_hartree": energies[1] if len(energies) > 1 else None,
                "s1_minus_s0_hartree": record["s1_minus_s0_hartree"],
                "scf_converged": record["scf_converged"],
                "casscf_converged": record["casscf_converged"],
                "selected_active_orbitals_1based": json.dumps(
                    record["selected_active_orbitals_1based"]
                ),
                "active_min_singular_value": record[
                    "active_min_singular_value"
                ],
                "projected_active_min_singular_value": record[
                    "projected_active_min_singular_value"
                ],
                "active_selection_gap": record["active_selection_gap"],
                "natural_occupations": json.dumps(
                    record["natural_occupations"]
                ),
                "gradient_norm": record["gradient_norm"],
                "gradient_rms": record["gradient_rms"],
                "gradient_max": record["gradient_max"],
            }
            writer.writerow(row)


def save_reference_checkpoint(
    path: Path,
    reference: dict[str, Any],
    target_root: int,
) -> None:
    mol = reference["mol"]
    ci_list = reference["ci"]
    # A ground-state-only CASSCF has no state-average weights; store an empty
    # array so the checkpoint round-trips for both setups.
    weights = state_settings(target_root=target_root)["state_weights"] or []
    np.savez(
        path,
        atom_symbols=np.asarray(mol.atom_charges(), dtype=int),
        coords_angstrom=np.asarray(
            mol.atom_coords(unit="Angstrom"),
            dtype=float,
        ),
        mo_coeff=np.asarray(reference["mo_coeff"]),
        e_states=np.asarray(reference["e_states"], dtype=float),
        target_root=int(target_root),
        ncas=NCAS,
        nelecas=np.asarray(NELECAS, dtype=int),
        state_weights=np.asarray(weights, dtype=float),
        **{
            f"ci_root_{index}": np.asarray(ci_vector)
            for index, ci_vector in enumerate(ci_list)
        },
    )


def load_reference_checkpoint(path: Path, verbose: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Reference checkpoint not found: {path}. "
            "Run --stage optimize first."
        )

    with np.load(path, allow_pickle=False) as saved:
        atom_charges = np.asarray(saved["atom_symbols"], dtype=int)
        coords_angstrom = np.asarray(saved["coords_angstrom"], dtype=float)
        mo_coeff = np.array(saved["mo_coeff"], copy=True)
        ci = []
        while f"ci_root_{len(ci)}" in saved:
            ci.append(np.array(saved[f"ci_root_{len(ci)}"], copy=True))
        e_states = np.asarray(saved["e_states"], dtype=float)
        saved_ncas = int(saved["ncas"])
        saved_nelecas = tuple(
            int(value) for value in np.asarray(saved["nelecas"], dtype=int)
        )
        saved_weights = np.asarray(saved["state_weights"], dtype=float)
        saved_target_root = int(saved["target_root"])

    expected_weights = np.asarray(
        state_settings(target_root=saved_target_root)["state_weights"] or [],
        dtype=float,
    )
    if saved_ncas != NCAS or saved_nelecas != NELECAS:
        raise ValueError("Checkpoint active space does not match this script.")
    if saved_weights.shape != expected_weights.shape or not np.allclose(
        saved_weights, expected_weights, atol=1e-12, rtol=0.0
    ):
        raise ValueError("Checkpoint state-average weights do not match this script.")

    atom_spec = [
        (int(charge), tuple(coords))
        for charge, coords in zip(atom_charges, coords_angstrom)
    ]
    mol = build_molecule(atom_spec, verbose=verbose)
    return {
        "mol": mol,
        "mo_coeff": mo_coeff,
        "ci": ci,
        "e_states": e_states,
        "target_root": saved_target_root,
    }


def reference_cache_key(
    reference: dict[str, Any],
    target_root: int,
) -> str:
    settings = state_settings(target_root=target_root)
    digest = hashlib.sha256()
    digest.update(settings["method"].encode())
    digest.update(BASIS.encode())
    digest.update(
        np.asarray(settings["state_weights"] or [], dtype=float).tobytes()
    )
    digest.update(np.asarray([target_root, NCAS, *NELECAS], dtype=int).tobytes())
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


def print_settings(state_label: str, target_root: int, output_dir: Path) -> None:
    settings = state_settings(state_label=state_label)
    print(f"===== PP {settings['method']} {state_label} geometry optimization =====")
    print(f"initial geometry = {INITIAL_GEOMETRY_SOURCE}")
    print(f"basis = {BASIS}")
    print(f"charge = {CHARGE}")
    print(f"spin = {SPIN}")
    print(f"unit = {UNIT}")
    print(f"active orbitals = {ACTIVE_ORBITALS_1BASED} (1-based)")
    print(f"ncas = {NCAS}")
    print(f"nelecas = {NELECAS}")
    print(f"nroots = {settings['nroots']}")
    if settings["state_average"]:
        print(f"state averaging = {settings['state_weights']} over {list(STATE_LABELS)}")
    else:
        print("state averaging = off (ground-state-only CASSCF)")
    print(f"target root = {target_root} ({state_label})")
    print("root tracking = False (fixed adiabatic root)")
    print(f"output directory = {output_dir}")


def optimization_metadata(
    state_label: str,
    target_root: int,
    output_dir: Path,
    reference: dict[str, Any],
    maxsteps: int,
) -> dict[str, Any]:
    energies = np.asarray(reference["e_states"], dtype=float)
    settings = state_settings(state_label=state_label)
    final_energies = {
        label: float(energies[index])
        for index, label in enumerate(STATE_LABELS[: len(energies)])
    }
    final_energies["target"] = float(energies[target_root])
    if len(energies) >= 2:
        final_energies["S1_minus_S0"] = float(energies[1] - energies[0])
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": f"pp_casscf_{state_label}.py",
        "driver": Path(__file__).name,
        "molecule": MOLECULE_NAME,
        "method": settings["method"],
        "state": state_label,
        "target_root": target_root,
        "initial_geometry_source": INITIAL_GEOMETRY_SOURCE,
        "initial_geometry_angstrom": INITIAL_ATOMS,
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
            "enabled": settings["state_average"],
            "states": (
                list(STATE_LABELS[: settings["nroots"]])
                if settings["state_average"]
                else [state_label]
            ),
            "nroots": settings["nroots"],
            "weights": settings["state_weights"],
        },
        "root_tracking": False,
        "project_previous_orbitals": True,
        "reuse_ci_guess": True,
        "strict_convergence": True,
        "conv_tol": SCF_CASSCF_CONV_TOL,
        "max_cycle": SCF_CASSCF_MAX_CYCLE,
        "geometry_optimization": {
            "maxsteps": maxsteps,
            "parameters": GEOM_PARAMS,
        },
        "final_energies_hartree": final_energies,
        "output_dir": str(output_dir),
        "files": {
            "optimized_xyz": f"pp_casscf_{state_label}_optimized.xyz",
            "history_csv": f"pp_casscf_{state_label}_optimization_history.csv",
            "history_json": f"pp_casscf_{state_label}_optimization_history.json",
            "reference_npz": f"pp_casscf_{state_label}_reference.npz",
            "orbitals_molden": f"pp_casscf_{state_label}_orbitals.molden",
            "natural_orbitals_molden": (
                f"pp_casscf_{state_label}_natural_orbitals.molden"
            ),
        },
        "frequency_check": {
            "completed": False,
        },
    }


def run_optimization(
    state_label: str,
    target_root: int,
    output_dir: Path,
    maxsteps: int,
    verbose: int,
) -> dict[str, Any]:
    mol = build_molecule(verbose=verbose)
    optimizer = build_optimizer(
        target_root=target_root,
        verbose=True,
        state_label=state_label,
    )
    fake_method = as_pyscf_method(mol, optimizer)

    print_settings(state_label, target_root, output_dir)
    mol_opt = optimize(
        fake_method,
        maxsteps=maxsteps,
        **GEOM_PARAMS,
    )

    # Evaluate the returned geometry once so the saved wavefunction and
    # metadata correspond exactly to mol_opt.
    optimizer(mol_opt)
    reference = optimizer.get_reference()

    optimized_xyz = output_dir / f"pp_casscf_{state_label}_optimized.xyz"
    write_xyz(mol_opt, optimized_xyz)
    print(f"\n===== Final {state_label} optimized geometry =====")
    print(mol_opt.tostring(format="xyz"))
    print(f"Saved optimized geometry: {optimized_xyz}")

    history_csv = (
        output_dir / f"pp_casscf_{state_label}_optimization_history.csv"
    )
    history_json = (
        output_dir / f"pp_casscf_{state_label}_optimization_history.json"
    )
    save_history(optimizer.history, history_csv, history_json)

    reference_path = output_dir / f"pp_casscf_{state_label}_reference.npz"
    save_reference_checkpoint(reference_path, reference, target_root)

    molden.from_mcscf(
        optimizer.last_mc,
        str(output_dir / f"pp_casscf_{state_label}_orbitals.molden"),
    )
    molden.from_mcscf(
        optimizer.last_mc,
        str(
            output_dir
            / f"pp_casscf_{state_label}_natural_orbitals.molden"
        ),
        cas_natorb=True,
    )

    metadata = optimization_metadata(
        state_label=state_label,
        target_root=target_root,
        output_dir=output_dir,
        reference=reference,
        maxsteps=maxsteps,
    )
    save_json(
        output_dir / f"pp_casscf_{state_label}_metadata.json",
        metadata,
    )
    return reference


def print_frequencies(freqs_cm: np.ndarray) -> None:
    print("Frequencies / cm^-1:")
    for index, frequency in enumerate(freqs_cm):
        if frequency < 0:
            print(f"mode {index:3d}: i{abs(frequency):.2f}")
        else:
            print(f"mode {index:3d}:  {frequency:.2f}")


def run_frequency_check(
    state_label: str,
    target_root: int,
    output_dir: Path,
    hessian_step: float,
    verbose: int,
    reference: dict[str, Any] | None = None,
) -> None:
    reference_path = output_dir / f"pp_casscf_{state_label}_reference.npz"
    if reference is None:
        reference = load_reference_checkpoint(reference_path, verbose=verbose)
        if reference["target_root"] != target_root:
            raise ValueError(
                "Checkpoint target root does not match the requested state."
            )

    reference_mol = reference["mol"].copy()
    reference_mo = np.array(reference["mo_coeff"], copy=True)
    reference_ci = [
        np.array(value, copy=True) for value in reference["ci"]
    ]
    reference_energies = np.asarray(reference["e_states"], dtype=float)

    def scanner_factory():
        scanner = build_optimizer(
            target_root=target_root,
            verbose=False,
            state_label=state_label,
        )
        scanner.set_reference(
            mol=reference_mol,
            mo_coeff=reference_mo,
            ci=reference_ci,
            e_states=reference_energies,
        )
        return scanner

    cache_dir = (
        output_dir / f"pp_casscf_{state_label}_hessian_displacements"
    )
    hess4, hess2 = numerical_hessian_from_scanner_factory(
        scanner_factory=scanner_factory,
        mol=reference_mol,
        step=hessian_step,
        cache_dir=cache_dir,
        cache_key=reference_cache_key(reference, target_root),
    )

    hessian_path = output_dir / f"pp_casscf_{state_label}_hessian.npz"
    np.savez(
        hessian_path,
        hess4=hess4,
        hess2=hess2,
        step_bohr=float(hessian_step),
    )

    modes_path = output_dir / f"pp_casscf_{state_label}_modes.npz"
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
        print("No imaginary frequency above the configured threshold was found.")
    else:
        plus_path = output_dir / f"pp_casscf_{state_label}_imag_plus.xyz"
        minus_path = output_dir / f"pp_casscf_{state_label}_imag_minus.xyz"
        write_xyz(mol_plus, plus_path)
        write_xyz(mol_minus, minus_path)
        print(f"Saved imaginary-mode + displacement: {plus_path}")
        print(f"Saved imaginary-mode - displacement: {minus_path}")

    metadata_path = output_dir / f"pp_casscf_{state_label}_metadata.json"
    metadata = load_json(metadata_path)
    metadata["frequency_check"] = {
        "completed": True,
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
    }
    save_json(metadata_path, metadata)


def main(default_state: str | None = None) -> None:
    args = parse_args(default_state=default_state)
    state_label = args.state
    target_root = STATE_LABELS.index(state_label)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else SCRIPT_DIR / "outputs" / f"pp_casscf_{state_label}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = None
    if args.stage in ("all", "optimize"):
        reference = run_optimization(
            state_label=state_label,
            target_root=target_root,
            output_dir=output_dir,
            maxsteps=args.maxsteps,
            verbose=args.verbose,
        )

    if args.stage in ("all", "frequency"):
        run_frequency_check(
            state_label=state_label,
            target_root=target_root,
            output_dir=output_dir,
            hessian_step=args.hessian_step,
            verbose=args.verbose,
            reference=reference,
        )


if __name__ == "__main__":
    main()
