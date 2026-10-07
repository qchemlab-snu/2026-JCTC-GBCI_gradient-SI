from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase import io
from pyscf import gto, mcscf, scf
from pyscf.data import nist
from pyscf.mcscf import addons


SCRIPT_DIR = Path(__file__).resolve().parent
PP_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.neb_calculator import build_pyscf_mol_from_ase  # noqa: E402


MOLECULE_NAME = "pp"
METHOD = "SA-CASSCF"
STATE_LABELS = ("S0", "S1")

BASIS = "ccpvdz"
CHARGE = 0
SPIN = 0
UNIT = "Angstrom"
VERBOSE = 4

NROOTS = 2
TARGET_ROOT = 1
STATE_WEIGHTS = [0.5, 0.5]
NCAS = 2
NELECAS = (1, 1)
ACTIVE_ORBITALS_1BASED = [38, 39]

SCF_CONV_TOL = 1e-6
SCF_MAX_CYCLE = 200
SCF_DAMP = 0.4
CASSCF_CONV_TOL = 1e-6
CASSCF_MAX_CYCLE = 200

INPUT_DIR = PP_DIR / "casci" / "pp_casci_S1_neb_outputs"
OUTPUT_DIR = SCRIPT_DIR / "pp_casscf_S1_neb_image_outputs"
INPUT_PATTERN = "pp_casci_S1_neb_*.xyz"
OUTPUT_PREFIX = "pp_casscf_S1_neb_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run equal-weight S0/S1 state-averaged CASSCF single-point "
            "calculations on PP CASCI S1 NEB xyz images."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help="Directory containing CASCI NEB image xyz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CASSCF energy summaries.",
    )
    parser.add_argument(
        "--pattern",
        default=INPUT_PATTERN,
        help="Glob pattern for input xyz images inside --input-dir.",
    )
    parser.add_argument(
        "--output-prefix",
        default=OUTPUT_PREFIX,
        help="Filename prefix for the npz/csv/json outputs inside --output-dir.",
    )
    parser.add_argument("--basis", default=BASIS, help="PySCF basis set name.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=VERBOSE,
        help="PySCF verbosity level.",
    )
    return parser.parse_args()


def find_image_paths(input_dir: Path, pattern: str) -> list[Path]:
    image_paths = sorted(input_dir.glob(pattern))
    if not image_paths:
        raise FileNotFoundError(f"No xyz images found in {input_dir} matching {pattern!r}")
    return image_paths


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def source_casci_energy_ev(atoms: Any) -> float | None:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        energy = atoms.info.get("energy")
        return None if energy is None else float(energy)


def geometry_records(atoms: Any) -> list[dict[str, Any]]:
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    return [
        {
            "element": symbol,
            "xyz": positions[idx].tolist(),
        }
        for idx, symbol in enumerate(symbols)
    ]


def build_reference(mol: gto.Mole) -> scf.hf.RHF:
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    mf.max_cycle = SCF_MAX_CYCLE
    mf.damp = SCF_DAMP
    return mf


def build_casscf(mf: scf.hf.RHF) -> mcscf.mc1step.CASSCF:
    mc = mcscf.CASSCF(mf, NCAS, NELECAS)
    mc.conv_tol = CASSCF_CONV_TOL
    mc.max_cycle = CASSCF_MAX_CYCLE
    mc.fcisolver.nroots = NROOTS
    mc.fix_spin_(ss=0)
    mc.state_average_(STATE_WEIGHTS)
    return mc


def root_energies(mc: mcscf.mc1step.CASSCF) -> np.ndarray:
    e_states = getattr(mc, "e_states", None)
    if e_states is None:
        raise RuntimeError("SA-CASSCF did not expose root energies in mc.e_states.")
    energies = np.atleast_1d(np.asarray(e_states, dtype=float))
    if len(energies) < NROOTS:
        raise RuntimeError(
            f"SA-CASSCF returned {len(energies)} root energies; expected {NROOTS}."
        )
    return energies


def ci_vectors(mc: mcscf.mc1step.CASSCF) -> list[Any]:
    if isinstance(mc.ci, (list, tuple)):
        return list(mc.ci)
    return [mc.ci]


def spin_square(
    mc: mcscf.mc1step.CASSCF,
    ci: Any,
) -> tuple[float | None, float | None]:
    try:
        s2, multiplicity = mc.fcisolver.spin_square(ci, mc.ncas, mc.nelecas)
    except Exception:
        return None, None
    return float(s2), float(multiplicity)


def run_image_calculation(
    image_path: Path,
    image_index: int,
    basis: str,
    verbose: int,
) -> dict[str, Any]:
    atoms = io.read(str(image_path))
    mol = build_pyscf_mol_from_ase(
        atoms,
        basis=basis,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=verbose,
    )

    print("\n" + "=" * 80)
    print(f"SA-CASSCF S1 image {image_index:03d}: {image_path}")
    print("=" * 80)

    mf = build_reference(mol)
    rhf_energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge for {image_path}.")

    mc = build_casscf(mf)
    mo_sorted = addons.sort_mo(
        mc,
        mf.mo_coeff,
        ACTIVE_ORBITALS_1BASED,
        base=1,
    )
    mc.kernel(mo_sorted)

    casscf_converged = bool(np.all(np.asarray(getattr(mc, "converged", True))))
    if not casscf_converged:
        print(f"WARNING: SA-CASSCF did not converge for {image_path}.")

    energies_hartree = root_energies(mc)
    energies_ev = energies_hartree * nist.HARTREE2EV
    ci_list = ci_vectors(mc)

    states = []
    for root, energy_hartree in enumerate(energies_hartree[:NROOTS]):
        ci = ci_list[root] if root < len(ci_list) else None
        s2, multiplicity = (None, None) if ci is None else spin_square(mc, ci)
        states.append(
            {
                "root": root,
                "label": STATE_LABELS[root],
                "energy_hartree": float(energy_hartree),
                "energy_ev": float(energy_hartree * nist.HARTREE2EV),
                "excitation_from_s0_ev": float(
                    (energy_hartree - energies_hartree[0]) * nist.HARTREE2EV
                ),
                "spin_square": s2,
                "multiplicity": multiplicity,
            }
        )

    s0_energy_hartree = float(energies_hartree[0])
    s1_energy_hartree = float(energies_hartree[TARGET_ROOT])
    s0_energy_ev = float(energies_ev[0])
    s1_energy_ev = float(energies_ev[TARGET_ROOT])
    s1_minus_s0_ev = float((s1_energy_hartree - s0_energy_hartree) * nist.HARTREE2EV)
    sa_energy_hartree = float(mc.e_tot)
    source_energy_ev = source_casci_energy_ev(atoms)

    print(f"S0 energy          : {s0_energy_ev:.8f} eV")
    print(f"S1 energy          : {s1_energy_ev:.8f} eV")
    print(f"S1 - S0            : {s1_minus_s0_ev:.8f} eV")
    if source_energy_ev is not None:
        print(f"Source CASCI energy: {source_energy_ev:.8f} eV")
    print(f"RHF converged      : {mf.converged}")
    print(f"SA-CASSCF converged: {casscf_converged}")

    return {
        "image_index": image_index,
        "input_xyz": relative_to_repo(image_path),
        "source_casci_energy_ev": source_energy_ev,
        "rhf_energy_hartree": rhf_energy,
        "sa_casscf_energy_hartree": sa_energy_hartree,
        "sa_casscf_energy_ev": float(sa_energy_hartree * nist.HARTREE2EV),
        "s0_energy_hartree": s0_energy_hartree,
        "s0_energy_ev": s0_energy_ev,
        "s1_energy_hartree": s1_energy_hartree,
        "s1_energy_ev": s1_energy_ev,
        "s1_relative_to_image0_ev": None,
        "s1_minus_s0_ev": s1_minus_s0_ev,
        "rhf_converged": bool(mf.converged),
        "casscf_converged": casscf_converged,
        "state_weights": STATE_WEIGHTS,
        "states": states,
        "geometry": geometry_records(atoms),
    }


def write_summary_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "image_index",
        "input_xyz",
        "source_casci_energy_ev",
        "rhf_energy_hartree",
        "sa_casscf_energy_hartree",
        "sa_casscf_energy_ev",
        "s0_energy_hartree",
        "s0_energy_ev",
        "s1_energy_hartree",
        "s1_energy_ev",
        "s1_relative_to_image0_ev",
        "s1_minus_s0_ev",
        "rhf_converged",
        "casscf_converged",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


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


def save_results(
    records: list[dict[str, Any]],
    output_dir: Path,
    input_dir: Path,
    pattern: str,
    basis: str,
    output_prefix: str = OUTPUT_PREFIX,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    s1_energies_ev = np.asarray([record["s1_energy_ev"] for record in records], dtype=float)
    s1_relative_ev = s1_energies_ev - s1_energies_ev[0]
    for record, relative_energy in zip(records, s1_relative_ev):
        record["s1_relative_to_image0_ev"] = float(relative_energy)

    np.savez(
        output_dir / f"{output_prefix}_results.npz",
        image_indices=np.asarray([record["image_index"] for record in records], dtype=int),
        s0_energies_hartree=np.asarray(
            [record["s0_energy_hartree"] for record in records], dtype=float
        ),
        s0_energies_ev=np.asarray([record["s0_energy_ev"] for record in records], dtype=float),
        s1_energies_hartree=np.asarray(
            [record["s1_energy_hartree"] for record in records], dtype=float
        ),
        s1_energies_ev=s1_energies_ev,
        s1_relative_to_image0_ev=s1_relative_ev,
        s1_minus_s0_ev=np.asarray(
            [record["s1_minus_s0_ev"] for record in records], dtype=float
        ),
        source_casci_energies_ev=np.asarray(
            [
                np.nan
                if record["source_casci_energy_ev"] is None
                else record["source_casci_energy_ev"]
                for record in records
            ],
            dtype=float,
        ),
    )

    write_summary_csv(records, output_dir / f"{output_prefix}_summary.csv")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "molecule": MOLECULE_NAME,
        "method": METHOD,
        "state": "S1",
        "basis": basis,
        "charge": CHARGE,
        "spin": SPIN,
        "unit": UNIT,
        "hartree_to_ev": float(nist.HARTREE2EV),
        "input_dir": relative_to_repo(input_dir),
        "input_pattern": pattern,
        "output_dir": relative_to_repo(output_dir),
        "active_space": {
            "ncas": NCAS,
            "nelecas": list(NELECAS),
            "active_orbitals_1based": ACTIVE_ORBITALS_1BASED,
        },
        "state_average": {
            "nroots": NROOTS,
            "state_labels": list(STATE_LABELS),
            "weights": STATE_WEIGHTS,
            "target_root": TARGET_ROOT,
            "target_state": STATE_LABELS[TARGET_ROOT],
        },
        "settings": {
            "scf_conv_tol": SCF_CONV_TOL,
            "scf_max_cycle": SCF_MAX_CYCLE,
            "scf_damp": SCF_DAMP,
            "casscf_conv_tol": CASSCF_CONV_TOL,
            "casscf_max_cycle": CASSCF_MAX_CYCLE,
            "fix_spin_ss": 0,
        },
        "n_images": len(records),
        "output_files": {
            "results_npz": f"{output_prefix}_results.npz",
            "summary_csv": f"{output_prefix}_summary.csv",
            "metadata_json": f"{output_prefix}_metadata.json",
        },
        "records": records,
    }
    with (output_dir / f"{output_prefix}_metadata.json").open("w") as fh:
        json.dump(to_serializable(metadata), fh, indent=2)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_paths = find_image_paths(input_dir, args.pattern)

    print("===== PP SA-CASSCF S1 energies on CASCI NEB images =====")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images          : {len(image_paths)}")
    print(f"Basis           : {args.basis}")
    print(f"Active orbitals : {ACTIVE_ORBITALS_1BASED} (1-based)")
    print(f"State average   : S0/S1 = {STATE_WEIGHTS}")

    records = [
        run_image_calculation(
            image_path=image_path,
            image_index=idx,
            basis=args.basis,
            verbose=args.verbose,
        )
        for idx, image_path in enumerate(image_paths)
    ]
    save_results(
        records,
        output_dir,
        input_dir,
        args.pattern,
        args.basis,
        output_prefix=args.output_prefix,
    )

    print("\n===== PP SA-CASSCF S1 NEB-image summary =====")
    print(f"Images calculated : {len(records)}")
    print(f"First S1 energy   : {records[0]['s1_energy_ev']:.8f} eV")
    print(f"Last S1 energy    : {records[-1]['s1_energy_ev']:.8f} eV")
    print(f"Output directory  : {output_dir}")


if __name__ == "__main__":
    main()
