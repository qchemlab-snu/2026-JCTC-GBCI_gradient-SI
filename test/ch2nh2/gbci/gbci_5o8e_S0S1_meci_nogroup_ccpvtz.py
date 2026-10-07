from pathlib import Path
import json
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.geomopt.addons import as_pyscf_method
from pyscf.geomopt.geometric_solver import optimize
from pyscf.tools import molden


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hessian import (
    imaginary_mode_to_displaced_mols,
    numerical_hessian_from_gradient_scanner,
)
from utils.meci_optimizer import (
    GBCIActiveRootTrackedPairScanner,
    HARTREE2EV,
    PenaltyFunctionMECIScanner,
)


RUN_NAME = "ch2nh2_gbci_S0S1_meci_5o8e_nogroup_ccpvtz"
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
ACT_LIST = [5, 6, 7, 8, 9]
ACT_BASE = 1
NROOTS = 3
TARGET_ROOTS = (0, 1)
# No grouping: every occupation pattern gets its own FASSCF-optimized bath
# (the original SFNOCI construction).  For (8e,5o) that is 15 baths per
# geometry against 3 for groupocc, so each step runs several times longer.
# Everything else is kept identical to
# ch2nh2_gbci_S0S1_meci_5o8e_groupocc_ccpvtz so the two are comparable.
GROUPA = None

SIGMA = 0.5
ALPHA = 1.0e-4
MAX_STEPS = 1000

GEOM_PARAMS = {
    "trust": 0.01,
    "tmax": 0.03,
    "convergence_energy": 1e-6,
    "convergence_grms": 3e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}

RUN_PENALTY_HESSIAN = True
HESSIAN_STEP = 1.0e-3
IMAG_THRESHOLD_CM = 30.0
IMAG_AMPLITUDE = 0.05
IMAG_AMPLITUDE_UNIT = "Angstrom"


def write_xyz(mol, path):
    path.write_text(mol.tostring(format="xyz") + "\n")


def write_json(data, path):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def optional_float(value):
    if value is None:
        return None
    return float(value)


def print_frequencies(freqs_cm):
    print("Penalty-objective frequencies / cm^-1:")
    for i, freq in enumerate(freqs_cm):
        if freq < 0:
            print(f"mode {i:3d}: i{abs(freq):.2f}")
        else:
            print(f"mode {i:3d}:  {freq:.2f}")


def build_molecule():
    return gto.M(
        atom=ATOMS,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=VERBOSE,
    )


def build_pair_scanner(
    act_list=ACT_LIST,
    act_base=ACT_BASE,
    root_tracking=True,
    active_tracking=True,
):
    return GBCIActiveRootTrackedPairScanner(
        ncas=NCAS,
        nelecas=NELECAS,
        act_list=act_list,
        nroots=NROOTS,
        target_roots=TARGET_ROOTS,
        groupA=GROUPA,
        conv_tol=1e-10,
        max_cycle=200,
        act_base=act_base,
        root_tracking=root_tracking,
        active_tracking=active_tracking,
    )


def build_penalty_scanner(pair_scanner):
    return PenaltyFunctionMECIScanner(
        pair_scanner,
        sigma=SIGMA,
        alpha=ALPHA,
        print_level=4,
    )


def base_metadata():
    return {
        "molecule": "CH2NH2+",
        "method": "GBCI penalty MECI",
        "states": "S0/S1",
        "start_geometry_source": START_GEOMETRY_SOURCE,
        "basis": BASIS,
        "charge": CHARGE,
        "spin": SPIN,
        "unit": UNIT,
        "ncas": NCAS,
        "nelecas": list(NELECAS),
        "act_list_1based_initial": ACT_LIST,
        "act_base": ACT_BASE,
        "nroots": NROOTS,
        "target_roots": list(TARGET_ROOTS),
        "groupA": GROUPA,
        "sigma": SIGMA,
        "alpha": ALPHA,
        "geom_params": GEOM_PARAMS,
        "max_steps": MAX_STEPS,
        "run_penalty_hessian": RUN_PENALTY_HESSIAN,
        "hessian_step_bohr": HESSIAN_STEP,
        "imag_threshold_cm": IMAG_THRESHOLD_CM,
        "imag_amplitude": IMAG_AMPLITUDE,
        "imag_amplitude_unit": IMAG_AMPLITUDE_UNIT,
    }


def save_final_pair_result(pair_result, output_dir):
    energies = np.asarray(pair_result.energies, dtype=float)
    gradients = np.asarray(pair_result.gradients, dtype=float)
    selected_act_0based = np.asarray(
        pair_result.diagnostics["selected_act"],
        dtype=int,
    )

    np.savez(
        output_dir / f"{RUN_NAME}_final_pair.npz",
        energies=energies,
        gradients=gradients,
        roots=np.asarray(pair_result.roots, dtype=int),
        selected_act_0based=selected_act_0based,
        selected_act_1based=selected_act_0based + 1,
    )

    summary = base_metadata()
    summary.update(
        {
            "final_roots": list(pair_result.roots),
            "final_energies_Eh": energies.tolist(),
            "final_gap_Eh": float(energies[1] - energies[0]),
            "final_gap_eV": float((energies[1] - energies[0]) * HARTREE2EV),
            "final_act_list_1based": (selected_act_0based + 1).tolist(),
        }
    )
    write_json(summary, output_dir / f"{RUN_NAME}_summary.json")
    return summary


def save_penalty_history(penalty_scanner, output_dir):
    history = []
    for item in penalty_scanner.history:
        row = dict(item)
        if "roots" in row:
            row["roots"] = list(row["roots"])
        history.append(row)
    write_json(history, output_dir / f"{RUN_NAME}_penalty_history.json")


def save_molden_files(pair_result, mol_opt, output_dir):
    rhf_molden = output_dir / f"{RUN_NAME}_final_RHF.molden"
    gbci_molden = output_dir / f"{RUN_NAME}_final_GBCI_orbitals.molden"

    molden.from_scf(pair_result.mf, str(rhf_molden))
    molden.from_mo(
        mol_opt,
        str(gbci_molden),
        pair_result.mc.mo_coeff,
        occ=getattr(pair_result.mc, "mo_occ", None),
    )

    print(f"Saved final RHF Molden file: {rhf_molden}")
    print(f"Saved final GBCI orbital Molden file: {gbci_molden}")

    return {
        "final_rhf_molden": str(rhf_molden),
        "final_gbci_molden": str(gbci_molden),
    }


def run_penalty_hessian_check(mol_opt, output_dir, act_list_1based):
    print("\n===== Penalty-objective numerical Hessian check =====")
    print(f"Finite-difference step = {HESSIAN_STEP} Bohr")
    print(f"Fixed Hessian active list, 1-based = {act_list_1based}")

    hessian_pair_scanner = build_pair_scanner(
        act_list=act_list_1based,
        act_base=1,
        root_tracking=False,
        active_tracking=False,
    )
    hessian_penalty_scanner = build_penalty_scanner(hessian_pair_scanner)

    hess4, hess2 = numerical_hessian_from_gradient_scanner(
        hessian_penalty_scanner,
        mol_opt,
        step=HESSIAN_STEP,
    )

    np.savez(
        output_dir / f"{RUN_NAME}_penalty_hessian.npz",
        hess4=hess4,
        hess2=hess2,
    )

    freqs_cm, _, imag_freq_cm, _, mol_plus, mol_minus = (
        imaginary_mode_to_displaced_mols(
            mol=mol_opt,
            hess2=hess2,
            imag_threshold_cm=IMAG_THRESHOLD_CM,
            amplitude=IMAG_AMPLITUDE,
            amplitude_unit=IMAG_AMPLITUDE_UNIT,
            save_path=output_dir / f"{RUN_NAME}_penalty_modes.npz",
        )
    )

    print_frequencies(freqs_cm)

    if mol_plus is None:
        print("No imaginary frequency above threshold was found.")
        return {"has_imaginary_frequency": False, "imag_frequency_cm": None}

    plus_xyz = output_dir / f"{RUN_NAME}_imag_plus.xyz"
    minus_xyz = output_dir / f"{RUN_NAME}_imag_minus.xyz"
    write_xyz(mol_plus, plus_xyz)
    write_xyz(mol_minus, minus_xyz)

    print(f"Imaginary frequency found: i{abs(imag_freq_cm):.2f} cm^-1")
    print(f"Saved mol_plus geometry: {plus_xyz}")
    print(f"Saved mol_minus geometry: {minus_xyz}")
    print("\n===== mol_plus =====")
    print(mol_plus.tostring(format="xyz"))
    print("\n===== mol_minus =====")
    print(mol_minus.tostring(format="xyz"))

    return {
        "has_imaginary_frequency": True,
        "imag_frequency_cm": optional_float(imag_freq_cm),
        "imag_plus_xyz": str(plus_xyz),
        "imag_minus_xyz": str(minus_xyz),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(base_metadata(), OUTPUT_DIR / f"{RUN_NAME}_input.json")

    mol = build_molecule()
    mf = scf.RHF(mol)
    pair_scanner = build_pair_scanner()
    penalty_scanner = build_penalty_scanner(pair_scanner)
    fake_opt = as_pyscf_method(mol, penalty_scanner)

    print("===== CH2NH2+ GBCI S0/S1 conical-intersection search =====")
    print(f"start_geometry_source = {START_GEOMETRY_SOURCE}")
    print(f"basis = {BASIS}")
    print(f"active space, 1-based = {ACT_LIST}")
    print(f"ncas = {NCAS}")
    print(f"nelecas = {NELECAS}")
    print(f"nroots = {NROOTS}")
    print(f"target_roots = {TARGET_ROOTS}")
    print(f"groupA = {GROUPA!r}")
    print(f"sigma = {SIGMA}")
    print(f"alpha = {ALPHA}")
    print(f"reference = {mf.__class__.__name__}")
    print(f"output_dir = {OUTPUT_DIR}")

    mol_opt = optimize(
        fake_opt,
        maxsteps=MAX_STEPS,
        **GEOM_PARAMS,
    )

    opt_xyz = OUTPUT_DIR / f"{RUN_NAME}_optimized.xyz"
    write_xyz(mol_opt, opt_xyz)

    print("\n===== Final S0/S1 MECI geometry =====")
    print(mol_opt.tostring(format="xyz"))
    print(f"Saved optimized geometry: {opt_xyz}")

    print("\n===== Final S0/S1 GBCI pair check =====")
    final_pair_result = pair_scanner(mol_opt)
    final_summary = save_final_pair_result(final_pair_result, OUTPUT_DIR)
    save_penalty_history(penalty_scanner, OUTPUT_DIR)
    molden_summary = save_molden_files(final_pair_result, mol_opt, OUTPUT_DIR)

    print(f"Final roots: {final_summary['final_roots']}")
    print(f"Final energies / Eh: {final_summary['final_energies_Eh']}")
    print(f"Final S1-S0 gap / Eh: {final_summary['final_gap_Eh']:.12e}")
    print(f"Final S1-S0 gap / eV: {final_summary['final_gap_eV']:.8f}")
    print(f"Final active list, 1-based: {final_summary['final_act_list_1based']}")

    hessian_summary = {}
    if RUN_PENALTY_HESSIAN:
        hessian_summary = run_penalty_hessian_check(
            mol_opt,
            OUTPUT_DIR,
            final_summary["final_act_list_1based"],
        )

    final_summary.update(
        {
            "optimized_xyz": str(opt_xyz),
            **molden_summary,
            **hessian_summary,
        }
    )
    write_json(final_summary, OUTPUT_DIR / f"{RUN_NAME}_summary.json")


if __name__ == "__main__":
    main()
