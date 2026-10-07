from pathlib import Path
import json
import sys

import numpy as np
from pyscf import gto
from pyscf.geomopt.addons import as_pyscf_method
from pyscf.geomopt.geometric_solver import optimize


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.active_space_tracking_optimizer import CASCI_Active_Root_Tracking_Optimizer
from utils.hessian import (
    imaginary_mode_to_displaced_mols,
    numerical_hessian_from_gradient_scanner,
)


RUN_NAME = "ch2nh2_casci_S0_opt_5o8e_ccpvtz"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / RUN_NAME
# RHF-optimized S0 minimum (planar iminium, C=N 1.272 A), not the twisted
# S2/S1 MECX guess the cc-pVDZ S0 scripts start from.  At this geometry the
# cc-pVTZ occupied orbitals are well separated (MO5-MO6 = 0.066 Eh against
# 0.026 Eh at the MECX structure), so the tracked active space is stable, and
# the CASCI gradient is already down to 6.1e-3 Eh/Bohr.
START_GEOMETRY_SOURCE = "../../casci/S0_opt_RHF.molden"

ATOMS = """N        0.59504934       0.00000418       0.00002110
C       -0.67688886       0.00000732      -0.00000853
H        1.12534395       0.22026262       0.83097781
H        1.12520035      -0.22028482      -0.83101477
H       -1.21148201       0.24130980       0.91030560
H       -1.21139777      -0.24130309      -0.91038122
"""

BASIS = "cc-pVTZ"
CHARGE = 1
SPIN = 0
UNIT = "Angstrom"
VERBOSE = 4

NCAS = 5
NELECAS = (4, 4)
# (8e,5o), matching the 5o8e MECI runs so the MECI energies can be referenced
# to this minimum.  Verified against a cc-pVTZ RHF at the starting geometry.
ACT_LIST = [5, 6, 7, 8, 9]
ACT_BASE = 1
NROOTS = 3
TARGET_ROOT = 0

CONV_TOL = 1.0e-10
MAX_CYCLE = 200
MAX_STEPS = 500

GEOM_PARAMS = {
    "trust": 0.02,
    "tmax": 0.03,
    "convergence_energy": 1e-6,
    "convergence_grms": 3e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}

HESSIAN_STEP = 1.0e-3
IMAG_THRESHOLD_CM = 30.0
IMAG_AMPLITUDE = 0.05
IMAG_AMPLITUDE_UNIT = "Angstrom"


def write_xyz(mol, path):
    path.write_text(mol.tostring(format="xyz") + "\n")


def write_json(data, path):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_molecule():
    return gto.M(
        atom=ATOMS,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=VERBOSE,
    )


def build_optimizer():
    return CASCI_Active_Root_Tracking_Optimizer(
        ncas=NCAS,
        nelecas=NELECAS,
        act_list=ACT_LIST,
        nroots=NROOTS,
        target_root=TARGET_ROOT,
        conv_tol=CONV_TOL,
        max_cycle=MAX_CYCLE,
        act_base=ACT_BASE,
    )


def base_metadata():
    return {
        "molecule": "CH2NH2+",
        "method": "CASCI",
        "state": "S0",
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
        "target_root": TARGET_ROOT,
        "conv_tol": CONV_TOL,
        "max_cycle": MAX_CYCLE,
        "geom_params": GEOM_PARAMS,
        "max_steps": MAX_STEPS,
        "hessian_step_bohr": HESSIAN_STEP,
        "imag_threshold_cm": IMAG_THRESHOLD_CM,
        "imag_amplitude": IMAG_AMPLITUDE,
        "imag_amplitude_unit": IMAG_AMPLITUDE_UNIT,
    }


def print_frequencies(freqs_cm):
    print("Frequencies / cm^-1:")
    for i, freq in enumerate(freqs_cm):
        if freq < 0:
            print(f"mode {i:3d}: i{abs(freq):.2f}")
        else:
            print(f"mode {i:3d}:  {freq:.2f}")


def save_hessian_check(opt_obj, mol_opt, output_dir):
    print("\n===== CASCI S0 numerical Hessian check =====")
    print(f"Finite-difference step = {HESSIAN_STEP} Bohr")

    hess4, hess2 = numerical_hessian_from_gradient_scanner(
        opt_obj,
        mol_opt,
        step=HESSIAN_STEP,
    )

    np.savez(
        output_dir / f"{RUN_NAME}_hessian.npz",
        hess4=hess4,
        hess2=hess2,
    )

    freqs_cm, modes_cart, imag_freq_cm, imag_mode_cart, mol_plus, mol_minus = (
        imaginary_mode_to_displaced_mols(
            mol=mol_opt,
            hess2=hess2,
            imag_threshold_cm=IMAG_THRESHOLD_CM,
            amplitude=IMAG_AMPLITUDE,
            amplitude_unit=IMAG_AMPLITUDE_UNIT,
        )
    )

    np.savez(
        output_dir / f"{RUN_NAME}_modes.npz",
        freqs_cm=freqs_cm,
        modes_cart=modes_cart,
        imag_freq_cm=imag_freq_cm,
        imag_mode_cart=imag_mode_cart,
    )

    print_frequencies(freqs_cm)

    if mol_plus is None:
        print("No imaginary frequency found above threshold.")
    else:
        write_xyz(mol_plus, output_dir / f"{RUN_NAME}_imag_plus.xyz")
        write_xyz(mol_minus, output_dir / f"{RUN_NAME}_imag_minus.xyz")
        print(f"Imaginary frequency found: i{abs(float(imag_freq_cm)):.2f} cm^-1")
        print("Displaced geometries were written for follow-up checks.")

    return {
        "frequencies_cm": np.asarray(freqs_cm, dtype=float).tolist(),
        "imaginary_frequency_cm": None
        if imag_freq_cm is None
        else float(imag_freq_cm),
        "has_imaginary_frequency": mol_plus is not None,
    }


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mol = build_molecule()
    write_xyz(mol, OUTPUT_DIR / f"{RUN_NAME}_input.xyz")
    write_json(base_metadata(), OUTPUT_DIR / f"{RUN_NAME}_input.json")

    opt_obj = build_optimizer()
    fake_opt = as_pyscf_method(mol, opt_obj)

    mol_opt = optimize(
        fake_opt,
        maxsteps=MAX_STEPS,
        **GEOM_PARAMS,
    )

    print("\n===== Final CASCI S0 optimized geometry =====")
    print(mol_opt.tostring(format="xyz"))
    write_xyz(mol_opt, OUTPUT_DIR / f"{RUN_NAME}_optimized.xyz")

    final_energy, final_gradient = opt_obj(mol_opt)
    final_act_0based = np.asarray(opt_obj.act_list, dtype=int)
    final_grad = np.asarray(final_gradient, dtype=float)

    np.savez(
        OUTPUT_DIR / f"{RUN_NAME}_final_gradient.npz",
        energy=np.asarray(final_energy, dtype=float),
        gradient=final_grad,
        final_act_0based=final_act_0based,
        final_act_1based=final_act_0based + 1,
    )

    hessian_summary = save_hessian_check(opt_obj, mol_opt, OUTPUT_DIR)

    summary = base_metadata()
    summary.update(
        {
            "final_energy_Eh": float(final_energy),
            "final_gradient_rms_Eh_per_Bohr": float(np.sqrt(np.mean(final_grad**2))),
            "final_gradient_max_Eh_per_Bohr": float(np.max(np.abs(final_grad))),
            "final_act_list_1based": (final_act_0based + 1).tolist(),
            "hessian_check": hessian_summary,
        }
    )
    write_json(summary, OUTPUT_DIR / f"{RUN_NAME}_summary.json")

    print("\nFiles written under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    run()
