"""RHF/6-31G reference for 3-hydroxyflavone at the B3LYP-optimized geometry.

Runs RHF, writes a chkfile and a Molden file, and screens the frontier MOs for
pi character so the GBCI (4e,4o) active space used by the gradient check in this directory can be
chosen from orbitals that actually carry the conjugated system.
"""

from pathlib import Path
import json
import sys

import numpy as np
from pyscf import gto, scf
from pyscf.tools import molden

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pi_character import molecular_plane_normal, pi_weights, suggest_pi_active_list

# ---- settings -------------------------------------------------------------
GEOM_FILE = Path("../dft/outputs/3hf_dft_S0_opt.xyz")

BASIS = "6-31G"
CHARGE = 0
SPIN = 0
VERBOSE = 4

SCF_CONV_TOL = 1e-10
SCF_MAX_CYCLE = 200

CHK_FILE = "3hf_rhf_631g.chk"
MOLDEN_FILE = "3hf_rhf_631g.molden"
OUTPUT_JSON = "00_rhf_631g_active.json"

# Active-space size suggested for the SA-CASSCF step: frontier pi / pi*.
# 4 occupied + 4 virtual pi orbitals span the chromone and phenyl pi systems,
# which is what the S0 -> S1 charge transfer needs.
N_OCC_PI = 2
N_VIR_PI = 2
PI_THRESHOLD = 0.5

# Window of MOs around the frontier to tabulate (counted from the HOMO).
REPORT_BELOW_HOMO = 8
REPORT_ABOVE_LUMO = 8
# ---------------------------------------------------------------------------


def build_molecule():
    return gto.M(
        atom=str(GEOM_FILE),
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        verbose=VERBOSE,
    )


def run_scf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    mf.max_cycle = SCF_MAX_CYCLE
    mf.chkfile = CHK_FILE
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")
    molden.from_scf(mf, MOLDEN_FILE)
    return mf


def report_frontier(mol, mf):
    normal, rms_dev = molecular_plane_normal(mol)
    print(f"\nHeavy-atom plane fit: rms out-of-plane deviation = {rms_dev:.4f} Angstrom")
    if rms_dev > 0.1:
        print("WARNING: molecule is not planar; the pi/sigma assignment below is unreliable.")

    pi_w, p_w = pi_weights(mol, mf.mo_coeff, normal=normal)
    nocc = int((np.asarray(mf.mo_occ) > 0).sum())
    homo, lumo = nocc, nocc + 1  # 1-based

    print(f"\nnelectron = {mol.nelectron}, nocc = {nocc}, nmo = {mf.mo_coeff.shape[1]}")
    print(f"HOMO = MO {homo}, LUMO = MO {lumo}  (1-based)")
    print("\n  MO   occ     energy/Ha    pi weight    p weight   label")
    lo = max(1, homo - REPORT_BELOW_HOMO)
    hi = min(mf.mo_coeff.shape[1], lumo + REPORT_ABOVE_LUMO)
    rows = []
    for i in range(lo, hi + 1):
        k = i - 1
        kind = "pi " if pi_w[k] >= PI_THRESHOLD else "   "
        tag = "HOMO" if i == homo else ("LUMO" if i == lumo else "")
        print(f"{i:5d} {mf.mo_occ[k]:5.2f} {mf.mo_energy[k]:13.6f} "
              f"{pi_w[k]:11.4f} {p_w[k]:11.4f}   {kind}{tag}")
        rows.append({
            "mo": i,
            "occ": float(mf.mo_occ[k]),
            "energy": float(mf.mo_energy[k]),
            "pi_weight": float(pi_w[k]),
            "p_weight": float(p_w[k]),
            "is_pi": bool(pi_w[k] >= PI_THRESHOLD),
        })

    act_list = suggest_pi_active_list(
        mol, mf.mo_coeff, mf.mo_occ,
        n_occ=N_OCC_PI, n_vir=N_VIR_PI,
        pi_threshold=PI_THRESHOLD, normal=normal,
    )
    ncas = N_OCC_PI + N_VIR_PI
    nelecas = 2 * N_OCC_PI
    print(f"\nSuggested active space: ({nelecas}e, {ncas}o)")
    print(f"act_list (1-based) = {act_list}")

    return {
        "geometry_file": str(GEOM_FILE),
        "basis": BASIS,
        "charge": CHARGE,
        "spin": SPIN,
        "e_rhf": float(mf.e_tot),
        "nocc": nocc,
        "homo": homo,
        "lumo": lumo,
        "plane_rms_dev_angstrom": rms_dev,
        "pi_threshold": PI_THRESHOLD,
        "frontier_mos": rows,
        "suggested_act_list": act_list,
        "suggested_ncas": ncas,
        "suggested_nelecas": nelecas,
    }


def main():
    mol = build_molecule()
    mf = run_scf(mol)
    print(f"\nRHF/{BASIS} energy: {mf.e_tot:.10f} Hartree")

    summary = report_frontier(mol, mf)
    Path(OUTPUT_JSON).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {CHK_FILE}, {MOLDEN_FILE}, {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
