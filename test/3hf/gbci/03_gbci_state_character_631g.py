"""GBCI(4e,4o) state character analysis for 3-hydroxyflavone in a 6-31G basis.

Answers which GBCI root carries charge transfer, and how much, by building the
AO density of every root and reporting Lowdin fragment charges, the dipole, the
active natural occupations, and the dominant hole -> particle pair of each
transition.  Fragment charges and dipoles come from the *state densities*, so
unlike the occupation-pattern populations in ``02_group_threshold_check_631g.py``
they include the grouped-bath relaxation.

The active space is the frontier pi/pi* window from ``00_rhf_631g_active.py``,
determined directly in this basis, so no cross-basis orbital matching is needed.
The grouped bath uses threshold 0.5, which ``02_group_threshold_check_631g.py``
shows gives 4 groups while still keeping the S0 and HOMO -> LUMO configurations
in separate groups (they merge only at 1.0).  The gradient cost carries a
ngroup**2 factor, so 4 groups instead of the default 7 is roughly 3x cheaper.
"""

import json
from functools import reduce
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.gbci.gbci import GBCI

from utils.bath_grouping import patch_gbci_grouping
from utils.gbci_setup import apply_act_list, check_active_window, is_contiguous_window

GEOMETRY = "../dft/outputs/3hf_dft_S0_opt.xyz"
BASIS = "6-31G"
SCF_CHK = "3hf_rhf_631g_character.chk"
OUTPUT_JSON = "03_gbci_state_character_631g.json"

# Frontier pi/pi* window from 00_rhf_631g_active.py, determined in this basis.
ACT_LIST = [61, 62, 63, 64]

NCAS = 4
NELECAS = (2, 2)
ACT_BASE = 1

NROOTS = 4

# Fragments of 3-hydroxyflavone (0-based atoms, hydrogens with their heavy atom).
# chromone = benzo-4-pyranone incl. the carbonyl and ring oxygens; the 3-OH and
# the 2-phenyl ring are kept separate so a charge shift between them is visible.
ACCEPTOR_ATOMS = [0, 1, 2, 4, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27]
DONOR_ATOMS = [5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23]
HYDROXY_ATOMS = [3, 18]

# Grouped bath keyed on the chromone population, the coordinate a charge-transfer
# state moves along. threshold 0.5 -> 4 groups (see 02_group_threshold_check).
GROUP_THRESHOLD = 0.5
GROUP_A = {"atom": [ACCEPTOR_ATOMS], "threshold": GROUP_THRESHOLD}

SCF_CONV_TOL = 1e-10
FASSCF_CONV_TOL = 1e-10
FCI_CONV_TOL = 1e-12

HARTREE2EV = 27.211386245988


def lowdin_atom_charges(mol, dm_ao):
    """Lowdin charge per atom from an AO-basis density matrix."""
    s = mol.intor_symmetric("int1e_ovlp")
    s12 = scipy.linalg.sqrtm(s).real
    pop = np.diag(reduce(np.dot, (s12, dm_ao, s12))).real

    charges = np.zeros(mol.natm)
    slices = mol.aoslice_by_atom()
    for atom in range(mol.natm):
        p0, p1 = slices[atom][2], slices[atom][3]
        charges[atom] = mol.atom_charge(atom) - pop[p0:p1].sum()
    return charges


def dipole_moment(mol, dm_ao):
    """Total dipole moment in Debye."""
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuclear = np.einsum("a,ax->x", charges, coords)
    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    electronic = np.einsum("xij,ji->x", ao_dip, dm_ao)
    return (nuclear - electronic) * 2.541746473


def total_density(gbci, ci_root):
    """AO density of one root. ``make_rdm1`` already includes the grouped core."""
    return gbci.make_rdm1(ci_root)


def main():
    patch_gbci_grouping()  # complete-linkage bath grouping, see utils/bath_grouping.py

    mol = gto.M(atom=GEOMETRY, basis=BASIS, charge=0, spin=0, unit="Angstrom",
                verbose=4)
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    mf.chkfile = SCF_CHK
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF/{BASIS} did not converge")

    act_list = list(ACT_LIST)
    print(f"act_list ({BASIS}, 1-based) = {act_list}")

    probe = GBCI(mf, ncas=NCAS, nelecas=NELECAS, group_a=GROUP_A)
    if not is_contiguous_window(probe, act_list, base=ACT_BASE):
        print(f"act_list {act_list} is not the default window "
              f"{list(range(probe.ncore + 1, probe.ncore + NCAS + 1))}; reordering")
        apply_act_list(mf, probe, act_list, base=ACT_BASE)

    gbci = GBCI(mf, ncas=NCAS, nelecas=NELECAS, group_a=GROUP_A)
    check_active_window(gbci)
    gbci.fix_spin_(ss=0)
    gbci.fcisolver.nroots = NROOTS
    gbci.fcisolver.conv_tol = FCI_CONV_TOL

    fasscf = gbci._get_fasscf(gbci.group_a, gbci.mo_coeff, gbci.ncas,
                              gbci.nelecas, gbci.ncore)
    fasscf.conv_tol = FASSCF_CONV_TOL

    gbci.kernel()

    ncore = gbci.ncore
    e_tot = np.asarray(gbci.e_tot).reshape(-1)
    ci = gbci.ci

    print(f"\nncore = {ncore}, active MOs (1-based) = {act_list}")
    print(f"GBCI energies (Eh): {e_tot}")

    # Reference: ground-state fragment charges
    dm0 = total_density(gbci, ci[0])
    q0 = lowdin_atom_charges(mol, dm0)
    q0_acc, q0_don = q0[ACCEPTOR_ATOMS].sum(), q0[DONOR_ATOMS].sum()
    q0_oh = q0[HYDROXY_ATOMS].sum()
    print(f"electron count check: trace(dm0 S) = "
          f"{np.einsum('ij,ji->', dm0, mol.intor_symmetric('int1e_ovlp')):.6f} "
          f"(nelectron = {mol.nelectron})")

    results = []
    for root in range(NROOTS):
        dm = total_density(gbci, ci[root])
        q = lowdin_atom_charges(mol, dm)
        q_acc, q_don = q[ACCEPTOR_ATOMS].sum(), q[DONOR_ATOMS].sum()
        q_oh = q[HYDROXY_ATOMS].sum()

        # make_rdm1_mo is the full nmo x nmo density; the active block is what
        # carries the excitation, the core diagonal shows the bath relaxation.
        rdm_mo = gbci.make_rdm1_mo(ci[root])
        act = slice(ncore, ncore + NCAS)
        rdm_act = 0.5 * (rdm_mo[act, act] + rdm_mo[act, act].T)
        occ_no = np.sort(np.linalg.eigvalsh(rdm_act))[::-1]
        occ_mo = np.diag(rdm_mo)[act]
        core_dev = float(np.abs(np.diag(rdm_mo)[:ncore] - 2.0).max())

        info = {
            "root": root,
            "e_tot": float(e_tot[root]),
            "exc_ev": float((e_tot[root] - e_tot[0]) * HARTREE2EV),
            "natural_occ": [float(x) for x in occ_no],
            "active_mo_occ": {str(act_list[i]): float(occ_mo[i]) for i in range(NCAS)},
            "max_core_occ_deviation": core_dev,
            "q_acceptor": float(q_acc),
            "q_donor": float(q_don),
            "dq_acceptor": float(q_acc - q0_acc),
            "dq_donor": float(q_don - q0_don),
            "q_hydroxy": float(q_oh),
            "dq_hydroxy": float(q_oh - q0_oh),
            "dipole_debye": [float(x) for x in dipole_moment(mol, dm)],
        }

        if root > 0:
            # trans_rdm1_mo(bra, ket)[p, q] = <bra| p^dagger q |ket>, so rows are
            # the particle (virtual active) and columns the hole (occupied active).
            tdm_mo = gbci.trans_rdm1_mo(ci[root], ci[0])
            nocc_act = NELECAS[0]
            t_vo = tdm_mo[ncore + nocc_act:ncore + NCAS, ncore:ncore + nocc_act]
            u_p, sv, vt_h = np.linalg.svd(t_vo)
            weights = sv**2 / max(float((sv**2).sum()), 1e-14)
            pairs = []
            for k in range(len(sv)):
                if weights[k] < 0.02:
                    continue
                hole = {str(act_list[i]): float(vt_h[k, i] ** 2)
                        for i in range(nocc_act) if vt_h[k, i] ** 2 > 0.02}
                part = {str(act_list[nocc_act + i]): float(u_p[i, k] ** 2)
                        for i in range(NCAS - nocc_act) if u_p[i, k] ** 2 > 0.02}
                pairs.append({"weight": float(weights[k]),
                              "singular_value": float(sv[k]),
                              "hole": hole, "particle": part})
            info["nto_pairs"] = pairs

        results.append(info)

        print(f"\n=== Root {root} (S{root}) ===")
        print(f"  E = {info['e_tot']:.10f} Eh, dE = {info['exc_ev']:.3f} eV")
        print("  active natural occupations: "
              + " ".join(f"{x:.4f}" for x in info["natural_occ"]))
        print("  occupation per active MO: "
              + " ".join(f"MO{act_list[i]}={occ_mo[i]:.3f}" for i in range(NCAS)))
        print(f"  max |core occ - 2| = {core_dev:.2e}")
        print(f"  Lowdin charge: chromone {q_acc:+.4f} (d {info['dq_acceptor']:+.4f}), "
              f"phenyl {q_don:+.4f} (d {info['dq_donor']:+.4f}), "
              f"3-OH {q_oh:+.4f} (d {info['dq_hydroxy']:+.4f})")
        d = info["dipole_debye"]
        print(f"  dipole = ({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}) D, "
              f"|mu| = {np.linalg.norm(d):.3f} D")
        for pair in info.get("nto_pairs", []):
            hole = ", ".join(f"MO{k}:{v:.2f}" for k, v in pair["hole"].items())
            part = ", ".join(f"MO{k}:{v:.2f}" for k, v in pair["particle"].items())
            print(f"  transition weight {pair['weight']:.3f}: "
                  f"[{hole}] -> [{part}]")

    Path(OUTPUT_JSON).write_text(json.dumps({
        "geometry": GEOMETRY,
        "basis": BASIS,
        "group_threshold": GROUP_THRESHOLD,
        "act_list": act_list,
        "ncas": NCAS,
        "nelecas": list(NELECAS),
        "group_a": GROUP_A,
        "acceptor_atoms": ACCEPTOR_ATOMS,
        "donor_atoms": DONOR_ATOMS,
        "hydroxy_atoms": HYDROXY_ATOMS,
        "roots": results,
    }, indent=2))
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
