"""How far can the Lowdin threshold be raised before 3HF's S1 stops being resolved?

The grouped bath relaxes one set of bath orbitals per group of active-space
occupation patterns, so an excited state is only described with its own bath
relaxation while the configurations it is built from sit in a *different* group
from the ones the ground state is built from.  Raising the threshold merges
groups, which is what makes the gradient cheaper -- its cost carries a
ngroup**2 factor -- but merging the ground-state and excited-state groups would
remove the very relaxation the excited state needs.

With CAS(4e,4o) = {61, 62, 63, 64} the grouping coordinate is the active-electron
population on the chromone acceptor, so a configuration that moves an electron
from the 2-phenyl donor onto the chromone shows up as a population jump.  This
reads the converged RHF/6-31G orbitals, reproduces the grouping under both rules,
and reports for each threshold whether the dominant S0 and S1 occupation patterns
stay apart.  No SCF, no GBCI kernel: it is a few seconds of integrals.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import scf

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]

gbci_module = importlib.import_module("pyscf.gbci.gbci")
GBCI = gbci_module.GBCI


def _load(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gbci_setup = _load("_gbci_setup", "utils/gbci_setup.py")
_bath_grouping = _load("_bath_grouping", "utils/bath_grouping.py")

CHK_FILE = SCRIPT_DIR / "3hf_rhf_631g.chk"
NCAS = 4
NELECAS = (2, 2)
ACT_LIST = [61, 62, 63, 64]
ACT_BASE = 1

# Chromone (benzo-4-pyranone) acceptor, as in 01_gbci_s1_grad_check_631g.py.
ACCEPTOR_ATOMS = [0, 1, 2, 4, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27]

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

# Occupations of the active window in act_list order [61, 62, 63, 64]
# = [HOMO-1, HOMO, LUMO, LUMO+1].  SA-CASSCF(8e,8o) puts 1.489/0.510 on the
# HOMO/LUMO pair and leaves everything else closed, so S1 is the HOMO -> LUMO
# single excitation; the other two are listed to show how the grouping treats
# the neighbouring excitations.
KEY_OCCUPATIONS = {
    "S0 closed shell": (2, 2, 0, 0),
    "HOMO->LUMO (S1)": (2, 1, 1, 0),
    "HOMO->LUMO+1": (2, 1, 0, 1),
    "HOMO-1->LUMO": (1, 2, 1, 0),
}

OUTPUT_JSON = SCRIPT_DIR / "outputs" / "02_group_threshold_check_631g.json"


def main() -> None:
    mol = scf.chkfile.load_mol(str(CHK_FILE))
    mf = scf.RHF(mol)
    mf.__dict__.update(scf.chkfile.load(str(CHK_FILE), "scf"))

    probe = GBCI(mf, ncas=NCAS, nelecas=NELECAS, group_a={"atom": [ACCEPTOR_ATOMS]})
    if not _gbci_setup.is_contiguous_window(probe, ACT_LIST, base=ACT_BASE):
        _gbci_setup.apply_act_list(mf, probe, ACT_LIST, base=ACT_BASE)

    gbci = GBCI(mf, ncas=NCAS, nelecas=NELECAS, group_a={"atom": [ACCEPTOR_ATOMS]})
    ncore = gbci.ncore
    active_mo = np.asarray(gbci.mo_coeff)[:, ncore:ncore + NCAS]
    po_list = np.asarray(gbci.possible_occ())

    values = _bath_grouping.population_matrix(
        mol, active_mo, po_list, [ACCEPTOR_ATOMS])
    acceptor = values[:, 0]

    print(f"{len(po_list)} possible occupation patterns, "
          f"acceptor population range {acceptor.min():.3f} .. {acceptor.max():.3f}")

    key_index = {}
    for name, occ in KEY_OCCUPATIONS.items():
        matches = np.where((po_list == np.asarray(occ)).all(axis=1))[0]
        if matches.size != 1:
            raise RuntimeError(f"occupation {occ} not found in po_list")
        key_index[name] = int(matches[0])
        print(f"  {name:<22} occ={occ}  acceptor population = "
              f"{acceptor[key_index[name]]:.4f}")

    sep = abs(acceptor[key_index["S0 closed shell"]]
              - acceptor[key_index["HOMO->LUMO (S1)"]])
    print(f"\nS0 <-> S1 acceptor population separation = {sep:.4f} electrons")
    print("A threshold at or above this merges the two into one group.\n")

    results = []
    for thres in THRESHOLDS:
        row: dict[str, Any] = {"threshold": thres}
        for rule_name, rule in (("leader (forge)", _bath_grouping.group_leader),
                                ("complete_linkage (utils)",
                                 _bath_grouping.group_complete_linkage)):
            groups = rule(values, thres)
            owner = {}
            for name, index in key_index.items():
                owner[name] = next(g for g, members in enumerate(groups)
                                   if index in members)
            diameters = [_bath_grouping.group_diameter(values, m) for m in groups]
            same = owner["S0 closed shell"] == owner["HOMO->LUMO (S1)"]
            row[rule_name] = {
                "n_groups": len(groups),
                "group_of": owner,
                "max_diameter": float(max(diameters)),
                "s0_s1_merged": bool(same),
            }
            flag = "MERGED - S1 bath relaxation lost" if same else "separate"
            print(f"  thres={thres:.1f}  {rule_name:<26} groups={len(groups):3d}  "
                  f"S0 in g{owner['S0 closed shell']}, "
                  f"S1 in g{owner['HOMO->LUMO (S1)']}, "
                  f"L+1 in g{owner['HOMO->LUMO+1']}, "
                  f"H-1 in g{owner['HOMO-1->LUMO']}  "
                  f"max diameter {max(diameters):.3f}  -> {flag}")
        results.append(row)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps({
        "chkfile": str(CHK_FILE),
        "act_list": ACT_LIST,
        "acceptor_atoms": ACCEPTOR_ATOMS,
        "po_list": po_list.tolist(),
        "acceptor_population": acceptor.tolist(),
        "key_occupations": {k: list(v) for k, v in KEY_OCCUPATIONS.items()},
        "s0_s1_separation": float(sep),
        "thresholds": results,
    }, indent=2))
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
