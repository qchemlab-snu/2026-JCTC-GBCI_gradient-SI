"""Tabulate the resulting bath groups for every GBCI system in the manuscript.

Reviewers asked which configurations end up in which group.  This script
reproduces the grouping for each system at its published input geometry and
prints, for every group, its member configurations and the quantity the grouping
was based on.  No geometry optimization is run: the grouping is a property of
the reference orbitals at one geometry, so only RHF plus the GBCI setup is
needed.

The four systems do not share one grouping mechanism, which is the main thing
the SI has to state:

    Li halides  {"atom": [0], "threshold": 0.4}        Lowdin population on Li
    pp          {"atom": [[0..8], [9..18]]}, 0.2       Lowdin on two fragments
    CH2NH2      {"mo": [[0,1,2,3], [4]]}               occupation sums, exact

Only the first three use a Lowdin population and a threshold.  CH2NH2 groups
configurations that share the same total occupation on each orbital subset, so
it has no threshold and admits no spread within a group.

Note on the Lowdin algorithm (pyscf.gbci.gbci.group_by_atom): it is a sequential
greedy scan, not a clustering.  The first unassigned configuration seeds a group
and absorbs every unassigned configuration within the threshold *of that seed*,
so members are close to the seed rather than to each other, and the result
depends on the order of the possible-occupation list.

Usage
-----
    OMP_NUM_THREADS=4 python3 report_bath_groups.py
    OMP_NUM_THREADS=4 python3 report_bath_groups.py --system LiF --system pp
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
from pyscf import gto, lib, scf
from pyscf.mcscf.addons import sort_mo


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]

_grad_module = Path(importlib.import_module("pyscf.grad.gbci").__file__).resolve()
if REPO_ROOT in _grad_module.parents:
    raise RuntimeError(
        f"pyscf.grad.gbci resolved to the local repo copy ({_grad_module}); "
        "point PYSCF_EXT_PATH at pyscf-forge and keep the repo root off sys.path"
    )

from pyscf.gbci.gbci import GBCI, fragment_aos_by_atoms

# utils/ is loaded by path for the same reason the other scripts do it: putting
# REPO_ROOT on sys.path makes pyscf treat this repo's pyscf/ tree as a plugin.
_spec = importlib.util.spec_from_file_location(
    "_bath_grouping", REPO_ROOT / "utils" / "bath_grouping.py")
_bath_grouping = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bath_grouping)

GROUPING_RULES = {
    "complete": _bath_grouping.group_complete_linkage,
    "leader": _bath_grouping.group_leader,
}

DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "bath_groups.json"


# ============================================================
# Systems
# ============================================================

@dataclass(frozen=True)
class SystemSpec:
    name: str
    atoms: str
    basis: str
    ncas: int
    nelecas: tuple[int, int]
    group_a: dict[str, Any]
    source: str
    charge: int = 0
    spin: int = 0
    act_list: tuple[int, ...] | None = None
    act_base: int = 1
    note: str = ""


PP_ATOMS = """C -3.74277916 0.68300914 0.0
C -2.53434166 1.29405070 0.0
C -3.50313646 -0.72793393 0.0
C -2.16078760 -0.90633919 0.0
H -4.69809009 1.18130123 0.0
H -4.24054134 -1.51352758 0.0
N -1.54283098 0.33045911 0.0
H -1.59821403 -1.81998326 0.0
H -2.30464184 2.34213309 0.0
C -0.16584886 0.56419685 0.0
C 0.75378635 -0.48400386 0.0
C 2.10568061 -0.18658172 0.0
H 0.44688705 -1.51636124 0.0
N 2.60807481 1.03480047 0.0
H 2.82251096 -0.99891916 0.0
C 1.73094162 2.02213037 0.0
C 0.35659599 1.85706993 0.0
H 2.13973853 3.02543900 0.0
H -0.27361675 2.73043999 0.0"""

# CH2NH2+ (2e,2o) S0/S1 MECI, from
# test/ch2nh2/gbci/outputs/ch2nh2_gbci_S0S1_meci_2o2e_groupN_ccpvtz/
#   ch2nh2_gbci_S0S1_meci_2o2e_groupN_ccpvtz_optimized.xyz
# The grouping is reported here, at the intersection, rather than at an S0
# geometry: MO8 and MO9 are the N- and C-localized p orbitals only near the
# seam, and that localization is what the nitrogen-seeded grouping rests on.
CH2NH2_MECI_2O2E_ATOMS = """N  0.72008165 -0.00836083 -0.00494257
C -0.74978247 -0.02238020 -0.01127616
H  1.25805170  0.71973138  0.44255403
H  1.23941784 -0.74638414 -0.46016712
H -1.16884282 -0.32366550  0.92068559
H -1.17157688  0.67354229 -0.69881478"""

CH2NH2_ATOMS = """N  0.59504934  0.00000418  0.00002110
C -0.67688886  0.00000732 -0.00000853
H  1.12534395  0.22026262  0.83097781
H  1.12520035 -0.22028482 -0.83101477
H -1.21148201  0.24130980  0.91030560
H -1.21139777 -0.24130309 -0.91038122"""

# 3-hydroxyflavone: B3LYP/6-31G(d) S0 minimum, from
# test/3HF/dft/outputs/3hf_dft_S0_opt.xyz (no imaginary frequencies).
HF3_ATOMS = """O -1.66557168 1.28041179 3.09806724
C -1.35864134 0.82956055 1.98137939
C 0.05398841 0.58682730 1.69471563
O 0.87455695 0.89837760 2.72438015
C 0.48347300 0.08976705 0.49295905
C 1.85524232 -0.20824449 0.06189313
C 2.08690585 -0.73332029 -1.22593008
C 3.37905304 -1.02394063 -1.65351822
C 4.46953144 -0.79874402 -0.81049584
C 4.25204805 -0.27879590 0.46668772
C 2.96286313 0.01558116 0.90523084
O -0.44533851 -0.19040719 -0.48323670
C -1.77631417 0.00730831 -0.29471728
C -2.62428446 -0.31654733 -1.36285660
C -3.99054052 -0.13022298 -1.21347847
C -4.52507800 0.37544519 -0.01224638
C -3.68127232 0.69297647 1.03796861
C -2.29040842 0.51232799 0.90974291
H 0.24434488 1.21884758 3.41510048
H 1.24892193 -0.91204147 -1.88863668
H 3.53390744 -1.42841860 -2.65018932
H 5.47770393 -1.02616703 -1.14598021
H 5.09249306 -0.09895316 1.13158798
H 2.81054264 0.41766010 1.89740807
H -2.19524864 -0.70493405 -2.28070448
H -4.65441322 -0.37898293 -2.03664113
H -5.59747014 0.51445352 0.08575575
H -4.05694913 1.08454874 1.97785486"""

# The chromone acceptor: carbonyl O, the pyranone ring, the fused benzo ring and
# its hydrogens. The 2-phenyl ring and the 3-OH stay outside, so the group
# boundary straddles the phenyl -> chromone charge transfer that S1 carries.
HF3_ACCEPTOR = [0, 1, 2, 4, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27]

def li_halide_geometry(partner: str, bond_length: float) -> str:
    return f"Li 0.0 0.0 0.0\n{partner} 0.0 0.0 {bond_length}"


LI_BOND_LENGTH = 1.50
LI_GROUP_A = {"atom": [0], "threshold": 0.4}

SYSTEMS = [
    SystemSpec(
        name="LiH", atoms=li_halide_geometry("H", LI_BOND_LENGTH), basis="ccpvdz",
        ncas=2, nelecas=(1, 1), act_list=(2, 3), group_a=dict(LI_GROUP_A),
        source="test/lix/li_halides_gbci_grad_check.py",
        note="sigma-only active space",
    ),
    SystemSpec(
        name="LiF", atoms=li_halide_geometry("F", LI_BOND_LENGTH), basis="ccpvdz",
        ncas=4, nelecas=(2, 2), act_list=(3, 4, 7, 10), group_a=dict(LI_GROUP_A),
        source="test/lix/li_halides_gbci_grad_check.py",
        note="sigma-only active space",
    ),
    SystemSpec(
        name="LiCl", atoms=li_halide_geometry("Cl", LI_BOND_LENGTH), basis="ccpvdz",
        ncas=4, nelecas=(2, 2), act_list=(7, 8, 11, 14), group_a=dict(LI_GROUP_A),
        source="test/lix/li_halides_gbci_grad_check.py",
        note="sigma-only active space",
    ),
    SystemSpec(
        name="pp", atoms=PP_ATOMS, basis="ccpvdz",
        ncas=2, nelecas=(1, 1), act_list=(38, 39),
        group_a={"atom": [list(range(0, 9)), list(range(9, 19))]},
        source="test/pp/gbci/pp_gbci_S1.py",
        note="two pyridine fragments; input geometry",
    ),
    SystemSpec(
        name="3-hydroxyflavone", atoms=HF3_ATOMS, basis="6-31G",
        ncas=4, nelecas=(2, 2), act_list=(61, 62, 63, 64),
        group_a={"atom": [HF3_ACCEPTOR], "threshold": 0.5},
        source="test/3hf/gbci/01_gbci_s1_grad_check_631g.py",
        note="frontier pi/pi* window; chromone acceptor fragment, threshold 0.5",
    ),
    SystemSpec(
        name="CH2NH2", atoms=CH2NH2_ATOMS, basis="cc-pVTZ",
        charge=1,  # methyleneiminium cation, CH2NH2+
        ncas=5, nelecas=(4, 4), act_list=(5, 6, 7, 8, 9),
        group_a={"mo": [[0, 1, 2, 3], [4]]},
        source="test/ch2nh2/gbci/ch2nh2_gbci_S0_opt_5o8e_groupocc_ccpvtz.py",
        note="occupation-sum grouping, no Lowdin and no threshold; input geometry",
    ),
    # The two (2e,2o) entries are the same active space grouped two ways.  They
    # are both listed because the SI claim is that they coincide, and the only
    # way to show that is to report both groupings side by side.  The Lowdin-on-N
    # one is the grouping the manuscript describes and the one Table S6 reports.
    SystemSpec(
        name="CH2NH2-2o2e-Lowdin-N", atoms=CH2NH2_MECI_2O2E_ATOMS, basis="cc-pVTZ",
        charge=1,
        ncas=2, nelecas=(1, 1), act_list=(8, 9),
        group_a={"atom": [[0]]},
        source="test/ch2nh2/gbci/gbci_2o2e_S0S1_meci_groupN_ccpvtz.py",
        note="Lowdin population on N, default threshold; S0/S1 MECI geometry",
    ),
    SystemSpec(
        name="CH2NH2-2o2e-MO", atoms=CH2NH2_MECI_2O2E_ATOMS, basis="cc-pVTZ",
        charge=1,
        ncas=2, nelecas=(1, 1), act_list=(8, 9),
        group_a={"mo": [[0], [1]]},
        source="test/ch2nh2/gbci/gbci_2o2e_S0S1_meci_groupocc_ccpvtz.py",
        note="occupation-sum grouping, one orbital per group; same MECI geometry",
    ),
]


# ============================================================
# Grouping
# ============================================================

def build_gbci(spec: SystemSpec, args: argparse.Namespace) -> tuple[gto.Mole, Any]:
    """RHF plus GBCI setup at the input geometry.  No optimization is run."""
    mol = gto.M(atom=spec.atoms, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, unit="Angstrom", symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = args.max_cycle
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{spec.name}: RHF did not converge")

    if spec.act_list is not None:
        probe = GBCI(mf, ncas=spec.ncas, nelecas=spec.nelecas, group_a=spec.group_a)
        mf.mo_coeff = sort_mo(probe, mf.mo_coeff, list(spec.act_list),
                              base=spec.act_base)

    gbci = GBCI(mf, ncas=spec.ncas, nelecas=spec.nelecas, group_a=spec.group_a)
    # Only {"atom": ...} systems go through group_by_atom; {"mo": ...} grouping
    # is exact and is not affected by the rule.
    original = _bath_grouping.patch_gbci_grouping(
        GROUPING_RULES[getattr(args, "grouping", "complete")])
    try:
        gbci.get_gbci_intermediates()
    finally:
        importlib.import_module("pyscf.gbci.gbci").group_by_atom = original
    return mol, gbci


def grouping_descriptor(
    spec: SystemSpec, mol: gto.Mole, gbci: Any, po_list: np.ndarray
) -> tuple[np.ndarray, list[str], str]:
    """The quantity the grouping is based on, one row per configuration.

    For atom grouping this is the Lowdin population of the active electrons on
    each fragment, recomputed exactly as ``group_by_atom`` does it.  For mo
    grouping it is the total occupation on each orbital subset, which is an
    integer and admits no spread within a group.
    """
    kind = "atom" if "atom" in spec.group_a else "mo"
    groups = spec.group_a[kind]
    if kind == "mo":
        values = np.array([[int(np.sum(occ[subset])) for subset in groups]
                           for occ in po_list], dtype=float)
        labels = [f"n(MO {'+'.join(str(i + 1) for i in subset)})" for subset in groups]
        return values, labels, "occupation sum"

    fragments = groups if isinstance(groups[0], (list, tuple)) else [groups]
    active = np.asarray(gbci.mo_coeff)[:, gbci.ncore:gbci.ncore + gbci.ncas]
    overlap = mol.intor_symmetric("cint1e_ovlp_sph")
    eigenvalues, vectors = np.linalg.eigh(overlap)
    s12 = (vectors * np.sqrt(eigenvalues)) @ vectors.T.conj()

    ao_sets = [fragment_aos_by_atoms(mol, atoms) for atoms in fragments]
    values = np.zeros((len(po_list), len(ao_sets)))
    for index, occ in enumerate(po_list):
        singly = np.where(occ == 1)[0]
        doubly = np.where(occ == 2)[0]
        density = (active[:, singly] @ active[:, singly].T
                   + 2.0 * (active[:, doubly] @ active[:, doubly].T))
        diagonal = np.diag(s12 @ density @ s12)
        for j, ao_set in enumerate(ao_sets):
            values[index, j] = np.sum(diagonal[ao_set])
    labels = [f"n(atoms {_atom_label(mol, atoms)})" for atoms in fragments]
    return values, labels, "Lowdin population"


def _atom_label(mol: gto.Mole, atoms: list[int]) -> str:
    """Compact 1-based description of a fragment.

    Contiguous stretches collapse to ranges; a fragment that is not one
    unbroken run is listed as its parts, so a scattered selection is not
    mislabelled as the range that merely spans it.
    """
    ordered = sorted(int(a) + 1 for a in atoms)
    if len(ordered) == 1:
        return f"{mol.atom_symbol(ordered[0] - 1)}{ordered[0]}"
    parts, start, previous = [], ordered[0], ordered[0]
    for value in ordered[1:] + [None]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        if value is not None:
            start = previous = value
    return ",".join(parts)


def analyse(spec: SystemSpec, args: argparse.Namespace) -> dict[str, object]:
    """Reproduce the grouping and describe every group."""
    mol, gbci = build_gbci(spec, args)
    cache = gbci._gbci_intermediates
    po_list = np.asarray(cache["po_list"], dtype=int)
    groups = [[int(i) for i in members] for members in cache["group"]]
    values, labels, quantity = grouping_descriptor(spec, mol, gbci, po_list)

    records = []
    for number, members in enumerate(groups, start=1):
        block = values[members]
        records.append({
            "group": number,
            "size": len(members),
            "configurations": ["".join(str(x) for x in po_list[i]) for i in members],
            "po_indices": members,
            "descriptor": block.tolist(),
            "descriptor_min": block.min(axis=0).tolist(),
            "descriptor_max": block.max(axis=0).tolist(),
            "descriptor_span": (block.max(axis=0) - block.min(axis=0)).tolist(),
            "diameter": _bath_grouping.group_diameter(values, members),
        })

    return {
        "name": spec.name,
        "source": spec.source,
        "note": spec.note,
        "geometry_angstrom": spec.atoms.splitlines(),
        "basis": spec.basis,
        "charge": spec.charge,
        "spin": spec.spin,
        "ncas": spec.ncas,
        "nelecas": list(spec.nelecas),
        "ncore": int(gbci.ncore),
        "act_list": list(spec.act_list) if spec.act_list else None,
        "act_base": spec.act_base,
        "group_a": spec.group_a,
        "grouping_quantity": quantity,
        "threshold": spec.group_a.get("threshold",
                                      0.2 if quantity == "Lowdin population" else None),
        "descriptor_labels": labels,
        "n_configurations": int(len(po_list)),
        "n_groups": len(groups),
        "grouping_rule": args.grouping if quantity == "Lowdin population" else "exact",
        "max_group_diameter": max(r["diameter"] for r in records),
        "groups": records,
    }


# ============================================================
# Reporting
# ============================================================

def print_report(result: dict[str, object]) -> None:
    labels = result["descriptor_labels"]
    print(f"\n{'=' * 78}")
    print(f"{result['name']}   CAS({sum(result['nelecas'])}e,{result['ncas']}o)   "
          f"basis={result['basis']}   ncore={result['ncore']}")
    print(f"  act_list {result['act_list']} (base {result['act_base']})"
          if result["act_list"] else "  default contiguous active window")
    threshold = result["threshold"]
    print(f"  group_a = {result['group_a']}")
    print(f"  grouped by {result['grouping_quantity']}"
          + (f", threshold {threshold}, rule '{result['grouping_rule']}'"
             if threshold is not None else " (exact)"))
    if threshold is not None:
        violation = ("  <-- exceeds the threshold"
                     if result["max_group_diameter"] > threshold + 1e-12 else "")
        print(f"  largest population difference within a group: "
              f"{result['max_group_diameter']:.3f}{violation}")
    print(f"  {result['n_groups']} groups from {result['n_configurations']} configurations")
    print(f"  source: {result['source']}")
    print(f"{'=' * 78}")

    width = max(len(label) for label in labels) + 2
    header = f"  {'group':>5} {'config':<10}" + "".join(f"{l:>{width}}" for l in labels)
    print(header)
    for record in result["groups"]:
        for n, (config, row) in enumerate(zip(record["configurations"],
                                              record["descriptor"])):
            tag = f"{record['group']}" if n == 0 else ""
            print(f"  {tag:>5} {config:<10}"
                  + "".join(f"{v:>{width}.3f}" for v in row))
        if record["size"] > 1:
            print(f"  {'':>5} {'span':<10}"
                  + "".join(f"{v:>{width}.3f}" for v in record["descriptor_span"]))


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    """One row per (system, group, configuration)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["system", "active_space", "basis", "group_a",
                         "grouping_quantity", "threshold", "n_groups",
                         "group", "group_size", "configuration", "po_index",
                         "descriptor_label", "descriptor_value"])
        for result in results:
            cas = f"CAS({sum(result['nelecas'])}e,{result['ncas']}o)"
            for record in result["groups"]:
                for config, index, row in zip(record["configurations"],
                                              record["po_indices"],
                                              record["descriptor"]):
                    for label, value in zip(result["descriptor_labels"], row):
                        writer.writerow([
                            result["name"], cas, result["basis"],
                            json.dumps(result["group_a"]),
                            result["grouping_quantity"], result["threshold"],
                            result["n_groups"], record["group"], record["size"],
                            config, index, label, value,
                        ])


def write_latex(results: list[dict[str, object]], path: Path) -> None:
    """A longtable per system, ready to paste into the SI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by test/bath_groups/report_bath_groups.py",
             "% Requires \\usepackage{longtable} only -- plain \\hline rules", ""]
    for result in results:
        labels = result["descriptor_labels"]
        cas = f"CAS({sum(result['nelecas'])}e,{result['ncas']}o)"
        threshold = result["threshold"]
        criterion = (f"{result['grouping_quantity']}, threshold {threshold}"
                     if threshold is not None
                     else f"{result['grouping_quantity']} (exact)")
        lines += [
            r"\begin{longtable}{c l " + "r" * len(labels) + "}",
            rf"\caption{{Bath groups for {result['name']}, {cas}, "
            rf"{result['basis']}. Grouped by {criterion}. "
            rf"{result['n_groups']} groups from {result['n_configurations']} "
            rf"configurations.}}\\",
            r"\hline",
            "Group & Configuration & " + " & ".join(labels) + r" \\",
            r"\hline", r"\endfirsthead", r"\hline",
            "Group & Configuration & " + " & ".join(labels) + r" \\",
            r"\hline", r"\endhead",
        ]
        # A rule between groups rather than a blank gap: the groups are the
        # point of the table, so their boundaries should be drawn, not implied
        # by spacing.  Plain \hline throughout, so the table needs no booktabs.
        for number, record in enumerate(result["groups"]):
            if number:
                lines.append(r"\hline")
            for n, (config, row) in enumerate(zip(record["configurations"],
                                                  record["descriptor"])):
                tag = str(record["group"]) if n == 0 else ""
                values = " & ".join(f"{v:.3f}" for v in row)
                lines.append(f"{tag} & {config} & {values} " + r"\\")
        lines += [r"\hline", r"\end{longtable}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabulate the resulting GBCI bath groups for every system.")
    parser.add_argument("--system", action="append", default=None,
                        help="Restrict to one system; repeatable.")
    # "complete" bounds the population difference *within* a group, which is how
    # the method is described.  pyscf-forge's own group_by_atom uses "leader",
    # which only bounds the distance to the first member of each group and can
    # exceed the threshold; see utils/bath_grouping.py.
    parser.add_argument("--grouping", choices=sorted(GROUPING_RULES),
                        default="complete")
    parser.add_argument("--scf-conv-tol", type=positive_float, default=1.0e-12)
    parser.add_argument("--max-cycle", type=int, default=400)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--latex-output", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lib.num_threads(args.threads)

    selected = SYSTEMS
    if args.system:
        wanted = {name.lower() for name in args.system}
        selected = [s for s in SYSTEMS if s.name.lower() in wanted]
        if not selected:
            raise SystemExit(f"No system matched {args.system}")

    results = []
    for spec in selected:
        result = analyse(spec, args)
        print_report(result)
        results.append(result)

    csv_path = args.csv_output or args.output.with_suffix(".csv")
    latex_path = args.latex_output or args.output.with_suffix(".tex")
    payload = {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pyscf_version": pyscf.__version__,
        "gbci_module": str(Path(importlib.import_module("pyscf.gbci.gbci").__file__).resolve()),
        "systems": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(results, csv_path)
    write_latex(results, latex_path)
    print(f"\nSaved results to {args.output}")
    print(f"Saved table to    {csv_path}")
    print(f"Saved LaTeX to    {latex_path}")


if __name__ == "__main__":
    main()
