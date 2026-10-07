"""Inspect RHF orbital character for LiH, LiF, and LiCl along the bond axis.

The sigma-only active spaces used by ``li_halides_gbci_grad_check.py`` are
specified through a hardcoded ``act_list``.  This script produces the table that
``act_list`` is read off: every MO is labelled by its sigma weight, so the Li-X
bonding and antibonding orbitals can be separated from the halogen pi lone
pairs that the default contiguous window would otherwise pick up.

Both molecules are placed on the z axis, so an AO contributes to a sigma orbital
only when its magnetic quantum number is zero (s, pz, dz^2, ...).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf
from pyscf.tools import molden


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# User settings
# ============================================================

BASIS = "ccpvdz"


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    partner: str
    bond_length: float  # Angstrom
    ncas: int
    nelecas: tuple[int, int]


# One bond length for all three, deliberately away from equilibrium (LiH 1.595,
# LiF 1.564, LiCl 2.021 A): a stationary point would drive both the analytic and
# the numerical gradient towards zero and weaken the comparison.
BOND_LENGTH = 1.50

MOLECULES = [
    MoleculeSpec("LiH", "H", BOND_LENGTH, 2, (1, 1)),
    MoleculeSpec("LiF", "F", BOND_LENGTH, 4, (2, 2)),
    MoleculeSpec("LiCl", "Cl", BOND_LENGTH, 4, (2, 2)),
]

# AO-label suffixes with magnetic quantum number zero.  Everything else (px, py,
# dxy, dxz, dyz, dx2-y2, ...) is pi or delta about the z axis.
SIGMA_SUFFIXES = ("s", "pz", "dz^2", "fz^3")


def build_molecule(spec: MoleculeSpec, basis: str, verbose: int) -> gto.Mole:
    """Build the neutral closed-shell molecule on the z axis."""
    return gto.M(
        atom=[("Li", (0.0, 0.0, 0.0)),
              (spec.partner, (0.0, 0.0, spec.bond_length))],
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry=False,
        verbose=verbose,
    )


def sigma_mask(mol: gto.Mole) -> np.ndarray:
    """Boolean mask selecting the m = 0 atomic orbitals."""
    labels = mol.ao_labels()
    return np.array(
        [label.split()[-1].endswith(SIGMA_SUFFIXES) for label in labels],
        dtype=bool,
    )


def sigma_weights(mol: gto.Mole, mo_coeff: np.ndarray) -> np.ndarray:
    """Fraction of each MO's squared coefficients carried by m = 0 AOs.

    Using bare squared coefficients rather than a Mulliken partition keeps this
    free of overlap-induced negative populations; for a sigma/pi split the two
    agree because the m = 0 and m != 0 blocks do not overlap on a linear molecule.
    """
    weight = mo_coeff ** 2
    mask = sigma_mask(mol)
    return weight[mask].sum(axis=0) / weight.sum(axis=0)


def dominant_labels(mol: gto.Mole, mo_coeff: np.ndarray, index: int, count: int = 3) -> str:
    """Return the ``count`` AO labels with the largest coefficient magnitude."""
    labels = mol.ao_labels()
    order = np.argsort(np.abs(mo_coeff[:, index]))[::-1][:count]
    return ", ".join(
        f"{labels[ao].strip()}({mo_coeff[ao, index]:+.2f})" for ao in order
    )


def suggested_act_list(
    spec: MoleculeSpec,
    mo_occ: np.ndarray,
    weights: np.ndarray,
    sigma_thresh: float,
) -> list[int]:
    """Highest occupied and lowest virtual sigma MOs, as 1-based indices.

    Ascending order puts the occupied members at the front of the active window,
    which is what the GBCI gradient's column-index occupied/virtual split needs.
    """
    is_sigma = weights >= sigma_thresh
    occupied = np.flatnonzero(is_sigma & (mo_occ > 0))
    virtual = np.flatnonzero(is_sigma & (mo_occ == 0))

    n_occ_active = spec.nelecas[0]
    n_vir_active = spec.ncas - n_occ_active
    if occupied.size < n_occ_active or virtual.size < n_vir_active:
        raise ValueError(
            f"{spec.name}: found {occupied.size} occupied and {virtual.size} virtual "
            f"sigma orbitals, need {n_occ_active} and {n_vir_active}"
        )

    selected = np.concatenate([occupied[-n_occ_active:], virtual[:n_vir_active]])
    return [int(index) + 1 for index in selected]


def report(spec: MoleculeSpec, args: argparse.Namespace) -> None:
    """Run RHF and print the per-MO character table for one molecule."""
    mol = build_molecule(spec, args.basis, args.verbose)
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv_tol
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{spec.name}: RHF did not converge")

    weights = sigma_weights(mol, mf.mo_coeff)
    ncore = (mol.nelectron - sum(spec.nelecas)) // 2
    default_window = set(range(ncore, ncore + spec.ncas))

    print(f"\n{'=' * 78}")
    print(
        f"{spec.name}  r = {spec.bond_length:.3f} A  basis = {args.basis}  "
        f"E(RHF) = {mf.e_tot:.10f} Eh"
    )
    print(f"  nelectron = {mol.nelectron}  ncore = {ncore}  "
          f"CAS({sum(spec.nelecas)}e,{spec.ncas}o)")
    print(f"{'=' * 78}")
    print(" MO(1b)  MO(0b)   occ   energy/Eh   w_sigma  type  default?  dominant AOs")
    for index in range(min(mf.mo_coeff.shape[1], args.max_mo)):
        kind = "sigma" if weights[index] >= args.sigma_thresh else "pi/del"
        marker = "  <-- " if index in default_window else "      "
        print(
            f" {index + 1:6d}  {index:6d}  {mf.mo_occ[index]:4.1f}  "
            f"{mf.mo_energy[index]:+11.6f}  {weights[index]:7.4f}  {kind:6s}{marker}"
            f"{dominant_labels(mol, mf.mo_coeff, index)}"
        )

    act_list = suggested_act_list(spec, mf.mo_occ, weights, args.sigma_thresh)
    print(f"\n  default contiguous window (1-based): "
          f"{[i + 1 for i in sorted(default_window)]}")
    print(f"  suggested sigma act_list   (1-based): {act_list}")
    print(f"  -> MoleculeSpec(\"{spec.name}\", \"{spec.partner}\", {spec.bond_length}, "
          f"{spec.ncas}, {spec.nelecas}, act_list={tuple(act_list)})")

    if args.molden:
        molden_path = SCRIPT_DIR / f"{spec.name.lower()}_rhf.molden"
        molden.from_mo(mol, str(molden_path), mf.mo_coeff, occ=mf.mo_occ)
        print(f"  molden: {molden_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print RHF orbital character for LiH/LiF/LiCl so the sigma-only "
            "act_list of the GBCI gradient check can be read off."
        )
    )
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--scf-conv-tol", type=float, default=1.0e-12)
    parser.add_argument("--sigma-thresh", type=float, default=0.9)
    parser.add_argument("--max-mo", type=int, default=20,
                        help="Number of MOs to print, counting from the lowest.")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--molecule", action="append", default=None,
                        help="Restrict to one molecule; repeatable.")
    parser.add_argument("--no-molden", dest="molden", action="store_false",
                        help="Skip the molden dumps.")
    return parser.parse_args()


def main() -> None:
    lib.num_threads(4)
    args = parse_args()
    selected = MOLECULES
    if args.molecule:
        wanted = {name.lower() for name in args.molecule}
        selected = [spec for spec in MOLECULES if spec.name.lower() in wanted]
        if not selected:
            raise SystemExit(f"No molecule matched {args.molecule}")
    for spec in selected:
        report(spec, args)


if __name__ == "__main__":
    main()
