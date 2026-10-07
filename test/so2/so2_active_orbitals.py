"""Characterise and export the SO2 GBCI(8e,5o) active orbitals.

Writes a Molden file for rendering and prints the assignment table that goes
with it.  The geometry is the asymmetric C1 one used for the gradient checks,
where symmetry labels are unavailable, so pi character is measured instead as
the fraction of the Lowdin-orthogonalised density carried by p functions along
the molecular-plane normal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.tools import molden

SYMBOLS = ("S", "O", "O")
BASIS = "ccpvdz"
NCAS, NELECAS = 5, (4, 4)
R1_ANG, R2_ANG, ANGLE_DEG, OOP_ANG = 1.38, 1.52, 112.0, 0.08


def reference_geometry() -> np.ndarray:
    half = np.deg2rad(ANGLE_DEG) / 2.0
    coords = np.array([
        [0.0, 0.0, 0.0],
        [R1_ANG * np.sin(half), 0.0, R1_ANG * np.cos(half)],
        [-R2_ANG * np.sin(half), 0.0, R2_ANG * np.cos(half)],
    ])
    coords[2, 1] += OOP_ANG
    return coords


def plane_normal(coords: np.ndarray) -> np.ndarray:
    """Unit normal of the best-fit plane through the nuclei."""
    centred = coords - coords.mean(axis=0)
    normal = scipy.linalg.svd(centred)[2][-1]
    return normal / np.linalg.norm(normal)


def out_of_plane_weights(mol: gto.Mole, normal: np.ndarray) -> np.ndarray:
    """Per-AO weight of the component along the plane normal (1 for a p along n)."""
    axes = {"x": 0, "y": 1, "z": 2}
    weights = np.zeros(mol.nao)
    for i, label in enumerate(mol.ao_labels()):
        token = label.split()[-1]
        if len(token) >= 3 and token[1] == "p" and token[2] in axes:
            direction = np.zeros(3)
            direction[axes[token[2]]] = 1.0
            weights[i] = float(direction @ normal) ** 2
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--molden", type=Path,
                        default=Path(__file__).resolve().parent / "outputs"
                        / "so2_cas8e5o_active.molden")
    args = parser.parse_args()

    coords = reference_geometry()
    mol = gto.M(atom=[(s, tuple(c)) for s, c in zip(SYMBOLS, coords)],
                basis=args.basis, charge=0, spin=0, symmetry=False, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-9, 800
    mf.kernel()

    ncore = (mol.nelectron - sum(NELECAS)) // 2
    active = range(ncore, ncore + NCAS)

    normal = plane_normal(mol.atom_coords())
    oop = out_of_plane_weights(mol, normal)
    overlap = mf.get_ovlp()
    orthogonal = scipy.linalg.sqrtm(overlap).real @ mf.mo_coeff
    labels = mol.ao_labels()
    atom_of = np.array([int(label.split()[0]) for label in labels])

    print(f"SO2 {args.basis}  distorted C1   E(RHF) = {mf.e_tot:.10f}")
    print(f"plane normal = {np.round(normal, 4)}")
    print(f"ncore = {ncore}  ->  CAS({sum(NELECAS)}e,{NCAS}o) = MO "
          f"{ncore + 1}..{ncore + NCAS}\n")
    print(f"{'MO':>3}{'e (Eh)':>11}{'occ':>5}{'pi(oop)':>9}{'S':>7}{'O':>7}   leading AOs")
    for k in active:
        weight = orthogonal[:, k] ** 2
        weight = weight / weight.sum()
        top = np.argsort(weight)[::-1][:3]
        print(f"{k + 1:>3}{mf.mo_energy[k]:>11.5f}{mf.mo_occ[k]:>5.1f}"
              f"{float((weight * oop).sum()):>9.3f}"
              f"{float(weight[atom_of == 0].sum()):>7.2f}"
              f"{float(weight[atom_of > 0].sum()):>7.2f}   "
              + ", ".join(f"{labels[i]} ({weight[i]:.2f})" for i in top))

    args.molden.parent.mkdir(parents=True, exist_ok=True)
    molden.from_mo(mol, str(args.molden), mf.mo_coeff[:, active],
                   ene=mf.mo_energy[list(active)], occ=mf.mo_occ[list(active)])
    print(f"\nWrote {args.molden}  ({NCAS} active orbitals, in window order)")


if __name__ == "__main__":
    main()
