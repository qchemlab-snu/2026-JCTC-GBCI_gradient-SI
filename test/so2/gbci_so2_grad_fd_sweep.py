"""Sweep finite-difference step sizes for the SO2 GBCI(8e,5o) gradient.

The analytic gradient does not depend on the step, so it is evaluated once and
reused; each step size then costs only the displaced energies.  That is the
whole point of a sweep script rather than repeated single-step runs: at
CAS(12e,9o) one energy point is the expensive part.

The active space is the default contiguous window, which at this geometry is
MO 13-17 (ncore = 12): the out-of-plane pi bond (13), an oxygen lone pair (14),
the in-plane n orbital (15), pi non-bonding (16) and pi* (17).  All three
orbitals of the pi system are in, so both n->pi* and pi->pi* are described.
No ``sort_mo`` is needed, so the orbital ordering the gradient driver assumes
is preserved.

Larger spaces are affordable for the energies but not for the reference
gradient: on this system the analytic gradient took 5s at (8e,5o), 67s at
(8e,7o) and 2587s at (12e,9o), since ``get_X`` loops over group pairs.  The
displaced energies stay cheap throughout (1-6s), so a sweep is dominated by
that single gradient.

Usage
-----
    OMP_NUM_THREADS=1 python3 gbci_so2_grad_fd_sweep.py
    OMP_NUM_THREADS=1 python3 gbci_so2_grad_fd_sweep.py --steps 1e-3,5e-4 --atoms 0
    python3 gbci_so2_grad_fd_sweep.py --analytic-only     # diagnostics, no FD

Run single-threaded: threaded BLAS moves the energy by ~1e-9 Eh between runs
and a central difference amplifies that by 1/(2h).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
from pyscf import gto, scf
from pyscf.gbci.gbci import GBCI

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

# Load utils/gbci_setup.py by path.  Putting REPO_ROOT on sys.path would make
# pyscf treat the repository's own pyscf/ tree as a plugin, which shadows
# pyscf-forge's pyscf.grad.gbci with the stale local copy.
_spec = importlib.util.spec_from_file_location(
    "_gbci_setup", REPO_ROOT / "utils" / "gbci_setup.py")
_gbci_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gbci_setup)
gradient_invariance = _gbci_setup.gradient_invariance
rotational_consistency = _gbci_setup.rotational_consistency

_grad_module = Path(importlib.import_module("pyscf.grad.gbci").__file__).resolve()
if REPO_ROOT in _grad_module.parents:
    raise RuntimeError(
        f"pyscf.grad.gbci resolved to the local repo copy ({_grad_module}); "
        "point PYSCF_EXT_PATH at pyscf-forge and keep the repo root off sys.path"
    )


# ============================================================
# User settings
# ============================================================

SYMBOLS = ("S", "O", "O")
BASIS = "ccpvdz"
CHARGE, SPIN = 0, 0

NCAS, NELECAS = 5, (4, 4)          # contiguous window -> MO 13..17 here
GROUP_A = {"atom": [[0]]}          # sulfur-centred grouping
NROOTS, TARGET_ROOT = 1, 0

# Deliberately asymmetric C1 geometry: unequal S-O bonds, off-equilibrium
# angle, and an out-of-plane offset, so no symmetry element survives and all
# nine Cartesian gradient components are nonzero.
R1_ANG, R2_ANG, ANGLE_DEG, OOP_ANG = 1.38, 1.52, 112.0, 0.08

DEFAULT_STEPS = (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.0e-4)
AXES = ("x", "y", "z")


def reference_geometry() -> np.ndarray:
    half = np.deg2rad(ANGLE_DEG) / 2.0
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [R1_ANG * np.sin(half), 0.0, R1_ANG * np.cos(half)],
            [-R2_ANG * np.sin(half), 0.0, R2_ANG * np.cos(half)],
        ]
    )
    coords[2, 1] += OOP_ANG
    return coords


def build_molecule(coords: np.ndarray, *, unit: str, args) -> gto.Mole:
    return gto.M(
        atom=[(s, tuple(c)) for s, c in zip(SYMBOLS, coords)],
        basis=args.basis, charge=CHARGE, spin=SPIN, unit=unit,
        symmetry=False, verbose=args.verbose,
    )


def run_gbci(mol: gto.Mole, args) -> Any:
    """RHF followed by GBCI on the default contiguous active window."""
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv_tol
    mf.conv_tol_grad = args.scf_conv_tol_grad
    mf.max_cycle = args.max_cycle
    if args.scf_damping:
        mf.damp, mf.level_shift = 0.4, 0.5
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    gbci = GBCI(mf, ncas=NCAS, nelecas=NELECAS, group_a=GROUP_A)
    gbci.fix_spin_(ss=0)
    gbci.fcisolver.nroots = args.nroots
    gbci.fcisolver.conv_tol = args.gbci_conv_tol
    gbci.fcisolver.max_cycle = args.max_cycle
    # The bath-orbital FASSCF is a separate optimization with its own
    # thresholds; an SCF stops only when energy *and* orbital gradient are
    # converged, so both matter for finite-difference smoothness.
    fasscf = gbci._get_fasscf(GROUP_A, gbci.mo_coeff, gbci.ncas,
                              gbci.nelecas, gbci.ncore)
    fasscf.conv_tol = args.fasscf_conv_tol
    fasscf.conv_tol_grad = args.fasscf_conv_tol_grad
    fasscf.max_cycle = args.max_cycle

    gbci.kernel()
    if not np.all(getattr(gbci, "converged", True)):
        raise RuntimeError("GBCI did not converge")
    return gbci


def root_energy(gbci: Any, root: int) -> float:
    values = np.asarray(gbci.e_tot, dtype=float).reshape(-1)
    return float(values[root])


def bath_groups(gbci: Any) -> list[list[int]]:
    return [[int(i) for i in group] for group in gbci._gbci_intermediates["group"]]


def displaced_energy(coords: np.ndarray, reference_groups, args) -> float:
    """Energy at one displaced geometry, with the bath grouping guarded.

    If the grouped-bath membership changes under the displacement the energy
    surface has a kink inside the stencil and the difference quotient is not
    approximating a derivative, so this refuses rather than returning a number.
    """
    gbci = run_gbci(build_molecule(coords, unit="Bohr", args=args), args)
    if bath_groups(gbci) != reference_groups:
        raise RuntimeError(
            "grouped-bath membership changed under displacement; the finite "
            "difference would straddle a kink in the energy surface"
        )
    return root_energy(gbci, args.target_root)


def stencil(order: int):
    """(multiple of h, weight) pairs; weights already divided by h."""
    if order == 2:
        return ((1, 0.5), (-1, -0.5))
    return ((-2, 1.0 / 12.0), (-1, -8.0 / 12.0), (1, 8.0 / 12.0), (2, -1.0 / 12.0))


def numerical_gradient(coords0, reference_groups, args, coordinate_list):
    """Central-difference the selected coordinates at one step size."""
    gradient = np.full((len(SYMBOLS), 3), np.nan)
    for atom, axis in coordinate_list:
        total = 0.0
        for multiple, weight in stencil(args.fd_order):
            shifted = coords0.copy()
            shifted[atom, axis] += multiple * args.step
            total += weight * displaced_energy(shifted, reference_groups, args) / args.step
        gradient[atom, axis] = total
        print(f"    {SYMBOLS[atom]}{atom} {AXES[axis]}  G_num = {total: .12e}", flush=True)
    return gradient


def compare(args) -> dict[str, object]:
    coords_ang = reference_geometry()
    mol = build_molecule(coords_ang, unit="Angstrom", args=args)
    coords0 = mol.atom_coords(unit="Bohr")

    print("Reference calculation")
    gbci = run_gbci(mol, args)
    reference_groups = bath_groups(gbci)
    energy = root_energy(gbci, args.target_root)
    spin_sq, _ = gbci.fcisolver.spin_square(gbci.ci, gbci.ncas, gbci.nelecas)

    analytic = np.asarray(
        gbci.nuc_grad_method().kernel(state=args.target_root), dtype=float
    )
    net_force, torque = gradient_invariance(mol, analytic)
    print(f"  E = {energy:.12f} Eh   <S^2> = {float(spin_sq):.2e}")
    print(f"  ncore = {gbci.ncore}, active MO "
          f"{gbci.ncore + 1}..{gbci.ncore + NCAS}, {len(reference_groups)} bath groups")
    print(f"  net force = {net_force:.3e} Eh/bohr   torque = {torque:.3e} Eh")

    rotational = None
    if not args.no_rotational_check:
        print("  rotational consistency (dE/dtheta vs n.tau, no Cartesian FD)")
        rotational = rotational_consistency(
            lambda c: root_energy(run_gbci(build_molecule(c, unit="Bohr", args=args), args),
                                  args.target_root),
            mol, analytic, step=args.rotation_step,
        )
        for key, value in rotational.items():
            print(f"    {key:<26} {value:+.6e}")

    coordinate_list = [(a, x) for a in args.atoms for x in args.axes]
    results = []
    if not args.analytic_only:
        for step in args.steps:
            args.step = step
            print(f"\nh = {step:.3e} bohr   ({args.fd_order}-point stencil, "
                  f"{len(coordinate_list) * len(stencil(args.fd_order))} energies)")
            # A step that trips the bath guard or fails to converge should cost
            # that step, not the whole sweep -- the analytic gradient above is
            # the expensive part and is already paid for.
            try:
                numerical = numerical_gradient(coords0, reference_groups, args,
                                               coordinate_list)
            except (RuntimeError, np.linalg.LinAlgError) as error:
                print(f"  skipped: {error}")
                results.append({"step_bohr": step, "error": str(error)})
                continue
            diff = analytic - numerical
            finite = ~np.isnan(numerical)
            results.append({
                "step_bohr": step,
                "numerical_gradient_hartree_per_bohr": numerical.tolist(),
                "difference_hartree_per_bohr": diff.tolist(),
                "max_abs_error_hartree_per_bohr": float(np.nanmax(np.abs(diff[finite]))),
                "rms_error_hartree_per_bohr": float(np.sqrt(np.nanmean(diff[finite] ** 2))),
            })
            print(f"  max |diff| = {results[-1]['max_abs_error_hartree_per_bohr']:.3e}   "
                  f"rms = {results[-1]['rms_error_hartree_per_bohr']:.3e}")

    return {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pyscf_version": pyscf.__version__,
        "gbci_gradient_module": str(Path(
            __import__("pyscf.grad.gbci", fromlist=["x"]).__file__).resolve()),
        "molecule": {
            "name": "SO2 (distorted, C1)",
            "geometry_angstrom": [[s, *map(float, c)]
                                  for s, c in zip(SYMBOLS, coords_ang)],
            "internal_coordinates": {"r_S_O1": R1_ANG, "r_S_O2": R2_ANG,
                                     "angle_deg": ANGLE_DEG, "oop_offset": OOP_ANG},
            "basis": args.basis, "charge": CHARGE, "spin": SPIN,
        },
        "calculation": {
            "method": "RHF / singlet GBCI",
            "ncas": NCAS, "nelecas": list(NELECAS),
            "ncore": int(gbci.ncore),
            "active_mo_indices": list(range(gbci.ncore + 1, gbci.ncore + NCAS + 1)),
            "group_a": GROUP_A, "n_bath_groups": len(reference_groups),
            "reference_groups": reference_groups,
            "fd_order": args.fd_order,
            "steps_bohr": list(args.steps),
            "coordinates": [[SYMBOLS[a], AXES[x]] for a, x in coordinate_list],
            "scf_conv_tol": args.scf_conv_tol,
            "scf_conv_tol_grad": args.scf_conv_tol_grad,
            "fasscf_conv_tol": args.fasscf_conv_tol,
            "fasscf_conv_tol_grad": args.fasscf_conv_tol_grad,
            "gbci_conv_tol": args.gbci_conv_tol,
            "max_cycle": args.max_cycle,
            "scf_damping": args.scf_damping,
        },
        "reference_energy_hartree": energy,
        "spin_square": float(spin_sq),
        "analytic_gradient_hartree_per_bohr": analytic.tolist(),
        "diagnostics": {
            "analytic_net_force_hartree_per_bohr": net_force,
            "analytic_torque_hartree": torque,
            "rotational_consistency": rotational,
        },
        "sweep": results,
    }


def report(result: dict[str, object]) -> None:
    analytic = np.asarray(result["analytic_gradient_hartree_per_bohr"])
    sweep = [e for e in result["sweep"] if "error" not in e]
    for entry in result["sweep"]:
        if "error" in entry:
            print(f"\n  h = {entry['step_bohr']:.3e} skipped: {entry['error']}")
    if not sweep:
        return

    print("\n" + "=" * 62)
    print("Step-size sweep")
    print("=" * 62)
    print(f"{'h (bohr)':>12}{'max |diff|':>18}{'rms':>18}")
    for entry in sweep:
        print(f"{entry['step_bohr']:>12.3e}"
              f"{entry['max_abs_error_hartree_per_bohr']:>18.3e}"
              f"{entry['rms_error_hartree_per_bohr']:>18.3e}")

    best = min(sweep, key=lambda e: e["max_abs_error_hartree_per_bohr"])
    numerical = np.asarray(best["numerical_gradient_hartree_per_bohr"])
    print(f"\nPer-component comparison at the best step, "
          f"h = {best['step_bohr']:.3e} bohr")
    print("-" * 74)
    print(f"{'atom':>5}{'axis':>6}{'analytic':>22}{'numerical':>22}{'diff':>13}")
    for atom in range(analytic.shape[0]):
        for axis in range(3):
            if np.isnan(numerical[atom, axis]):
                continue
            print(f"{SYMBOLS[atom] + str(atom):>5}{AXES[axis]:>6}"
                  f"{analytic[atom, axis]:>22.12e}{numerical[atom, axis]:>22.12e}"
                  f"{analytic[atom, axis] - numerical[atom, axis]:>13.2e}")


def save(result: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    analytic = np.asarray(result["analytic_gradient_hartree_per_bohr"])
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["atom", "axis", "analytic_Eh_per_bohr"]
        usable = [e for e in result["sweep"] if "error" not in e]
        for entry in usable:
            header += [f"numerical_h={entry['step_bohr']:.3e}",
                       f"diff_h={entry['step_bohr']:.3e}"]
        writer.writerow(header)
        for atom in range(analytic.shape[0]):
            for axis in range(3):
                row = [SYMBOLS[atom] + str(atom), AXES[axis],
                       f"{analytic[atom, axis]:.12e}"]
                for entry in usable:
                    num = np.asarray(entry["numerical_gradient_hartree_per_bohr"])
                    dif = np.asarray(entry["difference_hartree_per_bohr"])
                    row += [f"{num[atom, axis]:.12e}", f"{dif[atom, axis]:.6e}"]
                writer.writerow(row)
    print(f"\nSaved {output} and {csv_path}")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def step_list(value: str) -> tuple[float, ...]:
    steps = tuple(positive_float(v) for v in value.replace(",", " ").split())
    if not steps:
        raise argparse.ArgumentTypeError("give at least one step size")
    return steps


def index_list(value: str, limit: int, names) -> tuple[int, ...]:
    out = []
    for token in value.replace(",", " ").split():
        if token in names:
            out.append(names.index(token))
        else:
            out.append(int(token))
    if any(i < 0 or i >= limit for i in out):
        raise argparse.ArgumentTypeError(f"indices must be in 0..{limit - 1}")
    return tuple(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep finite-difference step sizes for the SO2 "
                    "GBCI(8e,5o) analytic gradient.")
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--steps", type=step_list, default=DEFAULT_STEPS,
                        help="comma-separated step sizes in bohr")
    parser.add_argument("--fd-order", type=int, choices=(2, 4), default=2,
                        help="2 costs 2 energies per coordinate, 4 costs 4 and "
                             "removes the leading h^2 term but amplifies energy "
                             "noise three times as much")
    parser.add_argument("--atoms", default="0,1,2",
                        help="atom indices to differentiate (default all)")
    parser.add_argument("--axes", default="x,y,z")
    parser.add_argument("--scf-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--scf-conv-tol-grad", type=positive_float, default=1.0e-9)
    parser.add_argument("--fasscf-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--fasscf-conv-tol-grad", type=positive_float, default=1.0e-12)
    parser.add_argument("--gbci-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--max-cycle", type=int, default=400)
    parser.add_argument("--nroots", type=int, default=NROOTS)
    parser.add_argument("--target-root", type=int, default=TARGET_ROOT)
    parser.add_argument("--scf-damping", action="store_true",
                        help="enable damp=0.4 / level_shift=0.5 on the RHF")
    parser.add_argument("--rotation-step", type=positive_float, default=1.0e-3)
    parser.add_argument("--no-rotational-check", action="store_true")
    parser.add_argument("--analytic-only", action="store_true",
                        help="diagnostics only; skip the displaced energies")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=SCRIPT_PATH.parent / "outputs" / "cas8e5o_fd_sweep.json")
    args = parser.parse_args()
    args.atoms = index_list(args.atoms, len(SYMBOLS), list(SYMBOLS))
    args.axes = index_list(args.axes, 3, list(AXES))
    args.step = args.steps[0]
    if args.target_root >= args.nroots:
        parser.error("--target-root must be smaller than --nroots")
    return args


def main() -> None:
    args = parse_args()
    result = compare(args)
    report(result)
    save(result, args.output)


if __name__ == "__main__":
    main()
