"""Compare analytic and finite-difference GBCI gradients for LiH, LiF, and LiCl.

Each molecule uses a sigma-only active space: CAS(2e,2o) for LiH and CAS(4e,4o)
for LiF and LiCl.  That is the point of this script.  The default contiguous
window ``[ncore, ncore + ncas)`` picks the halogen pi lone pairs for LiF and
LiCl -- spectators to the Li-X bond -- so the gradient would be validated on an
active space with no bonding character.  The explicit ``ACT_LIST`` below replaces
those pi orbitals with the sigma framework, which is what the Li-X bond is made
of and what the gradient should be exercised on.

The indices come from ``li_halides_orbital_character.py``; rerun it before
changing the geometry or the basis, since MO ordering is sensitive to both.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
import pyscf.grad
from pyscf import gto, lib, scf


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]

# Both pyscf-forge and this repo ship a pyscf/grad/gbci.py.  This script targets
# the pyscf-forge one, reached through PYSCF_EXT_PATH: the local repo copy
# predates the 2026-09-30 fixes (``get_X`` taking ``ref_mo``, ``Gradients.kernel``
# deriving ``ref_mo_energy`` from diag(C^T F C)), which are exactly what makes a
# non-default act_list safe.  Fail loudly rather than measure the wrong module.
gbci_module = importlib.import_module("pyscf.gbci.gbci")
gbci_grad = importlib.import_module("pyscf.grad.gbci")
GBCI = gbci_module.GBCI

if REPO_ROOT in Path(gbci_grad.__file__).resolve().parents:
    raise RuntimeError(
        f"pyscf.grad.gbci resolved to the local repo copy ({gbci_grad.__file__}); "
        "set PYSCF_EXT_PATH to pyscf-forge so the fixed implementation is used"
    )

# Load utils/gbci_setup.py by path.  Putting REPO_ROOT on sys.path would make
# pyscf treat the repository's own pyscf/ tree as a plugin, which shadows
# pyscf-forge's pyscf.grad.gbci with the stale local copy.
_spec = importlib.util.spec_from_file_location(
    "_gbci_setup", REPO_ROOT / "utils" / "gbci_setup.py")
_gbci_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gbci_setup)
apply_act_list = _gbci_setup.apply_act_list
check_active_window = _gbci_setup.check_active_window
gradient_invariance = _gbci_setup.gradient_invariance
is_contiguous_window = _gbci_setup.is_contiguous_window


# ============================================================
# User settings
# ============================================================

BASIS = "ccpvdz"
ACT_BASE = 1

# Li-centred grouped bath.  "threshold" is the largest Lowdin population
# difference (in active electrons on Li) allowed between two configurations in
# the same group; GBCI's own default is 0.2, so this has to be stated to get 0.4.
# It is not cosmetic: at 0.4 LiF and LiCl merge two groups into one (6 -> 5) and
# their energies move by 1.2e-08 and 1.9e-07 Eh.
LOWDIN_THRESHOLD = 0.4
GROUP_A = {"atom": [0], "threshold": LOWDIN_THRESHOLD}


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    partner: str
    bond_length: float  # Angstrom
    ncas: int
    nelecas: tuple[int, int]
    act_list: tuple[int, ...]  # 1-based, ascending, occupied members first
    act_comment: str
    group_a: dict[str, list[int]] = field(default_factory=lambda: dict(GROUP_A))


# One bond length for all three, deliberately away from equilibrium (LiH 1.595,
# LiF 1.564, LiCl 2.021 A).  A stationary point would drive both the analytic and
# the numerical gradient towards zero, which weakens the comparison; off
# equilibrium every component is large enough to test.  Re-run
# li_halides_orbital_character.py if this changes -- the act_list below is read
# off that script and MO ordering is geometry-sensitive.
BOND_LENGTH = 1.50

MOLECULES = [
    MoleculeSpec(
        "LiH", "H", BOND_LENGTH, 2, (1, 1), act_list=(2, 3),
        act_comment="2: sigma(Li2s/2pz-H1s); 3: sigma*(Li3s/3pz). "
                    "Same as the default window for this molecule.",
    ),
    MoleculeSpec(
        "LiF", "F", BOND_LENGTH, 4, (2, 2), act_list=(3, 4, 7, 10),
        act_comment="3: sigma(F2s); 4: sigma(F2pz-Li); 7: sigma*(Li3s); "
                    "10: sigma*(Li3pz).  Default window would be [5,6,7,8], "
                    "whose MOs 5 and 6 are the F 2p pi lone pairs.",
    ),
    MoleculeSpec(
        "LiCl", "Cl", BOND_LENGTH, 4, (2, 2), act_list=(7, 8, 11, 14),
        act_comment="7: sigma(Cl3s); 8: sigma(Cl3pz-Li); 11: sigma*(Li3s); "
                    "14: sigma*(Li3pz).  Default window would be [9,10,11,12], "
                    "whose MOs 9 and 10 are the Cl 3p pi lone pairs.",
    ),
]

# Tilt applied to the second reference geometry, in degrees about the y axis.
# On the z axis the transverse gradient components vanish by symmetry, so the
# rotational-invariance check would be satisfied by construction; tilting makes
# every Cartesian component nonzero and the torque a real test.
TILT_DEGREES = 30.0

DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "li_halides_gbci_grad_check.json"


# ============================================================
# Molecule and GBCI calculation
# ============================================================

def build_molecule(
    spec: MoleculeSpec,
    coords: np.ndarray,
    *,
    unit: str,
    basis: str,
    verbose: int,
) -> gto.Mole:
    """Build the neutral closed-shell molecule at the given coordinates."""
    symbols = ("Li", spec.partner)
    return gto.M(
        atom=[(symbol, tuple(coord)) for symbol, coord in zip(symbols, coords)],
        basis=basis,
        charge=0,
        spin=0,
        unit=unit,
        symmetry=False,
        verbose=verbose,
    )


def reference_coords(spec: MoleculeSpec, tilt_degrees: float = 0.0) -> np.ndarray:
    """Reference geometry in Angstrom, optionally tilted about the y axis."""
    angle = np.deg2rad(tilt_degrees)
    bond = np.array([np.sin(angle), 0.0, np.cos(angle)]) * spec.bond_length
    return np.array([[0.0, 0.0, 0.0], bond])


def scalar_root_energy(energy: Any) -> float:
    """Return the only requested GBCI root as a scalar."""
    values = np.asarray(energy, dtype=float).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one GBCI root, received {values.size}")
    return float(values[0])


def normalized_groups(groups: Any) -> list[list[int]]:
    """Convert the grouped-bath indices to JSON-comparable Python lists."""
    return [[int(index) for index in group] for group in groups]


def run_gbci(spec: MoleculeSpec, mol: gto.Mole, args: argparse.Namespace) -> Any:
    """Run RHF, install the sigma active space, and run GBCI."""
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv_tol
    # Set explicitly: pyscf otherwise derives it as sqrt(conv_tol), which at
    # conv_tol=1e-14 would be 1e-7.  An SCF stops only when the energy *and* the
    # orbital gradient criteria are both met, so this is the one that decides
    # how early it stops -- on SO2, loosening it from 1e-9 to 1e-6 moved the
    # converged energy by 8e-10 Eh and raised the finite-difference floor 20x.
    mf.conv_tol_grad = args.scf_conv_tol_grad
    mf.max_cycle = args.max_cycle
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{spec.name}: RHF did not converge")

    # The probe only supplies ncore/ncas to sort_mo.  Reordering has to happen
    # before the real GBCI object is built, so that gbci.mo_coeff and
    # gbci._scf.mo_coeff carry the same ordering.
    probe = GBCI(mf, ncas=spec.ncas, nelecas=spec.nelecas, group_a=spec.group_a)
    reordered = not is_contiguous_window(probe, spec.act_list, base=ACT_BASE)
    if reordered:
        apply_act_list(mf, probe, list(spec.act_list), base=ACT_BASE)

    gbci = GBCI(mf, ncas=spec.ncas, nelecas=spec.nelecas, group_a=spec.group_a)
    check_active_window(gbci)

    fasscf = gbci._get_fasscf(
        gbci.group_a, gbci.mo_coeff, gbci.ncas, gbci.nelecas, gbci.ncore
    )
    fasscf.conv_tol = args.fasscf_conv_tol
    fasscf.conv_tol_grad = args.fasscf_conv_tol_grad
    fasscf.max_cycle = args.max_cycle

    gbci.fcisolver.nroots = 1
    gbci.fcisolver.conv_tol = args.gbci_conv_tol
    gbci.fcisolver.max_cycle = args.max_cycle

    gbci.kernel()
    if not gbci.converged:
        raise RuntimeError(f"{spec.name}: GBCI did not converge")
    gbci._reordered_mo = reordered
    return gbci


def active_block(gbci: Any) -> np.ndarray:
    """The active-window columns of the GBCI reference orbitals."""
    return np.asarray(gbci.mo_coeff)[:, gbci.ncore:gbci.ncore + gbci.ncas]


def active_overlap_diagonal(reference: np.ndarray, gbci: Any, mol: gto.Mole) -> float:
    """Smallest |<phi_ref|S|phi_displaced>| over the active orbitals.

    The hardcoded act_list selects orbitals by index, so the finite difference is
    only meaningful while the displaced geometries keep the same orbitals in the
    active window.  The displaced overlap matrix is used for both sides, which is
    accurate to O(h) and far tighter than the 0.9 threshold needs.
    """
    overlap = reference.T @ mol.intor("int1e_ovlp") @ active_block(gbci)
    return float(np.min(np.abs(np.diag(overlap))))


# ============================================================
# Gradients
# ============================================================

def analytic_gradient(gbci: Any) -> np.ndarray:
    """Analytic GBCI gradient of the single requested root."""
    return np.asarray(gbci.nuc_grad_method().kernel(state=0), dtype=float)


def numerical_z_gradient(
    spec: MoleculeSpec,
    coords_bohr: np.ndarray,
    reference_groups: list[list[int]],
    reference_active: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Central-difference both atoms along z and guard the objective."""
    gradient = np.empty(len(coords_bohr))
    records: list[dict[str, object]] = []
    min_overlap = 1.0

    for atom_index in range(len(coords_bohr)):
        energies = {}
        for sign, key in ((-1.0, "minus"), (+1.0, "plus")):
            displaced = coords_bohr.copy()
            displaced[atom_index, 2] += sign * args.step_bohr
            mol = build_molecule(
                spec, displaced, unit="Bohr", basis=args.basis, verbose=args.verbose
            )
            gbci = run_gbci(spec, mol, args)

            groups = normalized_groups(gbci._gbci_intermediates["group"])
            if groups != reference_groups:
                raise RuntimeError(
                    f"{spec.name}: grouped-bath membership changed under the "
                    f"{key} displacement of atom {atom_index}; the numerical "
                    "derivative would compare different objectives"
                )
            overlap = active_overlap_diagonal(reference_active, gbci, mol)
            if overlap < args.active_overlap_thresh:
                raise RuntimeError(
                    f"{spec.name}: active orbital overlap with the reference "
                    f"dropped to {overlap:.4f} under the {key} displacement of "
                    f"atom {atom_index}; act_list no longer selects the same "
                    "sigma orbitals"
                )
            min_overlap = min(min_overlap, overlap)
            energies[key] = scalar_root_energy(gbci.e_tot)

        gradient[atom_index] = (
            energies["plus"] - energies["minus"]
        ) / (2.0 * args.step_bohr)
        records.append(
            {
                "atom_index": atom_index,
                "energy_minus_hartree": energies["minus"],
                "energy_plus_hartree": energies["plus"],
            }
        )

    for record in records:
        record["min_active_overlap"] = min_overlap
    return gradient, records


# ============================================================
# Driver
# ============================================================

def check_molecule(spec: MoleculeSpec, args: argparse.Namespace) -> dict[str, object]:
    """Run every check for one molecule and collect the diagnostics."""
    on_axis = build_molecule(
        spec, reference_coords(spec), unit="Angstrom",
        basis=args.basis, verbose=args.verbose,
    )
    coords_bohr = on_axis.atom_coords(unit="Bohr")
    gbci = run_gbci(spec, on_axis, args)
    reference_groups = normalized_groups(gbci._gbci_intermediates["group"])
    reference_active = active_block(gbci).copy()
    reference_energy = scalar_root_energy(gbci.e_tot)

    analytic = analytic_gradient(gbci)
    net_force, torque = gradient_invariance(on_axis, analytic)

    # Tilted copy: on the z axis the transverse components and hence the torque
    # vanish by symmetry, so the invariance test only bites off the axis.
    tilted = build_molecule(
        spec, reference_coords(spec, TILT_DEGREES), unit="Angstrom",
        basis=args.basis, verbose=args.verbose,
    )
    tilted_gbci = run_gbci(spec, tilted, args)
    tilted_analytic = analytic_gradient(tilted_gbci)
    tilted_net_force, tilted_torque = gradient_invariance(tilted, tilted_analytic)
    tilted_energy = scalar_root_energy(tilted_gbci.e_tot)

    numerical_z, displaced_records = numerical_z_gradient(
        spec, coords_bohr, reference_groups, reference_active, args
    )

    difference = analytic[:, 2] - numerical_z
    max_abs_z_error = float(np.max(np.abs(difference)))
    max_abs_transverse = float(np.max(np.abs(analytic[:, :2])))
    rotational_energy_error = abs(tilted_energy - reference_energy)
    passed = bool(
        max_abs_z_error <= args.atol
        and max_abs_transverse <= args.transverse_atol
        and tilted_torque <= args.torque_atol
        and rotational_energy_error <= args.rotation_energy_atol
    )

    return {
        "molecule": {
            "name": spec.name,
            "geometry_angstrom": [
                [symbol, *map(float, coord)]
                for symbol, coord in zip(("Li", spec.partner), reference_coords(spec))
            ],
            "bond_length_angstrom": spec.bond_length,
            "charge": 0,
            "spin": 0,
            "basis": args.basis,
            "unit": "Angstrom",
        },
        "calculation": {
            "method": "RHF / GBCI",
            "ncas": spec.ncas,
            "nelecas": list(spec.nelecas),
            "act_list": list(spec.act_list),
            "act_base": ACT_BASE,
            "act_list_character": spec.act_comment,
            "reordered_from_default_window": bool(gbci._reordered_mo),
            "group_a": spec.group_a,
            "lowdin_threshold": spec.group_a.get("threshold"),
            "reference_groups": reference_groups,
            "scf_conv_tol": args.scf_conv_tol,
            "scf_conv_tol_grad": args.scf_conv_tol_grad,
            "fasscf_conv_tol": args.fasscf_conv_tol,
            "fasscf_conv_tol_grad": args.fasscf_conv_tol_grad,
            "gbci_conv_tol": args.gbci_conv_tol,
            "max_cycle": args.max_cycle,
            "finite_difference": "two-point central difference",
            "step_bohr": args.step_bohr,
            "tilt_degrees": TILT_DEGREES,
        },
        "reference_energy_hartree": reference_energy,
        "tilted_energy_hartree": tilted_energy,
        "analytic_gradient_hartree_per_bohr": analytic.tolist(),
        "tilted_analytic_gradient_hartree_per_bohr": tilted_analytic.tolist(),
        "numerical_z_gradient_hartree_per_bohr": numerical_z.tolist(),
        "analytic_z_minus_numerical_hartree_per_bohr": difference.tolist(),
        "displaced_calculations": displaced_records,
        "diagnostics": {
            "max_abs_z_error_hartree_per_bohr": max_abs_z_error,
            "max_abs_analytic_transverse_hartree_per_bohr": max_abs_transverse,
            "net_force_norm_hartree_per_bohr": net_force,
            "torque_hartree": torque,
            "tilted_net_force_norm_hartree_per_bohr": tilted_net_force,
            "tilted_torque_hartree": tilted_torque,
            "rotational_energy_error_hartree": rotational_energy_error,
            "min_active_overlap": displaced_records[0]["min_active_overlap"],
            "passed": passed,
        },
    }


def print_result(result: dict[str, object]) -> None:
    """Print a compact gradient comparison for one molecule."""
    name = result["molecule"]["name"]
    partner = result["molecule"]["geometry_angstrom"][1][0]
    calculation = result["calculation"]
    analytic = np.asarray(result["analytic_gradient_hartree_per_bohr"])
    numerical = np.asarray(result["numerical_z_gradient_hartree_per_bohr"])
    difference = np.asarray(result["analytic_z_minus_numerical_hartree_per_bohr"])
    diagnostics = result["diagnostics"]

    print(f"\n{'=' * 78}")
    print(
        f"{name}  CAS({sum(calculation['nelecas'])}e,{calculation['ncas']}o)  "
        f"act_list={calculation['act_list']} (1-based)  basis={result['molecule']['basis']}"
    )
    print(f"  {calculation['act_list_character']}")
    print(f"  E(GBCI) = {result['reference_energy_hartree']:.10f} Eh")
    print(f"{'=' * 78}")
    print(" atom       analytic z          numerical z          difference")
    for atom_index, symbol in enumerate(("Li", partner)):
        print(
            f" {symbol:>2s}  "
            f"{analytic[atom_index, 2]:+20.12e}  "
            f"{numerical[atom_index]:+20.12e}  "
            f"{difference[atom_index]:+14.6e}"
        )
    print(f" max |analytic z - numerical z| : "
          f"{diagnostics['max_abs_z_error_hartree_per_bohr']:.6e} Eh/Bohr")
    print(f" max |analytic transverse|      : "
          f"{diagnostics['max_abs_analytic_transverse_hartree_per_bohr']:.6e} Eh/Bohr")
    print(f" net force (on axis / tilted)   : "
          f"{diagnostics['net_force_norm_hartree_per_bohr']:.6e} / "
          f"{diagnostics['tilted_net_force_norm_hartree_per_bohr']:.6e} Eh/Bohr")
    print(f" torque    (on axis / tilted)   : "
          f"{diagnostics['torque_hartree']:.6e} / "
          f"{diagnostics['tilted_torque_hartree']:.6e} Eh")
    print(f" |E(tilted) - E(on axis)|       : "
          f"{diagnostics['rotational_energy_error_hartree']:.6e} Eh")
    print(f" min active-orbital overlap     : {diagnostics['min_active_overlap']:.6f}")
    print(f" Result: {'PASS' if diagnostics['passed'] else 'FAIL'}")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare analytic and central-difference GBCI gradients for LiH, "
            "LiF, and LiCl on sigma-only active spaces."
        )
    )
    parser.add_argument("--basis", default=BASIS)
    # 1e-4 bohr, not the 2.5e-4 of test/SCRIPT_PATTERNS.md: that optimum was
    # measured on SO2, and these three molecules have a lower energy noise floor.
    # Measured max |analytic - numerical|, in Eh/bohr:
    #
    #   h (bohr)   LiH        LiF        LiCl
    #   1e-3                             3.7e-08
    #   5e-4                             1.0e-08
    #   2.5e-4     1.3e-09    6.5e-09    1.7e-07   <- one Li point lands 4e-11 Eh
    #   1e-4       2.4e-10    2.7e-09    9.6e-10      off, which 1/(2h) magnifies
    #
    # The LiCl outlier at 2.5e-4 reproduces bit-for-bit and disappears on either
    # side of it, so it is a converged-solution artifact at one displaced
    # geometry rather than gradient error: the analytic gradient is exactly
    # antisymmetric there and the torque stays at 1e-14.
    parser.add_argument("--step-bohr", type=positive_float, default=1.0e-4)
    parser.add_argument("--atol", type=positive_float, default=1.0e-5)
    parser.add_argument("--transverse-atol", type=positive_float, default=1.0e-8)
    parser.add_argument("--torque-atol", type=positive_float, default=1.0e-6)
    parser.add_argument("--rotation-energy-atol", type=positive_float, default=1.0e-8)
    parser.add_argument("--active-overlap-thresh", type=positive_float, default=0.9)
    # Tolerances are shared with the SO2 sweep
    # (test/so2/gbci_so2_grad_fd_sweep.py) except for the two energy
    # thresholds, which stay at 1e-12 here.  SO2's 1e-14 is below what these
    # molecules can resolve in double precision -- LiCl sits at -467 Eh, so
    # 1e-14 is a relative 2e-17 -- and an unreachable threshold makes the SCF
    # and the Davidson stop on numerical noise instead of converging.  Measured:
    # raising both to 1e-14 multiplied the energy noise by 10-20x (LiCl dE
    # 2e-13 -> 3.6e-12) and the finite-difference floor with it.  The orbital
    # gradient is stated rather than left to pyscf's sqrt(conv_tol) = 1e-6,
    # because that is the criterion that actually decides when the SCF stops.
    parser.add_argument("--scf-conv-tol", type=positive_float, default=1.0e-12)
    parser.add_argument("--scf-conv-tol-grad", type=positive_float, default=1.0e-9)
    parser.add_argument("--fasscf-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--fasscf-conv-tol-grad", type=positive_float, default=1.0e-12)
    parser.add_argument("--gbci-conv-tol", type=positive_float, default=1.0e-12)
    parser.add_argument("--max-cycle", type=int, default=400)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--molecule", action="append", default=None,
                        help="Restrict to one molecule; repeatable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-assert", action="store_true",
                        help="Report a failed comparison without a nonzero exit status.")
    return parser.parse_args()


def main() -> None:
    # Threaded BLAS moves the energy by ~1e-9 Eh between runs, which a central
    # difference amplifies by 1/(2h).  The finite-difference reference needs one
    # thread; see the finite-difference notes in test/SCRIPT_PATTERNS.md.
    lib.num_threads(1)

    args = parse_args()
    selected = MOLECULES
    if args.molecule:
        wanted = {name.lower() for name in args.molecule}
        selected = [spec for spec in MOLECULES if spec.name.lower() in wanted]
        if not selected:
            raise SystemExit(f"No molecule matched {args.molecule}")

    results = [check_molecule(spec, args) for spec in selected]
    for result in results:
        print_result(result)

    payload = {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pyscf_version": pyscf.__version__,
        "modules": {
            "gbci": str(Path(gbci_module.__file__).resolve()),
            "gbci_gradient": str(Path(gbci_grad.__file__).resolve()),
        },
        "num_threads": lib.num_threads(),
        "results": results,
        "passed": all(result["diagnostics"]["passed"] for result in results),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")

    if not payload["passed"] and not args.no_assert:
        failed = [
            result["molecule"]["name"]
            for result in results
            if not result["diagnostics"]["passed"]
        ]
        raise AssertionError(
            f"GBCI analytic gradient check failed for: {', '.join(failed)}"
        )


if __name__ == "__main__":
    main()
