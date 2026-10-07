"""Compare analytic and finite-difference GBCI gradients for a distorted SO2.

The reference geometry is deliberately asymmetric (C1): the two S-O bonds have
different lengths, the valence angle is off equilibrium, and one oxygen carries
an out-of-plane offset.  No symmetry element survives, so all nine Cartesian
gradient components are nonzero and an error in the analytic gradient cannot be
masked by a symmetry-enforced cancellation.

Unlike the tracking-optimizer based check in ``test/analytic_check.py`` the
active space here is fixed by an explicit ``act_list`` and the root index is
fixed, so every finite-difference point differentiates the same objective.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyscf
import pyscf.grad
from pyscf import gto, scf
from pyscf.mcscf.addons import sort_mo


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]

# The repo root holds the top-level ``utils`` package used below.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Both pyscf-forge and this repo ship a pyscf/grad/gbci.py.  This script targets
# the pyscf-forge one, which is reached through PYSCF_EXT_PATH; the local repo
# copy is deliberately NOT prepended to pyscf.grad.__path__.  Fail loudly rather
# than silently measuring the wrong module.
gbci_module = importlib.import_module("pyscf.gbci.gbci")
gbci_grad = importlib.import_module("pyscf.grad.gbci")
GBCI = gbci_module.GBCI

if REPO_ROOT in Path(gbci_grad.__file__).resolve().parents:
    raise RuntimeError(
        f"pyscf.grad.gbci resolved to the local repo copy ({gbci_grad.__file__}); "
        "set PYSCF_EXT_PATH to pyscf-forge so the forge implementation is used"
    )

from utils.group_a import legacy_group_a_to_dict


# ============================================================
# User settings
# ============================================================

SYMBOLS = ("S", "O", "O")
BASIS = "ccpvdz"
CHARGE = 0
SPIN = 0

NCAS = 4
NELECAS = (2, 2)
# ACT_LIST = None keeps the canonical RHF orbitals untouched, so the active space
# is the default contiguous window [ncore, ncore + NCAS) straddling the HOMO-LUMO
# gap.  That matters: the GBCI gradient driver loses rotational invariance as soon
# as sort_mo reorders the MOs, even for a permutation of the same active space
# (torque 9.9e-4 -> 3.4e-1 Eh on this geometry), so any explicit act_list makes
# the analytic gradient untrustworthy rather than merely selecting a different
# active space.  Pass --act-list only to reproduce that failure deliberately.
ACT_LIST = None
ACT_BASE = 1

GROUP_A = "S"  # AO label, translated to {"atom": [[0]]} by utils.group_a

NROOTS = 1
TARGET_ROOT = 0

# Distorted C1 geometry.  Equilibrium SO2 is r = 1.431 A, angle = 119.5 deg.
R1_ANG = 1.38
R2_ANG = 1.52
ANGLE_DEG = 112.0
OOP_ANG = 0.08  # out-of-plane offset on the second oxygen

DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "gbci_so2_grad_check.json"


def distorted_geometry() -> np.ndarray:
    """Return the asymmetric SO2 reference geometry in Angstrom."""
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


REFERENCE_COORDS_ANGSTROM = distorted_geometry()


# ============================================================
# Molecule and GBCI calculation
# ============================================================

def build_molecule(
    coords: np.ndarray,
    *,
    unit: str,
    basis: str,
    verbose: int,
) -> gto.Mole:
    """Build closed-shell SO2 at the given Cartesian coordinates."""
    return gto.M(
        atom=[(symbol, tuple(coord)) for symbol, coord in zip(SYMBOLS, coords)],
        basis=basis,
        charge=CHARGE,
        spin=SPIN,
        unit=unit,
        symmetry=False,
        verbose=verbose,
    )


def scalar_root_energy(energy: Any, root: int) -> float:
    """Return the requested GBCI root energy as a scalar."""
    values = np.asarray(energy, dtype=float).reshape(-1)
    if root >= values.size:
        raise ValueError(f"Root {root} requested but only {values.size} were solved")
    return float(values[root])


def root_ci(gbci: Any, root: int) -> np.ndarray:
    """Return the CI vector of one root, whatever the solver's CI layout is."""
    ci = gbci.ci
    if isinstance(ci, (list, tuple)):
        return np.array(ci[root], copy=True)
    ci = np.asarray(ci)
    if ci.ndim == 3:
        return np.array(ci[root], copy=True)
    if root != 0:
        raise ValueError(f"Root {root} requested but the solver returned one CI vector")
    return np.array(ci, copy=True)


def normalized_groups(groups: Any) -> list[list[int]]:
    """Convert the grouped-bath indices to JSON-comparable Python lists."""
    return [[int(index) for index in group] for group in groups]


def run_gbci(
    mol: gto.Mole,
    *,
    act_list: list[int] | None,
    nroots: int,
    scf_conv_tol: float,
    scf_conv_tol_grad: float,
    scf_damping: bool,
    fasscf_conv_tol: float,
    fasscf_conv_tol_grad: float,
    gbci_conv_tol: float,
    max_cycle: int,
) -> Any:
    """Run RHF followed by a singlet GBCI(4e,4o) on a contiguous active window.

    Every tolerance is tight on purpose.  A central difference divides an energy
    difference by 2h, so a residual convergence error in the energy shows up in
    the numerical gradient magnified by 1/(2h); at h = 1e-3 Bohr that is a factor
    of 500.  Loose tolerances put a floor of ~2.5e-6 Eh/Bohr on the comparison
    that shrinking h cannot remove.
    """
    mf = scf.RHF(mol)
    mf.conv_tol = scf_conv_tol
    mf.conv_tol_grad = scf_conv_tol_grad
    mf.max_cycle = max_cycle
    if scf_damping:
        # The settings utils/active_space_tracking_optimizer.py uses for SO2.
        # They help a hard optimization converge but measurably roughen the
        # energy surface (torque 3.5e-11 -> 1.7e-10, and ~9x worse finite
        # differences at h = 1e-5), so this check leaves them off by default.
        mf.damp = 0.4
        mf.level_shift = 0.5
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    gbci = GBCI(
        mf,
        ncas=NCAS,
        nelecas=NELECAS,
        group_a=legacy_group_a_to_dict(GROUP_A, mol),
    )
    gbci.fix_spin_(ss=0)
    gbci.fcisolver.nroots = nroots
    gbci.fcisolver.conv_tol = gbci_conv_tol
    gbci.fcisolver.max_cycle = max_cycle
    if act_list is None:
        gbci.mo_coeff = mf.mo_coeff
    else:
        gbci.mo_coeff = sort_mo(gbci, mf.mo_coeff, act_list, base=ACT_BASE)
    gbci._act_list_is_identity = bool(
        np.allclose(gbci.mo_coeff, mf.mo_coeff, atol=1e-12)
    )

    # The bath-orbital FASSCF is a separate optimization with its own tolerance;
    # it must be configured after mo_coeff is set and before the kernel runs.
    fasscf = gbci._get_fasscf(
        gbci.group_a, gbci.mo_coeff, gbci.ncas, gbci.nelecas, gbci.ncore
    )
    fasscf.conv_tol = fasscf_conv_tol
    # The orbital-gradient threshold matters as much as the energy one: an SCF
    # stops only when both are met, so a loose conv_tol_grad lets the bath
    # orbitals halt at slightly different points from geometry to geometry,
    # which a finite difference amplifies by 1/(2h).
    fasscf.conv_tol_grad = fasscf_conv_tol_grad
    fasscf.max_cycle = max_cycle

    gbci.kernel()
    if not np.all(getattr(gbci, "converged", True)):
        raise RuntimeError("GBCI did not converge")
    return gbci


def active_mo(gbci: Any) -> np.ndarray:
    """Return the active block of the CAS-sorted MO coefficients."""
    return np.asarray(gbci.mo_coeff)[:, gbci.ncore:gbci.ncore + NCAS]


def analytic_gradient(gbci: Any, root: int) -> np.ndarray:
    """Evaluate the GBCI analytic gradient through its public driver.

    The pyscf-forge driver pulls every intermediate, ``ecore_list`` included,
    from the cache the GBCI kernel left behind, so only the root index is needed.
    """
    return np.asarray(gbci.nuc_grad_method().kernel(state=root), dtype=float)


# ============================================================
# Consistency guards
# ============================================================

def active_space_overlap(
    reference_active_mo: np.ndarray,
    mol: gto.Mole,
    gbci: Any,
) -> float:
    """Smallest singular value of the reference/displaced active-space overlap.

    A value near 1 means ``act_list`` selected the same six orbitals at both
    geometries; a small value means the fixed indices picked up a different
    orbital under displacement.
    """
    ovlp = mol.intor("int1e_ovlp")
    matrix = reference_active_mo.T @ ovlp @ active_mo(gbci)
    return float(np.min(np.linalg.svd(matrix, compute_uv=False)))


# ============================================================
# Numerical gradient
# ============================================================

def energy_at_displaced_geometry(
    coords_bohr: np.ndarray,
    reference_groups: list[list[int]],
    reference_active_mo: np.ndarray,
    *,
    root: int,
    args: argparse.Namespace,
    act_list: list[int],
) -> tuple[float, list[list[int]], float]:
    """Return the GBCI energy, bath signature and active overlap at one point."""
    mol = build_molecule(
        coords_bohr, unit="Bohr", basis=args.basis, verbose=args.verbose
    )
    gbci = run_gbci(
        mol,
        act_list=act_list,
        nroots=args.nroots,
        scf_conv_tol=args.scf_conv_tol,
        scf_conv_tol_grad=args.scf_conv_tol_grad,
        scf_damping=args.scf_damping,
        fasscf_conv_tol=args.fasscf_conv_tol,
        fasscf_conv_tol_grad=args.fasscf_conv_tol_grad,
        gbci_conv_tol=args.gbci_conv_tol,
        max_cycle=args.max_cycle,
    )

    groups = normalized_groups(gbci._gbci_intermediates["group"])
    if groups != reference_groups:
        raise RuntimeError(
            "Grouped-bath membership changed under finite displacement; "
            "the numerical derivative would compare different objectives"
        )

    overlap = active_space_overlap(reference_active_mo, mol, gbci)
    return scalar_root_energy(gbci.e_tot, root), groups, overlap


def numerical_gradient(
    reference_coords_bohr: np.ndarray,
    reference_groups: list[list[int]],
    reference_active_mo: np.ndarray,
    *,
    root: int,
    args: argparse.Namespace,
    act_list: list[int] | None,
) -> tuple[np.ndarray, list[dict[str, object]], float]:
    """Central-difference every Cartesian coordinate of every atom.

    ``--fd-order 2`` is the usual two-point stencil.  ``--fd-order 4`` adds the
    +-2h points and cancels the leading h^2 truncation term, which dominates the
    residual once the electronic-structure tolerances are tight.
    """
    if args.fd_order == 2:
        stencil = ((1, 0.5), (-1, -0.5))
    else:
        stencil = ((-2, 1.0 / 12.0), (-1, -8.0 / 12.0),
                   (1, 8.0 / 12.0), (2, -1.0 / 12.0))

    natm = reference_coords_bohr.shape[0]
    gradient = np.zeros((natm, 3))
    records: list[dict[str, object]] = []
    min_overlap = 1.0
    labels = ("x", "y", "z")

    for atom_index in range(natm):
        for axis in range(3):
            derivative = 0.0
            energies: dict[int, float] = {}
            point_overlap = 1.0

            for multiple, weight in stencil:
                coords = reference_coords_bohr.copy()
                coords[atom_index, axis] += multiple * args.step_bohr
                energy, _, overlap = energy_at_displaced_geometry(
                    coords,
                    reference_groups,
                    reference_active_mo,
                    root=root,
                    args=args,
                    act_list=act_list,
                )
                derivative += weight * energy / args.step_bohr
                energies[multiple] = energy
                point_overlap = min(point_overlap, overlap)

            gradient[atom_index, axis] = derivative
            min_overlap = min(min_overlap, point_overlap)

            offsets = "  ".join(
                f"E({multiple:+d}h) = {energies[multiple]:.14f}"
                for multiple, _ in stencil
            )
            print(
                f"FD atom {atom_index:2d} coord {labels[axis]}: {offsets}  "
                f"G_num = {derivative: .10e}",
                flush=True,
            )
            if point_overlap < args.active_overlap_thresh:
                print(
                    f"  WARNING: active-space overlap dropped to "
                    f"{point_overlap:.6f}; the fixed act_list may have selected "
                    "a different orbital here"
                )

            records.append(
                {
                    "atom_index": atom_index,
                    "atom_symbol": SYMBOLS[atom_index],
                    "axis": labels[axis],
                    "energies_hartree": {
                        f"{multiple:+d}h": energies[multiple]
                        for multiple, _ in stencil
                    },
                    "numerical_derivative_hartree_per_bohr": derivative,
                    "min_active_overlap": point_overlap,
                }
            )

    return gradient, records, min_overlap


# ============================================================
# Comparison
# ============================================================

def torque_norm(coords_bohr: np.ndarray, gradient: np.ndarray) -> float:
    """Norm of the net torque, which vanishes for an exact gradient."""
    return float(np.linalg.norm(np.cross(coords_bohr, gradient).sum(axis=0)))


def active_window(gbci: Any, act_list: list[int] | None) -> list[int]:
    """Return the 1-based RHF MO indices that make up the active space."""
    if act_list is not None:
        return list(act_list)
    # No reordering: the active block is the default contiguous window.
    return [gbci.ncore + column + ACT_BASE for column in range(NCAS)]


def active_space_report(
    gbci: Any, act_list: list[int] | None
) -> list[dict[str, object]]:
    """Describe each active orbital so the active space can be verified."""
    mo_energy = np.asarray(gbci._scf.mo_energy, dtype=float)
    mo_occ = np.asarray(gbci._scf.mo_occ, dtype=float)
    ao_labels = gbci.mol.ao_labels()
    active = active_mo(gbci)

    report = []
    for column, mo_index in enumerate(active_window(gbci, act_list)):
        weights = active[:, column] ** 2
        leading = np.argsort(weights)[::-1][:3]
        zero_based = mo_index - ACT_BASE
        report.append(
            {
                "act_list_index": int(mo_index),
                "rhf_mo_index_zero_based": int(zero_based),
                "rhf_mo_energy_hartree": float(mo_energy[zero_based]),
                "rhf_mo_occupancy": float(mo_occ[zero_based]),
                "leading_ao_contributions": [
                    {"ao": ao_labels[i], "weight": float(weights[i])} for i in leading
                ],
            }
        )
    return report


def compare_gradients(args: argparse.Namespace) -> dict[str, object]:
    """Run the analytic and numerical calculations and collect diagnostics."""
    act_list = args.act_list
    root = args.target_root

    reference_mol = build_molecule(
        REFERENCE_COORDS_ANGSTROM,
        unit="Angstrom",
        basis=args.basis,
        verbose=args.verbose,
    )
    reference_coords_bohr = reference_mol.atom_coords(unit="Bohr")

    gbci = run_gbci(
        reference_mol,
        act_list=act_list,
        nroots=args.nroots,
        scf_conv_tol=args.scf_conv_tol,
        scf_conv_tol_grad=args.scf_conv_tol_grad,
        scf_damping=args.scf_damping,
        fasscf_conv_tol=args.fasscf_conv_tol,
        fasscf_conv_tol_grad=args.fasscf_conv_tol_grad,
        gbci_conv_tol=args.gbci_conv_tol,
        max_cycle=args.max_cycle,
    )
    reference_groups = normalized_groups(gbci._gbci_intermediates["group"])
    reference_active_mo = active_mo(gbci).copy()
    act_list_is_identity = bool(gbci._act_list_is_identity)
    if not act_list_is_identity:
        print(
            "\nWARNING: act_list reorders the canonical RHF MOs.  The GBCI\n"
            "         gradient driver splits occupied from virtual orbitals by\n"
            "         column index and indexes mo_energy the same way, so its\n"
            "         orbital-response term is evaluated on the wrong partition\n"
            "         once the active window is not the default one.  Expect the\n"
            "         analytic gradient to violate rotational invariance; check\n"
            "         the reported torque before reading the comparison.\n"
        )
    spin_square, multiplicity = gbci.fcisolver.spin_square(
        root_ci(gbci, root), gbci.ncas, gbci.nelecas
    )

    analytic = analytic_gradient(gbci, root)
    net_force = float(np.linalg.norm(analytic.sum(axis=0)))
    torque = torque_norm(reference_coords_bohr, analytic)

    if args.analytic_only:
        numerical = None
        difference = None
        displaced_records: list[dict[str, object]] = []
        min_overlap = 1.0
        max_abs_error = None
        rms_error = None
        passed = bool(net_force <= args.atol and torque <= args.torque_atol)
    else:
        numerical, displaced_records, min_overlap = numerical_gradient(
            reference_coords_bohr,
            reference_groups,
            reference_active_mo,
            root=root,
            args=args,
            act_list=act_list,
        )
        difference = analytic - numerical
        max_abs_error = float(np.max(np.abs(difference)))
        rms_error = float(np.sqrt(np.mean(difference**2)))
        passed = bool(
            max_abs_error <= args.atol
            and net_force <= args.atol
            and torque <= args.torque_atol
        )

    return {
        "script": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pyscf_version": pyscf.__version__,
        "modules": {
            "gbci": str(Path(gbci_module.__file__).resolve()),
            "gbci_gradient": str(Path(gbci_grad.__file__).resolve()),
        },
        "molecule": {
            "name": "SO2 (distorted, C1)",
            "geometry_angstrom": [
                [symbol, *map(float, coord)]
                for symbol, coord in zip(SYMBOLS, REFERENCE_COORDS_ANGSTROM)
            ],
            "internal_coordinates": {
                "r_S_O1_angstrom": R1_ANG,
                "r_S_O2_angstrom": R2_ANG,
                "angle_O_S_O_degree": ANGLE_DEG,
                "out_of_plane_offset_angstrom": OOP_ANG,
            },
            "charge": CHARGE,
            "spin": SPIN,
            "basis": args.basis,
        },
        "calculation": {
            "method": "RHF / singlet GBCI",
            "ncas": NCAS,
            "nelecas": list(NELECAS),
            "act_list": act_list,
            "act_base": ACT_BASE,
            "active_mo_indices": active_window(gbci, act_list),
            "act_list_is_identity_permutation": act_list_is_identity,
            "group_a": legacy_group_a_to_dict(GROUP_A, reference_mol),
            "reference_groups": reference_groups,
            "nroots": args.nroots,
            "target_root": root,
            "scf_conv_tol": args.scf_conv_tol,
            "scf_damping": args.scf_damping,
            "fasscf_conv_tol_grad": args.fasscf_conv_tol_grad,
            "gbci_conv_tol": args.gbci_conv_tol,
            "max_cycle": args.max_cycle,
            "finite_difference": f"central difference, order {args.fd_order}",
            "fd_order": args.fd_order,
            "step_bohr": args.step_bohr,
            "comparison_atol_hartree_per_bohr": args.atol,
            "torque_atol_hartree": args.torque_atol,
            "analytic_only": args.analytic_only,
        },
        "active_space_report": active_space_report(gbci, act_list),
        "reference_energy_hartree": scalar_root_energy(gbci.e_tot, root),
        "spin_square": float(spin_square),
        "multiplicity": float(multiplicity),
        "analytic_gradient_hartree_per_bohr": analytic.tolist(),
        "numerical_gradient_hartree_per_bohr": (
            None if numerical is None else numerical.tolist()
        ),
        "analytic_minus_numerical_hartree_per_bohr": (
            None if difference is None else difference.tolist()
        ),
        "displaced_calculations": displaced_records,
        "diagnostics": {
            "max_abs_error_hartree_per_bohr": max_abs_error,
            "rms_error_hartree_per_bohr": rms_error,
            "analytic_net_force_norm_hartree_per_bohr": net_force,
            "analytic_torque_norm_hartree": torque,
            "min_active_overlap": min_overlap,
            "passed": passed,
        },
    }


def print_results(result: dict[str, object]) -> None:
    """Print the gradient comparison."""
    analytic = np.asarray(result["analytic_gradient_hartree_per_bohr"])
    numerical_raw = result["numerical_gradient_hartree_per_bohr"]
    diagnostics = result["diagnostics"]

    print("\n" + "=" * 80)
    print("Distorted SO2 GBCI gradient check")
    print("=" * 80)
    print("Reference geometry / Angstrom")
    for entry in result["molecule"]["geometry_angstrom"]:
        print(f"  {entry[0]:<2s} {entry[1]:14.8f} {entry[2]:14.8f} {entry[3]:14.8f}")
    print(f"\nReference energy = {result['reference_energy_hartree']:.15f} Eh")
    print(f"<S^2> = {result['spin_square']:.10f}")

    print("\nActive space (from act_list)")
    for entry in result["active_space_report"]:
        leading = ", ".join(
            f"{item['ao']} ({item['weight']:.3f})"
            for item in entry["leading_ao_contributions"]
        )
        print(
            f"  MO {entry['act_list_index']:3d}  "
            f"e = {entry['rhf_mo_energy_hartree']:+10.5f} Eh  "
            f"occ = {entry['rhf_mo_occupancy']:.1f}  {leading}"
        )

    print("\n" + "=" * 80)
    print("Analytic gradient / Eh Bohr^-1")
    print("=" * 80)
    print(analytic)

    if numerical_raw is not None:
        numerical = np.asarray(numerical_raw)
        difference = np.asarray(result["analytic_minus_numerical_hartree_per_bohr"])

        print("\n" + "=" * 80)
        print("Numerical gradient / Eh Bohr^-1")
        print("=" * 80)
        print(numerical)

        print("\n" + "=" * 80)
        print("Difference = analytic - numerical / Eh Bohr^-1")
        print("=" * 80)
        print(difference)

        print("\nPer-coordinate comparison")
        print("-" * 80)
        print(" atom coord        analytic              numerical             diff")
        print("-" * 80)
        labels = ("x", "y", "z")
        for atom_index in range(analytic.shape[0]):
            for axis in range(3):
                print(
                    f"{atom_index:5d} {labels[axis]:>5s} "
                    f"{analytic[atom_index, axis]:20.12e} "
                    f"{numerical[atom_index, axis]:20.12e} "
                    f"{difference[atom_index, axis]:20.12e}"
                )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if diagnostics["max_abs_error_hartree_per_bohr"] is not None:
        print(
            "Max abs diff          = "
            f"{diagnostics['max_abs_error_hartree_per_bohr']:.12e} Eh/Bohr"
        )
        print(
            "RMS diff              = "
            f"{diagnostics['rms_error_hartree_per_bohr']:.12e} Eh/Bohr"
        )
    print(
        "Analytic net force    = "
        f"{diagnostics['analytic_net_force_norm_hartree_per_bohr']:.12e} Eh/Bohr"
    )
    print(
        "Analytic torque       = "
        f"{diagnostics['analytic_torque_norm_hartree']:.12e} Eh"
    )
    print(f"Min active overlap    = {diagnostics['min_active_overlap']:.6f}")
    if not result["calculation"]["act_list_is_identity_permutation"]:
        print(
            "NOTE: act_list is not the default active window, so the analytic\n"
            "      gradient is evaluated on a reordered MO set the driver does\n"
            "      not support.  A large torque here is that, not a geometry\n"
            "      or finite-difference problem."
        )
    print(f"Result: {'PASS' if diagnostics['passed'] else 'FAIL'}")


# ============================================================
# Command line
# ============================================================

def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def index_list(value: str) -> list[int]:
    parsed = [int(item) for item in value.replace(",", " ").split()]
    if len(parsed) != NCAS:
        raise argparse.ArgumentTypeError(f"act_list must hold {NCAS} MO indices")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare analytic and central-difference GBCI gradients for a "
            "deliberately asymmetric SO2 geometry."
        )
    )
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument(
        "--act-list",
        type=index_list,
        default=None if ACT_LIST is None else list(ACT_LIST),
        help=(
            "Explicit 1-based active MO indices.  Omit to keep the canonical "
            "contiguous window, which is the only arrangement the GBCI gradient "
            "driver handles correctly."
        ),
    )
    parser.add_argument("--step-bohr", type=positive_float, default=2.0e-3)
    parser.add_argument("--atol", type=positive_float, default=1.0e-7)
    parser.add_argument(
        "--torque-atol",
        type=positive_float,
        default=1.0e-8,
        help=(
            "Tolerance on the analytic net torque, which vanishes only for a "
            "rotationally invariant gradient."
        ),
    )
    parser.add_argument("--scf-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--scf-conv-tol-grad", type=positive_float, default=1.0e-9)
    parser.add_argument(
        "--scf-damping",
        action="store_true",
        help=(
            "Re-enable damp=0.4 / level_shift=0.5 on the RHF.  Only needed if "
            "the SCF fails to converge; they roughen the energy surface."
        ),
    )
    parser.add_argument("--fasscf-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--fasscf-conv-tol-grad", type=positive_float, default=1.0e-12)
    parser.add_argument("--gbci-conv-tol", type=positive_float, default=1.0e-14)
    parser.add_argument("--max-cycle", type=int, default=400)
    parser.add_argument(
        "--fd-order",
        type=int,
        choices=(2, 4),
        default=4,
        help=(
            "Central-difference order.  2 uses two energies per coordinate, 4 "
            "uses four and removes the leading h^2 truncation error."
        ),
    )
    parser.add_argument("--nroots", type=int, default=NROOTS)
    parser.add_argument("--target-root", type=int, default=TARGET_ROOT)
    parser.add_argument("--active-overlap-thresh", type=positive_float, default=0.8)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--analytic-only",
        action="store_true",
        help="Skip the 18 finite-difference energies; a cheap wiring smoke test.",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Report a failed comparison without returning a nonzero exit status.",
    )
    args = parser.parse_args()
    if args.target_root >= args.nroots:
        parser.error("--target-root must be smaller than --nroots")
    return args


def main() -> None:
    args = parse_args()
    result = compare_gradients(args)
    print_results(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")

    if not result["diagnostics"]["passed"] and not args.no_assert:
        raise AssertionError(
            "GBCI analytic gradient does not match the finite-difference "
            f"reference within {args.atol:.3e} Eh/Bohr"
        )


if __name__ == "__main__":
    main()
