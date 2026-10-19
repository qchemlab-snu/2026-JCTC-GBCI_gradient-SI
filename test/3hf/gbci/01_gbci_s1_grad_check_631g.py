"""Analytic vs finite-difference GBCI gradient for the 3-hydroxyflavone S1 state.

Geometry is the B3LYP/6-31G* minimum from ``test/3HF/dft`` (no imaginary modes).
The active space is the frontier pi/pi* window reported by
``00_rhf_631g_active.py``: CAS(4e,4o) = {61, 62, 63, 64} in 6-31G, four orbitals
with pi weight 1.000, already contiguous around the Fermi level so no MO
reordering is needed.

S1 is the right state to measure a gradient against here.
``03_gbci_state_character_631g.py`` gives it as a pure HOMO -> LUMO excitation
(transition weight 1.000) that moves 0.32 electrons from the 2-phenyl donor onto
the chromone acceptor, lifting the dipole from 4.09 to 9.46 D.  Widening the
active space to (6e,6o) leaves that description intact -- 5.55 -> 5.34 eV, 0.321
-> 0.345 electrons, 9.46 -> 9.00 D -- while the more strongly charge-transferred
S2 changes character outright (its hole delocalizes over two occupied orbitals)
and picks up a neighbouring state 0.23 eV away.  S1 is both active-space robust
and, at 1.05 eV below S2 here, cleanly separated for CI-overlap tracking.

The analytic gradient measured here is the pyscf-forge implementation; this
repository ships an older ``pyscf/grad/gbci.py`` that predates the 2026-09-30
fixes, and the script refuses to run against it.

Finite differences are the limiting factor, not the analytic gradient, so the
reference energies want a single thread and tight convergence everywhere
(see test/SCRIPT_PATTERNS.md).  The run is split into three modes driven by a
SLURM array:

    --mode reference        reference GBCI + analytic gradient, cached to .npz
    --mode fd --task-id K --num-tasks N      displacements of task K only
    --mode combine          merge the per-task partials and compare

Threads stay at one in every mode: the parallelism is over displacements, not
inside them, so the finite-difference noise floor is unaffected.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import gto, scf

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[3]

# Both pyscf-forge and this repo ship a pyscf/grad/gbci.py.  Target the forge one.
gbci_module = importlib.import_module("pyscf.gbci.gbci")
gbci_grad = importlib.import_module("pyscf.grad.gbci")
GBCI = gbci_module.GBCI

if REPO_ROOT in Path(gbci_grad.__file__).resolve().parents:
    raise RuntimeError(
        f"pyscf.grad.gbci resolved to the local repo copy ({gbci_grad.__file__}); "
        "set PYSCF_EXT_PATH to pyscf-forge only, and keep this repository off "
        "PYTHONPATH, so the fixed implementation is used"
    )


def _load(name: str, relative: str) -> Any:
    """Load a utils module by path; putting REPO_ROOT on sys.path would make
    pyscf load this repository's pyscf/ tree as a plugin and shadow forge."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_gbci_setup = _load("_gbci_setup", "utils/gbci_setup.py")
_bath_grouping = _load("_bath_grouping", "utils/bath_grouping.py")
_cphf_tuning = _load("_cphf_tuning", "utils/cphf_tuning.py")
_gbci_cache = _load("_gbci_cache", "utils/gbci_cache.py")
_bath_semicanonical = _load("_bath_semicanonical",
                           "utils/bath_semicanonical.py")
apply_act_list = _gbci_setup.apply_act_list
check_active_window = _gbci_setup.check_active_window
gradient_invariance = _gbci_setup.gradient_invariance
is_contiguous_window = _gbci_setup.is_contiguous_window


# ============================================================
# User settings
# ============================================================

GEOMETRY = SCRIPT_DIR / ".." / "dft" / "outputs" / "3hf_dft_S0_opt.xyz"
BASIS = "6-31G"
CHARGE = 0
SPIN = 0

NCAS = 4
NELECAS = (2, 2)
# Frontier pi/pi* from 00_rhf_631g_active.py: HOMO-1, HOMO, LUMO, LUMO+1.
# Contiguous around the Fermi level, so is_contiguous_window() is satisfied and
# the MO coefficients are left alone.
ACT_LIST = [61, 62, 63, 64]
ACT_BASE = 1

NROOTS = 4
STATE = 1  # S1, the HOMO -> LUMO pi -> pi* state

# Grouped bath keyed on the chromone (benzo-4-pyranone) acceptor: carbonyl O,
# the pyranone ring, and the fused benzo ring with its hydrogens. The 2-phenyl
# ring and the 3-OH stay outside, so the group boundary straddles the
# phenyl -> chromone charge transfer that S1 carries.
ACCEPTOR_ATOMS = [0, 1, 2, 4, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27]

# threshold 0.5 gives 4 groups instead of the 7 of the 0.2 default, and
# 02_group_threshold_check_631g.py shows the S0 and HOMO -> LUMO configurations
# stay in separate groups all the way to 0.8. The gradient cost carries a
# ngroup**2 factor, so this is roughly 3x cheaper.
GROUP_THRESHOLD = 0.5
GROUP_A = {"atom": [ACCEPTOR_ATOMS], "threshold": GROUP_THRESHOLD}

# Atoms displaced in the finite difference. All 28 atoms would be 168 single
# points and the reference energies have to run single-threaded, so the default
# is the intramolecular hydrogen bond that 3HF's excited-state proton transfer
# runs along:
#
#   O3-H18...O0=C1-C2(-O3),  H18...O0 = 1.937 A, O3...O0 = 2.596 A,
#   O3-H18...O0 angle = 121.6 deg
#
# Those are the coordinates an ESIPT moves along, so they are where the S1
# gradient has to be right. Widen with --fd-atoms all.
DEFAULT_FD_ATOMS = [0, 1, 2, 3, 18]

DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "01_gbci_s1_grad_check_631g.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-bohr", type=float, default=1.0e-4,
                        help="central-difference step (default 1e-4 bohr)")
    # The 4-point stencil trades truncation for noise: its error is O(h^4)
    # instead of O(h^2), but the coefficients amplify energy noise by 1.5/h
    # rather than 0.5/h, so it only pays off once truncation dominates.
    parser.add_argument("--stencil", type=int, choices=(2, 4), default=2,
                        help="2-point central difference, or 4-point O(h^4)")
    parser.add_argument("--fd-atoms", nargs="*", default=None,
                        help="0-based atom indices to displace, or 'all'")
    parser.add_argument("--state", type=int, default=STATE)
    parser.add_argument("--nroots", type=int, default=NROOTS)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-14)
    parser.add_argument("--scf-conv-tol-grad", type=float, default=1e-9)
    parser.add_argument("--fasscf-conv-tol", type=float, default=1e-14)
    # The bath SOSCF stalls near 1e-7 with delta_E already at 0, i.e. the
    # residual is the noise floor of the gradient evaluation rather than a real
    # distance from the stationary point.  Asking for less than that stops the
    # run on the iteration cap instead of on the criterion, which makes the
    # stopping point vary between displaced geometries -- exactly the
    # irreproducibility a finite difference amplifies by 1/(2h).
    parser.add_argument("--fasscf-conv-tol-grad", type=float, default=1e-6)
    parser.add_argument("--gbci-conv-tol", type=float, default=1e-14)
    # FASSCF routes a group through Newton SOSCF when its damped stage does not
    # converge, and that call leaves pyscf's canonicalization off, so the group
    # comes back with canonical energies on non-canonical orbitals.  The bath
    # CPHF assumes a semicanonical bath, so without this the analytic gradient
    # is ~2e-4 off the finite difference where a sound bath reaches 1e-7.
    parser.add_argument("--semicanonical-bath",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="re-diagonalize each group's core and virtual "
                             "bath blocks after FAS-SCF (default; "
                             "--no-semicanonical-bath reproduces the defect)")
    parser.add_argument("--max-cycle", type=int, default=400)
    # pyscf-forge calls its bath CPHF solve with max_cycle=30 at both call
    # sites, and lib.krylov returns silently when it runs out of cycles. The
    # solve spans nvir*nocc = 7440 here, so 30 Krylov vectors leave the
    # orbital-response term unconverged -- which shows up as a nonzero torque
    # while leaving the energy untouched.
    parser.add_argument("--cphf-max-cycle", type=int, default=200)
    parser.add_argument("--cphf-tol", type=float, default=1e-12)
    parser.add_argument("--cphf-level-shift", type=float, default=1e-5,
                        help="level shift used to precondition the bath CPHF")
    # The FASSCF bath optimization dominates a single point and depends only on
    # the orbitals, not on any gradient setting, so caching it makes repeated
    # runs that vary only the CPHF essentially free.
    parser.add_argument("--gbci-cache-dir", type=Path,
                        default=SCRIPT_DIR / "outputs" / "01_gbci_cache")
    parser.add_argument("--no-gbci-cache", action="store_true")
    # The group assignment is discrete, so the analytic gradient is necessarily
    # the derivative at fixed assignment. Freezing the displaced geometries to
    # the reference assignment makes the finite difference measure that same
    # surface; letting it float is what trips the membership guard at large h.
    # Basis and act_list are module constants above, but MO indices are not
    # transferable between basis sets, so a different basis needs both changed
    # together; these overrides keep that pairing in one command line.
    parser.add_argument("--basis", type=str, default=None,
                        help=f"override BASIS (default {BASIS})")
    parser.add_argument("--act-list", type=str, default=None,
                        help="override ACT_LIST, comma-separated 1-based MOs")
    # On by default: the analytic gradient is the derivative at fixed group
    # assignment, so a displaced point that reassigns a configuration is
    # differencing a different objective. At h = 1e-3 that guard tripped on
    # two of the fifteen components; freezing is what makes the finite
    # difference measure the surface the analytic gradient differentiates.
    parser.add_argument("--freeze-groups", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="hold the bath grouping at the reference one for "
                             "the displaced points (default; "
                             "--no-freeze-groups lets it float)")
    parser.add_argument("--cphf-level-shift-scan", type=str, default=None,
                        help="comma-separated shifts to solve and compare in "
                             "one run; diagnostic only, the best is used")
    # xvo is 2D, so lib.krylov runs its nroots == 1 branch and ends at the
    # first cycle with ||x1||**2 < max(lindep, tol**2).  At the call sites'
    # 1e-20 that leaves the projected metric spanning ~16 orders of magnitude
    # before numpy.linalg.solve inverts it, which is where the residual floor
    # comes from -- not the cycle count.  Raising lindep truncates the basis
    # while that solve is still conditioned.
    parser.add_argument("--cphf-lindep", type=float, default=1e-20,
                        help="Krylov linear-dependency threshold (forge "
                             "passes 1e-20)")
    parser.add_argument("--cphf-lindep-scan", type=str, default=None,
                        help="comma-separated lindep values to solve and "
                             "compare in one run; diagnostic only")
    parser.add_argument("--active-overlap-thresh", type=float, default=0.9)
    parser.add_argument("--ci-overlap-thresh", type=float, default=0.8)
    # verbose 5 keeps the per-point FASSCF/GBCI log; it is the run record.
    parser.add_argument("--verbose", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("reference", "fd", "combine", "serial"),
                        default="serial",
                        help="serial runs everything in one process")
    parser.add_argument("--reference-npz", type=Path,
                        default=SCRIPT_DIR / "outputs" / "01_reference.npz")
    parser.add_argument("--partial-dir", type=Path,
                        default=SCRIPT_DIR / "outputs" / "01_partials")
    parser.add_argument("--task-id", type=int, default=0,
                        help="0-based index of this array task")
    parser.add_argument("--num-tasks", type=int, default=1)
    return parser.parse_args()


def resolve_fd_atoms(args, natm: int) -> list[int]:
    if args.fd_atoms is None:
        return list(DEFAULT_FD_ATOMS)
    if len(args.fd_atoms) == 1 and str(args.fd_atoms[0]).lower() == "all":
        return list(range(natm))
    return [int(i) for i in args.fd_atoms]


def components(fd_atoms: list[int]) -> list[tuple[int, int]]:
    """Finite-difference components, the unit of work handed to one array task."""
    return [(atom, axis) for atom in fd_atoms for axis in range(3)]


def task_components(fd_atoms: list[int], task_id: int, num_tasks: int):
    """Round-robin share of the components, so a partial run stays spread out."""
    return components(fd_atoms)[task_id::num_tasks]


# ============================================================
# Molecule and GBCI calculation
# ============================================================

def reference_coords() -> tuple[list[str], np.ndarray]:
    """Symbols and Bohr coordinates of the B3LYP-optimized geometry."""
    mol = gto.M(atom=str(GEOMETRY), basis=BASIS, charge=CHARGE, spin=SPIN,
                unit="Angstrom", verbose=0)
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    return symbols, mol.atom_coords(unit="Bohr")


def build_molecule(symbols, coords_bohr, verbose: int) -> gto.Mole:
    return gto.M(
        atom=[(s, tuple(c)) for s, c in zip(symbols, coords_bohr)],
        basis=BASIS, charge=CHARGE, spin=SPIN, unit="Bohr",
        symmetry=False, verbose=verbose,
    )


def run_gbci(mol: gto.Mole, args: argparse.Namespace) -> Any:
    """RHF, install the active space, run GBCI with ``nroots`` roots."""
    mf = scf.RHF(mol)
    mf.conv_tol = args.scf_conv_tol
    # Set explicitly: pyscf derives it as sqrt(conv_tol) otherwise, and it is
    # this criterion that decides how early the SCF stops.
    mf.conv_tol_grad = args.scf_conv_tol_grad
    mf.max_cycle = args.max_cycle

    cache_dir = None if args.no_gbci_cache else args.gbci_cache_dir
    scf_key = None
    if cache_dir is not None:
        scf_key = _gbci_cache.scf_cache_key(mol, args.scf_conv_tol,
                                            args.scf_conv_tol_grad)
    if scf_key is None or not _gbci_cache.load_scf(mf, cache_dir, scf_key):
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("RHF did not converge")
        if scf_key is not None:
            _gbci_cache.save_scf(mf, cache_dir, scf_key)
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    probe = GBCI(mf, ncas=NCAS, nelecas=NELECAS, group_a=GROUP_A)
    reordered = not is_contiguous_window(probe, ACT_LIST, base=ACT_BASE)
    if reordered:
        apply_act_list(mf, probe, ACT_LIST, base=ACT_BASE)

    gbci = GBCI(mf, ncas=NCAS, nelecas=NELECAS, group_a=GROUP_A)
    check_active_window(gbci)
    gbci.fix_spin_(ss=0)

    fasscf = gbci._get_fasscf(gbci.group_a, gbci.mo_coeff, gbci.ncas,
                              gbci.nelecas, gbci.ncore)
    fasscf.conv_tol = args.fasscf_conv_tol
    fasscf.conv_tol_grad = args.fasscf_conv_tol_grad
    fasscf.max_cycle = args.max_cycle

    gbci.fcisolver.nroots = args.nroots
    gbci.fcisolver.conv_tol = args.gbci_conv_tol
    gbci.fcisolver.max_cycle = args.max_cycle

    if args.semicanonical_bath:
        _bath_semicanonical.patch_optimize_mo(gbci, verbose=args.verbose >= 4)

    gbci.kernel()
    gbci._reordered_mo = reordered
    return gbci


def normalized_groups(groups: Any) -> list[list[int]]:
    return [[int(i) for i in group] for group in groups]


def active_block(gbci: Any) -> np.ndarray:
    return np.asarray(gbci.mo_coeff)[:, gbci.ncore:gbci.ncore + gbci.ncas]


def active_overlap_diagonal(reference: np.ndarray, gbci: Any, mol: gto.Mole) -> float:
    """Smallest |<phi_ref|S|phi_displaced>| over the active orbitals.

    act_list selects orbitals by index, so the finite difference only means
    something while the displaced geometries keep the same orbitals in the
    window.  Using the displaced overlap for both sides is accurate to O(h).
    """
    overlap = reference.T @ mol.intor("int1e_ovlp") @ active_block(gbci)
    return float(np.min(np.abs(np.diag(overlap))))


def track_state(ci_reference: np.ndarray, gbci: Any) -> tuple[int, float]:
    """Index of the displaced root matching the reference state, and its overlap.

    S1 and S2 are 0.25 eV apart and share a hole, so their order can swap under a
    displacement; the CI coefficients themselves are the robust label at this
    step size.
    """
    ref = np.asarray(ci_reference).reshape(-1)
    ref = ref / np.linalg.norm(ref)
    overlaps = []
    for ci in gbci.ci:
        vec = np.asarray(ci).reshape(-1)
        overlaps.append(abs(float(ref @ (vec / np.linalg.norm(vec)))))
    best = int(np.argmax(overlaps))
    return best, float(overlaps[best])


# ============================================================
# Gradients
# ============================================================

def analytic_gradient(gbci: Any, state: int) -> np.ndarray:
    return np.asarray(gbci.nuc_grad_method().kernel(state=state), dtype=float)


def numerical_components(symbols, coords_bohr, reference, comps, args):
    """Central-difference the tracked state over the given (atom, axis) list."""
    records: list[dict[str, Any]] = []

    if args.stencil == 2:
        stencil = ((-1.0, -0.5), (+1.0, +0.5))          # offset/h, weight*h
    else:
        stencil = ((-2.0, +1.0 / 12.0), (-1.0, -8.0 / 12.0),
                   (+1.0, +8.0 / 12.0), (+2.0, -1.0 / 12.0))

    for atom, axis in comps:
        energies = {}
        for offset, weight in stencil:
            key = f"{'minus' if offset < 0 else 'plus'}{abs(offset):g}"
            print(f"\n--- displacement: atom {atom} axis {'xyz'[axis]} "
                  f"{key} {args.step_bohr:g} bohr ---", flush=True)
            displaced = coords_bohr.copy()
            displaced[atom, axis] += offset * args.step_bohr
            mol = build_molecule(symbols, displaced, args.verbose)
            gbci = run_gbci(mol, args)

            groups = normalized_groups(gbci._gbci_intermediates["group"])
            if groups != reference["groups"]:
                raise RuntimeError(
                    f"grouped-bath membership changed under the {key} "
                    f"displacement of atom {atom} axis {axis}; the numerical "
                    "derivative would compare different objectives")

            overlap = active_overlap_diagonal(reference["active"], gbci, mol)
            if overlap < args.active_overlap_thresh:
                raise RuntimeError(
                    f"active orbital overlap dropped to {overlap:.4f} under the "
                    f"{key} displacement of atom {atom} axis {axis}")

            root, ci_overlap = track_state(reference["ci"], gbci)
            if ci_overlap < args.ci_overlap_thresh:
                raise RuntimeError(
                    f"state tracking is ambiguous (best CI overlap "
                    f"{ci_overlap:.3f}) under the {key} displacement of atom "
                    f"{atom} axis {axis}")

            e_all = np.asarray(gbci.e_tot, dtype=float).reshape(-1)
            energies[key] = {
                "offset": offset, "weight": weight,
                "energy": float(e_all[root]), "root": root,
                "ci_overlap": ci_overlap, "active_overlap": overlap,
                "all_roots": [float(x) for x in e_all],
            }

        value = sum(e["weight"] * e["energy"] for e in energies.values()) / args.step_bohr
        neg = [e for e in energies.values() if e["offset"] < 0]
        pos = [e for e in energies.values() if e["offset"] > 0]
        records.append({
            "atom": int(atom), "axis": int(axis),
            "stencil": int(args.stencil),
            "points": energies,
            # Kept under the 2-point names so combine() stays unchanged; for the
            # 4-point stencil they are the worst value on each side.
            "ci_overlap_minus": min(e["ci_overlap"] for e in neg),
            "ci_overlap_plus": min(e["ci_overlap"] for e in pos),
            "active_overlap_minus": min(e["active_overlap"] for e in neg),
            "active_overlap_plus": min(e["active_overlap"] for e in pos),
            "numerical": float(value),
        })
        print(f"  atom {atom:2d} axis {'xyz'[axis]}: numerical = {value: .10f} "
              f"({args.stencil}-point, CI overlap >= "
              f"{min(e['ci_overlap'] for e in energies.values()):.3f})", flush=True)

    return records


# ============================================================
# Reference cache
# ============================================================

def save_reference(path: Path, reference: dict[str, Any]) -> None:
    """Cache the reference so the array tasks skip a 3-hour GBCI each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        active=reference["active"],
        ci=reference["ci"],
        analytic=reference["analytic"],
        coords_bohr=reference["coords_bohr"],
        e_roots=np.asarray(reference["e_roots"]),
        symbols=np.asarray(reference["symbols"]),
        groups_json=np.asarray(json.dumps(reference["groups"])),
        meta_json=np.asarray(json.dumps(reference["meta"])),
    )


def load_reference(path: Path) -> dict[str, Any]:
    data = np.load(path)
    return {
        "active": data["active"],
        "ci": data["ci"],
        "analytic": data["analytic"],
        "coords_bohr": data["coords_bohr"],
        "e_roots": [float(x) for x in data["e_roots"]],
        "symbols": [str(x) for x in data["symbols"]],
        "groups": json.loads(str(data["groups_json"])),
        "meta": json.loads(str(data["meta_json"])),
    }


def build_reference(args, symbols, coords_bohr) -> dict[str, Any]:
    """Reference GBCI plus the analytic gradient of the requested state."""
    mol = build_molecule(symbols, coords_bohr, args.verbose)
    gbci = run_gbci(mol, args)
    e_all = np.asarray(gbci.e_tot, dtype=float).reshape(-1)
    print(f"GBCI roots (Eh): {e_all}")
    print(f"S{args.state} = {e_all[args.state]:.10f}", flush=True)

    analytic = analytic_gradient(gbci, args.state)
    net_force, torque = gradient_invariance(mol, analytic)
    print(f"\nanalytic |sum g| = {net_force:.3e}, "
          f"|sum R x g| = {torque:.3e} Eh   (torque is the tight test)",
          flush=True)

    return {
        "active": active_block(gbci).copy(),
        "ci": np.asarray(gbci.ci[args.state]).copy(),
        "analytic": analytic,
        "coords_bohr": coords_bohr,
        "e_roots": [float(x) for x in e_all],
        "symbols": list(symbols),
        "groups": normalized_groups(gbci._gbci_intermediates["group"]),
        "meta": {
            "implementation": gbci_grad.__file__,
            "geometry": str(GEOMETRY), "basis": BASIS,
            "ncas": NCAS, "nelecas": list(NELECAS), "act_list": ACT_LIST,
            "reordered_mo": bool(gbci._reordered_mo), "group_a": GROUP_A,
            "state": args.state, "nroots": args.nroots,
            "step_bohr": args.step_bohr,
            "stencil": args.stencil,
            "cphf_max_cycle": args.cphf_max_cycle,
            "cphf_tol": args.cphf_tol,
            "cphf_level_shift": args.cphf_level_shift,
            "cphf_level_shift_scan": args.cphf_level_shift_scan,
            "cphf_lindep": args.cphf_lindep,
            "cphf_lindep_scan": args.cphf_lindep_scan,
            "freeze_groups": bool(args.freeze_groups),
            "semicanonical_bath": bool(args.semicanonical_bath),
            "scf_conv_tol": args.scf_conv_tol,
            "scf_conv_tol_grad": args.scf_conv_tol_grad,
            "fasscf_conv_tol": args.fasscf_conv_tol,
            "fasscf_conv_tol_grad": args.fasscf_conv_tol_grad,
            "gbci_conv_tol": args.gbci_conv_tol,
            "net_force": net_force, "torque": torque,
        },
    }


# ============================================================
# Combine
# ============================================================

# Settings that belong to the run that computed them.  The reference and the
# displaced points are separate jobs with separate command lines, so a combine
# must not describe one with the other's arguments: --freeze-groups is passed
# to the fd tasks, never to the combine, and reading it off the combine's own
# argv reported `false` for campaigns that were in fact frozen.
RUN_SETTINGS_KEYS = (
    "step_bohr", "stencil", "freeze_groups", "semicanonical_bath",
    "scf_conv_tol", "scf_conv_tol_grad",
    "fasscf_conv_tol", "fasscf_conv_tol_grad", "gbci_conv_tol",
    "cphf_max_cycle", "cphf_tol", "cphf_level_shift", "cphf_level_shift_scan",
    "cphf_lindep", "cphf_lindep_scan",
)


def run_settings(args) -> dict[str, Any]:
    """The settings of the process writing the file, for the record."""
    out: dict[str, Any] = {}
    for key in RUN_SETTINGS_KEYS:
        value = getattr(args, key)
        out[key] = bool(value) if isinstance(value, bool) else value
    return out


def merge_fd_settings(per_task: list[dict[str, Any]]) -> tuple[Any, str]:
    """One settings dict for the whole campaign, or None with the reason.

    Partials written before the fd tasks recorded their settings carry none;
    say so rather than substituting the combine's own arguments.
    """
    if not per_task:
        return None, "no partial files were read"
    missing = sum(1 for s in per_task if not s)
    if missing:
        return None, (f"{missing} of {len(per_task)} partials predate per-task "
                      "settings; see the slurm scripts beside this script for "
                      "the command that produced them")
    first = per_task[0]
    disagree = sorted({k for s in per_task for k in RUN_SETTINGS_KEYS
                       if s.get(k) != first.get(k)})
    if disagree:
        return None, f"the fd tasks disagree on {', '.join(disagree)}"
    return dict(first), "read from the partials"


def combine(args, reference, fd_atoms, records,
            fd_settings=None, fd_settings_note="computed in this process"
            ) -> dict[str, Any]:
    symbols = reference["symbols"]
    analytic = np.asarray(reference["analytic"])
    numerical = np.full_like(analytic, np.nan)
    for record in records:
        numerical[record["atom"], record["axis"]] = record["numerical"]

    mask = ~np.isnan(numerical)
    diff = analytic[mask] - numerical[mask]
    max_abs = float(np.max(np.abs(diff))) if diff.size else float("nan")
    rms = float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan")
    min_active = min((min(r["active_overlap_minus"], r["active_overlap_plus"])
                      for r in records), default=float("nan"))
    min_ci = min((min(r["ci_overlap_minus"], r["ci_overlap_plus"])
                  for r in records), default=float("nan"))

    print("\n atom axis      analytic        numerical       difference")
    for record in sorted(records, key=lambda r: (r["atom"], r["axis"])):
        a, x = record["atom"], record["axis"]
        print(f" {symbols[a]}{a:<3} {'xyz'[x]}   {analytic[a, x]: .10f} "
              f"{record['numerical']: .10f} "
              f"{analytic[a, x] - record['numerical']: .3e}")
    print(f"\ncomponents compared        = {int(mask.sum())} "
          f"of {3 * len(fd_atoms)}")
    print(f"max |analytic - numerical| = {max_abs:.3e} Eh/bohr")
    print(f"rms                        = {rms:.3e} Eh/bohr")
    print(f"min active-orbital overlap = {min_active:.6f}")
    print(f"min CI overlap (tracking)  = {min_ci:.6f}")
    print(f"analytic torque            = {reference['meta']['torque']:.3e} Eh")

    # The reference .npz carries the settings of the reference job.  Those
    # describe the analytic gradient, not the finite difference, so they go in
    # their own block instead of sitting at the top level where a reader would
    # take them for the campaign's.
    meta = dict(reference["meta"])
    reference_run = {k: meta.pop(k) for k in RUN_SETTINGS_KEYS if k in meta}
    step = fd_settings["step_bohr"] if fd_settings else args.step_bohr
    stencil = fd_settings["stencil"] if fd_settings else args.stencil
    if fd_settings and (fd_settings["step_bohr"] != args.step_bohr
                        or fd_settings["stencil"] != args.stencil):
        print(f"WARNING: --step-bohr {args.step_bohr:g} --stencil {args.stencil} "
              f"given, but the partials were written with "
              f"{fd_settings['step_bohr']:g} and {fd_settings['stencil']}; "
              f"the partials win", flush=True)
    print(f"fd settings                = {fd_settings_note}")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **meta,
        "step_bohr": step,
        "stencil": stencil,
        "fd_settings": fd_settings,
        "fd_settings_source": fd_settings_note,
        "reference_run": reference_run,
        "fd_atoms": fd_atoms,
        "e_roots": reference["e_roots"],
        "analytic_gradient": analytic.tolist(),
        "displacements": sorted(records, key=lambda r: (r["atom"], r["axis"])),
        "n_components": int(mask.sum()),
        "max_abs_difference": max_abs,
        "rms_difference": rms,
        "min_active_overlap": min_active,
        "min_ci_overlap": min_ci,
    }


# ============================================================
# Driver
# ============================================================

def main() -> None:
    args = parse_args()

    global BASIS, ACT_LIST
    if (args.basis is None) != (args.act_list is None):
        raise RuntimeError("--basis and --act-list must be given together: MO "
                           "indices are not transferable between basis sets")
    if args.basis is not None:
        BASIS = args.basis
        ACT_LIST = [int(x) for x in args.act_list.split(",")]
        print(f"basis overridden to {BASIS}, act_list = {ACT_LIST}", flush=True)
    _bath_grouping.patch_gbci_grouping()  # complete-linkage bath grouping
    if not args.no_gbci_cache:
        # The tag keeps frozen-grouping entries apart from floating ones:
        # optimize_mo returns the group assignment, so a cached entry written
        # under one rule must not be served under the other.
        _gbci_cache.patch_optimize_mo_cache(
            args.gbci_cache_dir,
            tag="frozen" if args.freeze_groups else "")
    shift_scan = None
    if args.cphf_level_shift_scan:
        shift_scan = [float(x) for x in args.cphf_level_shift_scan.split(",")]
    lindep_scan = None
    if args.cphf_lindep_scan:
        lindep_scan = [float(x) for x in args.cphf_lindep_scan.split(",")]
    _cphf_tuning.patch_bath_cphf(max_cycle=args.cphf_max_cycle,
                                 tol=args.cphf_tol,
                                 level_shift=args.cphf_level_shift,
                                 level_shift_scan=shift_scan,
                                 lindep=args.cphf_lindep,
                                 lindep_scan=lindep_scan)

    symbols, coords_bohr = reference_coords()
    fd_atoms = resolve_fd_atoms(args, len(symbols))

    print(f"3HF / {BASIS} / GBCI({sum(NELECAS)}e,{NCAS}o) act_list={ACT_LIST} "
          f"state=S{args.state}  mode={args.mode}")
    print(f"finite-difference step = {args.step_bohr:g} bohr, "
          f"atoms = {fd_atoms}", flush=True)
    print(f"bath CPHF: max_cycle = {args.cphf_max_cycle}, tol = {args.cphf_tol:g}, "
          f"level_shift = {shift_scan if shift_scan else args.cphf_level_shift}, "
          f"lindep = {lindep_scan if lindep_scan else args.cphf_lindep}",
          flush=True)
    print(f"FASSCF: conv_tol = {args.fasscf_conv_tol:g}, conv_tol_grad = "
          f"{args.fasscf_conv_tol_grad:g}, bath semicanonicalization = "
          f"{'on' if args.semicanonical_bath else 'OFF'}", flush=True)

    if args.mode in ("reference", "serial"):
        reference = build_reference(args, symbols, coords_bohr)
        if args.mode == "reference":
            save_reference(args.reference_npz, reference)
            print(f"\nSaved reference to {args.reference_npz}")
            print(f"next: --mode fd --task-id K --num-tasks N, then --mode combine")
            return
    else:
        reference = load_reference(args.reference_npz)
        coords_bohr = reference["coords_bohr"]
        symbols = reference["symbols"]

    if args.mode == "combine":
        records: list[dict[str, Any]] = []
        per_task: list[dict[str, Any]] = []
        for path in sorted(args.partial_dir.glob("task_*.json")):
            payload = json.loads(path.read_text())
            records.extend(payload["displacements"])
            per_task.append(payload.get("settings") or {})
            print(f"read {path.name}")
        fd_settings, note = merge_fd_settings(per_task)
        summary = combine(args, reference, fd_atoms, records,
                          fd_settings, note)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved {args.output}")
        return

    # Only fd and serial reach here; reference and combine have returned.
    # Serial computed its reference in this process, so it can freeze too.
    if args.freeze_groups:
        _bath_grouping.patch_gbci_grouping(
            _bath_grouping.frozen_rule(reference["groups"]))
        print(f"bath grouping frozen to the reference: "
              f"{[len(g) for g in reference['groups']]} patterns per group",
              flush=True)

    if args.mode == "fd":
        comps = task_components(fd_atoms, args.task_id, args.num_tasks)
        print(f"task {args.task_id}/{args.num_tasks} handles components "
              f"{[(a, 'xyz'[x]) for a, x in comps]}", flush=True)
    else:
        comps = components(fd_atoms)

    records = numerical_components(symbols, coords_bohr, reference, comps, args)

    if args.mode == "fd":
        args.partial_dir.mkdir(parents=True, exist_ok=True)
        path = args.partial_dir / f"task_{args.task_id:03d}.json"
        path.write_text(json.dumps({
            "task_id": args.task_id, "num_tasks": args.num_tasks,
            "settings": run_settings(args),
            "displacements": records,
        }, indent=2))
        print(f"\nSaved {path}")
        return

    summary = combine(args, reference, fd_atoms, records,
                      run_settings(args), "computed in this process")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
