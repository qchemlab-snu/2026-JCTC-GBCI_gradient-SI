"""PP S1 SA-CASSCF NEB along the planar -> TICT path, seeded from an existing band.

This replaces ``pp_casscf_S1_neb.py`` for the planar end of the path. That run
used the CASSCF S0 minimum as a fixed start endpoint, but the S0 minimum is a
Franck-Condon point on S1 with |F| = 4.32 eV/Angstrom, so image 1 was pulled
into the fixed endpoint (RMSD 0.000 Angstrom by step 80) and inherited that
force. The band never converged even though images 2-7 reached fmax < 0.09.

Two changes follow from that:

* The start endpoint is a planar geometry rather than the S0 minimum. Note that
  SA-CASSCF has no planar S1 minimum of its own: ``pp_casscf_S1_from_casci_planar``
  starts from the CASCI planar structure and relaxes all the way to the TICT
  minimum (same S1 energy as ``pp_casscf_S1`` to 1e-9 Hartree), and the planar
  saddle checks give 2 imaginary modes with a non-zero gradient. The planar
  endpoint is therefore a reference geometry from another method, not a
  stationary point of the surface this path walks on. The script reports the
  endpoint force so that assumption stays visible in the output.
* The band starts from an existing converged path instead of a fresh IDPP
  interpolation, so the optimizer begins near the answer rather than on a
  straight line through it.

Defaults seed from the GBCI planar -> TICT NEB images, which already carry
SA-CASSCF single-point energies in ``pp_casscf_S1_gbci_path_outputs``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import numpy as np
from ase import Atoms, io


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.neb_calculator import (  # noqa: E402
    build_neb_images,
    make_casscf_active_space_calculator_factory,
    run_neb_calculation,
    write_neb_images,
)


MOLECULE_NAME = "pp"
METHOD = "SA-CASSCF"
STATE = "S1"
PATH_LABEL = "planar_to_TICT"

BASIS = "ccpvdz"
CHARGE = 0
SPIN = 0
VERBOSE = 4

# Equal-weight S0/S1 state averaging, forces taken from root 1 (S1).
SCANNER_KWARGS = {
    "ncas": 2,
    "nelecas": (1, 1),
    "act_list": [38, 39],
    "nroots": 2,
    "target_root": 1,
    "conv_tol": 1e-6,
    "max_cycle": 200,
    "act_base": 1,
    "root_tracking": False,
    "state_average": True,
    "state_weights": [0.5, 0.5],
    "project_previous_orbitals": True,
    "reuse_ci_guess": True,
    "strict_convergence": False,
}

# The planar endpoint is not stationary on this surface, so the springs have to
# hold image 1 against a residual endpoint force of about 1 eV/Angstrom over a
# spacing of about 0.2 Angstrom. k = 0.6 was far too soft for that and let the
# band collapse into the endpoint; k * spacing ~ 1.2 eV/Angstrom restores it.
SPRING_CONSTANT = 6.0

# The reference GBCI band is downhill from planar to TICT with no interior
# maximum, and a climbing image on a monotonic band latches onto image 1 next to
# the non-stationary planar endpoint. Enable it with --climb only if a genuine
# barrier shows up.
CLIMB = False

# ASE warns that its old "aseneb" default frequently produces poor bands; the
# stopped pp_casscf_S1_neb run took that default.
NEB_METHOD = "improvedtangent"

OPTIMIZER = "FIRE"
FMAX = 0.05
MAX_STEPS = 500

SEED_PATH_DIR = (
    REPO_ROOT / "test" / "pp" / "gbci" / "pp_gbci_S1_planar_tict_neb_outputs"
)
SEED_PATH_PATTERN = "pp_gbci_S1_planar_tict_neb_*.xyz"
OUTPUT_DIR = SCRIPT_DIR / "pp_casscf_S1_planar_tict_neb_outputs"
IMAGE_PREFIX = "pp_casscf_S1_planar_tict_neb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an S1 SA-CASSCF NEB along the planar -> TICT path, starting "
            "from an existing converged band rather than an interpolation."
        )
    )
    parser.add_argument(
        "--path-dir",
        type=Path,
        default=SEED_PATH_DIR,
        help="Directory holding the existing path used as the initial band.",
    )
    parser.add_argument(
        "--path-pattern",
        default=SEED_PATH_PATTERN,
        help="Glob for the seed path image xyz files, sorted by name.",
    )
    parser.add_argument(
        "--seed-traj",
        type=Path,
        default=None,
        help=(
            "ASE trajectory to take the initial band from instead of --path-dir, "
            "e.g. a band from a run that was stopped before it converged."
        ),
    )
    parser.add_argument(
        "--seed-traj-step",
        type=int,
        default=-1,
        help=(
            "Optimizer step to read from --seed-traj. Use the last step before "
            "the band degraded, not necessarily the final one."
        ),
    )
    parser.add_argument(
        "--seed-traj-images",
        type=int,
        default=9,
        help="Images written per optimizer step in --seed-traj.",
    )
    parser.add_argument(
        "--neb-method",
        default=NEB_METHOD,
        help="ASE NEB tangent method.",
    )
    parser.add_argument(
        "--initial",
        type=Path,
        default=None,
        help=(
            "xyz file for the path start (planar). Defaults to the first image "
            "of the seed path."
        ),
    )
    parser.add_argument(
        "--final",
        type=Path,
        default=None,
        help=(
            "xyz file for the path end (TICT). Defaults to the last image of "
            "the seed path."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for NEB images and results.",
    )
    parser.add_argument(
        "--prefix",
        default=IMAGE_PREFIX,
        help="File name prefix for images, trajectory, log, and results.",
    )
    parser.add_argument(
        "--path-label",
        default=PATH_LABEL,
        help="Value recorded as 'path' in the metadata; set it when the "
             "endpoints are overridden.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=SPRING_CONSTANT,
        help="NEB spring constant in eV/Angstrom^2.",
    )
    parser.add_argument(
        "--climb",
        action="store_true",
        default=CLIMB,
        help="Enable the climbing image.",
    )
    parser.add_argument("--fmax", type=float, default=FMAX, help="ASE NEB force threshold.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Maximum NEB optimizer steps.",
    )
    parser.add_argument("--basis", default=BASIS, help="PySCF basis set name.")
    parser.add_argument("--verbose", type=int, default=VERBOSE, help="PySCF verbosity.")
    return parser.parse_args()


def relative_to_repo(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_geometry(path: Path, role: str) -> Atoms:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {role} geometry: {resolved}")
    return io.read(str(resolved))


def read_seed_path(path_dir: Path, pattern: str) -> List[Atoms]:
    """Read an existing NEB band, sorted by file name, as the initial images."""

    resolved = path_dir.expanduser().resolve()
    files = sorted(resolved.glob(pattern))
    if len(files) < 3:
        raise FileNotFoundError(
            f"Need at least 3 images matching {pattern!r} in {resolved}, "
            f"found {len(files)}."
        )

    frames = [io.read(str(path)) for path in files]
    symbols = frames[0].get_chemical_symbols()
    for path, frame in zip(files, frames):
        if frame.get_chemical_symbols() != symbols:
            raise ValueError(f"Atom order differs from the first image: {path}")
    return frames


def read_seed_traj(
    traj_path: Path,
    step: int,
    images_per_step: int,
) -> List[Atoms]:
    """Read one optimizer step of a band out of an ASE NEB trajectory.

    NEB trajectories hold every image of every step back to back, so a step is
    a contiguous slice. ``step`` may be negative to count from the end.
    """

    resolved = traj_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing seed trajectory: {resolved}")

    frames = io.read(str(resolved), index=":")
    n_steps = len(frames) // int(images_per_step)
    if n_steps < 1:
        raise ValueError(
            f"{resolved} holds {len(frames)} frames, fewer than one step of "
            f"{images_per_step} images."
        )

    index = int(step) % n_steps
    start = index * int(images_per_step)
    band = frames[start : start + int(images_per_step)]
    print(
        f"seed trajectory {relative_to_repo(resolved)}: {len(frames)} frames, "
        f"{n_steps} steps, using step {index}"
    )
    return band


def seed_band_positions(images: List[Atoms], seed: List[Atoms]) -> None:
    """Copy an existing band onto the interior images, keeping the endpoints."""

    if len(images) != len(seed):
        raise ValueError(
            f"Seed path has {len(seed)} images but the band has {len(images)}."
        )
    for image, frame in zip(images[1:-1], seed[1:-1]):
        image.set_positions(frame.get_positions())


def report_endpoint_forces(images: List[Atoms], fmax: float) -> dict[str, Any]:
    """Evaluate both fixed endpoints and flag any residual force above fmax.

    A fixed endpoint that is not stationary on this surface drags its neighbour
    in and puts a floor under the reported fmax, which is what stopped
    ``pp_casscf_S1_neb``. Two extra gradient calls make that visible up front.
    """

    report = {}
    for label, image in (("initial", images[0]), ("final", images[-1])):
        forces = image.get_forces()
        image_fmax = float(np.sqrt((np.asarray(forces) ** 2).sum(axis=1)).max())
        report[label] = image_fmax
        verdict = "stationary" if image_fmax <= fmax else "NOT stationary"
        print(f"{label} endpoint |F|max = {image_fmax:.4f} eV/Angstrom ({verdict})")
        if image_fmax > fmax:
            print(
                f"  warning: this endpoint is fixed with a residual force "
                f"{image_fmax / fmax:.0f}x the fmax target, so the band cannot "
                f"relax below it near that end."
            )
    return report


def collect_image_results(images: Any) -> dict[str, Any]:
    energies_ev = []
    forces_ev_per_angstrom = []
    scanner_steps = []
    converged = []

    for image in images:
        energy = image.get_potential_energy()
        forces = image.get_forces()
        record = getattr(image.calc, "last_record", None)

        energies_ev.append(float(energy))
        forces_ev_per_angstrom.append(np.asarray(forces, dtype=float))
        scanner_steps.append(None if record is None else record.scanner_step)
        converged.append(None if record is None else record.converged)

    return {
        "energies_ev": np.asarray(energies_ev, dtype=float),
        "forces_ev_per_angstrom": np.asarray(forces_ev_per_angstrom, dtype=float),
        "scanner_steps": scanner_steps,
        "converged": converged,
    }


def image_spacings(images: Any) -> List[float]:
    """RMS displacement between neighbouring images, as a collapse diagnostic."""

    spacings = []
    for left, right in zip(images[:-1], images[1:]):
        delta = right.get_positions() - left.get_positions()
        spacings.append(float(np.sqrt((delta ** 2).sum(axis=1).mean())))
    return spacings


def save_results(
    result: Any,
    images: Any,
    output_dir: Path,
    args: argparse.Namespace,
    seed_files: List[Path],
    endpoint_forces: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_results = collect_image_results(images)

    fit_arrays = {}
    if result.fit is not None:
        fit_names = [
            "fit_x",
            "fit_y",
            "fit_x_interpolated",
            "fit_y_interpolated",
            "fit_forces",
        ]
        for name, value in zip(fit_names, result.fit):
            fit_arrays[name] = np.asarray(value)

    energies_ev = image_results["energies_ev"]
    spacings = image_spacings(images)
    np.savez(
        output_dir / f"{args.prefix}_results.npz",
        barrier=np.asarray(result.barrier, dtype=float),
        max_force=np.asarray(result.max_force, dtype=float),
        energies_ev=energies_ev,
        relative_energies_ev=energies_ev - energies_ev[0],
        forces_ev_per_angstrom=image_results["forces_ev_per_angstrom"],
        image_spacings_angstrom=np.asarray(spacings, dtype=float),
        **fit_arrays,
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "molecule": MOLECULE_NAME,
        "method": METHOD,
        "state": STATE,
        "path": args.path_label,
        "basis": args.basis,
        "charge": CHARGE,
        "spin": SPIN,
        "seed_path_dir": relative_to_repo(
            args.seed_traj if args.seed_traj is not None else args.path_dir
        ),
        "seed_path_images": [relative_to_repo(path) for path in seed_files],
        "initial_geometry_source": relative_to_repo(
            args.initial if args.initial is not None else seed_files[0]
        ),
        "final_geometry_source": relative_to_repo(
            args.final if args.final is not None else seed_files[-1]
        ),
        "endpoint_fmax_ev_per_angstrom": endpoint_forces,
        "scanner_kwargs": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in SCANNER_KWARGS.items()
        },
        "neb": {
            "n_intermediate_images": len(images) - 2,
            "total_images": len(images),
            "climb": bool(args.climb),
            "spring_constant": args.k,
            "interpolate": None,
            "method": args.neb_method,
            "initial_band": (
                f"seed_traj_step_{args.seed_traj_step}"
                if args.seed_traj is not None
                else "existing_path"
            ),
            "optimizer": OPTIMIZER,
            "fmax": args.fmax,
            "max_steps": args.max_steps,
        },
        "barrier": np.asarray(result.barrier, dtype=float).tolist(),
        "max_force": float(result.max_force),
        "energies_ev": energies_ev.tolist(),
        "relative_energies_ev": (energies_ev - energies_ev[0]).tolist(),
        "image_spacings_angstrom": spacings,
        "scanner_steps": image_results["scanner_steps"],
        "converged": image_results["converged"],
        "output_files": {
            "results_npz": f"{args.prefix}_results.npz",
            "trajectory": f"{args.prefix}.traj",
            "ase_log": f"{args.prefix}_ase.log",
            "images": f"{args.prefix}_*.xyz",
        },
    }

    with (output_dir / f"{args.prefix}_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
    return image_results


def print_settings(
    args: argparse.Namespace,
    output_dir: Path,
    seed_files: List[Path],
) -> None:
    print(f"===== PP {METHOD} {STATE} NEB: planar -> TICT =====")
    source = args.seed_traj if args.seed_traj is not None else args.path_dir
    print(f"seed path        = {relative_to_repo(source)}")
    print(f"seed images      = {len(seed_files)}")
    print(f"initial geometry = {relative_to_repo(seed_files[0] if args.initial is None else args.initial)}")
    print(f"final geometry   = {relative_to_repo(seed_files[-1] if args.final is None else args.final)}")
    print(f"basis = {args.basis}")
    print(f"charge = {CHARGE}")
    print(f"spin = {SPIN}")
    print(f"active orbitals = {SCANNER_KWARGS['act_list']} (1-based)")
    print(f"ncas = {SCANNER_KWARGS['ncas']}")
    print(f"nelecas = {SCANNER_KWARGS['nelecas']}")
    print(f"nroots = {SCANNER_KWARGS['nroots']}")
    print(f"state averaging = {SCANNER_KWARGS['state_weights']} over ['S0', 'S1']")
    print(f"target root = {SCANNER_KWARGS['target_root']} ({STATE})")
    print(f"intermediate images = {len(seed_files) - 2}")
    print(f"climb = {bool(args.climb)}, k = {args.k}, tangent = {args.neb_method}")
    print(f"initial band = {'seed trajectory' if args.seed_traj is not None else 'existing path'}")
    print(f"optimizer = {OPTIMIZER}, fmax = {args.fmax}, max steps = {args.max_steps}")
    print(f"output directory = {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_traj is not None:
        seed = read_seed_traj(
            args.seed_traj, args.seed_traj_step, args.seed_traj_images
        )
        seed_files = [args.seed_traj] * len(seed)
    else:
        seed_files = sorted(args.path_dir.expanduser().resolve().glob(args.path_pattern))
        seed = read_seed_path(args.path_dir, args.path_pattern)
    print_settings(args, output_dir, seed_files)

    initial = seed[0] if args.initial is None else read_geometry(args.initial, "initial")
    final = seed[-1] if args.final is None else read_geometry(args.final, "final")
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("Initial and final geometries must have the same atom order.")
    if initial.get_chemical_symbols() != seed[0].get_chemical_symbols():
        raise ValueError("Endpoint atom order differs from the seed path.")

    calc_factory = make_casscf_active_space_calculator_factory(
        scanner_kwargs=SCANNER_KWARGS,
        basis=args.basis,
        charge=CHARGE,
        spin=SPIN,
        verbose=args.verbose,
        label_prefix=f"{args.prefix}_image",
    )

    images = build_neb_images(
        initial=initial,
        final=final,
        n_intermediate_images=len(seed) - 2,
        calculator_factory=calc_factory,
    )
    seed_band_positions(images, seed)

    print("\n===== endpoint check =====")
    endpoint_forces = report_endpoint_forces(images, args.fmax)

    result = run_neb_calculation(
        images=images,
        climb=bool(args.climb),
        k=args.k,
        interpolate=None,
        neb_kwargs={"method": args.neb_method},
        optimizer=OPTIMIZER,
        fmax=args.fmax,
        steps=args.max_steps,
        trajectory=output_dir / f"{args.prefix}.traj",
        logfile=output_dir / f"{args.prefix}_ase.log",
    )

    write_neb_images(images, output_dir, prefix=args.prefix)
    image_results = save_results(
        result, images, output_dir, args, seed_files, endpoint_forces
    )

    energies_ev = image_results["energies_ev"]
    relative_ev = energies_ev - energies_ev[0]
    spacings = image_spacings(images)
    print(f"\n===== PP {METHOD} {STATE} NEB summary =====")
    print(f"Images        : {len(images)}")
    print(f"Max force     : {result.max_force:.6f} eV/Angstrom")
    print(f"Barrier       : {np.asarray(result.barrier, dtype=float).tolist()}")
    print("Image energies relative to the planar endpoint (eV):")
    for index, value in enumerate(relative_ev):
        print(f"  image {index:02d} : {value: .6f}")
    print("Neighbour spacing (Angstrom), a collapsed image reads near zero:")
    for index, value in enumerate(spacings):
        print(f"  {index:02d}-{index + 1:02d} : {value:.4f}")
    print(f"Output        : {relative_to_repo(output_dir)}")


if __name__ == "__main__":
    main()
