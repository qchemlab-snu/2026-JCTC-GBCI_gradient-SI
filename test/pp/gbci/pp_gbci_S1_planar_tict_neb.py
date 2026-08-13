from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root = str(REPO_ROOT)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from utils.active_space_tracking_optimizer import (  # noqa: E402
    GBCI_Active_Root_Tracking_Optimizer,
)
from utils.neb_calculator import (  # noqa: E402
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    build_neb_images,
    make_gradient_scanner_calculator_factory,
    run_neb_calculation,
    write_neb_images,
)


BASIS = "ccpvdz"
CHARGE = 0
SPIN = 0
UNIT = "Angstrom"
VERBOSE = 4

N_INTERMEDIATE_IMAGES = 7
CLIMB = True
SPRING_CONSTANT = 0.6
INTERPOLATE = "idpp"
OPTIMIZER = "FIRE"
FMAX = 0.05
MAX_STEPS = 500

OUTPUT_DIR = SCRIPT_DIR / "pp_gbci_S1_planar_tict_neb_outputs"
OUTPUT_PREFIX = "pp_gbci_S1_planar_tict_neb"

SCANNER_KWARGS = {
    "ncas": 2,
    "nelecas": (1, 1),
    "act_list": [38, 39],
    "nroots": 3,
    "groupA": {
        "mol1": list(range(0, 9)),
        "mol2": list(range(9, 19)),
    },
    "target_root": 1,
    "conv_tol": 1e-10,
    "max_cycle": 200,
    "act_base": 1,
}

PLANAR_GEOMETRY = """
C          -3.82982877        0.63275106       -0.00000000
C          -2.53920964        1.24425728       -0.00000000
C          -3.60199675       -0.70944046        0.00000000
C          -2.18188051       -0.86081364        0.00000000
H          -4.76594101        1.16420457       -0.00000000
H          -4.31029725       -1.52002832        0.00000000
N          -1.57921306        0.32434146       -0.00000000
H          -1.61378850       -1.77431135        0.00000000
H          -2.30438178        2.29404036       -0.00000000
C          -0.12707750        0.57082026       -0.00000000
C           0.80402415       -0.51138841       -0.00000000
C           2.13410385       -0.20053361       -0.00000000
H           0.51117904       -1.55279152       -0.00000000
N           2.66314252        1.04412675        0.00000000
H           2.85312721       -1.01340300       -0.00000000
C           1.75323787        2.04461950        0.00000000
C           0.39502111        1.89959720        0.00000000
H           2.16398591        3.04913117        0.00000000
H          -0.22481973        2.78620063        0.00000000
"""

TICT_GEOMETRY = """
C          -3.56908686        0.38850017       -0.72618231
C          -2.17759969        0.71497823       -0.81369582
C          -3.71087983       -0.36762545        0.39568644
C          -2.40073116       -0.48574760        0.96857388
H          -4.32778300        0.69630776       -1.42649481
H          -4.60838875       -0.80455689        0.80057895
N          -1.51714476        0.17268826        0.21848606
H          -2.07622655       -1.00061387        1.85823017
H          -1.64648870        1.29643628       -1.55061414
C          -0.05923924        0.28201197        0.48515887
C           0.78637909       -0.40952208       -0.45824355
C           2.00533781        0.12432041       -0.76333160
H           0.49993679       -1.37505496       -0.86003379
N           2.48078204        1.30420741       -0.33110595
H           2.67052448       -0.43338105       -1.41639672
C           1.64196652        1.99984282        0.45468508
C           0.39655898        1.60246120        0.84837223
H           2.01223642        2.96437426        0.79020897
H          -0.20076643        2.24175305        1.48887950
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PP GBCI S1 NEB from the planar minimum to the TICT structure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for NEB trajectory, image xyz files, and summaries.",
    )
    parser.add_argument(
        "--n-intermediate-images",
        type=int,
        default=N_INTERMEDIATE_IMAGES,
        help="Number of NEB images between the planar and TICT endpoints.",
    )
    parser.add_argument("--fmax", type=float, default=FMAX, help="ASE NEB force threshold.")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Maximum ASE steps.")
    parser.add_argument(
        "--spring-constant",
        type=float,
        default=SPRING_CONSTANT,
        help="ASE NEB spring constant.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("FIRE", "LBFGS"),
        default=OPTIMIZER,
        help="ASE optimizer used for the NEB path.",
    )
    parser.add_argument(
        "--interpolate",
        choices=("idpp", "linear", "none"),
        default=INTERPOLATE,
        help="Initial image interpolation before NEB optimization.",
    )
    parser.add_argument(
        "--no-climb",
        action="store_true",
        default=not CLIMB,
        help="Disable climbing-image NEB.",
    )
    parser.add_argument(
        "--interpolate-only",
        action="store_true",
        help="Write interpolated endpoint images without running GBCI gradients.",
    )
    return parser.parse_args()


def atoms_from_geometry_block(geometry_block: str) -> Atoms:
    symbols = []
    positions = []
    for line in geometry_block.strip().splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"Invalid geometry line: {line!r}")
        symbols.append(fields[0])
        positions.append([float(value) for value in fields[1:]])
    return Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))


def make_calculator_factory(label_prefix: str) -> Any:
    def scanner_factory() -> GBCI_Active_Root_Tracking_Optimizer:
        return GBCI_Active_Root_Tracking_Optimizer(**SCANNER_KWARGS)

    return make_gradient_scanner_calculator_factory(
        scanner_factory=scanner_factory,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=VERBOSE,
        label_prefix=label_prefix,
    )


def get_image_energy_ev(image: Atoms) -> float:
    try:
        return float(image.get_potential_energy())
    except Exception:
        return float("nan")


def get_image_force_array(image: Atoms) -> np.ndarray:
    try:
        return np.asarray(image.get_forces(), dtype=float)
    except Exception:
        return np.full((len(image), 3), np.nan, dtype=float)


def collect_image_results(images: list[Atoms]) -> dict[str, Any]:
    energies_ev = []
    forces_ev_per_angstrom = []
    scanner_steps = []
    converged = []

    for image in images:
        energies_ev.append(get_image_energy_ev(image))
        forces_ev_per_angstrom.append(get_image_force_array(image))
        record = getattr(getattr(image, "calc", None), "last_record", None)
        scanner_steps.append(None if record is None else record.scanner_step)
        converged.append(None if record is None else record.converged)

    energies_array = np.asarray(energies_ev, dtype=float)
    return {
        "energies_ev": energies_array,
        "relative_energies_ev": energies_array - energies_array[0],
        "forces_ev_per_angstrom": np.asarray(forces_ev_per_angstrom, dtype=float),
        "scanner_steps": scanner_steps,
        "converged": converged,
    }


def to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def max_abs_force(force_array: np.ndarray) -> float:
    if np.all(np.isnan(force_array)):
        return float("nan")
    return float(np.nanmax(np.abs(force_array)))


def write_summary_csv(image_results: dict[str, Any], csv_path: Path) -> None:
    energies_ev = image_results["energies_ev"]
    relative_energies_ev = image_results["relative_energies_ev"]
    forces = image_results["forces_ev_per_angstrom"]

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "image_index",
                "point_label",
                "energy_ev",
                "relative_energy_ev",
                "max_force_abs_ev_per_angstrom",
                "scanner_step",
                "converged",
            ],
        )
        writer.writeheader()
        for idx, energy_ev in enumerate(energies_ev):
            if idx == 0:
                point_label = "planar"
            elif idx == len(energies_ev) - 1:
                point_label = "TICT"
            else:
                point_label = ""

            writer.writerow(
                {
                    "image_index": idx,
                    "point_label": point_label,
                    "energy_ev": energy_ev,
                    "relative_energy_ev": relative_energies_ev[idx],
                    "max_force_abs_ev_per_angstrom": max_abs_force(forces[idx]),
                    "scanner_step": image_results["scanner_steps"][idx],
                    "converged": image_results["converged"][idx],
                }
            )


def save_results(
    images: list[Atoms],
    output_dir: Path,
    args: argparse.Namespace,
    result: Any | None,
) -> None:
    image_results = collect_image_results(images)

    np.savez(
        output_dir / f"{OUTPUT_PREFIX}_results.npz",
        image_indices=np.arange(len(images), dtype=int),
        energies_ev=image_results["energies_ev"],
        relative_energies_ev=image_results["relative_energies_ev"],
        forces_ev_per_angstrom=image_results["forces_ev_per_angstrom"],
        barrier=np.asarray([] if result is None else result.barrier, dtype=float),
        max_force=np.asarray(np.nan if result is None else result.max_force, dtype=float),
    )
    write_summary_csv(image_results, output_dir / f"{OUTPUT_PREFIX}_summary.csv")

    interpolation = None if args.interpolate == "none" else args.interpolate
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "method": "GBCI",
        "state": "S1",
        "path": "planar_to_TICT",
        "basis": BASIS,
        "charge": CHARGE,
        "spin": SPIN,
        "unit": UNIT,
        "hartree_to_ev": HARTREE_TO_EV,
        "bohr_to_angstrom": BOHR_TO_ANGSTROM,
        "scanner_kwargs": SCANNER_KWARGS,
        "neb": {
            "n_intermediate_images": args.n_intermediate_images,
            "total_images": len(images),
            "climb": not args.no_climb,
            "spring_constant": args.spring_constant,
            "interpolate": interpolation,
            "optimizer": args.optimizer,
            "fmax": None if result is None else args.fmax,
            "max_steps": None if result is None else args.steps,
            "interpolate_only": bool(args.interpolate_only),
        },
        "barrier": None if result is None else np.asarray(result.barrier).tolist(),
        "max_force": None if result is None else float(result.max_force),
        "scanner_steps": image_results["scanner_steps"],
        "converged": image_results["converged"],
        "output_dir": relative_to_repo(output_dir),
        "output_files": {
            "trajectory": f"{OUTPUT_PREFIX}.traj",
            "ase_log": f"{OUTPUT_PREFIX}_ase.log",
            "results_npz": f"{OUTPUT_PREFIX}_results.npz",
            "summary_csv": f"{OUTPUT_PREFIX}_summary.csv",
            "image_xyz": f"{OUTPUT_PREFIX}_*.xyz",
        },
    }
    with (output_dir / f"{OUTPUT_PREFIX}_metadata.json").open("w") as fh:
        json.dump(to_serializable(metadata), fh, indent=2)


def linearly_interpolate_images(images: list[Atoms]) -> None:
    start = images[0].get_positions()
    end = images[-1].get_positions()
    for idx, image in enumerate(images):
        fraction = idx / (len(images) - 1)
        image.set_positions((1.0 - fraction) * start + fraction * end)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = atoms_from_geometry_block(PLANAR_GEOMETRY)
    final = atoms_from_geometry_block(TICT_GEOMETRY)
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("Planar and TICT geometries must have the same atom order.")

    calculator_factory = None
    if not args.interpolate_only:
        calculator_factory = make_calculator_factory(label_prefix=OUTPUT_PREFIX)

    images = build_neb_images(
        initial=initial,
        final=final,
        n_intermediate_images=args.n_intermediate_images,
        calculator_factory=calculator_factory,
    )

    result = None
    interpolation = None if args.interpolate == "none" else args.interpolate
    if args.interpolate_only:
        linearly_interpolate_images(images)
    else:
        result = run_neb_calculation(
            images=images,
            climb=not args.no_climb,
            k=args.spring_constant,
            interpolate=interpolation,
            optimizer=args.optimizer,
            fmax=args.fmax,
            steps=args.steps,
            trajectory=output_dir / f"{OUTPUT_PREFIX}.traj",
            logfile=output_dir / f"{OUTPUT_PREFIX}_ase.log",
        )

    write_neb_images(images, output_dir, prefix=OUTPUT_PREFIX)
    save_results(images, output_dir, args, result)

    print("\n===== PP GBCI S1 planar-to-TICT NEB summary =====")
    print(f"Images           : {len(images)}")
    print(f"Intermediate     : {args.n_intermediate_images}")
    print(f"Climbing image   : {not args.no_climb}")
    print(f"Interpolate only : {args.interpolate_only}")
    if result is not None:
        print(f"Barrier tuple    : {result.barrier}")
        print(f"Max NEB force    : {result.max_force}")
    print(f"Output directory : {output_dir}")


if __name__ == "__main__":
    main()
