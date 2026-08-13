from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms


REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root = str(REPO_ROOT)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from utils.neb_calculator import (  # noqa: E402
    build_neb_images,
    make_casci_active_space_calculator_factory,
    run_neb_calculation,
    write_neb_images,
)


BASIS = "ccpvdz"
CHARGE = 0
SPIN = 0
VERBOSE = 4

SCANNER_KWARGS = {
    "ncas": 2,
    "nelecas": (1, 1),
    "act_list": [38, 39],
    "nroots": 3,
    "target_root": 1,
    "conv_tol": 1e-10,
    "max_cycle": 200,
    "act_base": 1,
}

N_INTERMEDIATE_IMAGES = 7
CLIMB = True
SPRING_CONSTANT = 0.6
INTERPOLATE = "idpp"
OPTIMIZER = "FIRE"
FMAX = 0.05
MAX_STEPS = 500

OUTPUT_DIR = Path("pp_casci_S1_neb_outputs")
IMAGE_PREFIX = "pp_casci_S1_neb"


INITIAL_GEOMETRY = """
C          -3.79224212        0.61097518        0.00199758
C          -2.50707332        1.28018316        0.00307720
C          -3.53988514       -0.73978903       -0.00148764
C          -2.15332314       -0.89569832       -0.00256100
H          -4.74286408        1.11577968        0.00353995
H          -4.25182330       -1.54658550       -0.00320711
N          -1.52941228        0.39613039        0.00033073
H          -1.56379928       -1.78838714       -0.00516281
H          -2.32353667        2.33769803        0.00583277
C          -0.17956367        0.60189668        0.00021543
C           0.73914685       -0.49599094       -0.00257775
C           2.08869542       -0.20483711       -0.00134181
H           0.42998547       -1.52715106       -0.00602931
N           2.62590142        1.00379291        0.00222881
H           2.79077460       -1.03236321       -0.00346421
C           1.74830318        2.02790161        0.00455515
C           0.39026289        1.91311098        0.00377652
H           2.18962367        3.01785861        0.00724252
H          -0.21468493        2.80676781        0.00564510
"""

FINAL_GEOMETRY = """
C          -3.53324601        0.40167379       -0.73575608
C          -2.13148242        0.71623979       -0.80181565
C          -3.69996155       -0.35491854        0.37943138
C          -2.39069070       -0.49224367        0.98156585
H          -4.27875807        0.71821389       -1.44559747
H          -4.60760092       -0.78321477        0.76804609
N          -1.50530019        0.15731992        0.24562132
H          -2.08564048       -1.01117178        1.87160176
H          -1.57982871        1.29140436       -1.52279687
C          -0.07088887        0.26377329        0.50973059
C           0.77137766       -0.42346217       -0.43827043
C           1.97227918        0.13660367       -0.78625327
H           0.50615167       -1.40375303       -0.81301119
N           2.43189189        1.32889332       -0.38372425
H           2.63062902       -0.41583251       -1.44981018
C           1.60982503        2.00682730        0.42834032
C           0.38197841        1.58586326        0.86664244
H           1.97318325        2.97694371        0.75354607
H          -0.19453101        2.21222008        1.53527105
"""


def atoms_from_geometry_block(geometry_block: str) -> Atoms:
    symbols = []
    positions = []
    for line in geometry_block.strip().splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"Invalid geometry line: {line!r}")
        symbols.append(fields[0])
        positions.append([float(x) for x in fields[1:]])
    return Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))


def collect_image_results(images):
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


def save_results(result, images, output_dir: Path) -> None:
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

    np.savez(
        output_dir / "pp_casci_S1_neb_results.npz",
        barrier=np.asarray(result.barrier, dtype=float),
        max_force=np.asarray(result.max_force, dtype=float),
        energies_ev=image_results["energies_ev"],
        forces_ev_per_angstrom=image_results["forces_ev_per_angstrom"],
        **fit_arrays,
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "method": "CASCI",
        "state": "S1",
        "basis": BASIS,
        "charge": CHARGE,
        "spin": SPIN,
        "scanner_kwargs": SCANNER_KWARGS,
        "neb": {
            "n_intermediate_images": N_INTERMEDIATE_IMAGES,
            "total_images": len(images),
            "climb": CLIMB,
            "spring_constant": SPRING_CONSTANT,
            "interpolate": INTERPOLATE,
            "optimizer": OPTIMIZER,
            "fmax": FMAX,
            "max_steps": MAX_STEPS,
        },
        "barrier": np.asarray(result.barrier, dtype=float).tolist(),
        "max_force": float(result.max_force),
        "scanner_steps": image_results["scanner_steps"],
        "converged": image_results["converged"],
        "output_files": {
            "results_npz": "pp_casci_S1_neb_results.npz",
            "images": f"{IMAGE_PREFIX}_*.xyz",
        },
    }

    with (output_dir / "pp_casci_S1_neb_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    initial = atoms_from_geometry_block(INITIAL_GEOMETRY)
    final = atoms_from_geometry_block(FINAL_GEOMETRY)

    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("Initial and final geometries must have the same atom order.")

    calc_factory = make_casci_active_space_calculator_factory(
        scanner_kwargs=SCANNER_KWARGS,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        verbose=VERBOSE,
        label_prefix="pp_casci_S1_neb_image",
    )

    images = build_neb_images(
        initial=initial,
        final=final,
        n_intermediate_images=N_INTERMEDIATE_IMAGES,
        calculator_factory=calc_factory,
    )

    result = run_neb_calculation(
        images=images,
        climb=CLIMB,
        k=SPRING_CONSTANT,
        interpolate=INTERPOLATE,
        optimizer=OPTIMIZER,
        fmax=FMAX,
        steps=MAX_STEPS,
        trajectory=OUTPUT_DIR / "pp_casci_S1_neb.traj",
        logfile=OUTPUT_DIR / "pp_casci_S1_neb_ase.log",
    )

    write_neb_images(images, OUTPUT_DIR, prefix=IMAGE_PREFIX)
    save_results(result, images, OUTPUT_DIR)

    print("\n===== PP CASCI S1 NEB summary =====")
    print(f"Images              : {len(images)}")
    print(f"Barrier tuple       : {result.barrier}")
    print(f"Max NEB force       : {result.max_force}")
    print(f"Output directory    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
