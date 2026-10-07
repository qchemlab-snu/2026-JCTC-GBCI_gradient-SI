from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.neb_calculator import (  # noqa: E402
    build_neb_images,
    make_casci_active_space_calculator_factory,
    run_neb_calculation,
    write_neb_images,
)


RUN_NAME = "ch2nh2_casci_S0_meci_neb_5o8e_ccpvtz"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / RUN_NAME

# Endpoints are read from the finished runs rather than pasted in, the way the
# *_imag_followup.py and *_sigma1_continue.py scripts pick up their parent run.
# Both must have completed before this script is submitted.
INITIAL_XYZ = (
    SCRIPT_DIR / "outputs" / "ch2nh2_casci_S0_opt_5o8e_ccpvtz" / "ch2nh2_casci_S0_opt_5o8e_ccpvtz_optimized.xyz"
)
FINAL_XYZ = (
    SCRIPT_DIR / "outputs" / "ch2nh2_casci_S0S1_meci_5o8e_ccpvtz" / "ch2nh2_casci_S0S1_meci_5o8e_ccpvtz_optimized.xyz"
)

BASIS = "cc-pVTZ"
CHARGE = 1
SPIN = 0
UNIT = "Angstrom"
VERBOSE = 4

# The band is optimized on S0 (target_root=0): the path leaves the S0 minimum
# and climbs to the geometry where S0 and S1 touch.  S0 and S1 become
# degenerate at the final image, so root tracking is hardest there -- check the
# per-image energies in the summary CSV before trusting the barrier.
SCANNER_KWARGS = {
    "ncas": 5,
    "nelecas": (4, 4),
    "act_list": [5, 6, 7, 8, 9],
    "nroots": 3,
    "target_root": 0,
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

HARTREE2EV = 27.211386245988


def read_xyz(path: Path) -> Atoms:
    if not path.exists():
        raise FileNotFoundError(
            f"Endpoint geometry not found: {path}\n"
            "Run the S0 optimization and the MECI search for this method first."
        )
    lines = path.read_text().strip().splitlines()
    natm = int(lines[0].split()[0])
    symbols, positions = [], []
    for line in lines[2:2 + natm]:
        fields = line.split()
        symbols.append(fields[0])
        positions.append([float(x) for x in fields[1:4]])
    return Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))


def make_calculator_factory(label_prefix: str):
    return make_casci_active_space_calculator_factory(
        scanner_kwargs=SCANNER_KWARGS,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=VERBOSE,
        label_prefix=label_prefix,
    )


def collect_image_results(images):
    energies_ev, forces, steps, converged = [], [], [], []
    for image in images:
        energies_ev.append(float(image.get_potential_energy()))
        forces.append(np.asarray(image.get_forces(), dtype=float))
        record = getattr(image.calc, "last_record", None)
        steps.append(None if record is None else record.scanner_step)
        converged.append(None if record is None else record.converged)
    return {
        "energies_ev": np.asarray(energies_ev, dtype=float),
        "forces_ev_per_angstrom": np.asarray(forces, dtype=float),
        "scanner_steps": steps,
        "converged": converged,
    }


def write_summary_csv(image_results, path: Path) -> None:
    e = image_results["energies_ev"]
    rel = e - e[0]
    with path.open("w") as fh:
        fh.write("image,energy_eV,relative_eV,max_force_eV_per_A,scanner_step,converged\n")
        for i, energy in enumerate(e):
            fmax = float(np.abs(image_results["forces_ev_per_angstrom"][i]).max())
            fh.write(
                "%d,%.10f,%.6f,%.6e,%s,%s\n"
                % (i, energy, rel[i], fmax,
                   image_results["scanner_steps"][i], image_results["converged"][i])
            )


def save_results(result, images, output_dir: Path) -> dict:
    image_results = collect_image_results(images)

    fit_arrays = {}
    if result.fit is not None:
        names = ["fit_x", "fit_y", "fit_x_interpolated", "fit_y_interpolated", "fit_forces"]
        for name, value in zip(names, result.fit):
            fit_arrays[name] = np.asarray(value)

    np.savez(
        output_dir / f"{RUN_NAME}_results.npz",
        barrier=np.asarray(result.barrier, dtype=float),
        max_force=np.asarray(result.max_force, dtype=float),
        energies_ev=image_results["energies_ev"],
        forces_ev_per_angstrom=image_results["forces_ev_per_angstrom"],
        **fit_arrays,
    )
    write_summary_csv(image_results, output_dir / f"{RUN_NAME}_profile.csv")

    e = image_results["energies_ev"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "molecule": "CH2NH2+",
        "method": "CASCI",
        "band_state": "S0",
        "basis": BASIS,
        "charge": CHARGE,
        "spin": SPIN,
        "unit": UNIT,
        "initial_xyz": str(INITIAL_XYZ),
        "final_xyz": str(FINAL_XYZ),
        "scanner_kwargs": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in SCANNER_KWARGS.items()},
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
        "barrier_eV": np.asarray(result.barrier, dtype=float).tolist(),
        "max_neb_force_eV_per_A": float(result.max_force),
        "energies_eV": e.tolist(),
        "relative_energies_eV": (e - e[0]).tolist(),
        "endpoint_gap_eV": float(e[-1] - e[0]),
        "highest_image_index": int(np.argmax(e)),
        "highest_relative_eV": float(e.max() - e[0]),
        "scanner_steps": image_results["scanner_steps"],
        "image_converged": image_results["converged"],
    }
    (output_dir / f"{RUN_NAME}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    initial = read_xyz(INITIAL_XYZ)
    final = read_xyz(FINAL_XYZ)
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("Endpoint geometries must have the same atom order.")

    print("===== CH2NH2+ CASCI S0 -> MECI NEB (8e,5o / cc-pVTZ) =====")
    print(f"initial (S0 minimum) = {INITIAL_XYZ}")
    print(f"final   (S0/S1 MECI) = {FINAL_XYZ}")
    print(f"basis = {BASIS}")
    print(f"scanner_kwargs = {SCANNER_KWARGS}")
    print(f"images = {N_INTERMEDIATE_IMAGES + 2} (climb={CLIMB}, k={SPRING_CONSTANT})")
    print(f"output_dir = {OUTPUT_DIR}")

    images = build_neb_images(
        initial=initial,
        final=final,
        n_intermediate_images=N_INTERMEDIATE_IMAGES,
        calculator_factory=make_calculator_factory(f"{RUN_NAME}_image"),
    )

    result = run_neb_calculation(
        images=images,
        climb=CLIMB,
        k=SPRING_CONSTANT,
        interpolate=INTERPOLATE,
        optimizer=OPTIMIZER,
        fmax=FMAX,
        steps=MAX_STEPS,
        trajectory=OUTPUT_DIR / f"{RUN_NAME}.traj",
        logfile=OUTPUT_DIR / f"{RUN_NAME}_ase.log",
    )

    write_neb_images(images, OUTPUT_DIR, prefix=RUN_NAME)
    summary = save_results(result, images, OUTPUT_DIR)

    print("\n===== NEB summary =====")
    print(f"Images                : {len(images)}")
    print(f"Barrier (ASE)         : {result.barrier}")
    print(f"Max NEB force         : {result.max_force:.6e} eV/A")
    print(f"Highest image         : {summary['highest_image_index']} "
          f"at {summary['highest_relative_eV']:.4f} eV above the S0 minimum")
    print(f"Endpoint (MECI - S0)  : {summary['endpoint_gap_eV']:.4f} eV")
    print(f"Saved under           : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
