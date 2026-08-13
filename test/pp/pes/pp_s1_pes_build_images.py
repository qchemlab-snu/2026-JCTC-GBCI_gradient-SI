from __future__ import annotations

import argparse
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
    build_neb_images,
    make_casci_active_space_calculator_factory,
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

OUTPUT_ROOT = SCRIPT_DIR / "pp_s1_pes_image_outputs"

CASCI_SCANNER_KWARGS = {
    "ncas": 2,
    "nelecas": (1, 1),
    "act_list": [38, 39],
    "nroots": 3,
    "target_root": 1,
    "conv_tol": 1e-10,
    "max_cycle": 200,
    "act_base": 1,
}

GBCI_SCANNER_KWARGS = {
    **CASCI_SCANNER_KWARGS,
    "groupA": {
        "mol1": list(range(0, 9)),
        "mol2": list(range(9, 19)),
    },
}

CASCI_GROUND_STATE_GEOMETRY = """
C          -3.73143993        0.65188355       -0.21325648
C          -2.51817307        1.24685434       -0.32807972
C          -3.50289652       -0.69490677        0.21335648
C          -2.16127315       -0.85635017        0.32814287
H          -4.68168724        1.11637761       -0.41896743
H          -4.24671897       -1.44686974        0.41907899
N          -1.54245055        0.33052230       -0.00002689
H          -1.59175012       -1.70747153        0.65557748
H          -2.26130062        2.23819524       -0.65554408
C          -0.16244599        0.56467722       -0.00003783
C           0.74312415       -0.46184653       -0.25010711
C           2.09644794       -0.16766994       -0.23075471
H           0.41102441       -1.46229433       -0.48010942
N           2.59760033        1.03308530       -0.00002661
H           2.81355702       -0.95592807       -0.42551776
C           1.72840944        2.00130767        0.23070888
C           0.35380603        1.83246734        0.25004055
H           2.14530813        2.98201506        0.42549135
H          -0.28975431        2.66733240        0.48006407
"""

CASCI_PLANAR_GEOMETRY = """
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

GBCI_GROUND_STATE_GEOMETRY = """
C          -3.73165109        0.65261394        0.21046165
C          -2.51844176        1.24813744        0.32402918
C          -3.50280718       -0.69569851       -0.21064153
C          -2.16104397       -0.85765020       -0.32408962
H          -4.68207218        1.11787688        0.41364067
H          -4.24650532       -1.44846007       -0.41390279
N          -1.54229791        0.33057871       -0.00001358
H          -1.59150103       -1.70999197       -0.64820999
H          -2.26201418        2.24065297        0.64817473
C          -0.16256711        0.56471306        0.00000902
C           0.74322282       -0.46214327        0.24828788
C           2.09653597       -0.16797696        0.22910098
H           0.41137498       -1.46298616        0.47680941
N           2.59766239        1.03304989        0.00006957
H           2.81363506       -0.95656098        0.42262348
C           1.72840095        2.00153485       -0.22898633
C           0.35378966        1.83289711       -0.24823875
H           2.14526200        2.98252019       -0.42248307
H          -0.28959495        2.66827301       -0.47677336
"""

GBCI_PLANAR_GEOMETRY = """
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

GBCI_TICT_GEOMETRY = """
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

SEGMENTS = {
    "casci_ground_to_planar": {
        "method": "CASCI",
        "start_label": "ground_state_geometry",
        "end_label": "planar",
        "start_geometry": CASCI_GROUND_STATE_GEOMETRY,
        "end_geometry": CASCI_PLANAR_GEOMETRY,
    },
    "gbci_ground_to_planar": {
        "method": "GBCI",
        "start_label": "ground_state_geometry",
        "end_label": "planar",
        "start_geometry": GBCI_GROUND_STATE_GEOMETRY,
        "end_geometry": GBCI_PLANAR_GEOMETRY,
    },
    "gbci_planar_to_tict": {
        "method": "GBCI",
        "start_label": "planar",
        "end_label": "TICT",
        "start_geometry": GBCI_PLANAR_GEOMETRY,
        "end_geometry": GBCI_TICT_GEOMETRY,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the additional PP S1 PES NEB image segments for CASCI/GBCI."
    )
    parser.add_argument(
        "--segments",
        nargs="+",
        choices=sorted(SEGMENTS),
        default=sorted(SEGMENTS),
        help="Image segments to calculate.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for segment image outputs.",
    )
    parser.add_argument(
        "--n-intermediate-images",
        type=int,
        default=N_INTERMEDIATE_IMAGES,
        help="Number of images between the endpoints for each segment.",
    )
    parser.add_argument("--fmax", type=float, default=FMAX, help="ASE NEB force threshold.")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="ASE optimizer steps.")
    parser.add_argument(
        "--interpolate-only",
        action="store_true",
        help="Write linearly interpolated endpoint images without running CASCI/GBCI.",
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


def make_calculator_factory(method: str, label_prefix: str) -> Any:
    if method == "CASCI":
        return make_casci_active_space_calculator_factory(
            scanner_kwargs=CASCI_SCANNER_KWARGS,
            basis=BASIS,
            charge=CHARGE,
            spin=SPIN,
            unit=UNIT,
            verbose=VERBOSE,
            label_prefix=label_prefix,
        )

    if method == "GBCI":
        def scanner_factory() -> GBCI_Active_Root_Tracking_Optimizer:
            return GBCI_Active_Root_Tracking_Optimizer(**GBCI_SCANNER_KWARGS)

        return make_gradient_scanner_calculator_factory(
            scanner_factory=scanner_factory,
            basis=BASIS,
            charge=CHARGE,
            spin=SPIN,
            unit=UNIT,
            verbose=VERBOSE,
            label_prefix=label_prefix,
        )

    raise ValueError(f"Unknown method: {method}")


def save_segment_summary(
    segment_name: str,
    segment: dict[str, Any],
    images: list[Atoms],
    output_dir: Path,
    result: Any | None,
    fmax: float,
    max_steps: int,
) -> None:
    energies_ev = []
    scanner_steps = []
    converged = []
    for image in images:
        try:
            energies_ev.append(float(image.get_potential_energy()))
        except Exception:
            energies_ev.append(np.nan)
        record = getattr(getattr(image, "calc", None), "last_record", None)
        scanner_steps.append(None if record is None else record.scanner_step)
        converged.append(None if record is None else record.converged)

    energies_array = np.asarray(energies_ev, dtype=float)
    np.savez(
        output_dir / f"pp_s1_pes_{segment_name}_results.npz",
        image_indices=np.arange(len(images), dtype=int),
        energies_ev=energies_array,
        relative_to_first_ev=energies_array - energies_array[0],
    )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "segment": segment_name,
        "method": segment["method"],
        "state": "S1",
        "start_label": segment["start_label"],
        "end_label": segment["end_label"],
        "basis": BASIS,
        "charge": CHARGE,
        "spin": SPIN,
        "unit": UNIT,
        "scanner_kwargs": (
            CASCI_SCANNER_KWARGS if segment["method"] == "CASCI" else GBCI_SCANNER_KWARGS
        ),
        "neb": {
            "n_intermediate_images": len(images) - 2,
            "total_images": len(images),
            "climb": CLIMB,
            "spring_constant": SPRING_CONSTANT,
            "interpolate": INTERPOLATE,
            "optimizer": OPTIMIZER,
            "fmax": None if result is None else float(fmax),
            "max_steps": None if result is None else int(max_steps),
        },
        "barrier": None if result is None else np.asarray(result.barrier).tolist(),
        "max_force": None if result is None else float(result.max_force),
        "scanner_steps": scanner_steps,
        "converged": converged,
        "output_files": {
            "results_npz": f"pp_s1_pes_{segment_name}_results.npz",
            "images": f"pp_s1_pes_{segment_name}_*.xyz",
        },
    }

    with (output_dir / f"pp_s1_pes_{segment_name}_metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)


def run_segment(segment_name: str, args: argparse.Namespace) -> None:
    segment = SEGMENTS[segment_name]
    output_dir = args.output_root / segment_name
    image_prefix = f"pp_s1_pes_{segment_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    initial = atoms_from_geometry_block(segment["start_geometry"])
    final = atoms_from_geometry_block(segment["end_geometry"])
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError(f"Atom order mismatch in segment {segment_name}.")

    calculator_factory = None
    if not args.interpolate_only:
        calculator_factory = make_calculator_factory(
            method=segment["method"],
            label_prefix=image_prefix,
        )

    images = build_neb_images(
        initial=initial,
        final=final,
        n_intermediate_images=args.n_intermediate_images,
        calculator_factory=calculator_factory,
    )

    result = None
    if args.interpolate_only:
        start = initial.get_positions()
        end = final.get_positions()
        for idx, image in enumerate(images):
            fraction = idx / (len(images) - 1)
            image.set_positions((1.0 - fraction) * start + fraction * end)
    else:
        result = run_neb_calculation(
            images=images,
            climb=CLIMB,
            k=SPRING_CONSTANT,
            interpolate=INTERPOLATE,
            optimizer=OPTIMIZER,
            fmax=args.fmax,
            steps=args.steps,
            trajectory=output_dir / f"{image_prefix}.traj",
            logfile=output_dir / f"{image_prefix}_ase.log",
        )

    write_neb_images(images, output_dir, prefix=image_prefix)
    save_segment_summary(
        segment_name,
        segment,
        images,
        output_dir,
        result,
        fmax=args.fmax,
        max_steps=args.steps,
    )

    print("\n===== PP S1 PES image segment =====")
    print(f"Segment          : {segment_name}")
    print(f"Method           : {segment['method']}")
    print(f"Endpoints        : {segment['start_label']} -> {segment['end_label']}")
    print(f"Images           : {len(images)}")
    print(f"Output directory : {output_dir}")


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()

    print("===== PP S1 PES additional image builder =====")
    print(f"Output root       : {args.output_root}")
    print(f"Segments          : {', '.join(args.segments)}")
    print(f"Interpolate only  : {args.interpolate_only}")

    for segment_name in args.segments:
        run_segment(segment_name, args)


if __name__ == "__main__":
    main()
