"""Export the converged FC -> TICT NEB band as a plottable S1 profile.

The PES figure plots S1 relative to the S0 energy at the ground-state geometry,
so the SA-CASSCF S0 energy of image 0 supplies the reference. The NEB itself
only stores S1 (the ASE calculator energy), so that one number is read back out
of the run's stdout, where the endpoint check printed the full state pair.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from ase.io import read

H2EV = 27.21138602

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "pp_casscf_S1_fc_tict_neb_outputs"
PREFIX = "pp_casscf_S1_fc_tict_neb"
RUN_LOG = SCRIPT_DIR / f"{PREFIX}.out"

STATE_PAIR = re.compile(
    r"State 0 weight [\d.]+\s+E = (-?\d+\.\d+) S\^2 = -?[\d.]+\s*\n"
    r"\s*State 1 weight [\d.]+\s+E = (-?\d+\.\d+) S\^2 = -?[\d.]+"
)


def dih(p, i, j, k, l):
    b0, b1, b2 = p[i] - p[j], p[k] - p[j], p[l] - p[k]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def twist(p):
    a = abs(dih(p, 0, 6, 9, 10))
    return min(a, 180 - a)


def main() -> None:
    results = np.load(OUT_DIR / f"{PREFIX}_results.npz")
    s1 = np.asarray(results["energies_ev"], dtype=float)

    images = [read(str(OUT_DIR / f"{PREFIX}_{i:03d}.xyz")) for i in range(len(s1))]
    positions = [img.get_positions() for img in images]

    # The NEB stores only S1 (the ASE calculator energy). Both state energies
    # were printed for every evaluation, so each image's S0 is recovered by
    # matching on its known S1 rather than by trusting the print order.
    pairs = np.asarray(
        [[float(a), float(b)] for a, b in STATE_PAIR.findall(
            RUN_LOG.read_text(errors="replace"))],
        dtype=float,
    ) * H2EV
    if len(pairs) == 0:
        raise ValueError(f"No SA-CASSCF state pair found in {RUN_LOG}.")

    s0 = np.empty_like(s1)
    for i, energy in enumerate(s1):
        gaps = np.abs(pairs[:, 1] - energy)
        best = int(np.argmin(gaps))
        if gaps[best] > 1e-3:
            raise ValueError(
                f"No state pair matches image {i} (S1 {energy:.6f} eV; "
                f"closest {pairs[best, 1]:.6f} eV)."
            )
        s0[i] = pairs[best, 0]
    reference_s0_ev = float(s0[0])

    seg = [0.0] + [
        float(np.sqrt(((positions[i] - positions[i - 1]) ** 2).sum(1).mean()))
        for i in range(1, len(positions))
    ]
    s = np.cumsum(seg)

    rows = []
    for i, p in enumerate(positions):
        rows.append({
            "image_index": i,
            "xyz": f"{PREFIX}_{i:03d}.xyz",
            "path_length_angstrom": round(float(s[i]), 4),
            "twist_deg": round(twist(p), 2),
            "s0_reference_ev": reference_s0_ev,
            "s0_energy_ev": s0[i],
            "s1_energy_ev": s1[i],
            "s0_relative_to_s0_geometry_ev": s0[i] - reference_s0_ev,
            "s1_relative_to_s0_geometry_ev": s1[i] - reference_s0_ev,
            "s1_relative_to_start_ev": s1[i] - s1[0],
            "s1_minus_s0_ev": s1[i] - s0[i],
        })

    csv_path = OUT_DIR / f"{PREFIX}_profile.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {csv_path}")
    print(" img   s(A)  twist      S0      S1    gap")
    for r in rows:
        print("%4d  %5.3f  %5.1f  %6.3f  %6.3f  %5.3f"
              % (r["image_index"], r["path_length_angstrom"], r["twist_deg"],
                 r["s0_relative_to_s0_geometry_ev"],
                 r["s1_relative_to_s0_geometry_ev"], r["s1_minus_s0_ev"]))


if __name__ == "__main__":
    main()
