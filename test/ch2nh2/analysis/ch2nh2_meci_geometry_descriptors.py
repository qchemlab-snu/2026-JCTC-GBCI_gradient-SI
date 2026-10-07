"""Collect C-N bond lengths and pyramidalization angles for the CH2NH2 cc-pVTZ MECIs.

Reads the already-converged S0/S1 MECI geometries under
``test/ch2nh2/<method>/outputs/<run>/`` and reports, for each method, the two
coordinates that characterize this conical intersection: the C-N bond length and the
pyramidalization angle at carbon.  No quantum chemistry is re-run.

Usage::

    python ch2nh2_meci_geometry_descriptors.py
    python ch2nh2_meci_geometry_descriptors.py --output-dir somewhere/else
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.geometry_descriptors import (  # noqa: E402
    assign_hydrogens,
    bond_length,
    pyramidalization_angle,
    read_xyz,
    torsion_angle,
)

RUN_NAME = "ch2nh2_ccpvtz_meci_descriptors"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / RUN_NAME
DEFAULT_ALIGNED_DIR = SCRIPT_DIR / "outputs" / "ch2nh2_ccpvtz_meci_images"

# Rotation of the molecule about its own C-N axis, applied to the coordinates of the
# rendering copies.  The Jmol camera stays head-on, so this is the only knob that changes
# how far the CH2 end is turned toward the viewer, and the rotated coordinates are written
# to disk rather than hidden in the viewer state.
#
# 145 degrees reproduces the viewing angle of the existing comparison panels in
# test/ch2nh2/fig/.  It was fitted, not guessed: the four hydrogen centers were detected in
# fig/casci_s0s1_meci.png and in renders made over a full 360-degree scan, both through the
# same detector, and compared in the frame where C sits at the origin and N at (1, 0).  The
# residual summed over the four hydrogens bottoms out sharply at 145 degrees (0.15 in units
# of the C-N length, against 0.24 at 140 and 150 and >1 away from the minimum).
DEFAULT_SPIN_ABOUT_CN_DEG = 145.0

# One entry per geometry.  ``run_dir`` is relative to SCRIPT_DIR; the optimized geometry
# and the run summary are both named after the run directory, which is the repository
# convention for these outputs.
GEOMETRY_SPECS = [
    {
        "label": "CASCI",
        "method": "CASCI",
        "active_space": "5o8e",
        "run_dir": "../casci/outputs/ch2nh2_casci_S0S1_meci_5o8e_ccpvtz",
    },
    {
        "label": "CASSCF",
        "method": "CASSCF",
        "active_space": "5o8e",
        "run_dir": "../casscf/outputs/ch2nh2_casscf_S0S1_meci_5o8e_ccpvtz",
    },
    {
        "label": "GBCI (groupocc)",
        "method": "GBCI",
        "active_space": "5o8e",
        "run_dir": "../gbci/outputs/ch2nh2_gbci_S0S1_meci_5o8e_groupocc_ccpvtz",
    },
    {
        "label": "GBCI (nogroup)",
        "method": "GBCI",
        "active_space": "5o8e",
        "run_dir": "../gbci/outputs/ch2nh2_gbci_S0S1_meci_5o8e_nogroup_ccpvtz",
    },
]

# Published XMS reference geometry, used as a self-check: the pyramidalization convention
# in utils.geometry_descriptors must reproduce the value printed in the comparison figure.
REFERENCE_XYZ = SCRIPT_DIR / ".." / "fig" / "xms_geom.xyz"
REFERENCE_BOND = 1.412
REFERENCE_PYRAMIDALIZATION = 31.3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON and CSV results.",
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=DEFAULT_ALIGNED_DIR,
        help=(
            "Directory for the reorientation-normalized XYZ copies that the Jmol "
            "rendering script consumes."
        ),
    )
    parser.add_argument(
        "--spin-about-cn-deg",
        type=float,
        default=DEFAULT_SPIN_ABOUT_CN_DEG,
        help=(
            "Rotate the rendering copies about their own C-N axis by this angle. "
            "Changes only the viewing orientation, never the internal geometry."
        ),
    )
    return parser.parse_args()


def identify_ch2nh2_atoms(symbols, coords):
    """Return ``(n_index, c_index, c_hydrogens, n_hydrogens)`` for a CH2NH2 geometry.

    Hydrogens are assigned by proximity rather than by position in the file, so the
    analysis does not depend on the atom ordering of the input.
    """
    if symbols.count("N") != 1 or symbols.count("C") != 1:
        raise ValueError(f"expected exactly one N and one C, got {symbols}")
    n_idx = symbols.index("N")
    c_idx = symbols.index("C")
    groups = assign_hydrogens(symbols, coords, [n_idx, c_idx])
    c_hydrogens = groups[c_idx]
    n_hydrogens = groups[n_idx]
    if len(c_hydrogens) != 2 or len(n_hydrogens) != 2:
        raise ValueError(
            f"expected 2 H on C and 2 H on N, got {len(c_hydrogens)} and {len(n_hydrogens)}"
        )
    return n_idx, c_idx, c_hydrogens, n_hydrogens


def describe_geometry(symbols, coords):
    """Compute the CH2NH2 descriptors for one geometry."""
    n_idx, c_idx, c_h, n_h = identify_ch2nh2_atoms(symbols, coords)
    return {
        "cn_bond_angstrom": bond_length(coords, c_idx, n_idx),
        "pyramidalization_at_C_deg": pyramidalization_angle(coords, c_idx, n_idx, *c_h),
        "pyramidalization_at_N_deg": pyramidalization_angle(coords, n_idx, c_idx, *n_h),
        "hcnh_torsion_deg": torsion_angle(coords, c_h[0], c_idx, n_idx, n_h[0]),
    }


def align_for_rendering(symbols, coords, spin_about_cn_deg=0.0):
    """Put a CH2NH2 geometry into a common frame so renders are comparable.

    The C-N axis becomes +x, the planar NH2 end is laid into the screen plane (xy), and
    the handedness is fixed so the CH2 hydrogens always lean the same way.  A head-on
    camera then shows the C-N bond horizontal for every structure.

    ``spin_about_cn_deg`` turns the molecule about its own C-N axis afterwards, which is
    how the viewing angle is chosen: the camera stays fixed, so every geometry is turned
    by the same amount relative to its own C-N bond and the renders stay comparable.
    Bond lengths and angles are unchanged by either step.
    """
    n_idx, c_idx, c_h, n_h = identify_ch2nh2_atoms(symbols, coords)

    origin = 0.5 * (coords[c_idx] + coords[n_idx])
    shifted = coords - origin

    x_axis = shifted[n_idx] - shifted[c_idx]
    x_axis /= np.linalg.norm(x_axis)

    nh2_normal = np.cross(shifted[n_h[0]] - shifted[n_idx], shifted[n_h[1]] - shifted[n_idx])
    z_axis = nh2_normal - np.dot(nh2_normal, x_axis) * x_axis
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotated = shifted @ np.column_stack([x_axis, y_axis, z_axis])
    # Fix the remaining 180-degree ambiguity about the C-N axis using the CH2 bisector,
    # which points along the pyramidalization direction and is well separated from zero
    # (the individual CH2 hydrogens are nearly symmetric, so their own signs are not a
    # stable criterion).  Requiring a +y bisector makes every frame lean the same way.
    bisector = (rotated[c_h[0]] - rotated[c_idx]) + (rotated[c_h[1]] - rotated[c_idx])
    if bisector[1] < 0:
        rotated[:, 1] *= -1.0
        rotated[:, 2] *= -1.0

    if spin_about_cn_deg:
        theta = np.radians(spin_about_cn_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        spin = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_t, -sin_t], [0.0, sin_t, cos_t]]
        )
        rotated = rotated @ spin.T
    return rotated


def write_xyz(path, symbols, coords, comment=""):
    """Write a standard XYZ file."""
    lines = [str(len(symbols)), comment]
    for sym, xyz in zip(symbols, coords):
        lines.append(f"{sym:<2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}")
    Path(path).write_text("\n".join(lines) + "\n")


def read_summary(run_dir):
    """Load the run summary, normalizing the key names that differ between methods."""
    summary_path = run_dir / f"{run_dir.name}_summary.json"
    if not summary_path.exists():
        return {}, None
    raw = json.loads(summary_path.read_text())

    def pick(*names):
        for name in names:
            if name in raw:
                return raw[name]
        return None

    normalized = {
        "basis": raw.get("basis"),
        "charge": raw.get("charge"),
        "spin": raw.get("spin"),
        "unit": raw.get("unit"),
        "final_energies_hartree": pick("final_energies_Eh", "final_energies_hartree"),
        "final_gap_ev": pick("final_gap_eV", "final_gap_ev"),
        "has_imaginary_frequency": pick("has_imaginary_frequency"),
        "imag_frequency_cm": pick("imag_frequency_cm"),
        "active_orbitals_1based": pick(
            "final_act_list_1based", "final_active_orbitals_1based"
        ),
        "groupA": raw.get("groupA"),
    }
    # CASSCF records the frequency check in a nested block instead of at the top level.
    freq = raw.get("frequency_check")
    if isinstance(freq, dict):
        if normalized["has_imaginary_frequency"] is None:
            normalized["has_imaginary_frequency"] = freq.get("has_imaginary_frequency")
        if normalized["imag_frequency_cm"] is None:
            normalized["imag_frequency_cm"] = freq.get("imag_frequency_cm")
    return normalized, summary_path


def collect(spec):
    """Build the full record for one GEOMETRY_SPECS entry."""
    run_dir = (SCRIPT_DIR / spec["run_dir"]).resolve()
    xyz_path = run_dir / f"{run_dir.name}_optimized.xyz"
    symbols, coords = read_xyz(xyz_path)
    summary, summary_path = read_summary(run_dir)

    record = {
        "label": spec["label"],
        "method": spec["method"],
        "active_space": spec["active_space"],
        "states": "S0/S1 MECI",
        "run_name": run_dir.name,
        "source_xyz": str(xyz_path.relative_to(REPO_ROOT)),
        "source_summary": (
            str(summary_path.relative_to(REPO_ROOT)) if summary_path else None
        ),
        "geometry_angstrom": [
            [sym, *[float(v) for v in xyz]] for sym, xyz in zip(symbols, coords)
        ],
    }
    record.update(describe_geometry(symbols, coords))
    record.update(summary)
    return record, symbols, coords


def self_check():
    """Confirm the angle convention reproduces the published XMS reference values."""
    path = REFERENCE_XYZ.resolve()
    if not path.exists():
        return None
    symbols, coords = read_xyz(path)
    desc = describe_geometry(symbols, coords)
    return {
        "source_xyz": str(path.relative_to(REPO_ROOT)),
        "cn_bond_angstrom": desc["cn_bond_angstrom"],
        "pyramidalization_at_C_deg": desc["pyramidalization_at_C_deg"],
        "expected_cn_bond_angstrom": REFERENCE_BOND,
        "expected_pyramidalization_at_C_deg": REFERENCE_PYRAMIDALIZATION,
        "cn_bond_matches": abs(desc["cn_bond_angstrom"] - REFERENCE_BOND) < 5e-3,
        "pyramidalization_matches": (
            abs(desc["pyramidalization_at_C_deg"] - REFERENCE_PYRAMIDALIZATION) < 0.1
        ),
    }


def print_table(records):
    header = f"{'method':<18}{'active':<9}{'r(C-N)/A':>10}{'pyr(C)/deg':>12}{'pyr(N)/deg':>12}{'HCNH/deg':>11}"
    print(header)
    print("-" * len(header))
    for rec in records:
        print(
            f"{rec['label']:<18}{rec['active_space']:<9}"
            f"{rec['cn_bond_angstrom']:>10.4f}"
            f"{rec['pyramidalization_at_C_deg']:>12.2f}"
            f"{rec['pyramidalization_at_N_deg']:>12.2f}"
            f"{rec['hcnh_torsion_deg']:>11.2f}"
        )


def save_results(records, check, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": RUN_NAME,
        "script": Path(__file__).name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "molecule": "CH2NH2",
        "basis": "cc-pVTZ",
        "pyramidalization_convention": (
            "angle between the C->N bond vector and the H-C-H plane; 0 deg = planar sp2"
        ),
        "reference_self_check": check,
        "geometries": records,
    }
    json_path = output_dir / f"{RUN_NAME}_descriptors.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = output_dir / f"{RUN_NAME}_descriptors.csv"
    columns = [
        "label",
        "method",
        "active_space",
        "cn_bond_angstrom",
        "pyramidalization_at_C_deg",
        "pyramidalization_at_N_deg",
        "hcnh_torsion_deg",
        "final_gap_ev",
        "has_imaginary_frequency",
        "run_name",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return json_path, csv_path


def slug(spec):
    """Short file-name stem for one spec, e.g. ``gbci_groupocc_5o8e_ccpvtz``."""
    variant = spec["label"].partition("(")[2].rstrip(")").strip()
    parts = [spec["method"].lower()]
    if variant:
        parts.append(variant)
    parts += [spec["active_space"], "ccpvtz", "s0s1_meci"]
    return "_".join(parts)


def write_aligned_geometries(specs, geometries, aligned_dir, spin_about_cn_deg=0.0):
    """Write the reorientation-normalized XYZ copies used for rendering."""
    aligned_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for spec, (symbols, coords) in zip(specs, geometries):
        rotated = align_for_rendering(symbols, coords, spin_about_cn_deg)
        path = aligned_dir / f"{slug(spec)}_aligned.xyz"
        write_xyz(
            path,
            symbols,
            rotated,
            comment=(
                f"{spec['label']} {spec['active_space']} cc-pVTZ S0/S1 MECI, "
                f"C-N along +x, spun {spin_about_cn_deg:g} deg about C-N"
            ),
        )
        written.append(path)
    return written


def main():
    args = parse_args()
    collected = [collect(spec) for spec in GEOMETRY_SPECS]
    records = [item[0] for item in collected]
    geometries = [(item[1], item[2]) for item in collected]
    check = self_check()

    print("CH2NH2 cc-pVTZ S0/S1 MECI geometries")
    print("pyramidalization = angle between C->N bond and the H-C-H plane (0 = planar)")
    print()
    print_table(records)
    print()
    if check is None:
        print("reference self-check: skipped (xms_geom.xyz not found)")
    else:
        status = "OK" if check["cn_bond_matches"] and check["pyramidalization_matches"] else "MISMATCH"
        print(
            f"reference self-check ({status}): XMS r(C-N) = "
            f"{check['cn_bond_angstrom']:.4f} A (expected {check['expected_cn_bond_angstrom']}), "
            f"pyr(C) = {check['pyramidalization_at_C_deg']:.2f} deg "
            f"(expected {check['expected_pyramidalization_at_C_deg']})"
        )

    json_path, csv_path = save_results(records, check, args.output_dir)
    aligned = write_aligned_geometries(
        GEOMETRY_SPECS, geometries, args.aligned_dir, args.spin_about_cn_deg
    )
    print()
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    for path in aligned:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
