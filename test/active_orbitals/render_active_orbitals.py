"""Write Gaussian cube files for the active orbitals of each benchmark system.

Step one of the Supporting-Information orbital figures.  The orbitals come from
``export_active_space_moldens.build_system``, so the pictures show exactly the
orbitals that script writes to Molden -- nothing is recomputed here with
different settings.

The geometry is reoriented before the cubes are made rather than after: Jmol has
no reliable way to put an arbitrary molecule into a chosen frame from a script,
so the molecule is rotated into a canonical frame here and the camera in
``render_active_orbitals_jmol.spt`` stays fixed for every system.  This is the
same trick ``ch2nh2/analysis/ch2nh2_meci_geometry_descriptors.py`` uses
for the MECI renders.  Canonical frame:

* planar or three-dimensional -- best-fit plane normal along +z, longest in-plane
  principal axis along +x.  Rendered twice, ``front`` (looking down -z, molecular
  plane filling the screen) and ``side`` (rotated 90 degrees about x), so sigma
  and pi orbitals can be told apart.
* linear -- molecular axis along +x, rendered once as ``front``.

Rotating the geometry changes the MO coefficients, so the SCF is simply repeated
on the rotated molecule.  Every system here is a seconds-long RHF, and redoing it
is safer than rotating AO coefficients by hand through the l>0 shells.

Usage::

    python render_active_orbitals.py                      # the three SI systems
    python render_active_orbitals.py --systems so2_8e5o   # one system
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf.tools import cubegen

from export_active_space_moldens import (
    SYSTEMS,
    build_system,
    out_of_plane_fraction,
    plane_normal,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# The systems that go into the Supporting Information.  The lithium halides are in
# SYSTEMS too and can be rendered with --systems, but their active space is still
# undecided (the 1.50 A default window here vs the sigma-only act_list of
# test/lix/li_halides_gbci_grad_check.py), so they are not default.
DEFAULT_SYSTEMS = ("so2_8e5o", "pp_2e2o", "ch2nh2_8e5o")

BOHR = 0.52917721092


def canonical_frame(coords: np.ndarray) -> tuple[np.ndarray, bool]:
    """Rotation putting ``coords`` into the frame the Jmol camera expects.

    Returns ``(rotation, is_linear)``; apply it as ``coords @ rotation.T``.  For a
    linear molecule only the axis is fixed (to +x); otherwise the best-fit plane
    normal goes to +z and the longest in-plane extent to +x.
    """
    centred = coords - coords.mean(axis=0)
    normal = plane_normal(coords)

    if normal is None:
        axis = scipy.linalg.svd(centred)[2][0]
        rotation = _frame_from(x=axis, z=_any_perpendicular(axis))
        return rotation, True

    # Longest extent within the plane: the leading right singular vector of the
    # coordinates with the out-of-plane component projected out.
    in_plane = centred - np.outer(centred @ normal, normal)
    long_axis = scipy.linalg.svd(in_plane)[2][0]
    return _frame_from(x=long_axis, z=normal), False


def _any_perpendicular(vector: np.ndarray) -> np.ndarray:
    seed = np.array([0.0, 0.0, 1.0])
    if abs(vector @ seed) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    perpendicular = seed - (seed @ vector) * vector
    return perpendicular / np.linalg.norm(perpendicular)


def _frame_from(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Right-handed rotation sending ``x`` to +x and ``z`` to +z."""
    x = x / np.linalg.norm(x)
    z = z - (z @ x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def reoriented_atom_block(mol, rotation: np.ndarray) -> tuple[str, list[str], np.ndarray]:
    """``mol``'s geometry rotated into the canonical frame, as an atom block."""
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords = mol.atom_coords() * BOHR @ rotation.T
    lines = [f"{s:<3s} {c[0]:16.8f} {c[1]:16.8f} {c[2]:16.8f}"
             for s, c in zip(symbols, coords)]
    return "\n".join(lines), symbols, coords


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    lines = [str(len(symbols)), comment]
    lines += [f"{s:<3s} {c[0]:16.8f} {c[1]:16.8f} {c[2]:16.8f}"
              for s, c in zip(symbols, coords)]
    path.write_text("\n".join(lines) + "\n")


def render(name: str, spec: dict, out_dir: Path, grid: int, margin: float,
           verbose: int) -> dict:
    """Build one system, orient it, and write a cube per active orbital."""
    mol, _, _, _, window, _, _ = build_system(name, spec, verbose)
    rotation, is_linear = canonical_frame(mol.atom_coords())
    atom_block, symbols, coords = reoriented_atom_block(mol, rotation)

    # Same system, canonical frame.  The active window is unchanged by a rotation,
    # so `window` from the unrotated build is reused as the label.
    mol, mf, mo, active, window_rotated, energy, occ = build_system(
        name, spec, verbose, atom=atom_block)
    if window_rotated != window:
        raise RuntimeError(f"{name}: active window moved under rotation "
                           f"({window} -> {window_rotated})")

    system_dir = out_dir / name
    cube_dir = system_dir / "cubes"
    cube_dir.mkdir(parents=True, exist_ok=True)

    normal = plane_normal(mol.atom_coords())
    overlap_half = scipy.linalg.sqrtm(mf.get_ovlp()).real
    orthogonal = overlap_half @ mo
    views = ["front"] if is_linear else ["front", "side"]

    print(f"\n{name}   {mol.natm} atoms, {mol.nao} AOs, charge {spec['charge']}")
    print(f"  CAS({sum(spec['nelecas'])}e,{spec['ncas']}o)  RHF MOs {window}")
    print(f"  {'linear' if is_linear else 'planar/3D'}, views: {', '.join(views)}")

    orbitals = []
    for column, k in enumerate(active):
        mo_number = window[column]
        cube = cube_dir / f"{name}_mo{mo_number:02d}.cube"
        cubegen.orbital(mol, str(cube), mo[:, k], nx=grid, ny=grid, nz=grid,
                        margin=margin)
        pi = out_of_plane_fraction(mol, orthogonal[:, k], normal)
        orbitals.append({
            "mo": mo_number,
            "energy": float(energy[column]),
            "occ": float(occ[column]),
            "pi_out_of_plane": None if np.isnan(pi) else float(pi),
            "cube": cube.name,
        })
        pi_text = "  n/a" if np.isnan(pi) else f"{pi:5.3f}"
        print(f"  MO {mo_number:>3d}  e={energy[column]:>10.5f}  occ={occ[column]:>3.1f}"
              f"  pi(oop)={pi_text}  -> {cube.name}")

    write_xyz(system_dir / f"{name}_aligned.xyz", symbols, coords,
              f"{name}: {spec['note']} (canonical frame)")

    manifest = {
        "system": name,
        "note": spec["note"],
        "basis": spec["basis"],
        "charge": spec["charge"],
        "spin": spec["spin"],
        "ncas": spec["ncas"],
        "nelecas": list(spec["nelecas"]),
        "act_list": spec["act_list"],
        "window": window,
        "linear": bool(is_linear),
        "views": views,
        "grid": grid,
        "margin": margin,
        "orbitals": orbitals,
    }
    path = system_dir / f"{name}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  -> {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS),
                        help="comma-separated subset of " + ", ".join(SYSTEMS))
    parser.add_argument("--out-dir", type=Path,
                        default=SCRIPT_DIR / "reference" / "active_orbital_figures")
    parser.add_argument("--grid", type=int, default=80,
                        help="cube points per axis (default 80)")
    parser.add_argument("--margin", type=float, default=4.0,
                        help="cube box margin in Bohr (default 4.0)")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    for name in args.systems.replace(",", " ").split():
        render(name, SYSTEMS[name], args.out_dir, args.grid, args.margin,
               args.verbose)


if __name__ == "__main__":
    main()
