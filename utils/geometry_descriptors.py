"""Small geometry descriptors computed from Cartesian coordinates.

All functions take a ``(natm, 3)`` array of Cartesian coordinates in Angstrom and
zero-based atom indices, and return plain floats.  Nothing here depends on PySCF, so
these helpers can be used on any ``.xyz`` file produced by the repository.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_xyz(path):
    """Read a standard XYZ file.

    Returns ``(symbols, coords)`` where ``symbols`` is a list of element symbols and
    ``coords`` is an ``(natm, 3)`` float array in Angstrom.  The atom count on line 1 is
    enforced so that a truncated file fails loudly instead of silently losing atoms.
    """
    lines = Path(path).read_text().splitlines()
    natm = int(lines[0].split()[0])
    symbols = []
    coords = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 4:
            continue
        symbols.append(fields[0])
        coords.append([float(v) for v in fields[1:4]])
        if len(symbols) == natm:
            break
    if len(symbols) != natm:
        raise ValueError(f"{path}: expected {natm} atoms, parsed {len(symbols)}")
    return symbols, np.asarray(coords, dtype=float)


def bond_length(coords, i, j):
    """Distance between atoms ``i`` and ``j`` in Angstrom."""
    return float(np.linalg.norm(coords[i] - coords[j]))


def vector_angle(a, b):
    """Angle between two vectors in degrees, in ``[0, 180]``."""
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def bond_angle(coords, i, j, k):
    """Angle ``i-j-k`` in degrees, with ``j`` as the vertex."""
    return vector_angle(coords[i] - coords[j], coords[k] - coords[j])


def pyramidalization_angle(coords, center, apex, sub1, sub2):
    """Pyramidalization at ``center``, in degrees.

    Defined as the angle between the ``center -> apex`` bond vector and the plane spanned
    by ``sub1-center-sub2``.  A planar sp2 center gives 0 degrees; a fully pyramidal sp3
    center gives roughly 60 degrees.  The result is unsigned -- it does not distinguish
    which side of the plane the apex lies on.

    Note that other conventions exist (the supplement 90 - this value, the improper
    ``sub1-center-sub2...apex`` dihedral, or 360 - the sum of the three angles at the
    center) and they do not agree numerically.  This one is the convention used in the
    repository's CH2NH2 figures.
    """
    bond = coords[apex] - coords[center]
    normal = np.cross(coords[sub1] - coords[center], coords[sub2] - coords[center])
    if np.linalg.norm(normal) < 1e-10:
        raise ValueError("substituents are collinear with the center; plane is undefined")
    return abs(90.0 - vector_angle(bond, normal))


def torsion_angle(coords, i, j, k, l):
    """Signed dihedral ``i-j-k-l`` in degrees, in ``(-180, 180]``."""
    b0 = coords[i] - coords[j]
    b1 = coords[k] - coords[j]
    b2 = coords[l] - coords[k]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def assign_hydrogens(symbols, coords, heavy_indices):
    """Assign every H to its nearest heavy atom.

    Returns ``{heavy_index: [h_index, ...]}`` with each list sorted by index.  Used so
    that geometry analysis does not depend on the atom ordering of the input file.
    """
    groups = {idx: [] for idx in heavy_indices}
    for h, sym in enumerate(symbols):
        if sym != "H":
            continue
        nearest = min(heavy_indices, key=lambda idx: np.linalg.norm(coords[h] - coords[idx]))
        groups[nearest].append(h)
    return groups
