"""Pi/sigma character of molecular orbitals for (near-)planar molecules.

The molecular plane is obtained from a singular-value fit to the heavy-atom
positions, and an MO's pi weight is the Lowdin population carried by the
p-atomic-orbital component along the plane normal. This is only meaningful when
the fit residual is small; `molecular_plane_normal` returns that residual so the
caller can check it.

d functions are ignored in the pi weight. For the split-valence + polarisation
bases used here their contribution to frontier pi orbitals is a few percent.
"""

from functools import reduce

import numpy as np
import scipy.linalg


def molecular_plane_normal(mol, heavy_only=True):
    """Best-fit plane normal of the molecule.

    Returns
    -------
    normal : (3,) unit vector normal to the fitted plane.
    rms_dev : rms out-of-plane deviation of the fitted atoms, in Angstrom.
    """
    coords = mol.atom_coords(unit="Angstrom")
    if heavy_only:
        mask = np.array([mol.atom_symbol(i) != "H" for i in range(mol.natm)])
        if mask.sum() >= 3:
            coords = coords[mask]

    centred = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centred)
    normal = vt[-1]
    rms_dev = float(np.sqrt(((centred @ normal) ** 2).mean()))
    return normal, rms_dev


def lowdin_coeff(mol, mo_coeff):
    """MO coefficients in the Lowdin-orthogonalized AO basis."""
    s = mol.intor_symmetric("int1e_ovlp")
    s12 = scipy.linalg.sqrtm(s).real
    return reduce(np.dot, (s12, np.asarray(mo_coeff)))


def _p_shell_index(mol):
    """Map (atom, p-shell label) -> [ao_x, ao_y, ao_z] indices."""
    comp = {"x": 0, "y": 1, "z": 2}
    shells = {}
    for ao_id, label in enumerate(mol.ao_labels()):
        fields = label.split()
        atom_id = int(fields[0])
        shell = fields[2]  # e.g. '2px'
        if len(shell) < 3 or shell[1] != "p":
            continue
        key = (atom_id, shell[0])
        shells.setdefault(key, [None, None, None])[comp[shell[2]]] = ao_id
    return shells


def pi_weights(mol, mo_coeff, normal=None):
    """Per-MO pi and total-p Lowdin weights.

    Returns
    -------
    pi_w : (nmo,) weight carried by p components along the plane normal.
    p_w  : (nmo,) total p weight, for reference.
    """
    if normal is None:
        normal, _ = molecular_plane_normal(mol)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    c_lowdin = lowdin_coeff(mol, mo_coeff)
    nmo = c_lowdin.shape[1]
    pi_w = np.zeros(nmo)
    p_w = np.zeros(nmo)

    for aos in _p_shell_index(mol).values():
        if any(a is None for a in aos):
            continue
        block = c_lowdin[aos, :]          # (3, nmo)
        pi_w += (normal @ block) ** 2
        p_w += (block**2).sum(axis=0)

    return pi_w, p_w


def suggest_pi_active_list(mol, mo_coeff, mo_occ, n_occ, n_vir,
                           pi_threshold=0.5, normal=None):
    """Frontier pi orbitals as a 1-based MO list for `mcscf.sort_mo`.

    Picks the `n_occ` highest occupied and `n_vir` lowest virtual MOs whose pi
    weight is at least `pi_threshold`. Raises if not enough are found, so a
    silently wrong active space cannot slip through.
    """
    pi_w, _ = pi_weights(mol, mo_coeff, normal=normal)
    occ_idx = np.where(np.asarray(mo_occ) > 0)[0]
    vir_idx = np.where(np.asarray(mo_occ) == 0)[0]

    picked_occ = [i for i in occ_idx[::-1] if pi_w[i] >= pi_threshold][:n_occ]
    picked_vir = [i for i in vir_idx if pi_w[i] >= pi_threshold][:n_vir]

    if len(picked_occ) < n_occ or len(picked_vir) < n_vir:
        raise RuntimeError(
            f"found only {len(picked_occ)}/{n_occ} occupied and "
            f"{len(picked_vir)}/{n_vir} virtual pi orbitals at "
            f"pi_threshold={pi_threshold}"
        )

    return sorted(int(i) + 1 for i in picked_occ + picked_vir)
