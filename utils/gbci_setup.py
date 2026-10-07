"""Helpers for setting up a GBCI calculation whose analytic gradient is valid.

The GBCI gradient is correct only under two conditions that the code does not
check, and that fail silently -- the energy stays right while the gradient is
wrong.  See the "GBCI Gradients and ``act_list``" section of
``test/SCRIPT_PATTERNS.md`` for the measurements behind this module.

1. ``pyscf/grad/gbci.py::get_X`` reads ``gbci._scf.mo_coeff`` rather than the
   ``ref_mo_coeff`` threaded through the rest of ``grad_elec``.  Reordering the
   orbitals on the GBCI object alone mixes two orderings inside one function,
   so the SCF object has to be updated too.
2. ``grad_elec`` splits occupied from virtual by the column index ``neleca``,
   so after reordering the first ``neleca`` columns must still be exactly the
   SCF occupied set.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyscf.mcscf.addons import sort_mo


def orbital_permutation(mf: Any, mo_coeff: np.ndarray, *, thresh: float = 0.99) -> np.ndarray:
    """Return ``perm`` with ``mo_coeff[:, i] ~= mf.mo_coeff[:, perm[i]]``.

    Matching is done through the overlap matrix rather than orbital energies,
    which a converged level shift can contaminate.
    """
    overlap = np.abs(mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff)
    perm = overlap.argmax(axis=1)
    if len(set(perm.tolist())) != perm.size:
        raise ValueError("Reordered orbitals do not map one-to-one onto the SCF orbitals")
    worst = overlap.max(axis=1).min()
    if worst < thresh:
        raise ValueError(
            f"Reordered orbitals are not a pure permutation of the SCF orbitals "
            f"(weakest match {worst:.4f} < {thresh})"
        )
    return perm


def apply_act_list(mf: Any, cas_probe: Any, act_list, *, base: int = 1) -> np.ndarray:
    """Reorder ``mf`` in place so ``act_list`` becomes the active window.

    ``cas_probe`` is any CASCI/GBCI object built on ``mf``; it only supplies
    ``ncore``/``ncas`` to ``sort_mo``.  Returns the reordered coefficients.

    Build the GBCI object *after* calling this, so that ``gbci.mo_coeff`` and
    ``gbci._scf.mo_coeff`` agree.
    """
    mo = sort_mo(cas_probe, mf.mo_coeff, act_list, base=base)
    perm = orbital_permutation(mf, mo)
    mf.mo_coeff = mo
    mf.mo_energy = np.asarray(mf.mo_energy)[perm]
    mf.mo_occ = np.asarray(mf.mo_occ)[perm]
    return mo


def check_active_window(gbci: Any) -> None:
    """Raise if the occupied/virtual split by column index is not the SCF one.

    ``grad_elec`` treats columns ``0 .. neleca-1`` as the occupied orbitals.
    That holds only when the active space contains exactly ``nelecas[0]``
    occupied orbitals placed at the front of the active window.
    """
    mo_occ = np.asarray(gbci._scf.mo_occ)
    neleca = gbci.ncore + gbci.nelecas[0]
    n_occ = int((mo_occ > 0).sum())
    if neleca != n_occ:
        raise ValueError(
            f"ncore + nelecas[0] = {neleca} but the SCF has {n_occ} occupied "
            "orbitals; the gradient's occupied/virtual split would be wrong"
        )
    if not np.all(mo_occ[:neleca] > 0) or np.any(mo_occ[neleca:] > 0):
        raise ValueError(
            "Columns 0..neleca-1 are not the occupied orbitals.  Pass act_list "
            "in ascending order and include exactly nelecas[0] occupied "
            "orbitals, so they fill the front of the active window"
        )


def is_contiguous_window(gbci: Any, act_list, *, base: int = 1) -> bool:
    """True if ``act_list`` is the default window *in order*, needing no reordering.

    The order matters, not just the set: ``sort_mo`` reorders the columns for a
    scrambled list of the same orbitals, which is enough to break the gradient.
    """
    if act_list is None:
        return True
    zero_based = [int(i) - base for i in act_list]
    return zero_based == list(range(gbci.ncore, gbci.ncore + gbci.ncas))


def gradient_invariance(mol: Any, gradient: np.ndarray) -> tuple[float, float]:
    """Return (net force norm, net torque norm) of a nuclear gradient.

    Both vanish for an exact gradient of a translation- and rotation-invariant
    energy, with no finite differences involved.  The torque is the sensitive
    one: the translational sum rule holds even for the broken cases above, so
    only the torque exposes them.
    """
    coords = mol.atom_coords(unit="Bohr")
    net_force = float(np.linalg.norm(np.asarray(gradient).sum(axis=0)))
    torque = float(np.linalg.norm(np.cross(coords, gradient).sum(axis=0)))
    return net_force, torque


def check_gradient_invariance(
    mol: Any, gradient: np.ndarray, *, torque_atol: float = 1.0e-6
) -> tuple[float, float]:
    """Raise if the gradient breaks rotational invariance.

    The default tolerance is loose on purpose: a correct gradient on the
    contiguous window reaches ~3e-11, while the silent-failure modes land at
    1e-3 or worse, so anything in between deserves a look rather than a pass.
    """
    net_force, torque = gradient_invariance(mol, gradient)
    if torque > torque_atol:
        raise ValueError(
            f"Analytic gradient breaks rotational invariance (torque {torque:.3e} "
            f"Eh > {torque_atol:.1e}).  Check that mf.mo_coeff was reordered "
            "together with the GBCI object and that the active window keeps the "
            "occupied orbitals at the front."
        )
    return net_force, torque


def rotational_consistency(
    build_energy,
    mol: Any,
    gradient: np.ndarray,
    *,
    axis: np.ndarray | None = None,
    step: float = 1.0e-3,
) -> dict[str, float]:
    """Compare the analytic torque against dE/dtheta for a rigid rotation.

    The energy is invariant under rigid rotation, so ``dE/dtheta`` is zero and
    so is ``n . sum_A R_A x g_A``.  Differentiating along the rotation gives a
    finite-difference reference whose exact value is known to be zero, which
    makes this a far cheaper and better-conditioned check than a full 3N
    Cartesian comparison: three energies and one gradient, with no cancellation
    between large numbers.

    It also separates the two things a nonzero torque could mean.  If the two
    finite-rotation energies agree with the reference, the energy really is
    invariant, and any gap between ``dE/dtheta`` and ``n . tau`` is a defect in
    the analytic gradient rather than a broken premise.

    Args:
        build_energy: callable taking coordinates in bohr and returning the
            energy, rebuilding the whole calculation at that geometry.
        mol: the molecule at the reference geometry.
        gradient: the analytic gradient there, in Eh/bohr.
        axis: rotation axis; a fixed arbitrary direction by default.
        step: rotation half-step in radians.
    """
    from scipy.spatial.transform import Rotation

    if axis is None:
        axis = np.array([0.3, 0.5, 0.81])
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    coords = mol.atom_coords(unit="Bohr")
    analytic = float(np.dot(axis, np.cross(coords, gradient).sum(axis=0)))

    energy_plus = build_energy(Rotation.from_rotvec(+step * axis).apply(coords))
    energy_minus = build_energy(Rotation.from_rotvec(-step * axis).apply(coords))
    reference = build_energy(coords)
    numerical = (energy_plus - energy_minus) / (2.0 * step)

    return {
        "analytic_n_dot_torque": analytic,
        "numerical_dE_dtheta": numerical,
        "difference": analytic - numerical,
        "energy_drift_plus": energy_plus - reference,
        "energy_drift_minus": energy_minus - reference,
    }
