"""Grouping of GBCI possible occupations by active-electron population.

``pyscf.gbci.gbci.group_by_atom`` groups configurations with a sequential
*leader* rule: the first unassigned configuration seeds a group and absorbs
every unassigned configuration whose population is within ``thres`` of that
seed.  Membership is therefore decided against the seed alone, so the spread
inside a group can reach twice the threshold.  Measured on the distorted SO2
geometry at ``thres=0.2``, one group spans 0.335.

That contradicts the usual description of the method -- "a maximum population
difference of ``thres`` within each group" -- which asks for the group
*diameter* to be bounded.  :func:`group_complete_linkage` implements that
reading: a configuration joins only if it is within ``thres`` of *every*
member already in the group.

Both rules are greedy and single-pass, so both depend on the order of the
possible-occupation list; bounding the diameter optimally is NP-hard and not
worth it here, where the groups are small.

:func:`patch_gbci_grouping` swaps the corrected rule into ``pyscf.gbci`` at
runtime, so the fix can be measured without editing pyscf-forge.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def group_leader(values: np.ndarray, thres: float) -> list[list[int]]:
    """Reproduce ``group_by_atom``'s current rule: distance to the seed only."""
    values = np.atleast_2d(np.asarray(values, dtype=float).T).T
    count = len(values)
    groups: list[list[int]] = []
    visited: set[int] = set()
    for i in range(count):
        if i in visited:
            continue
        members = [i]
        visited.add(i)
        for j in range(count):
            if j not in visited and np.all(np.abs(values[i] - values[j]) <= thres):
                members.append(j)
                visited.add(j)
        groups.append(members)
    return groups


def group_complete_linkage(values: np.ndarray, thres: float) -> list[list[int]]:
    """Group so that every pair inside a group differs by at most ``thres``.

    This is the rule the method description implies: ``thres`` bounds the
    population difference within a group, not the distance to one chosen member.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float).T).T
    count = len(values)
    groups: list[list[int]] = []
    visited: set[int] = set()
    for i in range(count):
        if i in visited:
            continue
        members = [i]
        visited.add(i)
        for j in range(count):
            if j in visited:
                continue
            if all(np.all(np.abs(values[k] - values[j]) <= thres) for k in members):
                members.append(j)
                visited.add(j)
        groups.append(members)
    return groups


def group_diameter(values: np.ndarray, members: list[int]) -> float:
    """Largest population difference between any two members of a group."""
    values = np.atleast_2d(np.asarray(values, dtype=float).T).T
    block = values[members]
    if len(block) < 2:
        return 0.0
    return float(np.max(np.abs(block[:, None, :] - block[None, :, :])))


def population_matrix(
    mol: Any, active_mo: np.ndarray, po_list: np.ndarray, fragments: list[list[int]]
) -> np.ndarray:
    """Lowdin population of the active electrons on each fragment.

    Computed exactly as ``group_by_atom`` does, so the only difference between
    the two grouping rules is the rule itself.
    """
    from pyscf.gbci.gbci import fragment_aos_by_atoms

    overlap = mol.intor_symmetric("cint1e_ovlp_sph")
    eigenvalues, vectors = np.linalg.eigh(overlap)
    s12 = (vectors * np.sqrt(eigenvalues)) @ vectors.T.conj()
    ao_sets = [fragment_aos_by_atoms(mol, atoms) for atoms in fragments]

    values = np.zeros((len(po_list), len(ao_sets)))
    for index, occ in enumerate(np.asarray(po_list)):
        singly = np.where(occ == 1)[0]
        doubly = np.where(occ == 2)[0]
        density = (active_mo[:, singly] @ active_mo[:, singly].T
                   + 2.0 * (active_mo[:, doubly] @ active_mo[:, doubly].T))
        diagonal = np.diag(s12 @ density @ s12)
        for j, ao_set in enumerate(ao_sets):
            values[index, j] = np.sum(diagonal[ao_set])
    return values


def patch_gbci_grouping(rule: Callable[..., list[list[int]]] = group_complete_linkage):
    """Replace ``pyscf.gbci.gbci.group_by_atom``'s rule at runtime.

    Returns the original function so a caller can restore it.  The population
    part of the original is reused unchanged; only the grouping rule differs.
    """
    import importlib

    # ``pyscf.gbci.__init__`` defines a factory function named ``gbci``, which
    # shadows the submodule of the same name, so ``from pyscf.gbci import gbci``
    # returns the function.  Import the module explicitly.
    gbci_module = importlib.import_module("pyscf.gbci.gbci")
    _normalize_atom_groups = gbci_module._normalize_atom_groups

    original = gbci_module.group_by_atom

    def patched(mol, ac_mo_coeff, po_list, atom_groups, thres=0.2):
        fragments = _normalize_atom_groups(atom_groups)
        values = population_matrix(mol, ac_mo_coeff, po_list, fragments)
        return rule(values, thres)

    gbci_module.group_by_atom = patched
    return original
