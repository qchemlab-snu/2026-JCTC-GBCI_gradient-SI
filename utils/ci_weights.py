"""Configuration, determinant and grouped-bath weights of a CI vector.

The CI vectors of ``pyscf.mcscf`` (CASCI/CASSCF) and ``pyscf.gbci`` (GBCI) share
the same ``(nstringsa, nstringsb)`` determinant layout, so the same weight
analysis applies to both.  GBCI adds one extra partition: every determinant
belongs to a bath group, and ``conf_info_list[ia, ib]`` is that group index.

The GBCI Hamiltonian carries the bath overlaps inside its matrix elements and is
diagonalized in an orthonormal metric, so ``sum_I c_I**2 = 1`` holds for GBCI
exactly as it does for CASCI; the weights below need no overlap correction.

Three resolutions are available, coarse to fine:

* :func:`group_weights` -- weight per grouped bath (GBCI only).
* :func:`configuration_weights` -- weight per spatial occupation pattern,
  i.e. summed over the spin couplings of one configuration.
* :func:`determinant_weights` -- weight per alpha/beta determinant.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from pyscf.fci import cistring


def _unpack_nelecas(nelecas) -> Tuple[int, int]:
    if np.isscalar(nelecas):
        total = int(nelecas)
        nelecb = total // 2
        return total - nelecb, nelecb
    return int(nelecas[0]), int(nelecas[1])


def string_occupations(ncas: int, nelec: int) -> np.ndarray:
    """Occupation vectors (0/1 per active orbital) of all strings, in CI order."""
    strings = cistring.make_strings(range(ncas), nelec)
    occ = np.zeros((len(strings), ncas), dtype=np.int8)
    for i, s in enumerate(strings):
        for p in range(ncas):
            if (int(s) >> p) & 1:
                occ[i, p] = 1
    return occ


def determinant_weights(
    civec,
    ncas: int,
    nelecas,
    tol: float = 1.0e-3,
) -> List[Dict]:
    """Weights ``c**2`` of the individual determinants, largest first.

    Returns one record per determinant above ``tol`` with its alpha and beta
    occupation vectors, the CI coefficient and the weight.
    """
    neleca, nelecb = _unpack_nelecas(nelecas)
    civec = np.asarray(civec).reshape(
        cistring.num_strings(ncas, neleca), cistring.num_strings(ncas, nelecb)
    )
    occ_a = string_occupations(ncas, neleca)
    occ_b = string_occupations(ncas, nelecb)

    out = []
    for ia, ib in zip(*np.nonzero(civec**2 >= tol)):
        out.append(
            {
                "occ_a": occ_a[ia].tolist(),
                "occ_b": occ_b[ib].tolist(),
                "coeff": float(civec[ia, ib]),
                "weight": float(civec[ia, ib] ** 2),
            }
        )
    out.sort(key=lambda row: -row["weight"])
    return out


def configuration_weights(
    civec,
    ncas: int,
    nelecas,
    tol: float = 1.0e-3,
) -> List[Dict]:
    """Weights per spatial occupation pattern, summed over spin couplings.

    This is the "configuration" resolution: the open-shell determinants of one
    spatial occupation are added together, so a singlet open-shell configuration
    appears once with its full weight instead of twice with half of it.
    """
    neleca, nelecb = _unpack_nelecas(nelecas)
    civec = np.asarray(civec).reshape(
        cistring.num_strings(ncas, neleca), cistring.num_strings(ncas, nelecb)
    )
    occ_a = string_occupations(ncas, neleca)
    occ_b = string_occupations(ncas, nelecb)

    totals: Dict[Tuple[int, ...], float] = {}
    for ia in range(civec.shape[0]):
        for ib in range(civec.shape[1]):
            w = float(civec[ia, ib] ** 2)
            if w < 1.0e-14:
                continue
            key = tuple((occ_a[ia] + occ_b[ib]).tolist())
            totals[key] = totals.get(key, 0.0) + w

    out = [
        {"occ": list(key), "weight": value}
        for key, value in totals.items()
        if value >= tol
    ]
    out.sort(key=lambda row: -row["weight"])
    return out


def group_weights(civec, conf_info_list, ngroup: Optional[int] = None) -> np.ndarray:
    """Weight of each grouped bath: ``sum_{I in g} c_I**2``.

    ``conf_info_list`` is the GBCI determinant -> group map, as built by
    ``pyscf.gbci.gbci.group_info_list`` and stored in the scanner intermediates.
    """
    conf = np.asarray(conf_info_list, dtype=int)
    civec = np.asarray(civec).reshape(conf.shape)
    if ngroup is None:
        ngroup = int(conf.max()) + 1
    weights = np.zeros(ngroup)
    np.add.at(weights, conf.ravel(), (civec.ravel() ** 2))
    return weights


def group_occupation_labels(
    po_list,
    group: Sequence[Sequence[int]],
    group_a_mo: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict]:
    """Describe each grouped bath by the occupation patterns it collects.

    ``po_list`` are the possible active occupation patterns and ``group[g]`` the
    pattern indices of bath ``g`` (both from ``gbci.optimize_mo``).  With
    ``group_a_mo`` -- the MO-index groups of ``group_a={'mo': ...}`` -- the
    electron count per MO group is reported as well, which is the quantity the
    grouping is actually keyed on.
    """
    po_list = np.asarray(po_list, dtype=int)
    labels = []
    for ig, members in enumerate(group):
        members = [int(p) for p in members]
        patterns = [po_list[p].tolist() for p in members]
        record = {"group": ig, "n_patterns": len(members), "occ_patterns": patterns}
        if group_a_mo is not None:
            counts = [
                int(po_list[members[0]][list(block)].sum()) for block in group_a_mo
            ]
            record["mo_group_electrons"] = counts
        labels.append(record)
    return labels


def format_occ(occ: Sequence[int]) -> str:
    """Compact occupation string such as ``2 2 1 1 0``."""
    return " ".join(str(int(x)) for x in occ)


def group_configuration_labels(
    conf_info_list,
    ncas: int,
    nelecas,
    ngroup: Optional[int] = None,
) -> List[str]:
    """Label each grouped bath by the spatial occupations it collects.

    Derived from the determinant -> group map alone, so it needs neither the
    GBCI ``po_list`` nor ``group``: every determinant of the CI vector is
    assigned its spatial occupation pattern and the patterns are collected per
    group.  A group that collects more than one pattern is labelled with all of
    them, joined by ``" / "``; an empty group gets ``"-"``.
    """
    neleca, nelecb = _unpack_nelecas(nelecas)
    conf = np.asarray(conf_info_list, dtype=int)
    if ngroup is None:
        ngroup = int(conf.max()) + 1
    occ_a = string_occupations(ncas, neleca)
    occ_b = string_occupations(ncas, nelecb)

    patterns: List[set] = [set() for _ in range(ngroup)]
    for ia in range(conf.shape[0]):
        for ib in range(conf.shape[1]):
            patterns[int(conf[ia, ib])].add(
                tuple((occ_a[ia] + occ_b[ib]).tolist())
            )
    return [
        " / ".join(format_occ(p) for p in sorted(group_patterns, reverse=True))
        or "-"
        for group_patterns in patterns
    ]


def configuration_ci_vectors(ncas: int, nelecas) -> List[Tuple[str, np.ndarray]]:
    """One normalized CI vector per spatial occupation pattern.

    The determinants of a pattern are combined with equal weights.  That is not
    in general the spin-adapted CSF, but determinants of the same spatial
    pattern differ by two spin orbitals, so they do not couple through the
    one-particle density matrix: every such combination gives the same 1-RDM,
    which is what these vectors are used for.

    Returned in the order of :func:`configuration_weights`' occupation keys,
    each as ``(label, civec)`` with ``civec`` of shape ``(nstringsa,
    nstringsb)``.
    """
    neleca, nelecb = _unpack_nelecas(nelecas)
    occ_a = string_occupations(ncas, neleca)
    occ_b = string_occupations(ncas, nelecb)

    members: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    for ia in range(occ_a.shape[0]):
        for ib in range(occ_b.shape[0]):
            key = tuple((occ_a[ia] + occ_b[ib]).tolist())
            members.setdefault(key, []).append((ia, ib))

    out = []
    for key in sorted(members, reverse=True):
        civec = np.zeros((occ_a.shape[0], occ_b.shape[0]))
        dets = members[key]
        for ia, ib in dets:
            civec[ia, ib] = 1.0 / np.sqrt(len(dets))
        out.append((format_occ(key), civec))
    return out
