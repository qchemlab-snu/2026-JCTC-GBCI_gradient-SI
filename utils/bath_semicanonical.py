"""Restore semicanonical bath orbitals after GBCI's FAS-SCF.

The bath CPHF in `pyscf.grad.gbci._bath_rotation_zvec` divides by
`eps_v - eps_o` taken from `moe_list`.  That is the right equation only when
the group's bath orbitals are **semicanonical** -- the group Fock diagonal
within the core block and within the virtual block -- and when `moe_list`
holds the diagonal of that same Fock.

`optimize_mo` does not always deliver this.  It calls `FASSCF.mixed_routine`,
whose damped first stage is followed by `newton_soscf_kernel` when that stage
does not converge, and that call leaves pyscf's `canonicalization` at its
`False` default.  In `pyscf/soscf/newton_ah.py` the canonicalized energies are
assigned unconditionally while the coefficients are not::

    mo_energy, mo_coeff1 = mf._scf.canonicalize(mo_coeff, mo_occ, fock)
    if mf.canonicalization:
        mo_coeff = mo_coeff1

So a group routed through Newton comes back with canonical *energies* attached
to non-canonical *orbitals*.  Measured on 3HF / 6-31G / GBCI(4e,4o): a core
block off-diagonal by 0.17 Eh, `moe_list` missing its own Fock diagonal by
0.497 Eh, and an analytic gradient 2e-04 off the finite-difference reference
where a well-behaved bath reaches 1e-07.

:func:`patch_optimize_mo` re-diagonalizes each group's core and virtual bath
blocks and returns the matching orbital energies.  It leaves every space
untouched, so the GBCI energy does not move; only the gauge inside the blocks
and the energies that label it change.  The active orbitals are never
involved, which is why this is done here rather than by turning pyscf's
`canonicalization` on -- `canonicalize` splits by occupation and would be free
to rotate active orbitals into the core block, breaking the frozen active
space FAS-SCF is built on.

Harmless when the bath is already semicanonical: on such a bath the rotation
is the identity up to orbital phase, and the gradient is unchanged to 1e-10.
"""

from __future__ import annotations

import numpy as np

_PATCHED_ATTR = "_optimize_mo_uncanonicalized"


def group_average_occupations(gbci, po_list, group):
    """Equally weighted active occupation per group, as FAS-SCF defines it."""
    from pyscf.fci import cistring
    from pyscf.gbci.gbci import str2occ, group_info_list

    ncas, nelecas = gbci.ncas, gbci.nelecas
    stringsa = cistring.make_strings(range(ncas), nelecas[0])
    stringsb = cistring.make_strings(range(ncas), nelecas[1])
    nb = len(stringsb)
    conf = np.asarray(group_info_list(ncas, nelecas, po_list, group)).reshape(-1)

    occs = []
    for p in range(len(group)):
        target = np.where(conf == p)[0]
        if target.size == 0:
            raise ValueError(f"group {p} has no configurations assigned")
        occa = np.mean([str2occ(stringsa[c // nb], ncas) for c in target], axis=0)
        occb = np.mean([str2occ(stringsb[c % nb], ncas) for c in target], axis=0)
        occs.append(occa + occb)
    return occs


def semicanonicalize(gbci, mo_list, po_list, group, tol=None):
    """Return bath orbitals diagonal in the group Fock within each block,
    together with the orbital energies of that same Fock.

    ``tol``: when given, only groups whose block off-diagonality exceeds it are
    touched; the rest are returned unchanged.  Reports what it did as a list of
    per-group dicts.
    """
    from pyscf.scf import hf

    ncore, ncas = gbci.ncore, gbci.ncas
    nao = np.asarray(mo_list).shape[1]
    bath = list(range(ncore)) + list(range(ncore + ncas, nao))
    mol = gbci.mol
    h1e = gbci._scf.get_hcore()
    mo_cas = gbci.mo_coeff[:, ncore:ncore + ncas]
    occs = group_average_occupations(gbci, po_list, group)

    out = np.array(mo_list, copy=True)
    energies = np.zeros((len(group), nao))
    report = []
    for p in range(len(group)):
        coeff = np.asarray(mo_list[p])
        core = coeff[:, :ncore]
        dm = 2 * core @ core.conj().T + (mo_cas * occs[p]) @ mo_cas.conj().T
        fock = h1e + hf.get_veff(mol, dm)

        b = coeff[:, bath]
        fmo = b.conj().T @ fock @ b
        before = max(_offdiag(fmo[:ncore, :ncore]), _offdiag(fmo[ncore:, ncore:]))

        if tol is not None and before <= tol:
            out[p] = coeff
            energies[p] = np.diag(coeff.conj().T @ fock @ coeff).real
            report.append({"group": p, "offdiag_before": before,
                           "offdiag_after": before, "rotated": False})
            continue

        new = b.copy()
        for sl in (slice(0, ncore), slice(ncore, None)):
            _, vecs = np.linalg.eigh(fmo[sl, sl])
            new[:, sl] = b[:, sl] @ vecs
        out[p][:, bath] = new

        fmo2 = new.conj().T @ fock @ new
        after = max(_offdiag(fmo2[:ncore, :ncore]), _offdiag(fmo2[ncore:, ncore:]))
        energies[p] = np.diag(out[p].conj().T @ fock @ out[p]).real
        report.append({"group": p, "offdiag_before": before,
                       "offdiag_after": after, "rotated": True})
    return out, energies, report


def _offdiag(block):
    sub = np.array(block, copy=True)
    np.fill_diagonal(sub, 0.0)
    return float(np.abs(sub).max()) if sub.size else 0.0


def patch_optimize_mo(gbci, tol=None, verbose=True):
    """Wrap ``gbci.optimize_mo`` so every bath it returns is semicanonical.

    Returns the original bound method so a caller can restore it.
    """
    original = getattr(gbci, _PATCHED_ATTR, gbci.optimize_mo)
    setattr(gbci, _PATCHED_ATTR, original)

    def patched(*args, **kwargs):
        mo_list, mo_energy, po_list, group = original(*args, **kwargs)
        fixed, energies, report = semicanonicalize(
            gbci, mo_list, po_list, group, tol=tol)
        if verbose:
            for r in report:
                flag = "" if r["rotated"] else "  (already semicanonical)"
                print(f"  bath semicanonical [group {r['group']}]: block "
                      f"off-diagonal {r['offdiag_before']:.3e} -> "
                      f"{r['offdiag_after']:.3e}{flag}", flush=True)
        return fixed, energies, po_list, group

    gbci.optimize_mo = patched
    return original


def restore_optimize_mo(gbci):
    """Undo :func:`patch_optimize_mo`."""
    original = getattr(gbci, _PATCHED_ATTR, None)
    if original is not None:
        gbci.optimize_mo = original
        delattr(gbci, _PATCHED_ATTR)
