"""Disk cache for the FASSCF bath optimization inside GBCI.

``GBCI.kernel`` calls ``optimize_mo`` unconditionally and only writes
``_gbci_intermediates_cache`` afterwards, so forge's own cache helps a second
call inside one process and nothing else.  The bath optimization is most of the
cost of a GBCI single point, which makes re-running one just to vary a gradient
setting -- CPHF cycles, level shift -- needlessly expensive.

:func:`patch_optimize_mo_cache` wraps ``optimize_mo`` so its result is stored on
disk and reused whenever the inputs match.  The key covers the MO coefficients
byte for byte together with ``ncas``, ``nelecas``, ``ncore`` and ``group_a``, so
a cache entry can only be reused for the orbitals that produced it: displaced
geometries in a finite difference get their own entries automatically, and a
changed active space or grouping misses rather than returning stale data.

Reusing a cache entry requires bitwise-identical MO coefficients, so the SCF
must come from a chkfile rather than from a fresh ``kernel()`` -- otherwise the
key misses and the bath is recomputed (correct, just not faster).

The files are written with pickle and are only meant to be read back by the run
that wrote them; do not load a cache directory from an untrusted source.
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
from pathlib import Path

import numpy as np

_PATCHED_ATTR = "_optimize_mo_unpatched"


def cache_key(mo_coeff, ncas, nelecas, ncore, group_a, tag=""):
    """Content hash of everything the bath optimization depends on.

    ``tag`` identifies anything outside the arguments that changes the result --
    above all which grouping rule is installed, since ``optimize_mo`` returns
    the group assignment and a cached entry would otherwise serve the grouping
    from whichever rule happened to run first.
    """
    mo = np.ascontiguousarray(mo_coeff)
    digest = hashlib.sha256()
    digest.update(np.asarray(mo.shape, dtype=np.int64).tobytes())
    digest.update(str(mo.dtype).encode())
    digest.update(mo.tobytes())
    digest.update(repr((int(ncas), tuple(nelecas), int(ncore),
                        repr(group_a), str(tag))).encode())
    return digest.hexdigest()[:32]


def patch_optimize_mo_cache(cache_dir, tag="", verbose=True):
    """Cache ``GBCI.optimize_mo`` results under ``cache_dir``.

    ``tag`` goes into the key: pass something identifying the installed
    grouping rule, or entries written under one rule will be served under
    another.

    Returns the original method so a caller can restore it.
    """
    gbci_mod = importlib.import_module("pyscf.gbci.gbci")
    cls = gbci_mod.GBCI
    original = getattr(cls, _PATCHED_ATTR, cls.optimize_mo)
    setattr(cls, _PATCHED_ATTR, original)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def patched(self, mo_coeff=None, *args, **kwargs):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        key = cache_key(mo_coeff, self.ncas, self.nelecas, self.ncore,
                        getattr(self, "_group_a", None), tag)
        path = cache_dir / f"bath_{key}.pkl"

        if path.exists():
            try:
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
                if verbose:
                    print(f"  GBCI bath cache hit: {path.name}", flush=True)
                return payload["result"]
            except Exception as exc:        # a corrupt entry must not be fatal
                if verbose:
                    print(f"  GBCI bath cache unreadable ({exc}); recomputing",
                          flush=True)

        result = original(self, mo_coeff, *args, **kwargs)

        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as handle:
            pickle.dump({"key": key, "result": result}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)                   # atomic: concurrent tasks are safe
        if verbose:
            print(f"  GBCI bath cache written: {path.name}", flush=True)
        return result

    cls.optimize_mo = patched
    return original


def restore_optimize_mo():
    """Undo :func:`patch_optimize_mo_cache`."""
    gbci_mod = importlib.import_module("pyscf.gbci.gbci")
    cls = gbci_mod.GBCI
    original = getattr(cls, _PATCHED_ATTR, None)
    if original is not None:
        cls.optimize_mo = original


def scf_cache_key(mol, conv_tol, conv_tol_grad):
    """Content hash of the molecule and the SCF convergence settings."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mol.atom_coords()).tobytes())
    digest.update(repr([mol.atom_symbol(i) for i in range(mol.natm)]).encode())
    digest.update(repr(mol.basis).encode())
    digest.update(repr((mol.charge, mol.spin, float(conv_tol),
                        float(conv_tol_grad))).encode())
    return digest.hexdigest()[:32]


def load_scf(mf, cache_dir, key, verbose=True):
    """Restore a cached SCF solution into ``mf``. Returns True on a hit.

    The point is not the minute of SCF time but reproducibility: the bath cache
    keys on the MO coefficients byte for byte, so restoring the same orbitals is
    what lets it hit at all.
    """
    path = Path(cache_dir) / f"scf_{key}.pkl"
    if not path.exists():
        return False
    try:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
    except Exception as exc:
        if verbose:
            print(f"  SCF cache unreadable ({exc}); recomputing", flush=True)
        return False

    mf.mo_coeff = data["mo_coeff"]
    mf.mo_energy = data["mo_energy"]
    mf.mo_occ = data["mo_occ"]
    mf.e_tot = data["e_tot"]
    mf.converged = data["converged"]
    if verbose:
        print(f"  SCF cache hit: {path.name} (E = {mf.e_tot:.10f})", flush=True)
    return True


def save_scf(mf, cache_dir, key, verbose=True):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"scf_{key}.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as handle:
        pickle.dump({
            "mo_coeff": np.asarray(mf.mo_coeff),
            "mo_energy": np.asarray(mf.mo_energy),
            "mo_occ": np.asarray(mf.mo_occ),
            "e_tot": float(mf.e_tot),
            "converged": bool(mf.converged),
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    if verbose:
        print(f"  SCF cache written: {path.name}", flush=True)
