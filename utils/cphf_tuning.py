"""Convergence control for the bath CPHF solve in the GBCI analytic gradient.

``pyscf.grad.gbci._solve_bath_cphf`` defaults to ``max_cycle=100``, but both of
its call sites override that to 30.  The solve is a Krylov iteration over a
space of dimension ``nvir * nocc``, and ``lib.krylov`` returns silently when it
runs out of cycles, so on a molecule large enough for 30 vectors to be
insufficient the orbital-response term is simply wrong -- with no warning, and
with the energy unaffected.

:func:`patch_bath_cphf` forces the cycle count, tolerance and level shift from
outside pyscf-forge, and reports the residual of each solve so a solve that did
not converge is visible.  ``level_shift_scan`` solves the same equation at
several shifts in one run, which is the cheap way to tell an ill-conditioned
system from one that merely needs more cycles.

The level shift enters the preconditioner and is subtracted again inside the
operator, so it cancels from the equation actually being solved:

    x + (fvind(x) - x*s) * e_ai = -h1 * e_ai ,   e_ai = 1/(e_vir + s - e_occ)
    =>  x*(e_vir - e_occ) + fvind(x) = -h1

The solution is therefore independent of ``s``; only the conditioning changes.
:func:`bath_cphf_residual` measures that shift-free form, so residuals from
different shifts are directly comparable -- the preconditioned residual that
``lib.krylov`` works with is not.

``lindep`` is the knob that actually decides where the solve stops.  ``xvo`` is
2D, so ``lib.krylov`` takes its ``nroots == 1`` branch: no vector is ever
masked out, and the loop ends at the first cycle whose new Krylov direction
satisfies ``||x1||**2 < max(lindep, tol**2)``.  With the call sites' 1e-20 and
tol 1e-12 that is 1e-20, so ``innerprod`` spans roughly ``||b||**2`` down to
1e-20 before the loop breaks.  Those values land on the diagonal of the
projected matrix that ``numpy.linalg.solve(h, g)`` then inverts, so a span of
sixteen orders of magnitude costs the whole double-precision mantissa.  That is
why raising ``max_cycle`` changes nothing -- the loop breaks long before the
cycle limit, and ``lib.krylov`` would have raised ``RuntimeError`` had it not.

Raising ``lindep`` therefore stops the basis growing while the projected metric
is still invertible, which can lower the true residual even though it buys
fewer Krylov vectors.  ``lindep_scan`` measures that trade-off.  Note ``tol``
enters the same ``max(lindep, tol**2)`` test, so the two are one knob: a tol
of 1e-6 acts as a lindep of 1e-12.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np

_ORIGINAL_ATTR = "_solve_bath_cphf_unpatched"


def bath_cphf_residual(fvind, mo_energy, mo_occ, h1, mo1):
    """Relative residual of the shift-free bath CPHF equation.

    ``||x*(e_vir - e_occ) + fvind(x) + h1|| / ||h1||``.  Independent of the
    level shift used to obtain ``mo1``.
    """
    e_vir = mo_energy[mo_occ == 0]
    e_occ = mo_energy[mo_occ > 0]
    de = e_vir[:, None] - e_occ
    nvir, nocc = de.shape

    h1 = np.asarray(h1)
    x = np.asarray(mo1).reshape(-1, nvir, nocc)
    ax = x * de + np.asarray(fvind(x)).reshape(-1, nvir, nocc)

    denom = max(float(np.linalg.norm(h1)), 1e-300)
    return float(np.linalg.norm(ax.reshape(h1.shape) + h1) / denom)


def patch_bath_cphf(max_cycle=200, tol=1e-12, level_shift=1e-5,
                    level_shift_scan=None, lindep=1e-20, lindep_scan=None,
                    warn_residual=1e-8, verbose=True):
    """Force the bath CPHF solve settings, ignoring what the caller passes.

    Both call sites pass ``max_cycle=30`` and ``level_shift=1e-5`` explicitly,
    so the replacement discards the caller's values rather than only changing
    defaults.

    ``level_shift_scan`` and ``lindep_scan``: iterables of shifts and of Krylov
    linear-dependency thresholds.  Every combination is solved and its
    shift-free residual reported; the solution with the smallest residual is
    returned.  Use for diagnosis, not production.

    Returns the original function so a caller can restore it.
    """
    grad_mod = importlib.import_module("pyscf.grad.gbci")
    original = getattr(grad_mod, _ORIGINAL_ATTR, grad_mod._solve_bath_cphf)
    setattr(grad_mod, _ORIGINAL_ATTR, original)

    shifts = [level_shift] if level_shift_scan is None else list(level_shift_scan)
    lindeps = [lindep] if lindep_scan is None else list(lindep_scan)
    trials = [(s, ld) for ld in lindeps for s in shifts]

    def patched(fvind, mo_energy, mo_occ, h1, tol=None, lindep=1e-20,
                max_cycle=None, level_shift=None):
        # pyscf-forge solves this twice over: once per bath group inside
        # _bath_rotation_zvec, and once for the reference MOs inside grad_elec.
        # Naming the caller is the only way to tell the two apart in a log.
        caller = inspect.currentframe().f_back.f_code.co_name
        site = {"_bath_rotation_zvec": "group-bath",
                "grad_elec": "reference-MO"}.get(caller, caller)
        best_mo1, best_residual, best_trial = None, np.inf, None
        for shift, lindep_value in trials:
            # lib.krylov raises when the loop reaches max_cycle without the
            # new direction falling under max(lindep, tol**2).  At a lindep
            # deep enough to outrun double-precision orthogonality that is the
            # expected outcome, not a bug, so record it and keep scanning
            # instead of losing the whole run to one trial.
            try:
                mo1 = original(fvind, mo_energy, mo_occ, h1, tol=tol_forced,
                               lindep=lindep_value, max_cycle=max_cycle_forced,
                               level_shift=shift)
            except RuntimeError as exc:
                if verbose:
                    print(f"  bath CPHF [{site}]: max_cycle={max_cycle_forced} "
                          f"tol={tol_forced:g} level_shift={shift:g} "
                          f"lindep={lindep_value:g} "
                          f"RAISED {exc}", flush=True)
                continue
            residual = bath_cphf_residual(fvind, mo_energy, mo_occ, h1, mo1)
            if verbose:
                flag = "" if residual <= warn_residual else "   <-- NOT CONVERGED"
                print(f"  bath CPHF [{site}]: max_cycle={max_cycle_forced} "
                      f"tol={tol_forced:g} level_shift={shift:g} "
                      f"lindep={lindep_value:g} "
                      f"residual = {residual:.3e}{flag}", flush=True)
            if residual < best_residual:
                best_mo1, best_residual = mo1, residual
                best_trial = (shift, lindep_value)

        if best_mo1 is None:
            raise RuntimeError(
                f"bath CPHF [{site}]: every trial failed; widen lindep or "
                f"raise max_cycle (currently {max_cycle_forced})")

        if verbose and len(trials) > 1:
            print(f"  bath CPHF [{site}]: best level_shift = {best_trial[0]:g} "
                  f"lindep = {best_trial[1]:g} "
                  f"(residual {best_residual:.3e})", flush=True)
        return best_mo1

    tol_forced = tol
    max_cycle_forced = max_cycle
    grad_mod._solve_bath_cphf = patched
    return original


def restore_bath_cphf():
    """Undo :func:`patch_bath_cphf`."""
    grad_mod = importlib.import_module("pyscf.grad.gbci")
    original = getattr(grad_mod, _ORIGINAL_ATTR, None)
    if original is not None:
        grad_mod._solve_bath_cphf = original
