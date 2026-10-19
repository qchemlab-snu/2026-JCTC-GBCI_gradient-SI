"""The tight-FASSCF analytic gradient, with the bath semicanonicalized.

Runs `conv_tol_grad = 1e-11` -- the setting that has never yet reproduced
finite differences -- with `utils/bath_semicanonical.patch_optimize_mo`
installed, and compares against the finite-difference reference.  The patched
and unpatched gradients are computed in **one process from one FAS-SCF run**,
so the two differ only by the semicanonicalization.

Expected, from `21_`: unpatched ~2e-04, patched ~1e-07.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECK_SCRIPT = SCRIPT_DIR / "01_gbci_s1_grad_check_631g.py"
FD_DIR = SCRIPT_DIR / "outputs" / "01_partials_tight_frozen_h25e4"
OUTPUT_JSON = SCRIPT_DIR / "outputs" / "25_tight_gradient_fixed.json"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctg", type=float, default=1e-11)
    ap.add_argument("--fd-dir", type=Path, default=FD_DIR)
    ap.add_argument("--output", type=Path, default=OUTPUT_JSON)
    o = ap.parse_args()

    check = _load("_grad_check", CHECK_SCRIPT)
    semi = _load("_semi", REPO_ROOT / "utils" / "bath_semicanonical.py")
    check._bath_grouping.patch_gbci_grouping()

    # --no-semicanonical-bath is essential here: 01_ applies the fix by
    # default, and this script exists to measure what the fix is worth, so it
    # needs the bath exactly as FAS-SCF returns it and semicanonicalizes the
    # second variant itself.
    sys.argv = ["fixed", "--mode", "reference", "--no-gbci-cache",
                "--no-semicanonical-bath",
                "--fasscf-conv-tol-grad", repr(o.ctg), "--verbose", "3"]
    args = check.parse_args()
    symbols, coords = check.reference_coords()
    mol = check.build_molecule(symbols, coords, args.verbose)
    gbci = check.run_gbci(mol, args)

    inter = gbci._gbci_intermediates
    raw = (np.array(inter["mo_list"], copy=True), np.array(inter["mo_energy"], copy=True),
           inter["po_list"], inter["group"])
    print(f"\nFASSCF conv_tol_grad = {o.ctg:g}", flush=True)
    fixed_mo, fixed_e, report = semi.semicanonicalize(
        gbci, raw[0], raw[2], raw[3])
    for r in report:
        print(f"  group {r['group']}: block off-diagonal "
              f"{r['offdiag_before']:.3e} -> {r['offdiag_after']:.3e}", flush=True)
    moe_shift = np.abs(np.asarray(raw[1]) - fixed_e).max()
    print(f"  max |moe_as_returned - moe_semicanonical| = {moe_shift:.3e} Eh",
          flush=True)

    fd = {}
    for f in sorted(glob.glob(str(o.fd_dir / "*.json"))):
        for r in json.load(open(f))["displacements"]:
            fd[(r["atom"], r["axis"])] = r["numerical"]

    def score(ml, me, label):
        gbci.optimize_mo = (lambda a, b: (lambda *x, **k: (a, b, raw[2], raw[3])))(ml, me)
        gbci._clear_gbci_intermediates()
        gbci.kernel()
        e = np.asarray(gbci.e_tot, dtype=float).reshape(-1)
        g = check.analytic_gradient(gbci, args.state)
        nf, torque = check.gradient_invariance(mol, g)
        diffs = [g[a, x] - v for (a, x), v in fd.items()]
        mx = max(abs(d) for d in diffs)
        rms = float(np.sqrt(np.mean(np.square(diffs))))
        del gbci.optimize_mo
        print(f"{label:>26}{e[args.state]:20.12f}{torque:13.3e}"
              f"{mx:13.3e}{rms:12.3e}", flush=True)
        return {"label": label, "e_state": float(e[args.state]),
                "torque": float(torque), "net_force": float(nf),
                "max_abs_err": float(mx), "rms_err": rms, "analytic": g.tolist()}

    print(f"\n{'variant':>26}{'E(S1)':>20}{'torque':>13}"
          f"{'max|g-FD|':>13}{'rms':>12}", flush=True)
    rows = [score(raw[0], raw[1], "as returned"),
            score(fixed_mo, fixed_e, "semicanonicalized")]

    g_fixed = np.array(rows[1]["analytic"])
    print(f"\nenergy moved by {abs(rows[1]['e_state'] - rows[0]['e_state']):.2e} Eh "
          f"(the fix changes no space, so this should be ~0)")
    print(f"gradient moved by "
          f"{np.abs(g_fixed - np.array(rows[0]['analytic'])).max():.3e} Eh/bohr")

    print("\n--- semicanonicalized analytic gradient (Eh/bohr) ---")
    print(f"{'atom':>6}{'x':>16}{'y':>16}{'z':>16}")
    for i, sym in enumerate(symbols):
        print(f"{i:3d} {sym:<2}{g_fixed[i, 0]:16.10f}"
              f"{g_fixed[i, 1]:16.10f}{g_fixed[i, 2]:16.10f}")

    o.output.parent.mkdir(parents=True, exist_ok=True)
    o.output.write_text(json.dumps(
        {"ctg": o.ctg, "report": report, "moe_shift": float(moe_shift),
         "symbols": list(symbols), "runs": rows}, indent=2))
    print(f"\nSaved {o.output}")


if __name__ == "__main__":
    main()
