"""Run the SO2 GBCI finite-difference sweep at the published convergence settings.

``gbci_so2_grad_fd_sweep.py`` uses tighter thresholds than the manuscript
describes.  Rather than fork its 400 lines, this wrapper imports it, pins the
published settings, and calls its own ``compare``/``report``/``save``, so the two
runs differ only in the numbers listed under PAPER SETTINGS below.

Published settings (configurations grouped by the Lowdin population of active
electrons on the S atom):

    Lowdin population difference within a group   0.2
    reference-orbital energy / orbital gradient   1e-12 / 1e-6
    bath-orbital    energy / orbital gradient     1e-14 / 1e-12
    GBCI Davidson                                 1e-12
    CPHF Krylov convergence / linear dependence   1e-10 / 1e-20

Of these the sweep script already matched the bath-orbital pair and the Lowdin
threshold (0.2 is GBCI's default, set explicitly here so it is recorded); the
CPHF pair is fixed inside pyscf-forge's ``_solve_bath_cphf`` and is not settable
from here.  The three that differ are the reference-orbital pair and the
Davidson threshold, all of which the sweep script had tighter.

Usage
-----
    OMP_NUM_THREADS=1 python3 gbci_so2_grad_fd_sweep_paper.py
    OMP_NUM_THREADS=1 python3 gbci_so2_grad_fd_sweep_paper.py --steps 1e-3,5e-4

Any flag given on the command line overrides the pinned value, since argparse
takes the last occurrence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

# Import the sweep by path, exactly as it would run itself: it puts neither the
# repo root nor the local pyscf tree on sys.path, and its module-resolution
# guard runs on import, so a stale pyscf.grad.gbci fails here too.
_spec = importlib.util.spec_from_file_location(
    "_so2_fd_sweep", SCRIPT_DIR / "gbci_so2_grad_fd_sweep.py")
_sweep = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _sweep
_spec.loader.exec_module(_sweep)


# ============================================================
# PAPER SETTINGS
# ============================================================

LOWDIN_THRESHOLD = 0.2          # max population difference within a group
SCF_CONV_TOL = 1.0e-12          # reference-orbital energy
SCF_CONV_TOL_GRAD = 1.0e-6      # reference-orbital gradient
FASSCF_CONV_TOL = 1.0e-14       # bath-orbital energy
FASSCF_CONV_TOL_GRAD = 1.0e-12  # bath-orbital gradient
GBCI_CONV_TOL = 1.0e-12         # Davidson

# S-centred grouping, with the Lowdin threshold stated rather than defaulted.
GROUP_A = {"atom": [[0]], "threshold": LOWDIN_THRESHOLD}

# The sweep's own default starts at 4e-3, but 4e-3 and 2e-3 both move a
# configuration between bath groups on this geometry and are always skipped, so
# this grid starts where usable data begins and extends past the known optimum.
DEFAULT_STEPS = "1e-3,5e-4,2.5e-4,1e-4,5e-5"

DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "cas8e5o_fd_sweep_paper.json"


def pinned_argv() -> list[str]:
    """Paper settings as command-line flags, with user flags appended last."""
    return [
        "--scf-conv-tol", repr(SCF_CONV_TOL),
        "--scf-conv-tol-grad", repr(SCF_CONV_TOL_GRAD),
        "--fasscf-conv-tol", repr(FASSCF_CONV_TOL),
        "--fasscf-conv-tol-grad", repr(FASSCF_CONV_TOL_GRAD),
        "--gbci-conv-tol", repr(GBCI_CONV_TOL),
        "--steps", DEFAULT_STEPS,
        "--output", str(DEFAULT_OUTPUT),
    ] + sys.argv[1:]


def main() -> None:
    # compare() reads GROUP_A from the sweep module's globals, so pinning it
    # here is what puts the stated Lowdin threshold into the calculation and
    # into the saved JSON.
    _sweep.GROUP_A = GROUP_A

    argv, sys.argv = sys.argv, [sys.argv[0]] + pinned_argv()
    try:
        args = _sweep.parse_args()
    finally:
        sys.argv = argv

    print("Running the SO2 GBCI gradient sweep at the published settings:")
    print(f"  Lowdin threshold (S)        {LOWDIN_THRESHOLD}")
    print(f"  reference orbital E / grad  {args.scf_conv_tol:.0e} / "
          f"{args.scf_conv_tol_grad:.0e}")
    print(f"  bath orbital      E / grad  {args.fasscf_conv_tol:.0e} / "
          f"{args.fasscf_conv_tol_grad:.0e}")
    print(f"  GBCI Davidson               {args.gbci_conv_tol:.0e}")
    print(f"  steps (bohr)                {list(args.steps)}")

    result = _sweep.compare(args)
    _sweep.report(result)
    _sweep.save(result, args.output)


if __name__ == "__main__":
    main()
