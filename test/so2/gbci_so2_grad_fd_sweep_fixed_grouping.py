"""Rerun the SO2 GBCI gradient sweep with the corrected bath-grouping rule.

``pyscf.gbci.gbci.group_by_atom`` groups configurations by their distance to the
first member of each group, so the population spread inside a group can exceed
the threshold.  On this geometry it does: at ``thres=0.2`` one group spans 0.335,
and the configurations collapse into 4 groups instead of 6.

pyscf-forge is deliberately left unpatched, so this wrapper swaps the corrected
rule (``utils.bath_grouping.group_complete_linkage``, which bounds the spread
*within* a group) into ``pyscf.gbci`` for the duration of the run and otherwise
calls ``gbci_so2_grad_fd_sweep``'s own ``compare``/``report``/``save``.  The two
runs therefore differ only in the grouping rule.

The step grid is the sweep's own default, so the output is directly comparable
with ``outputs/cas8e5o_fd_sweep.json``.  Expect it to cost about twice as much:
``get_X`` loops over group pairs, and there are 6 groups rather than 4.

Usage
-----
    OMP_NUM_THREADS=1 python3 gbci_so2_grad_fd_sweep_fixed_grouping.py
    OMP_NUM_THREADS=1 python3 gbci_so2_grad_fd_sweep_fixed_grouping.py --grouping leader
    OMP_NUM_THREADS=1 python3 gbci_so2_grad_fd_sweep_fixed_grouping.py --steps 1e-3,5e-4

Run single-threaded: threaded BLAS moves the energy by ~1e-9 Eh between runs and
a central difference amplifies that by 1/(2h).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
REPO_ROOT = SCRIPT_PATH.parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Importing the sweep also runs its module-resolution guard, so a stale
# pyscf.grad.gbci fails here too.  Neither module puts REPO_ROOT on sys.path.
_sweep = _load("_so2_fd_sweep", SCRIPT_DIR / "gbci_so2_grad_fd_sweep.py")
_bath_grouping = _load("_bath_grouping", REPO_ROOT / "utils" / "bath_grouping.py")

GROUPING_RULES = {
    "complete": _bath_grouping.group_complete_linkage,
    "leader": _bath_grouping.group_leader,
}

DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "cas8e5o_fd_sweep_fixed_grouping.json"


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    """Take --grouping and --output here; everything else goes to the sweep."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--grouping", choices=sorted(GROUPING_RULES),
                        default="complete")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_known_args()


def main() -> None:
    wrapper_args, passthrough = parse_wrapper_args()

    argv, sys.argv = sys.argv, [sys.argv[0], "--output", str(wrapper_args.output),
                                *passthrough]
    try:
        args = _sweep.parse_args()
    finally:
        sys.argv = argv

    rule = GROUPING_RULES[wrapper_args.grouping]
    print(f"SO2 GBCI gradient sweep with the '{wrapper_args.grouping}' grouping rule")
    print(f"  group_a = {_sweep.GROUP_A}, threshold "
          f"{_sweep.GROUP_A.get('threshold', 0.2)}")
    print(f"  steps (bohr) {list(args.steps)}")
    print(f"  output       {args.output}")

    original = _bath_grouping.patch_gbci_grouping(rule)
    try:
        result = _sweep.compare(args)
        # Record which rule produced this, so the JSON cannot be mistaken for a
        # run against the unpatched pyscf-forge.
        result["calculation"]["grouping_rule"] = wrapper_args.grouping
        result["calculation"]["grouping_rule_source"] = str(
            (REPO_ROOT / "utils" / "bath_grouping.py").relative_to(REPO_ROOT))
        _sweep.report(result)
        _sweep.save(result, args.output)
    finally:
        importlib.import_module("pyscf.gbci.gbci").group_by_atom = original

    groups = result["calculation"]["reference_groups"]
    print(f"\n{len(groups)} bath groups: {[len(g) for g in groups]}")


if __name__ == "__main__":
    main()
