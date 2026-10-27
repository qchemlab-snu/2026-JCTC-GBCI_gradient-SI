"""Emit the supporting-information Cartesian-coordinate table for the CH2NH2 MECIs.

One longtable per active space, built from the converged ``*_optimized.xyz`` files so
the numbers cannot drift from the runs they describe.  The (8e,5o) table reproduces the
one already in the manuscript, which is also the check that the formatting is right.

Usage::

    python make_meci_si_geometry_table.py --active-space 2o2e
    python make_meci_si_geometry_table.py --active-space 5o8e --output-dir somewhere
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.geometry_descriptors import read_xyz  # noqa: E402
from utils.latex_tables import geometry_longtable  # noqa: E402

RUN_NAME = "ch2nh2_ccpvtz_meci_si_tables"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / RUN_NAME

ACTIVE_SPACE_LABEL = {"5o8e": "(8e,5o)", "2o2e": "(2e,2o)"}

# run_dir is relative to SCRIPT_DIR.  The CASCI (2e,2o) entry is the sigma = 0.25 run:
# the sigma = 1.0 run reached the same geometry to 0.0004 A but exhausted maxsteps
# without meeting the gradient criterion, so the converged one is the one to publish.
RUNS = {
    "5o8e": [
        ("CASCI", "../casci/outputs/ch2nh2_casci_S0S1_meci_5o8e_ccpvtz"),
        ("GBCI", "../gbci/outputs/ch2nh2_gbci_S0S1_meci_5o8e_groupocc_ccpvtz"),
        ("SA-CASSCF", "../casscf/outputs/ch2nh2_casscf_S0S1_meci_5o8e_ccpvtz"),
    ],
    # One GBCI row, from the MO-split (groupocc) run -- the one the published table
    # and Figure 5 both quote.  The Loewdin-on-N run
    # (ch2nh2_gbci_S0S1_meci_2o2e_groupN_ccpvtz) is not a second data point: with only
    # two active orbitals -- the N-localized and C-localized p pair -- Loewdin
    # clustering seeded on N puts them in separate groups, which is exactly the
    # [[0], [1]] split.  The two runs follow the same trajectory from step 0 and
    # converge to the same geometry within 8e-07 A, so listing both would imply an
    # agreement that was never tested.
    "2o2e": [
        ("CASCI", "../casci/outputs/ch2nh2_casci_S0S1_meci_2o2e_ccpvtz_sigma0.25"),
        ("GBCI", "../gbci/outputs/ch2nh2_gbci_S0S1_meci_2o2e_groupocc_ccpvtz"),
        ("SA-CASSCF", "../casscf/outputs/ch2nh2_casscf_S0S1_meci_2o2e_ccpvtz"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--active-space", choices=sorted(RUNS), default="2o2e")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_group(method: str, run_dir: str) -> tuple[str, list[str], list[list[float]]]:
    path = (SCRIPT_DIR / run_dir).resolve()
    symbols, coords = read_xyz(path / f"{path.name}_optimized.xyz")
    return method, symbols, coords


def main() -> None:
    args = parse_args()
    space = args.active_space
    groups = [load_group(method, run_dir) for method, run_dir in RUNS[space]]

    caption = (
        r"Optimized Cartesian coordinates (\AA) of the CH$_2$NH$_2^+$ "
        r"S$_0$/S$_1$ minimum-energy conical intersections located with CASCI, "
        r"GBCI and SA-CASSCF in the "
        f"{ACTIVE_SPACE_LABEL[space]} active space with the cc-pVTZ basis."
    )
    table = geometry_longtable(
        groups,
        caption=caption,
        label=f"tab:si_ch2nh2_ccpvtz_meci_geometries_{space}",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{RUN_NAME}_{space}.tex"
    out.write_text(table)
    print(table)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
