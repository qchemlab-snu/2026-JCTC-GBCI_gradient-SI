"""Put the SA-CASSCF NEB endpoints into the frames of the published structures.

The structure images already in the paper were rendered from the CASCI
geometries in whatever frame those files carry. To redraw the same structures
from the SA-CASSCF NEB endpoints without changing how they are composed, each
endpoint is rotated onto its reference by a Kabsch fit and written out in that
frame; the renderer then uses one fixed camera per structure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


SCRIPT_DIR = Path(__file__).resolve().parent
PP_DIR = SCRIPT_DIR.parent
NEB_DIR = PP_DIR / "casscf" / "pp_casscf_S1_fc_tict_neb_outputs"

PAIRS = {
    "start": {
        "neb": NEB_DIR / "pp_casscf_S1_fc_tict_neb_000.xyz",
        "reference": PP_DIR / "casci" / "pp_casci_S0.xyz",
        "out": SCRIPT_DIR / "pp_casscf_neb_start_aligned.xyz",
    },
    "end": {
        "neb": NEB_DIR / "pp_casscf_S1_fc_tict_neb_008.xyz",
        "reference": (
            PP_DIR / "casci" / "pp_casci_S1_neb_outputs" / "pp_casci_S1_neb_008.xyz"
        ),
        "out": SCRIPT_DIR / "pp_casscf_neb_end_aligned.xyz",
    },
}


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate ``mobile`` onto ``target``; both are centred first."""

    m = mobile - mobile.mean(axis=0)
    t = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(m.T @ t)
    d = np.sign(np.linalg.det(u @ vt))
    fitted = m @ (u @ np.diag([1.0, 1.0, d]) @ vt)
    return fitted, float(np.sqrt(((fitted - t) ** 2).sum(1).mean()))


def _swap(*pairs: tuple[int, int]) -> np.ndarray:
    order = np.arange(19)
    for i, j in pairs:
        order[i], order[j] = order[j], order[i]
    return order


# Each ring has a mirror symmetry that maps the molecule onto itself with the
# atoms relabelled: pyrrole N6-C1-C0-C2-C3 and pyridine C9-C10-C11-N13-C15-C16.
# Two optimizations can converge to the same structure with the rings flipped,
# which no rotation can undo, so the relabellings are part of the search.
RING_FLIPS = {
    "none": np.arange(19),
    "pyrrole": _swap((0, 2), (1, 3), (4, 5), (7, 8)),
    "pyridine": _swap((10, 16), (11, 15), (12, 18), (14, 17)),
}
RING_FLIPS["both"] = RING_FLIPS["pyrrole"][RING_FLIPS["pyridine"]]


def best_fit(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Fit with and without a mirror and keep the better one.

    The twisted structures are chiral and both rings are symmetric, so two
    optimizations can reach the same structure as a mirror image, with one or
    both rings relabelled, or both. A proper rotation alone then leaves a large
    residual even though the structures match. All eight combinations are tried
    and the closest is kept; for a structure figure they are equivalent.
    """

    best = None
    for flip_name, order in RING_FLIPS.items():
        permuted = mobile[order]
        for mirror_name, factor in (("", 1.0), ("+mirror", -1.0)):
            fitted, rmsd = kabsch(permuted * np.array([factor, 1.0, 1.0]), target)
            if best is None or rmsd < best[1]:
                best = (fitted, rmsd, f"{flip_name}{mirror_name}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    for key, spec in PAIRS.items():
        neb = read(str(spec["neb"]))
        ref = read(str(spec["reference"]))
        if neb.get_chemical_symbols() != ref.get_chemical_symbols():
            raise ValueError(f"Atom order differs for {key}.")

        target = ref.get_positions()
        fitted, rmsd, how = best_fit(neb.get_positions(), target)
        write(str(spec["out"]), Atoms(symbols=neb.get_chemical_symbols(),
                                      positions=fitted + target.mean(axis=0)))
        print(f"{key}: wrote {spec['out'].name}  "
              f"RMSD to reference {rmsd:.4f} A  [{how}]")


if __name__ == "__main__":
    main()
