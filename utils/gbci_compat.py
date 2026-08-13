"""Small compatibility helpers for the qchemlab-snu pyscf-forge GBCI API."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pyscf.grad
from pyscf.gbci.gbci import GBCI


_LOCAL_GRAD_PATH = str(Path(__file__).resolve().parents[1] / "pyscf" / "grad")
if _LOCAL_GRAD_PATH in pyscf.grad.__path__:
    pyscf.grad.__path__.remove(_LOCAL_GRAD_PATH)
pyscf.grad.__path__.insert(0, _LOCAL_GRAD_PATH)


def normalize_legacy_group_a(group_a):
    """Translate the legacy ``groupA`` forms used by the calculation scripts."""
    if group_a is None:
        return None

    if isinstance(group_a, Mapping):
        current_keys = {"atom", "mo", "occ", "kind"}
        if current_keys.intersection(group_a):
            return dict(group_a)

        # The old API accepted named molecular fragments whose values were
        # atom-index lists, for example {"mol1": [...], "mol2": [...]}.
        return {"atom": [list(atom_ids) for atom_ids in group_a.values()]}

    if isinstance(group_a, Sequence) and not isinstance(group_a, (str, bytes)):
        # The old list form grouped active-orbital indices by their summed
        # occupations.  The current API calls this the ``mo`` grouping mode.
        groups = []
        for group in group_a:
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes)):
                groups.append(list(group))
            else:
                groups.append([group])
        return {"mo": groups}

    raise TypeError(
        "groupA must be None, a mapping, or a sequence of active-orbital groups"
    )


def make_gbci(mf, ncas, nelecas, group_a=None, **kwargs):
    """Construct GBCI using the current ``group_a`` keyword and schema."""
    gbci = GBCI(
        mf,
        ncas,
        nelecas,
        group_a=normalize_legacy_group_a(group_a),
        **kwargs,
    )
    # qchemlab-snu's FASSCF driver is ROHF-based.  Its mixed routine defaults
    # to damping, while PySCF's ROHF get_fock explicitly rejects damping.
    gbci._fasscf.mixed_routine_options["damp"] = 0.0
    return gbci
