"""
CASCI active-space/root-tracked UBS/GPM MECI optimizer for PySCF.

This is a prototype, not a drop-in replacement for a production RFO MECI optimizer.
It combines
  1) active-space tracking by overlap with the previous active MO subspace,
  2) pair root tracking by CI-vector overlap,
  3) updated branching space/plane without CDV,
  4) gradient projection MECI optimization.

Assumptions:
  - RHF reference is used.
  - PySCF CASCI gradients are available for the requested states.
  - target_roots are CASCI root indices, 0-based. For S1/S2 with S0 included,
    use target_roots=(1, 2) and nroots >= 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf import addons as mcscf_addons
from pyscf.gbci.gbci import group_info_list

from utils.gbci_compat import make_gbci
HARTREE2EV = 27.211386245988
BOHR2ANG = 0.529177210903


@dataclass
class PairResult:
    """Result returned by a two-state scanner."""

    energies: np.ndarray          # shape (2,), Eh
    gradients: np.ndarray         # shape (2, natm, 3), Eh/Bohr
    roots: Tuple[int, int]
    mol: Any
    mf: Any
    mc: Any
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class CASCIActiveRootTrackedPairScanner:
    """
    CASCI scanner returning two tracked roots and their gradients.

    Compared with a single-root scanner, MECI optimization needs a pair of states.
    Therefore root tracking is performed as a 2 x nroots assignment problem:
    previous state A/B CI vectors are matched to distinct current roots by maximum
    total CI-vector overlap. The selected pair can then be ordered either by energy
    or by the tracked labels.
    """

    method_label = "CASCI"

    def __init__(
        self,
        ncas: int,
        nelecas: Any,
        target_roots: Sequence[int] = (0, 1),
        nroots: Optional[int] = None,
        act_list: Optional[Sequence[int]] = None,
        act_base: int = 0,
        candidate_idx: Optional[Sequence[int]] = None,
        root_tracking: bool = True,
        active_tracking: bool = True,
        pair_order: str = "energy",  # "energy" or "tracked"
        conv_tol: float = 1.0e-10,
        max_cycle: int = 50,
        mf_factory: Optional[Callable[[Any], Any]] = None,
        mc_factory: Optional[Callable[[Any, int, Any, int], Any]] = None,
        reuse_ci_guess: bool = False,
        root_overlap_warn: float = 0.50,
        root_margin_warn: float = 0.10,
        active_sv_warn: float = 0.70,
        verbose: bool = True,
    ):
        self.ncas = int(ncas)
        self.nelecas = nelecas

        if len(target_roots) != 2:
            raise ValueError("target_roots must contain exactly two root indices.")
        self.initial_target_roots = tuple(int(x) for x in target_roots)
        self.current_roots = self.initial_target_roots
        self.nroots = int(nroots) if nroots is not None else max(self.initial_target_roots) + 1

        self.act_list_is_auto = act_list is None
        if act_list is None:
            self.act_list = None
        else:
            arr = np.asarray(act_list, dtype=int)
            self.act_list = arr - 1 if act_base == 1 else arr

        self.candidate_idx = None if candidate_idx is None else np.asarray(candidate_idx, dtype=int)
        self.root_tracking = bool(root_tracking)
        self.active_tracking = bool(active_tracking)
        self.pair_order = str(pair_order).lower()
        if self.pair_order not in ("energy", "tracked"):
            raise ValueError("pair_order must be 'energy' or 'tracked'.")

        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.mf_factory = mf_factory
        self.mc_factory = mc_factory
        self.reuse_ci_guess = bool(reuse_ci_guess)

        self.root_overlap_warn = root_overlap_warn
        self.root_margin_warn = root_margin_warn
        self.active_sv_warn = active_sv_warn
        self.verbose = verbose

        self.step = 0
        self.converged = False

        # Previous-step memory for active/root tracking
        self.prev_mol = None
        self.prev_active_mo = None
        self.prev_ci_targets: Optional[List[np.ndarray]] = None
        self.prev_e_targets: Optional[np.ndarray] = None
        self.prev_ci_list: Optional[List[np.ndarray]] = None
        self.prev_coords = None

        self.last_result: Optional[PairResult] = None

    # ------------------------------------------------------------
    # Basic builders
    # ------------------------------------------------------------
    def make_mf(self, mol):
        if self.mf_factory is not None:
            mf = self.mf_factory(mol)
        else:
            mf = scf.RHF(mol)
        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.diis_start_cycle = 50
        mf.damp = 0.3
        mf.level_shift = 0.7
        return mf

    def make_cas(self, mf):
        if self.mc_factory is not None:
            mc = self.mc_factory(mf, self.ncas, self.nelecas, self.nroots)
        else:
            mc = mcscf.CASCI(mf, self.ncas, self.nelecas)
            mc.fcisolver.nroots = self.nroots
        mc.fix_spin_(ss=0.0, shift=0.5)
        mc.conv_tol = self.conv_tol
        mc.max_cycle = self.max_cycle
        return mc

    # ------------------------------------------------------------
    # Overlap utilities
    # ------------------------------------------------------------
    @staticmethod
    def cross_ao_overlap(mol_old, mol_new):
        return gto.intor_cross("int1e_ovlp", mol_old, mol_new)

    @staticmethod
    def cross_mo_overlap(mol_old, mo_old, mol_new, mo_new):
        s_old_new = gto.intor_cross("int1e_ovlp", mol_old, mol_new)
        return mo_old.T @ s_old_new @ mo_new

    @staticmethod
    def _as_ci_list(mc) -> List[Any]:
        ci = mc.ci
        if isinstance(ci, (list, tuple)):
            return list(ci)
        return [ci]

    @staticmethod
    def _ci_overlap(ci_old, ci_new) -> float:
        v_old = np.asarray(ci_old).ravel()
        v_new = np.asarray(ci_new).ravel()
        if v_old.shape != v_new.shape:
            return np.nan
        n_old = np.linalg.norm(v_old)
        n_new = np.linalg.norm(v_new)
        if n_old < 1.0e-14 or n_new < 1.0e-14:
            return np.nan
        return float(abs(np.vdot(v_old, v_new)) / (n_old * n_new))

    @staticmethod
    def get_root_energies(mc) -> np.ndarray:
        """
        Return state-resolved energies for CASCI-like and SA-CASSCF methods.

        CASCI stores its roots in ``e_tot``.  State-averaged CASSCF keeps the
        averaged optimization energy in ``e_tot`` and exposes the individual
        roots through ``e_states``.
        """
        e_states = getattr(mc, "e_states", None)
        if e_states is not None:
            return np.atleast_1d(np.asarray(e_states, dtype=float))
        return np.atleast_1d(np.asarray(mc.e_tot, dtype=float))

    def nelecas_total(self) -> int:
        if isinstance(self.nelecas, (tuple, list, np.ndarray)):
            return int(np.sum(self.nelecas))
        return int(self.nelecas)

    def get_default_act_list(self, mc, mf) -> np.ndarray:
        if getattr(mc, "ncore", None) is not None:
            ncore = int(mc.ncore)
        else:
            nelecas_total = self.nelecas_total()
            ncore_float = 0.5 * (mf.mol.nelectron - nelecas_total)
            if abs(ncore_float - round(ncore_float)) > 1.0e-10:
                raise ValueError(
                    "Cannot determine closed-shell ncore automatically. "
                    "For open-shell or odd-electron cases, give act_list explicitly."
                )
            ncore = int(round(ncore_float))

        act_list = np.arange(ncore, ncore + self.ncas, dtype=int)
        nmo = mf.mo_coeff.shape[1]
        if act_list[-1] >= nmo:
            raise ValueError(f"Automatic active list {act_list.tolist()} exceeds nmo={nmo}.")
        return act_list
    
    def sort_mo_by_active_list(self, method, mf, selected, ncore, copy_scf=True):
        """
        Sort MOs into the CAS/GBCI order and immediately attach consistent
        orbital metadata to both the method object and method._scf.

        This replaces mcscf.addons.sort_mo() for this optimizer.

        It moves selected orbitals into the active block:
            core | active | external

        The selected active orbitals keep the order already determined by
        active tracking.

        Important:
            This function also updates
                method._scf.mo_coeff
                method._scf.mo_energy
                method._scf.mo_occ
            to the same sorted order, so downstream code that reads
            mc._scf.mo_energy sees the correct order.

            The original mf object is not modified unless method._scf is the
            same object and copy_scf=False. By default, method._scf is copied
            before the sorted data are attached.
        """
        mo_coeff = mf.mo_coeff
        selected = list(np.asarray(selected, dtype=int))
        nmo = mo_coeff.shape[1]

        if len(set(selected)) != len(selected):
            raise ValueError(f"Duplicated active orbital indices: {selected}")

        if min(selected) < 0 or max(selected) >= nmo:
            raise ValueError(
                f"Active orbital index out of range. selected={selected}, nmo={nmo}"
            )

        selected_set = set(selected)
        rest = [i for i in range(nmo) if i not in selected_set]

        if len(rest) < ncore:
            raise ValueError(
                f"Not enough non-active orbitals to form ncore={ncore}. "
                f"selected={selected}, nmo={nmo}"
            )

        core = rest[:ncore]
        external = rest[ncore:]
        mo_order = np.asarray(core + selected + external, dtype=int)

        mo_sorted = mo_coeff[:, mo_order]
        mo_sorted = self.tag_sorted_mo_metadata(
            mf=mf,
            mo_sorted=mo_sorted,
            mo_order=mo_order,
            selected=selected,
        )
        self.attach_sorted_mo_to_method(
            method,
            mo_sorted,
            copy_scf=copy_scf,
        )

        return mo_sorted, mo_order

    @staticmethod
    def tag_sorted_mo_metadata(mf, mo_sorted, mo_order, selected=None):
        """
        Attach MO metadata that has been reordered consistently with mo_sorted.

        Important:
            Sorting mo_coeff columns does NOT sort mf.mo_energy or mf.mo_occ.
            Any code that reads orbital energies/occupations from the sorted
            orbital basis should read mo_sorted.mo_energy and mo_sorted.mo_occ,
            not the original mf.mo_energy/mf.mo_occ.

        The original mf object is not modified here, because mf.mo_coeff still
        refers to the canonical SCF order.
        """
        mo_order = np.asarray(mo_order, dtype=int)

        tags = {
            "mo_order": mo_order,
        }

        if selected is not None:
            tags["active_orbital_indices"] = np.asarray(selected, dtype=int)

        if getattr(mf, "mo_energy", None) is not None:
            tags["mo_energy"] = np.asarray(mf.mo_energy)[mo_order].copy()
        else:
            tags["mo_energy"] = None

        if getattr(mf, "mo_occ", None) is not None:
            tags["mo_occ"] = np.asarray(mf.mo_occ)[mo_order].copy()
        else:
            tags["mo_occ"] = None

        return lib.tag_array(mo_sorted, **tags)

    @staticmethod
    def attach_sorted_mo_to_method(method, mo_sorted, copy_scf=True):
        """
        Store the sorted MO basis and its metadata on the CAS/GBCI object.

        This makes the following mutually consistent in the sorted CAS/GBCI
        orbital order:
            method.mo_coeff
            method.mo_energy
            method.mo_occ
            method._scf.mo_coeff
            method._scf.mo_energy
            method._scf.mo_occ

        By default, method._scf is shallow-copied before editing. This avoids
        mutating the original mf object outside this optimizer while still
        ensuring that downstream code reading method._scf.mo_energy sees the
        sorted orbital-energy order.
        """
        mo_energy = getattr(mo_sorted, "mo_energy", None)
        mo_occ = getattr(mo_sorted, "mo_occ", None)
        mo_order = getattr(mo_sorted, "mo_order", None)

        method.mo_coeff = mo_sorted
        method.mo_energy = mo_energy
        method.mo_occ = mo_occ
        method.mo_order = mo_order

        if getattr(method, "_scf", None) is not None:
            if copy_scf:
                try:
                    method._scf = method._scf.copy()
                except Exception:
                    import copy
                    method._scf = copy.copy(method._scf)

            method._scf.mo_coeff = mo_sorted
            method._scf.mo_energy = mo_energy
            method._scf.mo_occ = mo_occ
            method._scf.mo_order = mo_order

        return method


    # ------------------------------------------------------------
    # Active-space tracking
    # ------------------------------------------------------------
    def _reorder_active_orbitals(self, overlap_matrix: np.ndarray, selected: np.ndarray) -> np.ndarray:
        selected = np.asarray(selected, dtype=int)
        try:
            from scipy.optimize import linear_sum_assignment
            sub = np.abs(overlap_matrix[:, selected])
            row_ind, col_ind = linear_sum_assignment(-sub)
            order = np.argsort(row_ind)
            selected = selected[col_ind[order]]
        except Exception:
            selected = np.asarray(sorted(selected), dtype=int)
        return selected

    def select_active_by_previous_active_mo(self, mol_new, mo_new) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.prev_mol is None or self.prev_active_mo is None:
            raise RuntimeError("No previous active orbitals are available.")

        s_old_new = self.cross_ao_overlap(self.prev_mol, mol_new)
        a = self.prev_active_mo.T @ s_old_new @ mo_new
        score = np.sum(np.abs(a) ** 2, axis=0)

        nmo = mo_new.shape[1]
        candidate_idx = np.arange(nmo) if self.candidate_idx is None else self.candidate_idx
        selected = candidate_idx[np.argsort(score[candidate_idx])[-self.ncas:]]
        selected = self._reorder_active_orbitals(a, selected)

        a_sel = a[:, selected]
        try:
            svals = np.linalg.svd(a_sel, compute_uv=False)
        except np.linalg.LinAlgError:
            svals = np.full(self.ncas, np.nan)

        selected_scores = score[selected]
        mask = np.ones(nmo, dtype=bool)
        mask[selected] = False
        if self.candidate_idx is not None:
            candidate_mask = np.zeros(nmo, dtype=bool)
            candidate_mask[candidate_idx] = True
            mask = mask & candidate_mask

        best_unselected_score = float(np.max(score[mask])) if np.any(mask) else np.nan
        active_selection_gap = float(np.min(selected_scores) - best_unselected_score)

        diag = {
            "active_overlap_matrix": a_sel,
            "active_abs_overlap_matrix": np.abs(a_sel),
            "active_scores_all": score,
            "active_selected_scores": selected_scores,
            "active_singular_values": svals,
            "active_min_singular_value": float(np.nanmin(svals)),
            "active_selection_gap": active_selection_gap,
            "best_unselected_score": best_unselected_score,
        }
        return selected, diag

    def make_cas_mo(self, mc, mf) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        if self.step == 0 or not self.active_tracking:
            if self.act_list is None:
                selected = self.get_default_act_list(mc, mf)
                self.act_list = selected.copy()
            else:
                selected = np.asarray(self.act_list, dtype=int)

            active_diag = {
                "active_overlap_matrix": None,
                "active_abs_overlap_matrix": None,
                "active_scores_all": None,
                "active_selected_scores": None,
                "active_singular_values": None,
                "active_min_singular_value": None,
                "active_selection_gap": None,
                "best_unselected_score": None,
            }
        else:
            selected, active_diag = self.select_active_by_previous_active_mo(
                mol_new=mf.mol,
                mo_new=mf.mo_coeff,
            )
            self.act_list = selected.copy()

        ncore = int(mc.ncore)
        mo_cas, mo_order = self.sort_mo_by_active_list(
            method=mc,
            mf=mf,
            selected=selected,
            ncore=ncore,
        )

        self.last_mo_order = mo_order
        self.last_mo_energy = getattr(mo_cas, "mo_energy", None)
        self.last_mo_occ = getattr(mo_cas, "mo_occ", None)

        return mo_cas, selected, active_diag

    # ------------------------------------------------------------
    # Pair root tracking
    # ------------------------------------------------------------
    def choose_root_pair(self, mc) -> Tuple[Tuple[int, int], Dict[str, Any]]:
        e_tot = self.get_root_energies(mc)
        ci_list = self._as_ci_list(mc)
        nroots_actual = len(ci_list)

        if nroots_actual < 2:
            raise RuntimeError(
                f"{self.method_label} returned fewer than two roots; "
                "cannot optimize MECI."
            )
        if max(self.initial_target_roots) >= nroots_actual:
            raise RuntimeError(
                f"target_roots={self.initial_target_roots} but "
                f"{self.method_label} returned {nroots_actual} roots."
            )

        use_initial = (
            self.step == 0
            or self.prev_ci_targets is None
            or not self.root_tracking
        )

        if use_initial:
            tracked_roots = tuple(self.initial_target_roots)
            overlap_matrix = np.full((2, nroots_actual), np.nan)
            pair_score = np.nan
            pair_margin = np.nan
        else:
            overlap_matrix = np.array(
                [
                    [self._ci_overlap(prev_ci, ci) for ci in ci_list]
                    for prev_ci in self.prev_ci_targets
                ],
                dtype=float,
            )
            ov = np.nan_to_num(overlap_matrix, nan=-1.0e6)

            assignments = list(permutations(range(nroots_actual), 2))
            scores = np.array([ov[0, a[0]] + ov[1, a[1]] for a in assignments], dtype=float)
            order = np.argsort(scores)[::-1]
            best_idx = int(order[0])
            best_assignment = assignments[best_idx]
            best_score = float(scores[best_idx])
            second_score = float(scores[order[1]]) if len(order) > 1 else np.nan
            pair_margin = best_score - second_score if np.isfinite(second_score) else np.nan

            # If pair overlap assignment is ambiguous, use energy continuity as tie-breaker.
            if self.prev_e_targets is not None:
                near = [idx for idx in order if scores[idx] >= best_score - self.root_margin_warn]
                if len(near) > 1:
                    costs = []
                    for idx in near:
                        a = assignments[int(idx)]
                        costs.append(np.sum(np.abs(e_tot[list(a)] - self.prev_e_targets)))
                    best_assignment = assignments[int(near[int(np.argmin(costs))])]
                    best_score = float(scores[assignments.index(best_assignment)])

            tracked_roots = tuple(int(x) for x in best_assignment)
            pair_score = best_score

        if self.pair_order == "energy":
            roots = tuple(int(r) for r in sorted(tracked_roots, key=lambda r: e_tot[r]))
        else:
            roots = tracked_roots

        selected_overlaps = np.full(2, np.nan)
        if not np.all(np.isnan(overlap_matrix)):
            # In tracked-label order, not necessarily the final energy order.
            selected_overlaps = np.array(
                [overlap_matrix[0, tracked_roots[0]], overlap_matrix[1, tracked_roots[1]]],
                dtype=float,
            )

        root_diag = {
            "overlap_matrix": overlap_matrix,
            "tracked_roots_before_pair_order": tracked_roots,
            "selected_roots": roots,
            "selected_overlaps_tracked_order": selected_overlaps,
            "pair_score": pair_score,
            "pair_margin": pair_margin,
            "pair_order": self.pair_order,
        }
        return roots, root_diag

    # ------------------------------------------------------------
    # Optional diagnostics
    # ------------------------------------------------------------
    def get_spin_square(self, mc, ci):
        try:
            s2, mult = mc.fcisolver.spin_square(ci, mc.ncas, mc.nelecas)
            return float(s2), float(mult)
        except Exception:
            return np.nan, np.nan

    def get_active_natural_occupations(self, mc, ci):
        try:
            dm1 = mc.fcisolver.make_rdm1(ci, mc.ncas, mc.nelecas)
            occ = np.linalg.eigvalsh(dm1)
            return np.sort(occ)[::-1]
        except Exception:
            return None

    def print_diagnostics(self, mf, mc, selected_act, active_diag, root_diag, energies, gradients):
        if not self.verbose:
            return

        e_tot = self.get_root_energies(mc)
        ci_list = self._as_ci_list(mc)
        roots = root_diag["selected_roots"]

        print("\n" + "=" * 78)
        print(
            f"{self.method_label} active/root-pair tracking diagnostics: "
            f"scanner step {self.step}"
        )
        print("=" * 78)
        print(f"SCF converged   : {bool(mf.converged)}")
        print(
            f"{self.method_label} converged : "
            f"{bool(np.all(getattr(mc, 'converged', True)))}"
        )
        print(f"Selected roots  : {roots}  pair_order={root_diag['pair_order']}")
        print(f"Pair energies   : {energies[0]:.12f}, {energies[1]:.12f} Eh")
        print(f"Pair gap        : {(energies[1] - energies[0]) * HARTREE2EV:+.8f} eV")

        print("\nRoot energies:")
        for i, e in enumerate(e_tot):
            marker = " <== selected" if i in roots else ""
            print(f"  root {i:2d}: {e:.12f} Eh{marker}")

        print("\nActive orbital indices, 0-based:")
        print(" ", np.asarray(selected_act, dtype=int))
        print("Active orbital indices, 1-based:")
        print(" ", (np.asarray(selected_act, dtype=int) + 1))

        if active_diag.get("active_selected_scores") is not None:
            print("\nActive selected scores:")
            print(" ", np.array2string(active_diag["active_selected_scores"], precision=6))
            print("\n|<old active_i | new active_j>|:")
            print(np.array2string(active_diag["active_abs_overlap_matrix"], precision=4))
            print("\nActive-space singular values:")
            print(" ", np.array2string(active_diag["active_singular_values"], precision=6))
            print(f"Minimum active singular value : {active_diag['active_min_singular_value']:.6f}")
            print(f"Active selection gap          : {active_diag['active_selection_gap']:.6e}")
            if active_diag["active_min_singular_value"] < self.active_sv_warn:
                print("WARNING: Active space overlap is poor; active space may be discontinuous.")

        if root_diag.get("overlap_matrix") is not None:
            ov = root_diag["overlap_matrix"]
            if not np.all(np.isnan(ov)):
                print("\nCI overlap matrix, rows = previous selected states A/B, columns = current roots:")
                print(np.array2string(ov, precision=6))
                print(f"Tracked assignment before pair ordering: {root_diag['tracked_roots_before_pair_order']}")
                print(f"Selected overlaps in tracked order     : {root_diag['selected_overlaps_tracked_order']}")
                print(f"Root-pair score                         : {root_diag['pair_score']:.6f}")
                print(f"Root-pair margin                        : {root_diag['pair_margin']:.6f}")
                if np.nanmin(root_diag["selected_overlaps_tracked_order"]) < self.root_overlap_warn:
                    print("WARNING: One selected CI-vector overlap is small; root tracking may be unreliable.")
                if not np.isnan(root_diag["pair_margin"]) and root_diag["pair_margin"] < self.root_margin_warn:
                    print("WARNING: Root-pair assignment is ambiguous.")

        print("\nSpin diagnostics:")
        for i, ci in enumerate(ci_list):
            s2, mult = self.get_spin_square(mc, ci)
            print(f"  root {i:2d}: <S^2> = {s2:.6f}, mult = {mult:.6f}")

        print("\nNatural occupations for selected roots:")
        for r in roots:
            occ = self.get_active_natural_occupations(mc, ci_list[r])
            if occ is not None:
                print(f"  root {r:2d}: {np.array2string(occ, precision=6)}")

        print("\nGradient diagnostics for selected roots:")
        for k, r in enumerate(roots):
            gg = np.asarray(gradients[k])
            print(
                f"  root {r:2d}: norm={np.linalg.norm(gg):.6e}, "
                f"rms={np.sqrt(np.mean(gg**2)):.6e}, max={np.max(np.abs(gg)):.6e} Eh/Bohr"
            )
        print("=" * 78 + "\n")

    # ------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------
    def __call__(self, mol) -> PairResult:
        mf = self.make_mf(mol)
        mf.kernel()
        if not mf.converged:
            mf2 = scf.newton(mf)
            mf2.kernel()
            
            mf.mo_coeff = mf2.mo_coeff
            mf.mo_energy = mf2.mo_energy
            if not mf2.converged:
                print("WARNING: RHF did not converge.")

        mc = self.make_cas(mf)
        mo_cas, selected_act, active_diag = self.make_cas_mo(mc, mf)

        if self.reuse_ci_guess and self.prev_ci_list is not None:
            try:
                mc.kernel(mo_cas, ci0=self.prev_ci_list)
            except TypeError:
                mc.kernel(mo_cas)
        else:
            mc.kernel(mo_cas)

        casci_converged = bool(np.all(getattr(mc, "converged", True)))
        self.converged = bool(mf.converged) and casci_converged
        if not casci_converged:
            print("WARNING: CASCI did not converge.")

        roots, root_diag = self.choose_root_pair(mc)
        self.current_roots = roots

        e_tot = self.get_root_energies(mc)
        ci_list = self._as_ci_list(mc)
        energies = e_tot[list(roots)].astype(float)

        grad_method = mc.nuc_grad_method()
        gradients = []
        for r in roots:
            gradients.append(np.asarray(grad_method.kernel(mo_coeff = mc.mo_coeff, state=int(r)), dtype=float))
        gradients = np.stack(gradients, axis=0)

        self.print_diagnostics(
            mf=mf,
            mc=mc,
            selected_act=selected_act,
            active_diag=active_diag,
            root_diag=root_diag,
            energies=energies,
            gradients=gradients,
        )

        # Store previous-step data after root/active selection.
        self.prev_ci_targets = [np.array(ci_list[r], copy=True) for r in roots]
        self.prev_e_targets = energies.copy()
        self.prev_ci_list = [np.array(ci, copy=True) for ci in ci_list]
        self.prev_mol = mol.copy()
        self.prev_active_mo = mo_cas[:, mc.ncore:mc.ncore + mc.ncas].copy()
        self.prev_coords = mol.atom_coords().copy()
        self.act_list = np.asarray(selected_act, dtype=int).copy()

        result = PairResult(
            energies=energies,
            gradients=gradients,
            roots=roots,
            mol=mol,
            mf=mf,
            mc=mc,
            diagnostics={
                "active": active_diag,
                "root": root_diag,
                "selected_act": np.asarray(selected_act, dtype=int),
                "scanner_step": self.step,
            },
        )
        self.last_result = result
        self.step += 1
        return result


class CASSCFActiveRootTrackedPairScanner(CASCIActiveRootTrackedPairScanner):
    """
    Equal-weight SA-CASSCF scanner returning two state energies and gradients.

    A single state-averaged CASSCF wavefunction is optimized at each geometry.
    The selected state energies and analytic gradients are then taken from that
    same SA-CASSCF result and returned through the common :class:`PairResult`
    interface used by the MECI penalty scanners.

    Notes
    -----
    PySCF 2.10 supports analytic state-specific SA-CASSCF gradients only for
    equal state-average weights.  This class validates that restriction early
    so a geometry optimization cannot silently switch to an unsupported setup.
    """

    method_label = "SA-CASSCF"

    def __init__(
        self,
        ncas: int,
        nelecas: Any,
        target_roots: Sequence[int] = (0, 1),
        nroots: Optional[int] = None,
        state_weights: Optional[Sequence[float]] = None,
        act_list: Optional[Sequence[int]] = None,
        act_base: int = 0,
        candidate_idx: Optional[Sequence[int]] = None,
        root_tracking: bool = False,
        active_tracking: bool = True,
        pair_order: str = "energy",
        conv_tol: float = 1.0e-10,
        max_cycle: int = 100,
        mf_factory: Optional[Callable[[Any], Any]] = None,
        reuse_ci_guess: bool = True,
        project_previous_orbitals: bool = True,
        strict_convergence: bool = True,
        root_overlap_warn: float = 0.50,
        root_margin_warn: float = 0.10,
        active_sv_warn: float = 0.70,
        verbose: bool = True,
    ):
        resolved_nroots = (
            int(nroots)
            if nroots is not None
            else max(int(root) for root in target_roots) + 1
        )
        if state_weights is None:
            weights = np.full(resolved_nroots, 1.0 / resolved_nroots)
        else:
            weights = np.asarray(state_weights, dtype=float)

        if weights.ndim != 1 or len(weights) != resolved_nroots:
            raise ValueError(
                "state_weights must be a one-dimensional sequence with "
                f"length nroots={resolved_nroots}."
            )
        if np.any(weights < 0.0):
            raise ValueError("state_weights must be non-negative.")
        if abs(float(np.sum(weights)) - 1.0) > 1.0e-10:
            raise ValueError(
                "state_weights must sum to 1.0; "
                f"got {float(np.sum(weights)):.12f}."
            )
        if float(np.max(weights) - np.min(weights)) > 1.0e-8:
            raise NotImplementedError(
                "Analytic SA-CASSCF state gradients require equal state "
                f"weights in this PySCF build; got {weights.tolist()}."
            )

        super().__init__(
            ncas=ncas,
            nelecas=nelecas,
            target_roots=target_roots,
            nroots=resolved_nroots,
            act_list=act_list,
            act_base=act_base,
            candidate_idx=candidate_idx,
            root_tracking=root_tracking,
            active_tracking=active_tracking,
            pair_order=pair_order,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
            mf_factory=mf_factory,
            mc_factory=None,
            reuse_ci_guess=reuse_ci_guess,
            root_overlap_warn=root_overlap_warn,
            root_margin_warn=root_margin_warn,
            active_sv_warn=active_sv_warn,
            verbose=verbose,
        )

        self.state_weights = weights.tolist()
        self.project_previous_orbitals = bool(project_previous_orbitals)
        self.strict_convergence = bool(strict_convergence)

        self.prev_casscf_mo = None
        self.last_mf = None
        self.last_mc = None
        self.last_selected_act = None
        self.last_active_diag = None
        self.last_root_diag = None
        self.history: List[Dict[str, Any]] = []

    def make_mf(self, mol):
        if self.mf_factory is not None:
            mf = self.mf_factory(mol)
        else:
            mf = scf.RHF(mol)
        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.damp = 0.4
        mf.level_shift = 0.5
        return mf

    def make_cas(self, mf):
        mc = mcscf.CASSCF(mf, self.ncas, self.nelecas)
        mc.conv_tol = self.conv_tol
        mc.max_cycle = self.max_cycle
        mc.fcisolver.nroots = self.nroots
        mc.fix_spin_(ss=0.0, shift=0.5)
        mc.state_average_(self.state_weights)
        return mc

    @staticmethod
    def _empty_active_diagnostics() -> Dict[str, Any]:
        return {
            "active_overlap_matrix": None,
            "active_abs_overlap_matrix": None,
            "active_scores_all": None,
            "active_selected_scores": None,
            "active_singular_values": None,
            "active_min_singular_value": None,
            "active_selection_gap": None,
            "best_unselected_score": None,
        }

    def make_cas_mo(self, mc, mf):
        """
        Build the orbital guess, projecting the previous optimized CASSCF MOs.

        At the first geometry, the requested RHF orbitals are sorted into the
        active block.  Later geometries use PySCF's CASSCF projection helper so
        that the complete optimized orbital set, particularly the active
        subspace, remains continuous.
        """
        if (
            not self.project_previous_orbitals
            or self.prev_casscf_mo is None
            or self.prev_mol is None
        ):
            return super().make_cas_mo(mc, mf)

        if self.active_tracking:
            selected, active_diag = self.select_active_by_previous_active_mo(
                mol_new=mf.mol,
                mo_new=mf.mo_coeff,
            )
        else:
            selected = np.asarray(self.act_list, dtype=int)
            active_diag = self._empty_active_diagnostics()

        mo_cas = mcscf_addons.project_init_guess(
            mc,
            self.prev_casscf_mo,
            prev_mol=self.prev_mol,
            priority="active",
            use_hf_core=False,
        )

        ncore = int(mc.ncore)
        active_projected = mo_cas[:, ncore:ncore + mc.ncas]
        active_cross_overlap = self.cross_mo_overlap(
            self.prev_mol,
            self.prev_active_mo,
            mf.mol,
            active_projected,
        )
        try:
            projected_svals = np.linalg.svd(
                active_cross_overlap,
                compute_uv=False,
            )
        except np.linalg.LinAlgError:
            projected_svals = np.full(self.ncas, np.nan)

        active_diag["projected_active_overlap_matrix"] = active_cross_overlap
        active_diag["projected_active_singular_values"] = projected_svals
        active_diag["projected_active_min_singular_value"] = float(
            np.nanmin(projected_svals)
        )

        self.last_mo_order = None
        self.last_mo_energy = None
        self.last_mo_occ = None
        return mo_cas, np.asarray(selected, dtype=int), active_diag

    def run_casscf(self, mc, mo_cas) -> None:
        ci0 = None
        if self.reuse_ci_guess and self.prev_ci_list is not None:
            if len(self.prev_ci_list) == self.nroots:
                ci0 = [
                    np.array(ci_vector, copy=True)
                    for ci_vector in self.prev_ci_list
                ]

        if ci0 is None:
            mc.kernel(mo_cas)
        else:
            mc.kernel(mo_cas, ci0=ci0)

    @staticmethod
    def _as_reference_ci_list(ci, nroots: int) -> List[np.ndarray]:
        if isinstance(ci, (list, tuple)):
            ci_list = list(ci)
        else:
            ci_array = np.asarray(ci)
            if nroots == 1:
                ci_list = [ci_array]
            elif ci_array.ndim >= 1 and ci_array.shape[0] == nroots:
                ci_list = [ci_array[index] for index in range(nroots)]
            else:
                raise ValueError(
                    "A multi-root reference CI array must have its root "
                    "dimension first."
                )

        if len(ci_list) != nroots:
            raise ValueError(
                f"Reference CI has {len(ci_list)} roots; expected {nroots}."
            )
        return [np.array(ci_vector, copy=True) for ci_vector in ci_list]

    def set_reference(self, mol, mo_coeff, ci, e_states=None):
        """
        Seed an independent evaluation from a converged SA-CASSCF reference.

        This is primarily used to give every numerical-Hessian displacement a
        fresh scanner initialized from the same final MECI wavefunction.
        """
        ci_list = self._as_reference_ci_list(ci, self.nroots)
        mo_coeff = np.asarray(mo_coeff)

        ncore_float = 0.5 * (mol.nelectron - self.nelecas_total())
        if abs(ncore_float - round(ncore_float)) > 1.0e-10:
            raise ValueError(
                "Cannot determine the closed-shell CASSCF core size for "
                "the supplied reference."
            )
        ncore = int(round(ncore_float))

        self.prev_mol = mol.copy()
        self.prev_casscf_mo = np.array(mo_coeff, copy=True)
        self.prev_active_mo = np.array(
            mo_coeff[:, ncore:ncore + self.ncas],
            copy=True,
        )
        self.prev_ci_list = ci_list
        self.prev_ci_targets = [
            np.array(ci_list[root], copy=True)
            for root in self.initial_target_roots
        ]
        self.prev_coords = mol.atom_coords(unit="Bohr").copy()
        self.current_roots = self.initial_target_roots

        if e_states is None:
            self.prev_e_targets = None
        else:
            energies = np.atleast_1d(np.asarray(e_states, dtype=float))
            if len(energies) != self.nroots:
                raise ValueError(
                    f"Reference has {len(energies)} state energies; "
                    f"expected {self.nroots}."
                )
            self.prev_e_targets = energies[
                list(self.initial_target_roots)
            ].copy()
        return self

    def get_reference(self) -> Dict[str, Any]:
        if self.last_mc is None:
            raise RuntimeError("No converged SA-CASSCF result is available.")

        return {
            "mol": self.last_mc.mol.copy(),
            "mo_coeff": np.array(self.last_mc.mo_coeff, copy=True),
            "ci": [
                np.array(ci_vector, copy=True)
                for ci_vector in self._as_ci_list(self.last_mc)
            ],
            "e_states": np.array(
                self.get_root_energies(self.last_mc),
                copy=True,
            ),
            "selected_act_0based": np.array(
                self.last_selected_act,
                dtype=int,
                copy=True,
            ),
        }

    def compute_gradients(self, mc, roots) -> np.ndarray:
        gradients = []
        for root in roots:
            try:
                gradient = mc.nuc_grad_method(
                    state=int(root)
                ).kernel()
            except TypeError:
                gradient = mc.nuc_grad_method().kernel(state=int(root))
            gradients.append(np.asarray(gradient, dtype=float))
        return np.stack(gradients, axis=0)

    def __call__(self, mol) -> PairResult:
        mf = self.make_mf(mol)
        mf.kernel()
        if not mf.converged:
            message = "RHF did not converge for the requested SA-CASSCF geometry."
            if self.strict_convergence:
                raise RuntimeError(message)
            print(f"WARNING: {message}")

        mc = self.make_cas(mf)
        mo_cas, selected_act, active_diag = self.make_cas_mo(mc, mf)
        self.run_casscf(mc, mo_cas)

        casscf_converged = bool(
            np.all(getattr(mc, "converged", True))
        )
        self.converged = bool(mf.converged) and casscf_converged
        if not casscf_converged:
            message = "SA-CASSCF did not converge for the requested geometry."
            if self.strict_convergence:
                raise RuntimeError(message)
            print(f"WARNING: {message}")

        roots, root_diag = self.choose_root_pair(mc)
        self.current_roots = roots

        all_energies = self.get_root_energies(mc)
        ci_list = self._as_ci_list(mc)
        energies = all_energies[list(roots)].astype(float)
        gradients = self.compute_gradients(mc, roots)

        self.print_diagnostics(
            mf=mf,
            mc=mc,
            selected_act=selected_act,
            active_diag=active_diag,
            root_diag=root_diag,
            energies=energies,
            gradients=gradients,
        )

        self.prev_ci_targets = [
            np.array(ci_list[root], copy=True)
            for root in roots
        ]
        self.prev_e_targets = energies.copy()
        self.prev_ci_list = [
            np.array(ci_vector, copy=True)
            for ci_vector in ci_list
        ]
        self.prev_mol = mol.copy()
        self.prev_casscf_mo = np.array(mc.mo_coeff, copy=True)
        self.prev_active_mo = np.array(
            mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas],
            copy=True,
        )
        self.prev_coords = mol.atom_coords(unit="Bohr").copy()
        self.act_list = np.asarray(selected_act, dtype=int).copy()

        result = PairResult(
            energies=energies,
            gradients=gradients,
            roots=roots,
            mol=mol,
            mf=mf,
            mc=mc,
            diagnostics={
                "active": active_diag,
                "root": root_diag,
                "selected_act": np.asarray(selected_act, dtype=int),
                "scanner_step": self.step,
                "state_weights": list(self.state_weights),
            },
        )
        self.last_result = result
        self.last_mf = mf
        self.last_mc = mc
        self.last_selected_act = np.asarray(
            selected_act,
            dtype=int,
        ).copy()
        self.last_active_diag = active_diag
        self.last_root_diag = root_diag

        self.history.append(
            {
                "step": int(self.step),
                "roots": tuple(int(root) for root in roots),
                "energies_hartree": energies.copy(),
                "gap_hartree": float(energies[1] - energies[0]),
                "gap_ev": float(
                    (energies[1] - energies[0]) * HARTREE2EV
                ),
                "selected_act_0based": np.asarray(
                    selected_act,
                    dtype=int,
                ).copy(),
                "scf_converged": bool(mf.converged),
                "casscf_converged": casscf_converged,
                "active_min_singular_value": active_diag.get(
                    "active_min_singular_value"
                ),
                "projected_active_min_singular_value": active_diag.get(
                    "projected_active_min_singular_value"
                ),
            }
        )

        if self.verbose:
            print(
                f"SA-CASSCF pair scanner step {self.step}: "
                f"roots={roots}, "
                f"gap={(energies[1] - energies[0]) * HARTREE2EV:+.8f} eV"
            )

        self.step += 1
        return result


class GBCIActiveRootTrackedPairScanner(CASCIActiveRootTrackedPairScanner):
    """
    GBCI scanner returning two tracked roots and their nuclear gradients.

    This class is the GBCI analogue of CASCIActiveRootTrackedPairScanner.
    It reuses the parent class for
        - RHF construction,
        - active-space tracking,
        - two-root CI-overlap assignment,
        - diagnostics and PairResult construction logic,
    and overrides only the GBCI-specific pieces:
        - make_cas(): construct GBCI,
        - _as_ci_list(): accept GBCI CI-vector layouts,
        - get_ncore()/make_cas_mo(): avoid assuming a PySCF CASCI object,
        - run_gbci(): explicit GBCI Hamiltonian/overlap construction,
        - compute_gbci_gradients(): call your GBCI gradient API.

    Notes
    -----
    If groupA is None, this follows your existing convention and constructs
    GBCI(mf, ncas, nelecas, **gbci_kwargs).  If groupA is not None, it constructs
    GBCI(mf, ncas, nelecas, group_a=groupA, **gbci_kwargs).
    """

    def __init__(
        self,
        ncas: int,
        nelecas: Any,
        target_roots: Sequence[int] = (0, 1),
        nroots: Optional[int] = None,
        groupA=None,
        gbci_kwargs: Optional[Dict[str, Any]] = None,
        act_list: Optional[Sequence[int]] = None,
        act_base: int = 0,
        candidate_idx: Optional[Sequence[int]] = None,
        root_tracking: bool = True,
        active_tracking: bool = True,
        pair_order: str = "energy",
        conv_tol: float = 1.0e-10,
        max_cycle: int = 100,
        mf_factory=None,
        reuse_ci_guess: bool = False,
        root_overlap_warn: float = 0.50,
        root_margin_warn: float = 0.10,
        active_sv_warn: float = 0.70,
        verbose: bool = True,
    ):
        super().__init__(
            ncas=ncas,
            nelecas=nelecas,
            target_roots=target_roots,
            nroots=nroots,
            act_list=act_list,
            act_base=act_base,
            candidate_idx=candidate_idx,
            root_tracking=root_tracking,
            active_tracking=active_tracking,
            pair_order=pair_order,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
            mf_factory=mf_factory,
            mc_factory=None,
            reuse_ci_guess=reuse_ci_guess,
            root_overlap_warn=root_overlap_warn,
            root_margin_warn=root_margin_warn,
            active_sv_warn=active_sv_warn,
            verbose=verbose,
        )

        self.groupA = groupA
        self.gbci_kwargs = {} if gbci_kwargs is None else dict(gbci_kwargs)

        self.last_mf = None
        self.last_gbci = None
        self.last_mo_gbci = None
        self.last_selected_act = None
        self.last_active_diag = None
        self.last_root_diag = None
        self.last_gbci_intermediates = None

    # ------------------------------------------------------------
    # GBCI builder
    # ------------------------------------------------------------
    def make_cas(self, mf):
        gbci = make_gbci(
            mf,
            self.ncas,
            self.nelecas,
            group_a=self.groupA,
            **self.gbci_kwargs,
        )

        for obj in (gbci, getattr(gbci, "fcisolver", None)):
            if obj is None:
                continue
            if hasattr(obj, "conv_tol"):
                obj.conv_tol = self.conv_tol
            if hasattr(obj, "max_cycle"):
                obj.max_cycle = self.max_cycle
            if hasattr(obj, "nroots"):
                obj.nroots = self.nroots
            if hasattr(obj, "nroot"):
                obj.nroot = self.nroots

        try:
            gbci.fix_spin_(ss=0.0, shift=0.5)
        except TypeError:
            try:
                gbci.fix_spin_(ss=0.0)
            except Exception:
                pass
        except Exception:
            pass

        return gbci

    # ------------------------------------------------------------
    # CI-vector layout helper
    # ------------------------------------------------------------
    def _as_ci_list(self, mc):
        ci = getattr(mc, "ci", None)
        if ci is None:
            return []

        if isinstance(ci, (list, tuple)):
            return list(ci)

        arr = np.asarray(ci)
        e_tot = np.atleast_1d(np.asarray(getattr(mc, "e_tot", []), dtype=float))
        nroots_actual = len(e_tot) if len(e_tot) > 0 else self.nroots

        # GBCI code above uses mc.ci[root], so root-first is the preferred layout.
        if arr.ndim >= 2 and arr.shape[0] == nroots_actual:
            return [np.array(arr[i, ...], copy=True) for i in range(nroots_actual)]

        # Alternative layout: columns are roots.
        if arr.ndim >= 2 and arr.shape[-1] == nroots_actual:
            return [np.array(arr[..., i], copy=True) for i in range(nroots_actual)]

        # Fallback: single-root vector.
        return [arr]

    # ------------------------------------------------------------
    # MO sorting / ncore handling
    # ------------------------------------------------------------
    def get_ncore(self, mc, mf) -> int:
        if getattr(mc, "ncore", None) is not None:
            return int(mc.ncore)

        nelecas_total = self.nelecas_total()
        ncore_float = 0.5 * (mf.mol.nelectron - nelecas_total)
        if abs(ncore_float - round(ncore_float)) > 1.0e-10:
            raise ValueError(
                "Cannot determine closed-shell ncore from "
                f"mol.nelectron={mf.mol.nelectron}, nelecas={self.nelecas}. "
                "For open-shell or odd-electron cases, set ncore in the GBCI object "
                "or give a custom active-orbital selection."
            )
        return int(round(ncore_float))

    def make_cas_mo(self, mc, mf):
        if self.step == 0 or not self.active_tracking:
            if self.act_list is None:
                selected = self.get_default_act_list(mc, mf)
                self.act_list = selected.copy()
                if self.verbose:
                    print("\nInitial active space was not provided.")
                    print("Automatically selected active orbitals from ncas/nelecas:")
                    print(f"  ncas    = {self.ncas}")
                    print(f"  nelecas = {self.nelecas}")
                    print(f"  ncore   = {selected[0]}")
                    print(f"  act_list, 0-based = {selected.tolist()}")
                    print(f"  act_list, 1-based = {(selected + 1).tolist()}")
            else:
                selected = np.asarray(self.act_list, dtype=int)
                if self.verbose:
                    print("\nInitial active space was provided manually.")
                    print(f"  act_list, 0-based = {selected.tolist()}")
                    print(f"  act_list, 1-based = {(selected + 1).tolist()}")

            active_diag = {
                "active_overlap_matrix": None,
                "active_abs_overlap_matrix": None,
                "active_scores_all": None,
                "active_selected_scores": None,
                "active_singular_values": None,
                "active_min_singular_value": None,
                "active_selection_gap": None,
                "best_unselected_score": None,
            }
        else:
            selected, active_diag = self.select_active_by_previous_active_mo(
                mol_new=mf.mol,
                mo_new=mf.mo_coeff,
            )
            self.act_list = selected.copy()

        ncore = self.get_ncore(mc, mf)
        mo_cas, mo_order = self.sort_mo_by_active_list(
            method=mc,
            mf=mf,
            selected=selected,
            ncore=ncore,
        )

        self.last_mo_order = mo_order
        self.last_mo_energy = getattr(mo_cas, "mo_energy", None)
        self.last_mo_occ = getattr(mo_cas, "mo_occ", None)

        return mo_cas, selected, active_diag

    # ------------------------------------------------------------
    # GBCI kernel and gradient hooks
    # ------------------------------------------------------------
    def _get_ci0_for_gbci(self):
        if not self.reuse_ci_guess:
            return None

        # Prefer all current roots as a guess if available.  If the underlying
        # solver does not accept this shape, run_gbci falls back to no ci0.
        if self.prev_ci_list is not None:
            return self.prev_ci_list
        if self.prev_ci_targets is not None:
            return self.prev_ci_targets
        return None

    def run_gbci(self, gbci, mo_cas):
        mol = gbci.mol

        ncas = gbci.ncas
        nelecas = gbci.nelecas

        mo_list, moe_list, po_list, group = gbci.optimize_mo(mo_cas)

        # Mirror pyscf.gbci.gbci.kernel: the ungrouped path keys the CI
        # configurations off po_list, the grouped path off group.
        if gbci.group_a is None:
            conf_info_list = group_info_list(ncas, nelecas, po_list)
            svd_basis = po_list
        else:
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
            svd_basis = group

        dmet_core_list, ov_list = gbci.get_svd_matrices(mo_list, svd_basis)
        dmet_act_list = gbci.get_active_dm(mo_cas)
        h1e, ecore_list = gbci.get_h1cas(dmet_act_list, mo_list, dmet_core_list)
        eri = gbci.get_h2eff(mo_cas)

        # pyscf.grad.gbci reads these off the GBCI object.  We drive the
        # fcisolver directly (to inject a tracked ci0), so publish them here the
        # same way pyscf.gbci.gbci.kernel does.
        gbci._cache_gbci_intermediates(
            mo_cas,
            ncas,
            nelecas,
            gbci.ncore,
            {
                "mo_list": mo_list,
                "mo_energy": moe_list,
                "po_list": po_list,
                "group": group,
                "svd_basis": svd_basis,
                "conf_info_list": conf_info_list,
                "dmet_core_list": dmet_core_list,
                "ov_list": ov_list,
                "ecore_list": ecore_list,
            },
        )

        ci0 = self._get_ci0_for_gbci()
        try:
            if ci0 is None:
                e_tot, gbci.ci = gbci.fcisolver.kernel(
                    h1e,
                    eri,
                    ncas,
                    nelecas,
                    conf_info_list,
                    ov_list,
                    ecore_list,
                    verbose=mol.verbose,
                )
            else:
                e_tot, gbci.ci = gbci.fcisolver.kernel(
                    h1e,
                    eri,
                    ncas,
                    nelecas,
                    conf_info_list,
                    ov_list,
                    ecore_list,
                    ci0=ci0,
                    verbose=mol.verbose,
                )
        except TypeError:
            # Some solver versions do not accept ci0.
            e_tot, gbci.ci = gbci.fcisolver.kernel(
                h1e,
                eri,
                ncas,
                nelecas,
                conf_info_list,
                ov_list,
                ecore_list,
                verbose=mol.verbose,
            )

        gbci.e_tot = np.atleast_1d(np.asarray(e_tot, dtype=float))
        self.attach_sorted_mo_to_method(gbci, mo_cas)

        if getattr(gbci.fcisolver, "converged", None) is not None:
            gbci.converged = bool(np.all(gbci.fcisolver.converged))
        else:
            gbci.converged = True

        intermediates = {
            "mo_list": mo_list,
            "moe_list": moe_list,
            "po_list": po_list,
            "group": group,
            "conf_info_list": conf_info_list,
            "dmet_core_list": dmet_core_list,
            "ov_list": ov_list,
            "ecore_list": ecore_list,
        }
        self.last_gbci_intermediates = intermediates

        return e_tot, intermediates

    def compute_gbci_gradients(self, gbci, mo_cas, roots, intermediates):
        grad_method = gbci.nuc_grad_method()
        ci_list = self._as_ci_list(gbci)

        mo_list = intermediates["mo_list"]
        moe_list = intermediates["moe_list"]
        conf_info_list = intermediates["conf_info_list"]
        dmet_core_list = intermediates["dmet_core_list"]
        ov_list = intermediates["ov_list"]
        ecore_list = intermediates["ecore_list"]

        gradients = []
        for r in roots:
            r = int(r)
            ci_r = ci_list[r]

            try:
                grad = grad_method.kernel(
                    mo_cas,
                    gbci._scf.mo_energy,
                    mo_list,
                    moe_list,
                    conf_info_list,
                    dmet_core_list,
                    ov_list,
                    ecore_list,
                    ci_r,
                )
            except TypeError:
                # Fallbacks for PySCF-like gradient APIs.
                try:
                    grad = grad_method.kernel(mo_coeff=mo_cas, state=r)
                except TypeError:
                    try:
                        grad = grad_method.kernel(state=r)
                    except TypeError:
                        grad = grad_method.kernel(r)

            gradients.append(np.asarray(grad, dtype=float))

        return np.stack(gradients, axis=0)

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    def print_diagnostics(self, mf, mc, selected_act, active_diag, root_diag, energies, gradients):
        if not self.verbose:
            return

        e_tot = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))
        roots = root_diag["selected_roots"]

        print("\n" + "=" * 78)
        print(f"GBCI active/root-pair tracking diagnostics: scanner step {self.step}")
        print("=" * 78)
        print(f"SCF converged  : {bool(mf.converged)}")
        print(f"GBCI converged : {bool(np.all(getattr(mc, 'converged', True)))}")
        print(f"Selected roots : {roots}  pair_order={root_diag['pair_order']}")
        print(f"Pair energies  : {energies[0]:.12f}, {energies[1]:.12f} Eh")
        print(f"Pair gap       : {(energies[1] - energies[0]) * HARTREE2EV:+.8f} eV")

        print("\nRoot energies:")
        for i, e in enumerate(e_tot):
            marker = " <== selected" if i in roots else ""
            print(f"  root {i:2d}: {e:.12f} Eh{marker}")

        print("\nActive orbital indices, 0-based:")
        print(" ", np.asarray(selected_act, dtype=int))
        print("Active orbital indices, 1-based:")
        print(" ", np.asarray(selected_act, dtype=int) + 1)

        if active_diag.get("active_selected_scores") is not None:
            print("\nActive selected scores:")
            print(" ", np.array2string(active_diag["active_selected_scores"], precision=6))
            print("\n|<old active_i | new active_j>|:")
            print(np.array2string(active_diag["active_abs_overlap_matrix"], precision=4))
            print("\nActive-space singular values:")
            print(" ", np.array2string(active_diag["active_singular_values"], precision=6))
            print(f"Minimum active singular value : {active_diag['active_min_singular_value']:.6f}")
            print(f"Active selection gap          : {active_diag['active_selection_gap']:.6e}")
            if active_diag["active_min_singular_value"] < self.active_sv_warn:
                print("WARNING: Active space overlap is poor; active space may be discontinuous.")

        ov = root_diag.get("overlap_matrix")
        if ov is not None and not np.all(np.isnan(ov)):
            print("\nCI overlap matrix, rows = previous selected states A/B, columns = current roots:")
            print(np.array2string(ov, precision=6))
            print(f"Tracked assignment before pair ordering: {root_diag['tracked_roots_before_pair_order']}")
            print(f"Selected overlaps in tracked order     : {root_diag['selected_overlaps_tracked_order']}")
            print(f"Root-pair score                         : {root_diag['pair_score']:.6f}")
            print(f"Root-pair margin                        : {root_diag['pair_margin']:.6f}")
            if np.nanmin(root_diag["selected_overlaps_tracked_order"]) < self.root_overlap_warn:
                print("WARNING: One selected CI-vector overlap is small; root tracking may be unreliable.")
            if not np.isnan(root_diag["pair_margin"]) and root_diag["pair_margin"] < self.root_margin_warn:
                print("WARNING: Root-pair assignment is ambiguous.")

        print("\nGradient diagnostics for selected roots:")
        for k, r in enumerate(roots):
            gg = np.asarray(gradients[k])
            print(
                f"  root {r:2d}: norm={np.linalg.norm(gg):.6e}, "
                f"rms={np.sqrt(np.mean(gg**2)):.6e}, "
                f"max={np.max(np.abs(gg)):.6e} Eh/Bohr"
            )
        print("=" * 78 + "\n")

    # ------------------------------------------------------------
    # Main scanner call
    # ------------------------------------------------------------
    def __call__(self, mol):
        mf = self.make_mf(mol)
        mf.kernel()
        if not mf.converged:
            print("WARNING: RHF did not converge.")

        gbci = self.make_cas(mf)
        mo_cas, selected_act, active_diag = self.make_cas_mo(gbci, mf)

        e_tot, intermediates = self.run_gbci(gbci, mo_cas)

        gbci_converged = bool(np.all(getattr(gbci, "converged", True)))
        self.converged = bool(mf.converged) and gbci_converged
        if not gbci_converged:
            print("WARNING: GBCI did not converge.")

        roots, root_diag = self.choose_root_pair(gbci)
        self.current_roots = roots

        e_tot = np.atleast_1d(np.asarray(gbci.e_tot, dtype=float))
        ci_list = self._as_ci_list(gbci)
        energies = e_tot[list(roots)].astype(float)
        gradients = self.compute_gbci_gradients(gbci, mo_cas, roots, intermediates)

        self.print_diagnostics(
            mf=mf,
            mc=gbci,
            selected_act=selected_act,
            active_diag=active_diag,
            root_diag=root_diag,
            energies=energies,
            gradients=gradients,
        )

        # Store previous-step data after root/active selection.
        self.prev_ci_targets = [np.array(ci_list[r], copy=True) for r in roots]
        self.prev_e_targets = energies.copy()
        self.prev_ci_list = [np.array(ci, copy=True) for ci in ci_list]
        self.prev_mol = mol.copy()

        ncore = self.get_ncore(gbci, mf)
        self.prev_active_mo = mo_cas[:, ncore:ncore + self.ncas].copy()
        self.prev_coords = mol.atom_coords().copy()
        self.act_list = np.asarray(selected_act, dtype=int).copy()

        result = PairResult(
            energies=energies,
            gradients=gradients,
            roots=roots,
            mol=mol,
            mf=mf,
            mc=gbci,
            diagnostics={
                "active": active_diag,
                "root": root_diag,
                "selected_act": np.asarray(selected_act, dtype=int),
                "scanner_step": self.step,
                "gbci_intermediates": intermediates,
            },
        )
        self.last_result = result

        self.last_mf = mf
        self.last_gbci = gbci
        self.last_mo_gbci = mo_cas
        self.last_selected_act = np.asarray(selected_act, dtype=int).copy()
        self.last_active_diag = active_diag
        self.last_root_diag = root_diag

        print(
            f"GBCI pair scanner step {self.step}: roots={roots}, "
            f"gap={(energies[1] - energies[0]) * HARTREE2EV:+.8f} eV"
        )

        self.step += 1
        return result


class PenaltyFunctionMECIScanner:
    """
    Turn a two-state PairResult scanner into a scalar penalty-function scanner.

    Given a pair scanner returning
        E1, E2, G1, G2,
    this scanner returns
        E_obj = 0.5 * (E1 + E2) + sigma * gap^2 / (gap^2 + alpha)
        G_obj = 0.5 * (G1 + G2) + dE_penalty/dR

    where
        gap = E2 - E1,
        dE_penalty/dR = [2*sigma*alpha*gap/(gap^2+alpha)^2] * (G2 - G1).

    Units
    -----
    E1/E2/sigma are Hartree, gradients are Eh/Bohr, and alpha is Eh^2.
    """

    def __init__(
        self,
        pair_scanner,
        sigma: float = 0.02,
        alpha: float = 1.0e-4,
        verbose: bool = True,
        print_level: int = 1,
    ):
        if alpha <= 0.0:
            raise ValueError("alpha must be positive. It has units of Eh^2.")

        self.pair_scanner = pair_scanner
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.verbose = bool(verbose)
        self.print_level = int(print_level)

        self.step = 0
        self.converged = False
        self.mol = None
        self.e_tot = None
        self.de = None

        self.last_pair_result = None
        self.last_info: Dict[str, Any] = {}
        self.history = []

    @staticmethod
    def _gradient_stats(grad):
        grad = np.asarray(grad)
        return {
            "norm": float(np.linalg.norm(grad)),
            "rms": float(np.sqrt(np.mean(grad**2))),
            "max": float(np.max(np.abs(grad))),
        }

    def objective_from_pair_result(self, pair_result):
        energies = np.asarray(pair_result.energies, dtype=float)
        gradients = np.asarray(pair_result.gradients, dtype=float)

        if energies.shape != (2,):
            raise ValueError(f"pair_result.energies must have shape (2,), got {energies.shape}.")
        if gradients.shape[0] != 2:
            raise ValueError(
                "pair_result.gradients must have first dimension 2, "
                f"got shape {gradients.shape}."
            )

        e1, e2 = energies
        g1, g2 = gradients

        gap = float(e2 - e1)
        gap2 = gap * gap
        denom = gap2 + self.alpha

        eavg = 0.5 * (e1 + e2)
        epen = self.sigma * gap2 / denom
        eobj = eavg + epen

        coeff = 2.0 * self.sigma * self.alpha * gap / (denom * denom)
        gavg = 0.5 * (g1 + g2)
        gpen = coeff * (g2 - g1)
        gobj = gavg + gpen

        info = {
            "e1": float(e1),
            "e2": float(e2),
            "eavg": float(eavg),
            "gap_Eh": float(gap),
            "gap_eV": float(gap * HARTREE2EV),
            "epen": float(epen),
            "eobj": float(eobj),
            "sigma": self.sigma,
            "alpha": self.alpha,
            "penalty_grad_coeff": float(coeff),
            "gavg_stats": self._gradient_stats(gavg),
            "gpen_stats": self._gradient_stats(gpen),
            "gobj_stats": self._gradient_stats(gobj),
            "roots": tuple(pair_result.roots),
        }
        return float(eobj), np.asarray(gobj, dtype=float), info

    def __call__(self, mol):
        pair_result = self.pair_scanner(mol)
        eobj, gobj, info = self.objective_from_pair_result(pair_result)

        self.converged = bool(getattr(self.pair_scanner, "converged", True))
        self.mol = mol
        self.e_tot = eobj
        self.de = gobj
        self.last_pair_result = pair_result
        self.last_info = info
        self.history.append(dict(info, penalty_step=self.step))

        if self.verbose:
            print("\n" + "-" * 78)
            print(f"Penalty MECI scanner step {self.step}")
            print(f"Roots       : {info['roots']}")
            print(f"E1, E2 / Eh : {info['e1']:.12f}, {info['e2']:.12f}")
            print(f"Eavg / Eh   : {info['eavg']:.12f}")
            print(f"Gap / eV    : {info['gap_eV']:+.8f}")
            print(f"Epen / Eh   : {info['epen']:.12e}")
            print(f"Eobj / Eh   : {info['eobj']:.12f}")
            print(
                "|Gobj| / Eh Bohr^-1: "
                f"norm={info['gobj_stats']['norm']:.6e}, "
                f"rms={info['gobj_stats']['rms']:.6e}, "
                f"max={info['gobj_stats']['max']:.6e}"
            )
            if self.print_level >= 2:
                print(
                    "|Gpen| / Eh Bohr^-1: "
                    f"norm={info['gpen_stats']['norm']:.6e}, "
                    f"rms={info['gpen_stats']['rms']:.6e}, "
                    f"max={info['gpen_stats']['max']:.6e}"
                )
                print(f"Penalty gradient coefficient: {info['penalty_grad_coeff']:+.6e}")
            print("-" * 78 + "\n")

        self.step += 1
        return eobj, gobj

    def as_scanner(self):
        return self


class UBSGradientProjectionMECI:
    """
    Updated Branching Space / Gradient Projection MECI optimizer.

    The pair_scanner must return PairResult with two energies and two gradients.
    The optimizer itself does not know about CASCI, root tracking, or active tracking;
    those are handled inside the scanner above.
    """

    def __init__(
        self,
        pair_scanner: Callable[[Any], PairResult],
        max_steps: int = 100,
        max_step_bohr: float = 0.05,
        gap_tol_ev: float = 1.0e-3,
        max_pg_tol: float = 3.0e-4,
        rms_pg_tol: float = 2.0e-4,
        max_disp_tol_bohr: float = 3.0e-3,
        initial_hinv_scale: float = 1.0,
        use_bfgs: bool = True,
        remove_translation: bool = True,
        orient_x_continuously: bool = True,
        xyz_path: Optional[str] = "ubs_gpm_meci_path.xyz",
        verbose: bool = True,
    ):
        self.pair_scanner = pair_scanner
        self.max_steps = int(max_steps)
        self.max_step_bohr = float(max_step_bohr)
        self.gap_tol_ev = float(gap_tol_ev)
        self.max_pg_tol = float(max_pg_tol)
        self.rms_pg_tol = float(rms_pg_tol)
        self.max_disp_tol_bohr = float(max_disp_tol_bohr)
        self.initial_hinv_scale = float(initial_hinv_scale)
        self.use_bfgs = bool(use_bfgs)
        self.remove_translation = bool(remove_translation)
        self.orient_x_continuously = bool(orient_x_continuously)
        self.xyz_path = xyz_path
        self.verbose = verbose

        self.x_prev = None
        self.y_prev = None
        self.Hinv = None
        self.prev_step = None
        self.prev_gobj = None

    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            raise RuntimeError("Cannot normalize near-zero vector.")
        return v / n

    @staticmethod
    def _remove_components(v: np.ndarray, bases: Iterable[np.ndarray]) -> np.ndarray:
        out = v.copy()
        for b in bases:
            out -= np.dot(out, b) * b
        return out

    def _arbitrary_perp(self, x: np.ndarray) -> np.ndarray:
        idx = int(np.argmin(np.abs(x)))
        v = np.zeros_like(x)
        v[idx] = 1.0
        v = self._remove_components(v, [x])
        return self._safe_normalize(v)

    def _initial_y(self, x: np.ndarray, gmean: np.ndarray) -> np.ndarray:
        y = self._remove_components(gmean, [x])
        if np.linalg.norm(y) < 1.0e-10:
            y = self._arbitrary_perp(x)
        else:
            y = self._safe_normalize(y)
        return y

    def _update_y_ubs(self, x_new: np.ndarray, gmean: np.ndarray) -> np.ndarray:
        if self.x_prev is None or self.y_prev is None:
            return self._initial_y(x_new, gmean)

        x_old = self.x_prev
        y_old = self.y_prev
        a = float(np.dot(y_old, x_new))
        b = float(np.dot(x_old, x_new))
        denom = np.sqrt(a * a + b * b)

        if denom < 1.0e-10:
            y_new = self._initial_y(x_new, gmean)
        else:
            # Maeda-Ohno-Morokuma updated BP formula, Eq. 5 form.
            y_new = (a * x_old - b * y_old) / denom
            y_new = self._remove_components(y_new, [x_new])
            if np.linalg.norm(y_new) < 1.0e-10:
                y_new = self._initial_y(x_new, gmean)
            else:
                y_new = self._safe_normalize(y_new)

        if self.y_prev is not None and np.dot(y_new, self.y_prev) < 0.0:
            y_new *= -1.0
        return y_new

    @staticmethod
    def _project_xy(v: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return v - np.dot(v, x) * x - np.dot(v, y) * y

    @staticmethod
    def _project_y(v: np.ndarray, y: np.ndarray) -> np.ndarray:
        return v - np.dot(v, y) * y

    @staticmethod
    def _max_atom_norm(v_flat: np.ndarray, natm: int) -> float:
        vv = v_flat.reshape(natm, 3)
        return float(np.max(np.linalg.norm(vv, axis=1)))

    @staticmethod
    def _rms_cart(v_flat: np.ndarray) -> float:
        return float(np.sqrt(np.mean(v_flat * v_flat)))

    def _limit_step(self, step: np.ndarray, natm: int) -> np.ndarray:
        if self.remove_translation:
            step3 = step.reshape(natm, 3)
            step3 -= step3.mean(axis=0, keepdims=True)
            step = step3.reshape(-1)

        max_atom_step = self._max_atom_norm(step, natm)
        if max_atom_step > self.max_step_bohr:
            step = step * (self.max_step_bohr / max_atom_step)
        return step

    def _bfgs_update(self, s: np.ndarray, ygrad: np.ndarray) -> None:
        sy = float(np.dot(s, ygrad))
        if sy <= 1.0e-10:
            return
        Hy = self.Hinv @ ygrad
        yHy = float(np.dot(ygrad, Hy))
        if yHy <= 1.0e-14:
            return
        rho = 1.0 / sy
        I = np.eye(s.size)
        V = I - rho * np.outer(s, ygrad)
        self.Hinv = V @ self.Hinv @ V.T + rho * np.outer(s, s)

    def _take_step(self, gobj: np.ndarray, y: np.ndarray, natm: int) -> np.ndarray:
        if self.use_bfgs:
            step = -self.Hinv @ gobj
            if np.dot(step, gobj) > 0.0 or not np.all(np.isfinite(step)):
                step = -gobj.copy()
                self.Hinv = np.eye(gobj.size) * self.initial_hinv_scale
        else:
            step = -gobj.copy()

        # The projected gradient has no y component; remove numerical/bfgs leakage.
        step = self._project_y(step, y)
        
        return self._limit_step(step, natm)

    def _append_xyz(self, mol, comment: str) -> None:
        if self.xyz_path is None:
            return
        coords_ang = mol.atom_coords() * BOHR2ANG
        with open(self.xyz_path, "a") as f:
            f.write(f"{mol.natm}\n")
            f.write(comment + "\n")
            for ia in range(mol.natm):
                sym = mol.atom_symbol(ia)
                x, y, z = coords_ang[ia]
                f.write(f"{sym:2s} {x:18.10f} {y:18.10f} {z:18.10f}\n")

    @staticmethod
    def _make_mol(mol0, coords_flat: np.ndarray):
        coords = coords_flat.reshape(mol0.natm, 3)
        return mol0.set_geom_(coords, unit="Bohr", inplace=False)

    def kernel(self, mol0):
        natm = mol0.natm
        ndim = 3 * natm
        coords = mol0.atom_coords().reshape(-1).copy()

        self.x_prev = None
        self.y_prev = None
        self.Hinv = np.eye(ndim) * self.initial_hinv_scale
        self.prev_step = None
        self.prev_gobj = None

        if self.xyz_path is not None:
            open(self.xyz_path, "w").close()

        history: List[Dict[str, Any]] = []
        last_max_step = np.inf

        for istep in range(self.max_steps + 1):
            mol = self._make_mol(mol0, coords)
            res = self.pair_scanner(mol)

            e = np.asarray(res.energies, dtype=float)
            g = np.asarray(res.gradients, dtype=float)
            if e.shape != (2,):
                raise ValueError("PairResult.energies must have shape (2,).")
            if g.shape != (2, natm, 3):
                raise ValueError(f"PairResult.gradients must have shape (2, {natm}, 3).")

            EA, EB = float(e[0]), float(e[1])
            gA = g[0].reshape(-1)
            gB = g[1].reshape(-1)

            raw_gap = EB - EA
            gap_ev = raw_gap * HARTREE2EV
            gdiff = gB - gA
            R = float(np.linalg.norm(gdiff))
            if R < 1.0e-12:
                raise RuntimeError("DGV norm is nearly zero; UBS/GPM cannot define x reliably.")

            x = gdiff / R
            signed_gap = raw_gap
            if self.orient_x_continuously and self.x_prev is not None and np.dot(x, self.x_prev) < 0.0:
                x = -x
                signed_gap = -signed_gap

            gmean = 0.5 * (gA + gB)
            y = self._update_y_ubs(x, gmean)
            Pgmean = self._project_xy(gmean, x, y)
            ggap = 2.0 * signed_gap * x
            gobj = ggap + Pgmean

            if self.use_bfgs and self.prev_step is not None and self.prev_gobj is not None:
                self._bfgs_update(self.prev_step, gobj - self.prev_gobj)

            max_pg = self._max_atom_norm(Pgmean, natm)
            rms_pg = self._rms_cart(Pgmean)
            max_gobj = self._max_atom_norm(gobj, natm)
            rms_gobj = self._rms_cart(gobj)

            rec = {
                "step": istep,
                "roots": res.roots,
                "EA": EA,
                "EB": EB,
                "Emean": 0.5 * (EA + EB),
                "gap_ev": gap_ev,
                "abs_gap_ev": abs(gap_ev),
                "R_DGV": R,
                "max_Pgmean": max_pg,
                "rms_Pgmean": rms_pg,
                "max_gobj": max_gobj,
                "rms_gobj": rms_gobj,
                "max_step_bohr": last_max_step,
                "scanner_diagnostics": res.diagnostics,
            }
            history.append(rec)

            self._append_xyz(
                mol,
                comment=(
                    f"step={istep} roots={res.roots} EA={EA:.12f} EB={EB:.12f} "
                    f"gap_eV={gap_ev:+.8f} maxPg={max_pg:.3e} rmsPg={rms_pg:.3e}"
                ),
            )

            if self.verbose:
                print("--------- UBS Gradient Projection MECI Optimization ---------")
                print(
                " step  roots        E0/Eh           E1/Eh        gap/eV   "
                "max|Pgmean|   rms|Pgmean|  maxstep/Bohr"
                )

                print(
                    f"{istep:5d}  {str(res.roots):>9s}  {EA:14.8f} {EB:14.8f}  "
                    f"{gap_ev:+9.5f}  {max_pg:11.3e} {rms_pg:11.3e}  {last_max_step:11.3e}"
                )

            converged = (
                istep > 0
                and abs(gap_ev) < self.gap_tol_ev
                and max_pg < self.max_pg_tol
                and rms_pg < self.rms_pg_tol
                and last_max_step < self.max_disp_tol_bohr
            )
            if converged:
                if self.verbose:
                    print("\nConverged.")
                self.x_prev = x.copy()
                self.y_prev = y.copy()
                return mol, history

            if istep == self.max_steps:
                break

            step = self._take_step(gobj, y, natm)
            last_max_step = self._max_atom_norm(step, natm)
            self.prev_step = step.copy()
            self.prev_gobj = gobj.copy()
            coords = coords + step

            self.x_prev = x.copy()
            self.y_prev = y.copy()

        if self.verbose:
            print("\nNot converged within max_steps.")
        return self._make_mol(mol0, coords), history
@dataclass
class PenaltyStepRecord:
    step: int
    roots: Any
    EA: float
    EB: float
    Emean: float
    gap_ev: float
    penalty: float
    objective: float
    coef_gap: float
    max_gpen: float
    rms_gpen: float
    max_gmean: float
    rms_gmean: float
    max_step_bohr: float
    trust_radius_bohr: float
    scanner_diagnostics: Dict[str, Any] = field(default_factory=dict)


class PenaltyFunctionMECI:
    """
    Penalty-function MECI optimizer for a two-state PySCF-like scanner.

    Parameters
    ----------
    pair_scanner
        Callable object. Typically CASCIActiveRootTrackedPairScanner(...).
    sigma
        Dimensionless penalty weight. Larger values enforce degeneracy more
        strongly but can make steps unstable.
    alpha_ev
        Alpha parameter in eV for gap^2 / (|gap| + alpha). Converted to Eh.
    max_step_bohr
        Initial per-atom trust radius in Bohr.
    adaptive_trust
        If True, adjust future trust radius based on whether the previous
        objective decreased. This does not reject steps, because rejecting would
        require rolling back the stateful active/root-tracking scanner.
    """

    def __init__(
        self,
        pair_scanner: Callable[[Any], Any],
        sigma: float = 3.5,
        alpha_ev: float = 0.50,
        max_steps: int = 100,
        max_step_bohr: float = 0.03,
        min_step_bohr: float = 0.002,
        max_step_bohr_cap: Optional[float] = None,
        gap_tol_ev: float = 1.0e-3,
        max_grad_tol: float = 3.0e-4,
        rms_grad_tol: float = 2.0e-4,
        max_disp_tol_bohr: float = 3.0e-3,
        initial_hinv_scale: float = 1.0,
        use_bfgs: bool = True,
        remove_translation: bool = True,
        adaptive_trust: bool = True,
        trust_shrink: float = 0.50,
        trust_grow: float = 1.15,
        xyz_path: Optional[str] = "penalty_meci_path.xyz",
        verbose: bool = True,
    ):
        self.pair_scanner = pair_scanner
        self.sigma = float(sigma)
        self.alpha_ev = float(alpha_ev)
        self.alpha = self.alpha_ev / HARTREE2EV
        if self.alpha <= 0.0:
            raise ValueError("alpha_ev must be positive.")

        self.max_steps = int(max_steps)
        self.max_step_bohr = float(max_step_bohr)
        self.current_trust_bohr = float(max_step_bohr)
        self.min_step_bohr = float(min_step_bohr)
        self.max_step_bohr_cap = float(max_step_bohr_cap) if max_step_bohr_cap is not None else float(max_step_bohr)

        self.gap_tol_ev = float(gap_tol_ev)
        self.max_grad_tol = float(max_grad_tol)
        self.rms_grad_tol = float(rms_grad_tol)
        self.max_disp_tol_bohr = float(max_disp_tol_bohr)
        self.initial_hinv_scale = float(initial_hinv_scale)
        self.use_bfgs = bool(use_bfgs)
        self.remove_translation = bool(remove_translation)
        self.adaptive_trust = bool(adaptive_trust)
        self.trust_shrink = float(trust_shrink)
        self.trust_grow = float(trust_grow)
        self.xyz_path = xyz_path
        self.verbose = bool(verbose)

        self.Hinv: Optional[np.ndarray] = None
        self.prev_step: Optional[np.ndarray] = None
        self.prev_gpen: Optional[np.ndarray] = None
        self.prev_objective: Optional[float] = None

    # -----------------------------
    # Small utilities
    # -----------------------------
    @staticmethod
    def _make_mol(mol0, coords_flat: np.ndarray):
        coords = coords_flat.reshape(mol0.natm, 3)
        return mol0.set_geom_(coords, unit="Bohr", inplace=False)

    @staticmethod
    def _max_atom_norm(v_flat: np.ndarray, natm: int) -> float:
        vv = v_flat.reshape(natm, 3)
        return float(np.max(np.linalg.norm(vv, axis=1)))

    @staticmethod
    def _rms_cart(v_flat: np.ndarray) -> float:
        return float(np.sqrt(np.mean(v_flat * v_flat)))

    def _remove_translation_from_step(self, step: np.ndarray, natm: int) -> np.ndarray:
        if not self.remove_translation:
            return step
        step3 = step.reshape(natm, 3)
        step3 = step3 - step3.mean(axis=0, keepdims=True)
        return step3.reshape(-1)

    def _limit_step(self, step: np.ndarray, natm: int) -> np.ndarray:
        step = self._remove_translation_from_step(step, natm)
        max_atom_step = self._max_atom_norm(step, natm)
        if max_atom_step > self.current_trust_bohr:
            step = step * (self.current_trust_bohr / max_atom_step)
        return step

    def _append_xyz(self, mol, comment: str) -> None:
        if self.xyz_path is None:
            return
        coords_ang = mol.atom_coords() * BOHR2ANG
        with open(self.xyz_path, "a") as f:
            f.write(f"{mol.natm}\n")
            f.write(comment + "\n")
            for ia in range(mol.natm):
                sym = mol.atom_symbol(ia)
                x, y, z = coords_ang[ia]
                f.write(f"{sym:2s} {x:18.10f} {y:18.10f} {z:18.10f}\n")

    # -----------------------------
    # Penalty objective and gradient
    # -----------------------------
    def penalty_value_and_coeff(self, gap: float) -> Tuple[float, float]:
        """
        Return penalty value and d(penalty)/d(gap), both in Hartree units.

        penalty(gap) = sigma * gap^2 / (|gap| + alpha)
        coeff         = d penalty / d gap
        """
        abs_gap = abs(float(gap))
        denom = abs_gap + self.alpha
        raw = (gap * gap) / denom

        # derivative of gap^2 / (|gap| + alpha)
        # at gap=0 this formula gives 0, which is the continuous first derivative.
        d_raw = gap * (abs_gap + 2.0 * self.alpha) / (denom * denom)

        return self.sigma * raw, self.sigma * d_raw

    def build_penalty_gradient(self, EA: float, EB: float, gA: np.ndarray, gB: np.ndarray):
        gap = float(EB - EA)
        gmean = 0.5 * (gA + gB)
        gdiff = gB - gA
        penalty, coef = self.penalty_value_and_coeff(gap)
        objective = 0.5 * (EA + EB) + penalty
        gpen = gmean + coef * gdiff
        return objective, penalty, coef, gap, gpen, gmean, gdiff

    # -----------------------------
    # BFGS stepper
    # -----------------------------
    def _bfgs_update(self, s: np.ndarray, ygrad: np.ndarray) -> None:
        if self.Hinv is None:
            return

        sy = float(np.dot(s, ygrad))
        if sy <= 1.0e-10:
            return

        Hy = self.Hinv @ ygrad
        yHy = float(np.dot(ygrad, Hy))
        if yHy <= 1.0e-14:
            return

        rho = 1.0 / sy
        I = np.eye(s.size)
        V = I - rho * np.outer(s, ygrad)
        self.Hinv = V @ self.Hinv @ V.T + rho * np.outer(s, s)

    def _take_step(self, gpen: np.ndarray, natm: int) -> np.ndarray:
        if self.use_bfgs and self.Hinv is not None:
            step = -self.Hinv @ gpen
            if np.dot(step, gpen) > 0.0 or not np.all(np.isfinite(step)):
                # Fallback to steepest descent and reset inverse Hessian.
                step = -gpen.copy()
                self.Hinv = np.eye(gpen.size) * self.initial_hinv_scale
        else:
            step = -gpen.copy()

        return self._limit_step(step, natm)

    def _update_future_trust(self, objective: float) -> None:
        if not self.adaptive_trust or self.prev_objective is None:
            return

        if objective > self.prev_objective:
            self.current_trust_bohr = max(
                self.min_step_bohr,
                self.current_trust_bohr * self.trust_shrink,
            )
        else:
            self.current_trust_bohr = min(
                self.max_step_bohr_cap,
                self.current_trust_bohr * self.trust_grow,
            )

    # -----------------------------
    # Main optimization loop
    # -----------------------------
    def kernel(self, mol0):
        natm = mol0.natm
        ndim = 3 * natm
        coords = mol0.atom_coords().reshape(-1).copy()

        self.Hinv = np.eye(ndim) * self.initial_hinv_scale
        self.prev_step = None
        self.prev_gpen = None
        self.prev_objective = None
        self.current_trust_bohr = self.max_step_bohr

        if self.xyz_path is not None:
            open(self.xyz_path, "w").close()

        history: List[PenaltyStepRecord] = []
        last_max_step = np.inf
            
        for istep in range(self.max_steps + 1):
            mol = self._make_mol(mol0, coords)
            res = self.pair_scanner(mol)

            e = np.asarray(res.energies, dtype=float)
            g = np.asarray(res.gradients, dtype=float)

            if e.shape != (2,):
                raise ValueError("pair_scanner result must have energies with shape (2,).")
            if g.shape != (2, natm, 3):
                raise ValueError(f"pair_scanner result must have gradients with shape (2, {natm}, 3).")

            EA, EB = float(e[0]), float(e[1])
            gA = g[0].reshape(-1)
            gB = g[1].reshape(-1)

            objective, penalty, coef, gap, gpen, gmean, gdiff = self.build_penalty_gradient(
                EA, EB, gA, gB
            )

            # Update inverse Hessian using the last accepted step and current gradient.
            if self.use_bfgs and self.prev_step is not None and self.prev_gpen is not None:
                self._bfgs_update(self.prev_step, gpen - self.prev_gpen)

            # Adapt future trust radius based on previous objective. No step rejection is
            # done because pair_scanner is usually stateful for active/root tracking.
            self._update_future_trust(objective)

            gap_ev = gap * HARTREE2EV
            max_gpen = self._max_atom_norm(gpen, natm)
            rms_gpen = self._rms_cart(gpen)
            max_gmean = self._max_atom_norm(gmean, natm)
            rms_gmean = self._rms_cart(gmean)
            roots = getattr(res, "roots", None)
            diagnostics = getattr(res, "diagnostics", {}) or {}

            rec = PenaltyStepRecord(
                step=istep,
                roots=roots,
                EA=EA,
                EB=EB,
                Emean=0.5 * (EA + EB),
                gap_ev=gap_ev,
                penalty=penalty,
                objective=objective,
                coef_gap=coef,
                max_gpen=max_gpen,
                rms_gpen=rms_gpen,
                max_gmean=max_gmean,
                rms_gmean=rms_gmean,
                max_step_bohr=last_max_step,
                trust_radius_bohr=self.current_trust_bohr,
                scanner_diagnostics=diagnostics,
            )
            history.append(rec)

            self._append_xyz(
                mol,
                comment=(
                    f"step={istep} roots={roots} EA={EA:.12f} EB={EB:.12f} "
                    f"gap_eV={gap_ev:+.8f} Fpen={objective:.12f} "
                    f"penalty={penalty:.6e} coef={coef:.6e} "
                    f"maxG={max_gpen:.3e} rmsG={rms_gpen:.3e}"
                ),
            )

            if self.verbose:
                print("--------- Penalty function MECI Optimization ---------")
                print(
                " step  roots        E0/Eh           E1/Eh        gap/eV   "
                "Fpen/Eh      coef      max|gpen|    rms|gpen|  trust/Bohr  maxstep/Bohr"
            )

                print(
                    f"{istep:5d}  {str(roots):>9s}  {EA:14.8f} {EB:14.8f}  "
                    f"{gap_ev:+9.5f}  {objective:11.6f} {coef:9.3e}  "
                    f"{max_gpen:11.3e} {rms_gpen:11.3e}  "
                    f"{self.current_trust_bohr:10.3e} {last_max_step:12.3e}"
                )

            converged = (
                istep > 0
                and abs(gap_ev) < self.gap_tol_ev
                and max_gpen < self.max_grad_tol
                and rms_gpen < self.rms_grad_tol
                and last_max_step < self.max_disp_tol_bohr
            )

            if converged:
                if self.verbose:
                    print("\nConverged.")
                return mol, history

            if istep == self.max_steps:
                break

            step = self._take_step(gpen, natm)
            last_max_step = self._max_atom_norm(step, natm)

            self.prev_step = step.copy()
            self.prev_gpen = gpen.copy()
            self.prev_objective = objective

            coords = coords + step

        if self.verbose:
            print("\nNot converged within max_steps.")
        return self._make_mol(mol0, coords), history
