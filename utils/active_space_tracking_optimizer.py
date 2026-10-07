from pyscf import gto, scf, mcscf, lib
from pyscf.mcscf import addons
from pyscf.gbci.gbci import group_info_list

from utils.gbci_compat import make_gbci
import numpy as np
import numpy


HARTREE2EV = 27.211386245988


class CASCI_Active_Root_Tracking_Optimizer:
    method_label = "CASCI"

    def __init__(
        self,
        ncas,
        nelecas,
        act_list=None,
        nroots=2,
        target_root=0,
        conv_tol=1e-10,
        max_cycle=100,
        act_base=0,
        candidate_idx=None,
        root_tracking=True,
        root_overlap_warn=0.50,
        root_margin_warn=0.10,
        active_sv_warn=0.70,
        verbose=True,
        ):
        self.ncas = ncas
        self.nelecas = nelecas

        self.act_list_is_auto = act_list is None

        if act_list is None:
            self.act_list = None
        else:
            if act_base == 1:
                self.act_list = np.asarray(act_list, dtype=int) - 1
            else:
                self.act_list = np.asarray(act_list, dtype=int)

        self.nroots = nroots
        self.initial_target_root = target_root
        self.current_root = target_root

        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.candidate_idx = candidate_idx

        self.root_tracking = root_tracking
        self.root_overlap_warn = root_overlap_warn
        self.root_margin_warn = root_margin_warn
        self.active_sv_warn = active_sv_warn
        self.verbose = verbose

        self.step = 0
        self.converged = False

        self.prev_mol = None
        self.prev_active_mo = None
        self.prev_ci_target = None
        self.prev_e_target = None
        self.prev_coords = None

        # Metadata for the sorted MO basis used in the current CAS/GBCI step.
        self.last_mo_order = None
        self.last_mo_energy = None
        self.last_mo_occ = None

    # ------------------------------------------------------------
    # Basic builders
    # ------------------------------------------------------------
    def make_mf(self, mol):
        mf = scf.RHF(mol)
        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.damp = 0.4
        mf.level_shift = 0.5
        return mf

    def make_cas(self, mf):
        mc = mcscf.CASCI(mf, self.ncas, self.nelecas)
        mc.conv_tol = self.conv_tol
        mc.max_cycle = self.max_cycle
        mc.fcisolver.nroots = self.nroots
        mc.fix_spin_(ss=0)
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
    def _as_ci_list(mc):
        """
        PySCF CASCI에서 nroots=1이면 mc.ci가 단일 vector,
        nroots>1이면 list/tuple일 수 있으므로 통일해서 list로 반환.
        """
        ci = mc.ci
        if isinstance(ci, (list, tuple)):
            return list(ci)
        return [ci]

    @staticmethod
    def _ci_overlap(ci_old, ci_new):
        v_old = np.asarray(ci_old).ravel()
        v_new = np.asarray(ci_new).ravel()

        if v_old.shape != v_new.shape:
            return np.nan

        n_old = np.linalg.norm(v_old)
        n_new = np.linalg.norm(v_new)

        if n_old < 1e-14 or n_new < 1e-14:
            return np.nan

        return abs(np.vdot(v_old, v_new)) / (n_old * n_new)

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
    # Helper
    # ------------------------------------------------------------

    def nelecas_total(self):
        """
        nelecas가 int 또는 (nalpha, nbeta) tuple 둘 다 가능하게 처리.
        """
        if isinstance(self.nelecas, (tuple, list, np.ndarray)):
            return int(np.sum(self.nelecas))
        return int(self.nelecas)


    def get_default_act_list(self, mc, mf):
        """
        act_list=None일 때 첫 step의 active orbital index를 자동 생성.

        기준:
            ncore = (total electrons - active electrons) / 2
            active = ncore : ncore + ncas

        반환값은 0-based MO index.
        """
        if getattr(mc, "ncore", None) is not None:
            ncore = int(mc.ncore)
        else:
            nelecas_total = self.nelecas_total()
            ncore_float = 0.5 * (mf.mol.nelectron - nelecas_total)

            if abs(ncore_float - round(ncore_float)) > 1e-10:
                raise ValueError(
                    "Cannot determine closed-shell ncore from "
                    f"mol.nelectron={mf.mol.nelectron}, nelecas={self.nelecas}. "
                    "For open-shell or odd-electron cases, give act_list explicitly."
                )

            ncore = int(round(ncore_float))

        act_list = np.arange(ncore, ncore + self.ncas, dtype=int)

        nmo = mf.mo_coeff.shape[1]
        if act_list[-1] >= nmo:
            raise ValueError(
                f"Automatic active list {act_list.tolist()} exceeds number of MOs {nmo}."
            )

        return act_list

    # ------------------------------------------------------------
    # Active-space tracking
    # ------------------------------------------------------------

    def select_active_by_previous_active_mo(self, mol_new, mo_new):
        """
        이전 step의 active MO block과 현재 canonical MO 사이 overlap을 비교해서
        현재 active orbital index를 고른다.
        """
        if self.prev_mol is None or self.prev_active_mo is None:
            raise RuntimeError("No previous active orbitals are available.")

        s_old_new = self.cross_ao_overlap(self.prev_mol, mol_new)

        # A[i, j] = <previous active orbital i | current canonical MO j>
        a = self.prev_active_mo.T @ s_old_new @ mo_new

        # 현재 canonical MO j가 이전 active subspace에 얼마나 들어가는지
        score = np.sum(np.abs(a) ** 2, axis=0)

        nmo = mo_new.shape[1]

        if self.candidate_idx is None:
            candidate_idx = np.arange(nmo)
        else:
            candidate_idx = np.asarray(self.candidate_idx, dtype=int)

        cand_scores = score[candidate_idx]
        selected = candidate_idx[np.argsort(cand_scores)[-self.ncas:]]

        # 이전 active orbital 순서와 최대한 맞게 새 active orbital 순서 정렬
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

        if np.any(mask):
            best_unselected_score = np.max(score[mask])
        else:
            best_unselected_score = np.nan

        active_selection_gap = np.min(selected_scores) - best_unselected_score

        diag = {
            "active_overlap_matrix": a_sel,
            "active_abs_overlap_matrix": np.abs(a_sel),
            "active_scores_all": score,
            "active_selected_scores": selected_scores,
            "active_singular_values": svals,
            "active_min_singular_value": np.nanmin(svals),
            "active_selection_gap": active_selection_gap,
            "best_unselected_score": best_unselected_score,
        }

        return selected, diag

    def _reorder_active_orbitals(self, overlap_matrix, selected):
        """
        Hungarian assignment로 이전 active 순서와 새 active 순서를 맞춘다.
        """
        selected = np.asarray(selected, dtype=int)

        try:
            from scipy.optimize import linear_sum_assignment

            sub = np.abs(overlap_matrix[:, selected])
            row_ind, col_ind = linear_sum_assignment(-sub)

            order = np.argsort(row_ind)
            selected = selected[col_ind[order]]

        except Exception:
            # scipy가 없거나 실패하면 index 순서로 fallback
            selected = np.asarray(sorted(selected), dtype=int)

        return selected

    def make_cas_mo(self, mc, mf):
        """
        현재 step에서 CASCI에 넣을 mo_coeff 생성.

        step 0:
            - act_list가 주어졌으면 그것 사용
            - act_list=None이면 ncas/nelecas 기준으로 자동 생성

        step > 0:
            - 이전 active MO block과 현재 canonical MO overlap으로 active space tracking
        """
        if self.step == 0:
            if self.act_list is None:
                selected = self.get_default_act_list(mc, mf)
                self.act_list = selected.copy()

                if self.verbose:
                    print("\nInitial active space was not provided.")
                    print("Automatically selected active orbitals from ncas/nelecas:")
                    print(f"  ncas   = {self.ncas}")
                    print(f"  nelecas = {self.nelecas}")
                    print(f"  ncore  = {selected[0]}")
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
    # Root tracking
    # ------------------------------------------------------------
    def choose_root(self, mc):
        e_tot = self.get_root_energies(mc)
        ci_list = self._as_ci_list(mc)

        nroots_actual = len(ci_list)

        if self.step == 0 or self.prev_ci_target is None or not self.root_tracking:
            root = self.initial_target_root
            ci_overlaps = np.full(nroots_actual, np.nan)
            margin = np.nan
        else:
            ci_overlaps = np.array(
                [self._ci_overlap(self.prev_ci_target, ci) for ci in ci_list],
                dtype=float,
            )

            if np.all(np.isnan(ci_overlaps)):
                root = self.current_root
                margin = np.nan
            else:
                # 기본은 CI vector overlap 최대 root
                root = int(np.nanargmax(ci_overlaps))

                # 1등과 2등 차이
                sorted_ov = np.sort(ci_overlaps[~np.isnan(ci_overlaps)])
                if len(sorted_ov) >= 2:
                    margin = sorted_ov[-1] - sorted_ov[-2]
                else:
                    margin = np.nan

                # overlap이 비슷하면 energy continuity로 tie-break
                if self.prev_e_target is not None:
                    best_ov = ci_overlaps[root]
                    near = np.where(ci_overlaps >= best_ov - self.root_margin_warn)[0]
                    if len(near) > 1:
                        dE = np.abs(e_tot[near] - self.prev_e_target)
                        root = int(near[np.argmin(dE)])

        root_diag = {
            "ci_overlaps": ci_overlaps,
            "root_tracking_margin": margin,
            "selected_root": root,
        }

        return root, root_diag

    @staticmethod
    def get_root_energies(mc):
        """
        Return state-resolved energies when available.

        CASCI stores root energies in mc.e_tot.  State-averaged CASSCF keeps the
        optimization energy in mc.e_tot and stores individual root energies in
        mc.e_states, so root tracking must read mc.e_states there.
        """
        e_states = getattr(mc, "e_states", None)
        if e_states is not None:
            return np.atleast_1d(np.asarray(e_states, dtype=float))

        return np.atleast_1d(np.asarray(mc.e_tot, dtype=float))

    # ------------------------------------------------------------
    # Optional state diagnostics
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

    # ------------------------------------------------------------
    # Diagnostics printing
    # ------------------------------------------------------------
    def print_diagnostics(
        self,
        mf,
        mc,
        selected_act,
        active_diag,
        root_diag,
        e_target,
        grad,
    ):
        if not self.verbose:
            return

        e_tot = self.get_root_energies(mc)
        ci_list = self._as_ci_list(mc)

        print("\n" + "=" * 72)
        print(f"{self.method_label} active/root tracking diagnostics: step {self.step}")
        print("=" * 72)

        print(f"SCF converged   : {bool(mf.converged)}")
        print(f"{self.method_label} converged : {bool(np.all(getattr(mc, 'converged', True)))}")
        print(f"Selected root   : {root_diag['selected_root']}")
        print(f"Target energy   : {e_target:.12f} Eh")

        if self.prev_e_target is not None:
            dE = e_target - self.prev_e_target
            print(f"dE from previous tracked state: {dE:+.6e} Eh  ({dE * HARTREE2EV:+.6e} eV)")

        print("\nRoot energies:")
        for i, e in enumerate(e_tot):
            marker = " <== tracked" if i == root_diag["selected_root"] else ""
            print(f"  root {i:2d}: {e:.12f} Eh{marker}")

        if len(e_tot) >= 2:
            print("\nAdjacent root gaps:")
            for i in range(len(e_tot) - 1):
                gap = (e_tot[i + 1] - e_tot[i]) * HARTREE2EV
                print(f"  root {i+1} - root {i}: {gap:.6f} eV")

        print("\nActive orbital indices, 0-based:")
        print(" ", np.asarray(selected_act, dtype=int))

        if active_diag["active_selected_scores"] is not None:
            print("\nActive selected scores:")
            print(" ", np.array2string(active_diag["active_selected_scores"], precision=6))

            print("\n|<old active_i | new active_j>|:")
            print(np.array2string(active_diag["active_abs_overlap_matrix"], precision=4))

            print("\nActive-space singular values:")
            print(" ", np.array2string(active_diag["active_singular_values"], precision=6))

            print(f"Minimum active singular value : {active_diag['active_min_singular_value']:.6f}")
            print(f"Active selection gap          : {active_diag['active_selection_gap']:.6e}")
            print(f"Best unselected score         : {active_diag['best_unselected_score']:.6f}")

            if active_diag["active_min_singular_value"] < self.active_sv_warn:
                print("WARNING: Active space overlap is poor. Active space may have changed discontinuously.")

        if root_diag["ci_overlaps"] is not None:
            print("\nCI overlaps with previous tracked root:")
            for i, ov in enumerate(root_diag["ci_overlaps"]):
                print(f"  root {i:2d}: {ov:.6f}")

            margin = root_diag["root_tracking_margin"]
            print(f"Root tracking margin: {margin:.6f}")

            selected_root = root_diag["selected_root"]
            selected_ov = root_diag["ci_overlaps"][selected_root]

            if not np.isnan(selected_ov) and selected_ov < self.root_overlap_warn:
                print("WARNING: CI-vector overlap is small. Root tracking may be unreliable.")

            if not np.isnan(margin) and margin < self.root_margin_warn:
                print("WARNING: Root tracking is ambiguous. Two roots have similar CI overlap.")

        print("\nSpin diagnostics:")
        for i, ci in enumerate(ci_list):
            s2, mult = self.get_spin_square(mc, ci)
            print(f"  root {i:2d}: <S^2> = {s2:.6f}, mult = {mult:.6f}")

        ci_target = ci_list[root_diag["selected_root"]]
        occ = self.get_active_natural_occupations(mc, ci_target)
        if occ is not None:
            print("\nActive natural occupations of tracked root:")
            print(" ", np.array2string(occ, precision=6))

        grad = np.asarray(grad)
        grad_norm = np.linalg.norm(grad)
        grad_rms = np.sqrt(np.mean(grad**2))
        grad_max = np.max(np.abs(grad))

        print("\nGradient diagnostics:")
        print(f"  |grad| norm : {grad_norm:.6e} Eh/Bohr")
        print(f"  |grad| RMS  : {grad_rms:.6e} Eh/Bohr")
        print(f"  |grad| max  : {grad_max:.6e} Eh/Bohr")

        coords = mf.mol.atom_coords()
        if self.prev_coords is not None and self.prev_coords.shape == coords.shape:
            disp = coords - self.prev_coords
            disp_rms = np.sqrt(np.mean(disp**2))
            disp_max = np.max(np.abs(disp))
            print("\nGeometry displacement from previous step:")
            print(f"  RMS displacement : {disp_rms:.6e} Bohr")
            print(f"  Max displacement : {disp_max:.6e} Bohr")

        print("=" * 72 + "\n")

    # ------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------
    def compute_gradient(self, mc, root):
        """
        Compute the nuclear gradient of the selected root.

        CASCI and state-averaged CASSCF accept state=... in kernel().  Plain
        single-root CASSCF does not, so root 0 falls back to kernel().
        """
        grad_method = mc.nuc_grad_method()

        try:
            return grad_method.kernel(state=root)
        except TypeError:
            if root != 0:
                raise
            return grad_method.kernel()

    def get_active_mo_for_tracking(self, mc, mo_cas):
        """
        Return the active MO block to carry into the next geometry step.

        For CASCI this is the sorted input MO block.  For orbital-optimized
        methods such as CASSCF, mc.mo_coeff contains the final active orbitals.
        """
        mo_coeff = getattr(mc, "mo_coeff", None)
        if mo_coeff is None:
            mo_coeff = mo_cas

        ncore = int(mc.ncore)
        return mo_coeff[:, ncore:ncore + mc.ncas]

    def __call__(self, mol):
        mf = self.make_mf(mol)
        mf.kernel()

        if not mf.converged:
            print("WARNING: RHF did not converge.")

        mc = self.make_cas(mf)

        mo_cas, selected_act, active_diag = self.make_cas_mo(mc, mf)

        mc.kernel(mo_cas)

        method_converged = bool(np.all(getattr(mc, "converged", True)))
        self.converged = bool(mf.converged) and method_converged

        if not method_converged:
            print(f"WARNING: {self.method_label} did not converge.")

        root, root_diag = self.choose_root(mc)
        self.current_root = root

        e_tot = self.get_root_energies(mc)
        e_target = float(e_tot[root])

        cas_grad = self.compute_gradient(mc, root)

        self.print_diagnostics(
            mf=mf,
            mc=mc,
            selected_act=selected_act,
            active_diag=active_diag,
            root_diag=root_diag,
            e_target=e_target,
            grad=cas_grad,
        )

        # Store previous-step data after diagnostics
        ci_list = self._as_ci_list(mc)
        self.prev_ci_target = np.array(ci_list[root], copy=True)
        self.prev_e_target = e_target

        self.prev_mol = mol.copy()
        self.prev_active_mo = self.get_active_mo_for_tracking(mc, mo_cas).copy()
        self.prev_coords = mol.atom_coords().copy()

        self.act_list = np.asarray(selected_act, dtype=int).copy()

        print(
            f"Step {self.step}: tracked root {root}, "
            f"energy = {e_target:.12f} Eh"
        )

        self.step += 1

        return e_target, cas_grad


class CASSCF_Active_Root_Tracking_Optimizer(CASCI_Active_Root_Tracking_Optimizer):
    """
    CASSCF version implemented as a subclass of CASCI_Active_Root_Tracking_Optimizer.

    This class intentionally mirrors the CASCI optimizer structure.  It reuses
    the parent class for SCF construction, active-space tracking, root tracking,
    diagnostics, convergence bookkeeping, and the main __call__ flow.

    The main CASSCF-specific difference is in make_cas():
      - construct mcscf.CASSCF instead of mcscf.CASCI
      - use state_average_ automatically when nroots > 1, because plain PySCF
        CASSCF cannot optimize an unspecified multi-root fcisolver
      - allow either explicit state_weights or a scalar state_average_weight

    After the CASSCF kernel, the parent tracking hook stores the optimized
    active orbitals from mc.mo_coeff for the next geometry step.
    """

    method_label = "CASSCF"

    def __init__(
        self,
        ncas,
        nelecas,
        act_list=None,
        nroots=2,
        target_root=0,
        conv_tol=1e-10,
        max_cycle=100,
        act_base=0,
        candidate_idx=None,
        root_tracking=True,
        root_overlap_warn=0.50,
        root_margin_warn=0.10,
        active_sv_warn=0.70,
        verbose=True,
        state_average=True,
        state_weights=None,
        state_average_weight=None,
        project_previous_orbitals=True,
        reuse_ci_guess=True,
        strict_convergence=True,
    ):
        if state_weights is not None and state_average_weight is not None:
            raise ValueError(
                "Give either state_weights or state_average_weight, not both."
            )

        super().__init__(
            ncas=ncas,
            nelecas=nelecas,
            act_list=act_list,
            nroots=nroots,
            target_root=target_root,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
            act_base=act_base,
            candidate_idx=candidate_idx,
            root_tracking=root_tracking,
            root_overlap_warn=root_overlap_warn,
            root_margin_warn=root_margin_warn,
            active_sv_warn=active_sv_warn,
            verbose=verbose,
        )

        self.state_average = state_average
        self.state_weights = state_weights
        self.state_average_weight = state_average_weight
        self.project_previous_orbitals = bool(project_previous_orbitals)
        self.reuse_ci_guess = bool(reuse_ci_guess)
        self.strict_convergence = bool(strict_convergence)

        self.prev_casscf_mo = None
        self.prev_ci_list = None

        self.last_mf = None
        self.last_mc = None
        self.last_selected_act = None
        self.last_active_diag = None
        self.last_root_diag = None
        self.last_gradient = None
        self.history = []

    def get_state_weights(self):
        """
        Return CASSCF state-average weights.

        Defaults to equal weights, matching the root count requested by the
        CASCI-style nroots option.  If state_average_weight is given as a
        scalar, it is assigned to the currently tracked root and the remaining
        weight is distributed evenly over the other roots.
        """
        if self.state_weights is not None:
            return self.validate_state_weights(self.state_weights, "state_weights")

        if self.state_average_weight is None:
            return [1.0 / self.nroots] * self.nroots

        weight = float(self.state_average_weight)

        if weight < 0.0 or weight > 1.0:
            raise ValueError(
                f"state_average_weight must be between 0 and 1. Got {weight}."
            )

        if self.nroots == 1:
            if abs(weight - 1.0) > 1e-10:
                raise ValueError(
                    "state_average_weight must be 1.0 when nroots=1."
                )
            return [1.0]

        root = int(self.current_root)
        if root < 0 or root >= self.nroots:
            raise ValueError(
                f"current_root={root} is out of range for nroots={self.nroots}."
            )

        weights = np.full(self.nroots, (1.0 - weight) / (self.nroots - 1))
        weights[root] = weight
        return weights.tolist()

    def validate_state_weights(self, state_weights, label):
        weights = np.asarray(state_weights, dtype=float)

        if weights.ndim != 1 or len(weights) != self.nroots:
            raise ValueError(
                f"{label} must be a one-dimensional sequence with length "
                f"nroots={self.nroots}."
            )

        if np.any(weights < 0):
            raise ValueError(f"{label} must be non-negative.")

        weight_sum = float(np.sum(weights))
        if abs(weight_sum - 1.0) > 1e-10:
            raise ValueError(
                f"{label} must sum to 1.0. Got sum={weight_sum:.12f}."
            )

        return weights.tolist()

    def make_cas(self, mf):
        """
        Build a PySCF CASSCF object using the same visible settings as CASCI.
        """
        mc = mcscf.CASSCF(mf, self.ncas, self.nelecas)
        mc.conv_tol = self.conv_tol
        mc.max_cycle = self.max_cycle
        mc.fcisolver.nroots = self.nroots
        mc.fix_spin_(ss=0)

        if self.nroots > 1:
            if not self.state_average:
                raise ValueError(
                    "PySCF CASSCF with nroots > 1 requires state_average=True "
                    "for this CASCI-style root-tracking optimizer."
                )
            mc.state_average_(self.get_state_weights())

        return mc

    def make_cas_mo(self, mc, mf):
        """
        Build the CASSCF orbital guess.

        The first geometry uses the requested canonical-RHF active orbitals.
        Later geometries project the complete optimized CASSCF orbital set from
        the previous geometry, prioritizing the active block.  The closest
        canonical RHF orbitals are still identified for diagnostics, but they
        are not used to replace the projected CASSCF active subspace.
        """
        if (
            not self.project_previous_orbitals
            or self.prev_casscf_mo is None
            or self.prev_mol is None
        ):
            return super().make_cas_mo(mc, mf)

        selected, active_diag = self.select_active_by_previous_active_mo(
            mol_new=mf.mol,
            mo_new=mf.mo_coeff,
        )
        mo_cas = addons.project_init_guess(
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

        return mo_cas, selected, active_diag

    def run_casscf(self, mc, mo_cas):
        """Run CASSCF, reusing the previous multi-root CI vectors when possible."""
        ci0 = None
        if self.reuse_ci_guess and self.prev_ci_list is not None:
            if len(self.prev_ci_list) == self.nroots:
                ci0 = [np.array(ci, copy=True) for ci in self.prev_ci_list]

        if ci0 is None:
            mc.kernel(mo_cas)
        else:
            mc.kernel(mo_cas, ci0=ci0)

    @staticmethod
    def _finite_float_or_none(value):
        if value is None:
            return None
        value = float(value)
        return value if np.isfinite(value) else None

    @classmethod
    def _serializable_vector(cls, values):
        if values is None:
            return None
        return [cls._finite_float_or_none(value) for value in np.ravel(values)]

    def make_history_record(
        self,
        mol,
        mf,
        mc,
        selected_act,
        active_diag,
        root_diag,
        e_target,
        grad,
    ):
        energies = self.get_root_energies(mc)
        grad = np.asarray(grad, dtype=float)
        ci_list = self._as_ci_list(mc)
        natural_occupations = self.get_active_natural_occupations(
            mc,
            ci_list[int(root_diag["selected_root"])],
        )

        return {
            "step": int(self.step),
            "target_root": int(root_diag["selected_root"]),
            "target_energy_hartree": float(e_target),
            "root_energies_hartree": [float(value) for value in energies],
            "s1_minus_s0_hartree": (
                float(energies[1] - energies[0]) if len(energies) >= 2 else None
            ),
            "scf_converged": bool(mf.converged),
            "casscf_converged": bool(
                np.all(getattr(mc, "converged", True))
            ),
            "selected_active_orbitals_0based": [
                int(value) for value in np.asarray(selected_act, dtype=int)
            ],
            "selected_active_orbitals_1based": [
                int(value) + 1 for value in np.asarray(selected_act, dtype=int)
            ],
            "active_min_singular_value": self._finite_float_or_none(
                active_diag.get("active_min_singular_value")
            ),
            "projected_active_min_singular_value": self._finite_float_or_none(
                active_diag.get("projected_active_min_singular_value")
            ),
            "active_selection_gap": self._finite_float_or_none(
                active_diag.get("active_selection_gap")
            ),
            "ci_overlaps": self._serializable_vector(
                root_diag.get("ci_overlaps")
            ),
            "natural_occupations": (
                None
                if natural_occupations is None
                else [float(value) for value in natural_occupations]
            ),
            "gradient_norm": float(np.linalg.norm(grad)),
            "gradient_rms": float(np.sqrt(np.mean(grad**2))),
            "gradient_max": float(np.max(np.abs(grad))),
            "geometry_angstrom": np.asarray(
                mol.atom_coords(unit="Angstrom"),
                dtype=float,
            ).tolist(),
        }

    def set_reference(
        self,
        mol,
        mo_coeff,
        ci,
        e_states=None,
    ):
        """
        Seed an independent evaluation from a converged SA-CASSCF reference.

        This is intended for numerical-Hessian displacements.  Each displaced
        gradient can use a fresh optimizer seeded from exactly the same
        optimized geometry, orbitals, and CI vectors.
        """
        mo_coeff = np.asarray(mo_coeff)
        if isinstance(ci, (list, tuple)):
            ci_list = list(ci)
        elif self.nroots == 1:
            ci_list = [ci]
        else:
            ci_array = np.asarray(ci)
            if ci_array.shape[0] != self.nroots:
                raise ValueError(
                    "A multi-root reference CI array must have its root "
                    "dimension first."
                )
            ci_list = list(ci_array)

        if len(ci_list) != self.nroots:
            raise ValueError(
                f"Reference CI has {len(ci_list)} roots; expected {self.nroots}."
            )

        ncore_float = 0.5 * (mol.nelectron - self.nelecas_total())
        if abs(ncore_float - round(ncore_float)) > 1e-10:
            raise ValueError(
                "Cannot determine the CASSCF core size for the reference."
            )
        ncore = int(round(ncore_float))

        self.prev_mol = mol.copy()
        self.prev_casscf_mo = np.array(mo_coeff, copy=True)
        self.prev_active_mo = np.array(
            mo_coeff[:, ncore:ncore + self.ncas],
            copy=True,
        )
        self.prev_ci_list = [np.array(value, copy=True) for value in ci_list]
        self.prev_ci_target = np.array(
            self.prev_ci_list[self.initial_target_root],
            copy=True,
        )
        self.prev_coords = mol.atom_coords().copy()

        if e_states is None:
            self.prev_e_target = None
        else:
            energies = np.atleast_1d(np.asarray(e_states, dtype=float))
            self.prev_e_target = float(energies[self.initial_target_root])

        self.current_root = self.initial_target_root
        return self

    def get_reference(self):
        """Return copies of the latest converged SA-CASSCF reference data."""
        if self.last_mc is None:
            raise RuntimeError("No converged CASSCF result is available.")

        return {
            "mol": self.last_mc.mol.copy(),
            "mo_coeff": np.array(self.last_mc.mo_coeff, copy=True),
            "ci": [
                np.array(value, copy=True)
                for value in self._as_ci_list(self.last_mc)
            ],
            "e_states": np.array(
                self.get_root_energies(self.last_mc),
                copy=True,
            ),
        }

    def compute_gradient(self, mc, root):
        """
        Compute CASSCF gradients with a clearer message for unsupported SA weights.
        """
        try:
            return mc.nuc_grad_method(state=int(root)).kernel()
        except TypeError:
            return super().compute_gradient(mc, root)
        except NotImplementedError as err:
            weights = getattr(mc, "weights", None)
            if weights is not None:
                weights = np.asarray(weights, dtype=float)
                if np.max(weights) - np.min(weights) > 1e-8:
                    raise NotImplementedError(
                        "This PySCF build does not support analytic SA-CASSCF "
                        "gradients with unequal state-average weights. "
                        f"Requested weights were {weights.tolist()}. "
                        "Use equal weights for geometry optimization, or add a "
                        "numerical gradient path for unequal-weight SA-CASSCF."
                    ) from err
            raise

    def __call__(self, mol):
        mf = self.make_mf(mol)
        mf.kernel()

        if not mf.converged:
            message = "RHF did not converge for the requested CASSCF geometry."
            if self.strict_convergence:
                raise RuntimeError(message)
            print(f"WARNING: {message}")

        mc = self.make_cas(mf)
        mo_cas, selected_act, active_diag = self.make_cas_mo(mc, mf)
        self.run_casscf(mc, mo_cas)

        method_converged = bool(np.all(getattr(mc, "converged", True)))
        self.converged = bool(mf.converged) and method_converged

        if not method_converged:
            message = "SA-CASSCF did not converge for the requested geometry."
            if self.strict_convergence:
                raise RuntimeError(message)
            print(f"WARNING: {message}")

        root, root_diag = self.choose_root(mc)
        self.current_root = root

        energies = self.get_root_energies(mc)
        e_target = float(energies[root])
        casscf_grad = self.compute_gradient(mc, root)

        self.print_diagnostics(
            mf=mf,
            mc=mc,
            selected_act=selected_act,
            active_diag=active_diag,
            root_diag=root_diag,
            e_target=e_target,
            grad=casscf_grad,
        )

        record = self.make_history_record(
            mol=mol,
            mf=mf,
            mc=mc,
            selected_act=selected_act,
            active_diag=active_diag,
            root_diag=root_diag,
            e_target=e_target,
            grad=casscf_grad,
        )
        self.history.append(record)

        ci_list = self._as_ci_list(mc)
        self.prev_ci_list = [np.array(ci, copy=True) for ci in ci_list]
        self.prev_ci_target = np.array(ci_list[root], copy=True)
        self.prev_e_target = e_target
        self.prev_mol = mol.copy()
        self.prev_casscf_mo = np.array(mc.mo_coeff, copy=True)
        self.prev_active_mo = self.get_active_mo_for_tracking(
            mc,
            mo_cas,
        ).copy()
        self.prev_coords = mol.atom_coords().copy()
        self.act_list = np.asarray(selected_act, dtype=int).copy()

        self.last_mf = mf
        self.last_mc = mc
        self.last_selected_act = np.asarray(selected_act, dtype=int).copy()
        self.last_active_diag = active_diag
        self.last_root_diag = root_diag
        self.last_gradient = np.asarray(casscf_grad, dtype=float).copy()
        self.last_mo_energy = getattr(mc, "mo_energy", None)
        self.last_mo_occ = getattr(mc, "mo_occ", None)

        if self.verbose:
            print(
                f"Step {self.step}: tracked root {root}, "
                f"energy = {e_target:.12f} Eh"
            )
        self.step += 1

        return e_target, casscf_grad


class GBCI_Active_Root_Tracking_Optimizer(CASCI_Active_Root_Tracking_Optimizer):
    """
    GBCI version implemented as a subclass of CASCI_Active_Root_Tracking_Optimizer.

    This class reuses the parent class for:
      - SCF construction
      - active-space tracking diagnostics
      - root tracking logic
      - CI-vector overlap logic
      - printing diagnostics
      - convergence bookkeeping

    It overrides only the parts where GBCI can differ from PySCF CASCI:
      - make_cas(): construct a GBCI object instead of mcscf.CASCI
      - make_cas_mo(): sort active orbitals without relying on mcscf.addons.sort_mo
      - _as_ci_list(): handle GBCI-style CI/eigenvector layouts
      - run_gbci(): optionally pass previous CI vector as ci0
      - compute_gradient(): allow slightly different gradient APIs
      - __call__(): same parent flow, but calls run_gbci() instead of mc.kernel(mo)

    Expected GBCI-like API:
      gbci = gbci_factory(mf, ncas, nelecas, **gbci_kwargs)
      gbci.kernel(mo_coeff) or gbci.kernel(mo_coeff, ci0=ci0)
      gbci.e_tot
      gbci.ci
      gbci.nuc_grad_method().kernel(state=root)

    If your actual GBCI API differs, usually only make_cas(), run_gbci(),
    and compute_gradient() need small edits.
    """

    def __init__(
        self,
        ncas,
        nelecas,
        groupA = None,
        gbci_kwargs=None,
        act_list=None,
        nroots=2,
        target_root=0,
        conv_tol=1e-10,
        max_cycle=100,
        act_base=0,
        candidate_idx=None,
        root_tracking=True,
        root_overlap_warn=0.50,
        root_margin_warn=0.10,
        active_sv_warn=0.70,
        verbose=True,
    ):
        super().__init__(
            ncas=ncas,
            nelecas=nelecas,
            act_list=act_list,
            nroots=nroots,
            target_root=target_root,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
            act_base=act_base,
            candidate_idx=candidate_idx,
            root_tracking=root_tracking,
            root_overlap_warn=root_overlap_warn,
            root_margin_warn=root_margin_warn,
            active_sv_warn=active_sv_warn,
            verbose=verbose,
        )

        self.groupA = groupA
        self.gbci_kwargs = {} if gbci_kwargs is None else dict(gbci_kwargs)

        # Useful for post-step inspection.
        self.last_mf = None
        self.last_gbci = None
        self.last_mo_gbci = None
        self.last_mo_order = None
        self.last_mo_energy = None
        self.last_mo_occ = None
        self.last_selected_act = None
        self.last_active_diag = None
        self.last_root_diag = None

    # ------------------------------------------------------------
    # GBCI builder
    # ------------------------------------------------------------
    def make_cas(self, mf):
        """
        Override the parent CASCI builder.

        The parent __call__ expects a method called make_cas(), so we keep the
        same name even though this actually returns a GBCI object.
        """

        gbci = make_gbci(
            mf,
            self.ncas,
            self.nelecas,
            group_a=self.groupA,
            **self.gbci_kwargs,
        )

        # Try to set common convergence / root-count attributes if present.
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
        gbci.fix_spin_(ss=0)
        gbci._thres = 0.1
        return gbci

    # ------------------------------------------------------------
    # CI-vector layout helper for GBCI
    # ------------------------------------------------------------
    def _as_ci_list(self, mc):
        """
        Convert mc.ci / gbci.ci into a list of root-specific vectors.

        The parent CASCI version assumes PySCF CASCI layout.
        GBCI/NOCI codes often store eigenvectors as a matrix, e.g.
            ci[:, root]
        so this method accepts a few common layouts.
        """
        ci = getattr(mc, "ci", None)
        if ci is None:
            return []

        if isinstance(ci, (list, tuple)):
            return list(ci)

        arr = np.asarray(ci)
        e_tot = np.atleast_1d(np.asarray(getattr(mc, "e_tot", []), dtype=float))
        nroots_actual = len(e_tot) if len(e_tot) > 0 else self.nroots

        # Common layout: columns are roots.
        if arr.ndim >= 2 and arr.shape[-1] == nroots_actual:
            return [np.array(arr[..., i], copy=True) for i in range(nroots_actual)]

        # Alternative layout: first axis is root index.
        if arr.ndim >= 2 and arr.shape[0] == nroots_actual:
            return [np.array(arr[i, ...], copy=True) for i in range(nroots_actual)]

        # Fallback: single-root vector.
        return [arr]

    # ------------------------------------------------------------
    # MO sorting without mcscf.addons.sort_mo
    # ------------------------------------------------------------
    def get_ncore(self, mc, mf):
        """
        Determine ncore from the GBCI object if possible; otherwise infer it.
        """
        if getattr(mc, "ncore", None) is not None:
            return int(mc.ncore)

        nelecas_total = self.nelecas_total()
        ncore_float = 0.5 * (mf.mol.nelectron - nelecas_total)

        if abs(ncore_float - round(ncore_float)) > 1e-10:
            raise ValueError(
                "Cannot determine closed-shell ncore from "
                f"mol.nelectron={mf.mol.nelectron}, nelecas={self.nelecas}. "
                "For open-shell or odd-electron cases, set ncore inside the GBCI object "
                "or give a custom make_cas_mo()."
            )

        return int(round(ncore_float))

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

    def make_cas_mo(self, mc, mf):
        """
        Override parent make_cas_mo().

        Parent version uses mcscf.addons.sort_mo(mc, ...), which may fail for
        a non-PySCF GBCI object. The active-space selection logic is kept the
        same as the parent code.
        """
        if self.step == 0:
            if self.act_list is None:
                selected = self.get_default_act_list(mc, mf)
                self.act_list = selected.copy()

                if self.verbose:
                    print("\nInitial active space was not provided.")
                    print("Automatically selected active orbitals from ncas/nelecas:")
                    print(f"  ncas   = {self.ncas}")
                    print(f"  nelecas = {self.nelecas}")
                    print(f"  ncore  = {selected[0]}")
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

        # Save for inspection because parent make_cas_mo() returns only 3 objects.
        self.last_mo_order = mo_order
        self.last_mo_energy = getattr(mo_cas, "mo_energy", None)
        self.last_mo_occ = getattr(mo_cas, "mo_occ", None)

        return mo_cas, selected, active_diag

    # ------------------------------------------------------------
    # GBCI kernel / gradient hooks
    # ------------------------------------------------------------
    def run_gbci(self, gbci, mo_cas):
        """
        Run GBCI.

        If a previous tracked CI vector exists, pass it as ci0 when the GBCI
        kernel accepts it. If not, gracefully fall back to kernel(mo_cas).
        """
        ci0 = self.prev_ci_target if self.prev_ci_target is not None else None

        mol = gbci.mol
        ncas = gbci.ncas
        nelecas = gbci.nelecas

        mo_list, moe_list, po_list, group = gbci.optimize_mo(mo_cas)
        p = mo_list.shape[0]

        # Mirror pyscf.gbci.gbci.kernel.
        if gbci.group_a is None:
            conf_info_list = group_info_list(ncas, nelecas, po_list)
            svd_basis = po_list
        else:
            conf_info_list = group_info_list(ncas, nelecas, po_list, group)
            svd_basis = group

        dmet_core_list, ov_list = gbci.get_svd_matrices(mo_list, svd_basis)
        dmet_act_list = gbci.get_active_dm(mo_cas)
        h1e, ecore_list = gbci.get_h1cas(dmet_act_list , mo_list , dmet_core_list)
        eri = gbci.get_h2eff(mo_cas)

        # pyscf.grad.gbci reads these off the GBCI object.
        gbci._cache_gbci_intermediates(
            mo_cas, ncas, nelecas, gbci.ncore,
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

        e_tot, gbci.ci = gbci.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                        conf_info_list, ov_list, ecore_list,
                                        ci0=ci0, verbose=mol.verbose)
        gbci.e_tot = np.atleast_1d(np.asarray(e_tot, dtype=float))
        self.attach_sorted_mo_to_method(gbci, mo_cas)
        if getattr(gbci.fcisolver, 'converged', None) is not None:
            gbci.converged = numpy.all(gbci.fcisolver.converged)
        else:
            gbci.converged = True

        return e_tot, mo_list, moe_list, conf_info_list, dmet_core_list, ov_list, ecore_list

    def compute_gradient(self, gbci, root):
        """
        Compute gradient of the selected GBCI root.
        """
        grad_method = gbci.nuc_grad_method()

        try:
            return grad_method.kernel(state=root)
        except TypeError:
            return grad_method.kernel(root)

    # ------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------
    def __call__(self, mol):
        """
        Mostly copied from the parent __call__.

        The only essential difference is that we call self.run_gbci(mc, mo_cas)
        instead of mc.kernel(mo_cas), so that GBCI can receive ci0 if supported.
        """
        mf = self.make_mf(mol)
        mf.damp = 0.4
        mf.level_shift = 0.5
        mf.kernel()

        if not mf.converged:
            print("WARNING: RHF did not converge.")

        # Name kept as mc to reuse parent helper methods and diagnostics.
        mc = self.make_cas(mf)
        mo_cas, selected_act, active_diag = self.make_cas_mo(mc, mf)

        e_tot, mo_list, moe_list, conf_info_list, dmet_core_list, ov_list, ecore_list = self.run_gbci(mc, mo_cas)

        gbci_converged = bool(np.all(getattr(mc, "converged", True)))
        self.converged = bool(mf.converged) and gbci_converged

        if not gbci_converged:
            print("WARNING: GBCI did not converge.")

        root, root_diag = self.choose_root(mc)
        self.current_root = root

        e_tot = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))
        e_target = float(e_tot[root])

        grad_method = mc.nuc_grad_method()
        try:
            # Legacy signature: this repo's pyscf.grad.gbci takes the
            # intermediates positionally.
            gbci_grad = grad_method.kernel(
                mo_cas, mc._scf.mo_energy, mo_list, moe_list, conf_info_list,
                dmet_core_list, ov_list, ecore_list, mc.ci[root])
        except TypeError:
            # pyscf-forge's pyscf.grad.gbci reads them off the GBCI object
            # instead; run_gbci() published them via _cache_gbci_intermediates.
            gbci_grad = grad_method.kernel(state=root)

        self.print_diagnostics(
            mf=mf,
            mc=mc,
            selected_act=selected_act,
            active_diag=active_diag,
            root_diag=root_diag,
            e_target=e_target,
            grad=gbci_grad,
        )

        # Store previous-step data after diagnostics.
        ci_list = self._as_ci_list(mc)
        self.prev_ci_target = np.array(ci_list[root], copy=True)
        self.prev_e_target = e_target

        ncore = self.get_ncore(mc, mf)
        self.prev_mol = mol.copy()
        self.prev_active_mo = mo_cas[:, ncore:ncore + self.ncas].copy()
        self.prev_coords = mol.atom_coords().copy()

        self.act_list = np.asarray(selected_act, dtype=int).copy()

        self.last_mf = mf
        self.last_gbci = mc
        self.last_mo_gbci = mo_cas
        self.last_selected_act = np.asarray(selected_act, dtype=int).copy()
        self.last_active_diag = active_diag
        self.last_root_diag = root_diag

        print(
            f"Step {self.step}: tracked root {root}, "
            f"energy = {e_target:.12f} Eh"
        )

        self.step += 1

        return e_target, gbci_grad
