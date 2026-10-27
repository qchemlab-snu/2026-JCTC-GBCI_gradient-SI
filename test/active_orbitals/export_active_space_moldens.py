"""Export the active orbitals of each benchmark system as a Molden file.

One file per system, holding only the active orbitals in window order, so the
figures can be rendered directly. Each system is rebuilt from the geometry and
settings its own production or benchmark script uses.

For the lithium halides the active space is the default contiguous
window, so the RHF orbitals are used as they come. For pp and CH2NH2 the
scripts pass an explicit ``act_list``, which is applied here with ``sort_mo``;
those spaces are the *initial* selection at the starting geometry, since the
tracking optimizer re-selects orbitals as the geometry changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf import gto, scf
from pyscf.mcscf.addons import sort_mo
from pyscf.tools import molden

SCRIPT_DIR = Path(__file__).resolve().parent


# One bond length for all three, deliberately off equilibrium (LiH 1.595, LiF 1.564,
# LiCl 2.021 A), as in test/li_halides/gbci/li_halides_gbci_grad_check.py: at a
# stationary point both the analytic and the numerical gradient go to zero, which
# weakens the comparison.
LI_BOND = 1.50


def li_halide(partner: str, ncas: int, nelecas, act_list, character: str,
              bond: float = LI_BOND):
    """Li-X with the sigma-only active space of li_halides_gbci_grad_check.py.

    ``act_list`` is read off li_halides_orbital_character.py and is geometry
    sensitive, so it travels with the bond length rather than being recomputed.
    For LiF and LiCl it differs from the default contiguous window, which would
    pull in the halogen 2p/3p pi lone pairs instead of the Li-X sigma* orbitals.
    """
    return dict(
        atom=[["Li", (0.0, 0.0, 0.0)], [partner, (0.0, 0.0, bond)]],
        basis="ccpvdz", charge=0, spin=0, ncas=ncas, nelecas=nelecas,
        act_list=list(act_list),
        note=f"Li-{partner} {bond} A, sigma-only act_list {list(act_list)}; {character}",
    )


# pp: CASCI S0-optimised geometry (planar), from
# test/pp/casci/pp_casci_S0.out "Final ground state optimized geometry".
PP_ATOMS = """C          -3.75064441        0.68165263        0.00000163
C          -2.54053493        1.28505925       -0.00000080
C          -3.51115646       -0.72936345        0.00000163
C          -2.16974625       -0.89984775       -0.00000081
H          -4.70409556        1.18347729        0.00000475
H          -4.24568116       -1.51763696        0.00000473
N          -1.55818470        0.32786364       -0.00000124
H          -1.60531308       -1.81301549       -0.00000223
H          -2.30897516        2.33331211       -0.00000220
C          -0.16128410        0.56490823       -0.00000629
C           0.75838653       -0.48442428       -0.00000450
C           2.11600054       -0.19081593        0.00000433
H           0.45206236       -1.51753460       -0.00000538
N           2.61798215        1.03653268       -0.00001123
H           2.83205053       -1.00295181        0.00001161
C           1.73933795        2.02966975        0.00000429
C           0.36087625        1.85885465       -0.00000450
H           2.14739911        3.03255902        0.00001156
H          -0.26909246        2.73308097       -0.00000537"""

# CH2NH2+: CASCI(8e,5o)/cc-pVTZ S0-optimised geometry, from
# test/CH2NH2/ccpvtz/casci/outputs/ch2nh2_casci_S0_opt_5o8e_ccpvtz/
#   ch2nh2_casci_S0_opt_5o8e_ccpvtz_optimized.xyz
CH2NH2_ATOMS = """N           0.59255061       -0.00001238       -0.00004047
C          -0.67410400       -0.00005196       -0.00022958
H           1.11801778        0.21864286        0.82486620
H           1.11847219       -0.21860303       -0.82467092
H          -1.20415283        0.23914922        0.90215389
H          -1.20495876       -0.23912870       -0.90217914"""

# CH2NH2+ (2e,2o) S0/S1 MECI, from
# test/ch2nh2/gbci/outputs/ch2nh2_gbci_S0S1_meci_2o2e_groupN_ccpvtz/
#   ch2nh2_gbci_S0S1_meci_2o2e_groupN_ccpvtz_optimized.xyz
# Unlike the (8e,5o) entry above, this one is taken at the intersection rather
# than at the S0 minimum: MO8 and MO9 are the N- and C-localized p orbitals only
# near the seam, and that is what the figure has to show.
CH2NH2_MECI_2O2E_ATOMS = """N  0.72008165 -0.00836083 -0.00494257
C -0.74978247 -0.02238020 -0.01127616
H  1.25805170  0.71973138  0.44255403
H  1.23941784 -0.74638414 -0.46016712
H -1.16884282 -0.32366550  0.92068559
H -1.17157688  0.67354229 -0.69881478"""

# 3-hydroxyflavone: B3LYP/6-31G(d) S0 minimum, from
# test/3HF/dft/outputs/3hf_dft_S0_opt.xyz (no imaginary frequencies).
HF3_ATOMS = """O -1.66557168 1.28041179 3.09806724
C -1.35864134 0.82956055 1.98137939
C 0.05398841 0.58682730 1.69471563
O 0.87455695 0.89837760 2.72438015
C 0.48347300 0.08976705 0.49295905
C 1.85524232 -0.20824449 0.06189313
C 2.08690585 -0.73332029 -1.22593008
C 3.37905304 -1.02394063 -1.65351822
C 4.46953144 -0.79874402 -0.81049584
C 4.25204805 -0.27879590 0.46668772
C 2.96286313 0.01558116 0.90523084
O -0.44533851 -0.19040719 -0.48323670
C -1.77631417 0.00730831 -0.29471728
C -2.62428446 -0.31654733 -1.36285660
C -3.99054052 -0.13022298 -1.21347847
C -4.52507800 0.37544519 -0.01224638
C -3.68127232 0.69297647 1.03796861
C -2.29040842 0.51232799 0.90974291
H 0.24434488 1.21884758 3.41510048
H 1.24892193 -0.91204147 -1.88863668
H 3.53390744 -1.42841860 -2.65018932
H 5.47770393 -1.02616703 -1.14598021
H 5.09249306 -0.09895316 1.13158798
H 2.81054264 0.41766010 1.89740807
H -2.19524864 -0.70493405 -2.28070448
H -4.65441322 -0.37898293 -2.03664113
H -5.59747014 0.51445352 0.08575575
H -4.05694913 1.08454874 1.97785486"""

# The chromone acceptor: carbonyl O, the pyranone ring, the fused benzo ring and
# its hydrogens. The 2-phenyl ring and the 3-OH stay outside, so the group
# boundary straddles the phenyl -> chromone charge transfer that S1 carries.
HF3_ACCEPTOR = [0, 1, 2, 4, 11, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27]

SYSTEMS = {
    "lih_2e2o": li_halide(
        "H", 2, (1, 1), (2, 3),
        "2: sigma(Li2s/2pz-H1s), 3: sigma*(Li3s/3pz), same as the default window"),
    "lif_4e4o": li_halide(
        "F", 4, (2, 2), (3, 4, 7, 10),
        "3: sigma(F2s), 4: sigma(F2pz-Li), 7: sigma*(Li3s), 10: sigma*(Li3pz)"),
    "licl_4e4o": li_halide(
        "Cl", 4, (2, 2), (7, 8, 11, 14),
        "7: sigma(Cl3s), 8: sigma(Cl3pz-Li), 11: sigma*(Li3s), 14: sigma*(Li3pz)"),
    "pp_8e6o": dict(atom=PP_ATOMS, basis="ccpvdz", charge=0, spin=0,
                    ncas=6, nelecas=(4, 4), act_list=[35, 36, 37, 38, 39, 40],
                    note="CASCI(8e,6o)/cc-pVDZ S0-optimised geometry"),
    # The (2e,2o) space of test/pp/gbci/pp_gbci_S1.py and test/pp/casci/pp_casci_S1.py,
    # shown at the same S0 geometry as pp_8e6o so the two spaces are comparable.
    "pp_2e2o": dict(atom=PP_ATOMS, basis="ccpvdz", charge=0, spin=0,
                    ncas=2, nelecas=(1, 1), act_list=[38, 39],
                    note="CAS(2e,2o) of pp_gbci_S1.py, at the CASCI S0-optimised geometry"),
    "3hf_4e4o": dict(atom=HF3_ATOMS, basis="6-31G", charge=0, spin=0,
                      ncas=4, nelecas=(2, 2), act_list=[61, 62, 63, 64],
                      note="GBCI(4e,4o)/6-31G frontier pi/pi* window, at the "
                           "B3LYP/6-31G(d) S0 minimum"),
    "ch2nh2_8e5o": dict(atom=CH2NH2_ATOMS, basis="cc-pVTZ", charge=1, spin=0,
                        ncas=5, nelecas=(4, 4), act_list=[5, 6, 7, 8, 9],
                        note="CASCI(8e,5o)/cc-pVTZ S0-optimised geometry"),
    "ch2nh2_2e2o": dict(atom=CH2NH2_MECI_2O2E_ATOMS, basis="cc-pVTZ", charge=1,
                        spin=0, ncas=2, nelecas=(1, 1), act_list=[8, 9],
                        note="GBCI(2e,2o)/cc-pVTZ S0/S1 MECI geometry"),
}


def plane_normal(coords: np.ndarray) -> np.ndarray | None:
    """Unit normal of the best-fit plane, or None when the system is linear."""
    centred = coords - coords.mean(axis=0)
    if centred.shape[0] < 3:
        return None
    singular = scipy.linalg.svd(centred, compute_uv=False)
    if singular[-1] / singular[0] > 0.5:
        return None
    return scipy.linalg.svd(centred)[2][-1]


def out_of_plane_fraction(mol: gto.Mole, orthogonal_mo: np.ndarray,
                          normal: np.ndarray | None) -> float:
    """Fraction of the Lowdin density on p functions along the plane normal."""
    if normal is None:
        return float("nan")
    axes = {"x": 0, "y": 1, "z": 2}
    weight = orthogonal_mo ** 2
    weight = weight / weight.sum()
    total = 0.0
    for i, label in enumerate(mol.ao_labels()):
        token = label.split()[-1]
        if len(token) >= 3 and token[1] == "p" and token[2] in axes:
            direction = np.zeros(3)
            direction[axes[token[2]]] = 1.0
            total += weight[i] * float(direction @ normal) ** 2
    return total


def build_system(name: str, spec: dict, verbose: int = 0, atom=None):
    """Rebuild one SYSTEMS entry and put its active orbitals in window order.

    Returns ``(mol, mf, mo, active, window, energy, occ)``: ``active`` are the
    columns of ``mo`` holding the active orbitals, ``window`` the 1-based RHF MO
    numbers they came from, and ``energy``/``occ`` those orbitals' RHF values in
    window order.  ``atom`` overrides ``spec["atom"]``, which is how the figure
    renderer feeds back a reoriented copy of the same geometry.

    ``energy`` and ``occ`` are indexed by ``window``, not by the columns of ``mo``:
    ``sort_mo`` moves orbitals but leaves ``mf.mo_energy`` in the original RHF
    order, so reading ``mf.mo_energy[active]`` reports whichever orbitals happened
    to sit in the active columns beforehand.  For LiF that is the pair of
    degenerate F 2p pi lone pairs rather than the sigma orbitals actually selected.
    """
    mol = gto.M(atom=spec["atom"] if atom is None else atom,
                basis=spec["basis"], charge=spec["charge"],
                spin=spec["spin"], unit="Angstrom", symmetry=False, verbose=verbose)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-8, 800
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{name}: RHF did not converge")

    ncas, nelecas = spec["ncas"], spec["nelecas"]
    ncore = (mol.nelectron - sum(nelecas)) // 2
    if spec["act_list"] is None:
        mo = mf.mo_coeff
        window = list(range(ncore + 1, ncore + ncas + 1))
    else:
        probe = type("CAS", (), {"ncore": ncore, "ncas": ncas, "mol": mol})()
        mo = sort_mo(probe, mf.mo_coeff, spec["act_list"], base=1)
        window = list(spec["act_list"])

    active = list(range(ncore, ncore + ncas))
    original = [index - 1 for index in window]
    return (mol, mf, mo, active, window,
            mf.mo_energy[original], mf.mo_occ[original])


def export(name: str, spec: dict, out_dir: Path, verbose: int) -> None:
    mol, mf, mo, active, window, energy, occ = build_system(name, spec, verbose)
    ncas, nelecas = spec["ncas"], spec["nelecas"]
    ncore = active[0]
    normal = plane_normal(mol.atom_coords())
    overlap_half = scipy.linalg.sqrtm(mf.get_ovlp()).real
    orthogonal = overlap_half @ mo
    labels = mol.ao_labels()

    print(f"\n{name}   {mol.natm} atoms, {mol.nao} AOs, {mol.nelectron} electrons, "
          f"charge {spec['charge']}")
    print(f"  CAS({sum(nelecas)}e,{ncas}o)  ncore={ncore}  RHF MOs {window}")
    print(f"  {spec['note']}")
    print(f"  {'col':>4}{'e (Eh)':>11}{'occ':>5}{'pi(oop)':>9}   leading AOs")
    for column, k in enumerate(active):
        weight = orthogonal[:, k] ** 2
        weight = weight / weight.sum()
        top = np.argsort(weight)[::-1][:3]
        pi = out_of_plane_fraction(mol, orthogonal[:, k], normal)
        pi_text = "    n/a" if np.isnan(pi) else f"{pi:>9.3f}"
        print(f"  {column + 1:>4}{energy[column]:>11.5f}{occ[column]:>5.1f}{pi_text}   "
              + ", ".join(f"{labels[i]} ({weight[i]:.2f})" for i in top))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_active.molden"
    molden.from_mo(mol, str(path), mo[:, active], ene=energy, occ=occ)
    print(f"  -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(SYSTEMS),
                        help="comma-separated subset of " + ", ".join(SYSTEMS))
    parser.add_argument("--out-dir", type=Path,
                        default=SCRIPT_DIR / "reference" / "active_space_moldens")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    for name in args.systems.replace(",", " ").split():
        export(name, SYSTEMS[name], args.out_dir, args.verbose)


if __name__ == "__main__":
    main()
