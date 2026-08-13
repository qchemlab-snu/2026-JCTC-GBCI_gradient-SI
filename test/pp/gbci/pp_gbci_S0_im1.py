from pathlib import Path
import sys

from pyscf import gto, scf
from pyscf.geomopt.addons import as_pyscf_method
from pyscf.geomopt.geometric_solver import optimize

REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root = str(REPO_ROOT)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from utils.active_space_tracking_optimizer import GBCI_Active_Root_Tracking_Optimizer
from utils.hessian import (
    imaginary_mode_to_displaced_mols,
    numerical_hessian_from_gradient_scanner,
)


atoms = """C          -3.74277916        0.68300914       -0.01527714
C          -2.53434166        1.29405070       -0.02371808
C          -3.50313646       -0.72793393        0.01546647
C          -2.16078760       -0.90633919        0.02375761
H          -4.69809009        1.18130123       -0.03056988
H          -4.24054134       -1.51352758        0.03085215
N          -1.54283098        0.33045911       -0.00002139
H          -1.59821403       -1.81998326        0.04999808
H          -2.30464184        2.34213309       -0.05000000
C          -0.16584886        0.56419685       -0.00003673
C           0.75378635       -0.48400386       -0.01765853
C           2.10568061       -0.18658172       -0.01567141
H           0.44688705       -1.51636124       -0.03585632
N           2.60807481        1.03480047       -0.00002873
H           2.82251096       -0.99891916       -0.02898598
C           1.73094162        2.02213037        0.01560406
C           0.35659599        1.85706993        0.01758082
H           2.13973853        3.02543900        0.02892225
H          -0.27361675        2.73043999        0.03577521"""

mol = gto.M(
    atom=atoms,
    basis="ccpvdz",
    charge=0,
    spin=0,
    unit="Angstrom",
    verbose=4,
)
mf = scf.RHF(mol)

act_list = [38, 39]
opt_obj = GBCI_Active_Root_Tracking_Optimizer(
    ncas=2,
    nelecas=(1, 1),
    act_list=act_list,
    nroots=3,
    groupA={
        "mol1": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "mol2": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    },
    target_root=0,
    conv_tol=1e-10,
    max_cycle=200,
    act_base=1,
)

fake_opt = as_pyscf_method(mol, opt_obj)

geom_params = {
    "trust": 0.02,
    "tmax": 0.03,
    "convergence_energy": 1e-6,
    "convergence_grms": 3e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}

mol_opt = optimize(
    fake_opt,
    maxsteps=500,
    **geom_params,
)

output_dir = Path(__file__).resolve().parent
opt_xyz_path = output_dir / "pp_gbci_S0_im1_optimized.xyz"
freq_npz_path = output_dir / "pp_gbci_S0_im1_freq_check.npz"

opt_xyz = mol_opt.tostring(format="xyz")
opt_xyz_path.write_text(opt_xyz, encoding="utf-8")

print("\n===== Final S0 optimized geometry from mol_plus =====")
print(opt_xyz)
print(f"Optimized geometry saved to: {opt_xyz_path}")

_, hess2 = numerical_hessian_from_gradient_scanner(
    opt_obj,
    mol_opt,
    step=1e-3,
)

freqs_cm, _, imag_freq_cm, _, mol_plus, mol_minus = imaginary_mode_to_displaced_mols(
    mol=mol_opt,
    hess2=hess2,
    imag_threshold_cm=30.0,
    amplitude=0.05,
    amplitude_unit="Angstrom",
    save_path=freq_npz_path,
)

print("Frequencies / cm^-1:")
for i, freq in enumerate(freqs_cm):
    if freq < 0:
        print(f"mode {i:3d}: i{abs(freq):.2f}")
    else:
        print(f"mode {i:3d}:  {freq:.2f}")

print(f"Frequency-check data saved to: {freq_npz_path}")

if mol_plus is None:
    print("No imaginary frequency found.")
else:
    print(f"Imaginary frequency found: i{abs(imag_freq_cm):.2f} cm^-1")
    print("mol_plus and mol_minus were generated for a possible follow-up.")
    print("\n===== mol_plus =====")
    print(mol_plus.tostring(format="xyz"))
    print("\n===== mol_minus =====")
    print(mol_minus.tostring(format="xyz"))
