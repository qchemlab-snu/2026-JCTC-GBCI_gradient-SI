from pyscf import gto, scf
import sys
from pathlib import Path
from pyscf.geomopt.addons import as_pyscf_method
from pyscf.geomopt.geometric_solver import optimize

REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root = str(REPO_ROOT)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from utils.active_space_tracking_optimizer import CASCI_Active_Root_Tracking_Optimizer
from utils.hessian import (
    imaginary_mode_to_displaced_mols,
    numerical_hessian_from_gradient_scanner,
)


atoms = """C          -3.80553899        0.65455830       -0.03120472
C          -2.54540190        1.27490231       -0.05241133
C          -3.58171572       -0.68430060        0.02921156
C          -2.12090935       -0.85431945        0.05338503
H          -4.74676350        1.17409662       -0.06314108
H          -4.28859503       -1.49527257        0.05988009
N          -1.51806168        0.28420363        0.00387647
H          -1.59197767       -1.78672155        0.11758369
H          -2.29648093        2.31488396       -0.11709883
C          -0.16080513        0.55215703       -0.00115892
C           0.78895303       -0.50870300       -0.04342246
C           2.11327681       -0.19399273       -0.04182399
H           0.49944719       -1.54716664       -0.09461350
N           2.62886993        1.05161835        0.00115726
H           2.84031426       -0.99749308       -0.08050745
C           1.73584701        2.02551549        0.04211325
C           0.36038947        1.88138983        0.04050077
H           2.14054302        3.03244033        0.07858119
H          -0.25710244        2.76367297        0.08654812"""

mol = gto.M(atom=atoms, basis="ccpvdz", charge=0, spin=0, verbose=4)
mf = scf.RHF(mol)

act_list = [38, 39]
opt_obj = CASCI_Active_Root_Tracking_Optimizer(
    ncas=2,
    nelecas=(1, 1),
    act_list=act_list,
    nroots=3,
    target_root=1,
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

print("\n===== Final S1 im2 optimized geometry =====")
print(mol_opt.tostring(format="xyz"))

hess4, hess2 = numerical_hessian_from_gradient_scanner(
    opt_obj,
    mol_opt,
    step=1e-3,
)

freqs_cm, _, _, _, mol_plus, mol_minus = imaginary_mode_to_displaced_mols(
    mol=mol_opt,
    hess2=hess2,
    imag_threshold_cm=30.0,
    amplitude=0.05,
    amplitude_unit="Angstrom",
)

print("Frequencies / cm^-1:")
for i, freq in enumerate(freqs_cm):
    if freq < 0:
        print(f"mode {i:3d}: i{abs(freq):.2f}")
    else:
        print(f"mode {i:3d}:  {freq:.2f}")

if mol_plus is None:
    print("No imaginary frequency found.")
else:
    print("Imaginary frequency found. mol_plus and mol_minus were generated.")
    print("\n===== mol_plus =====")
    print(mol_plus.tostring(format="xyz"))
    print("\n===== mol_minus =====")
    print(mol_minus.tostring(format="xyz"))
