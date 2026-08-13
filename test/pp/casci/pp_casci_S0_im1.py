from pyscf import gto, scf, mcscf
import numpy as np
import sys
from pathlib import Path

from pyscf.geomopt.geometric_solver import optimize
from pyscf.geomopt.addons import as_pyscf_method
from time import time

REPO_ROOT = Path(__file__).resolve().parents[3]
repo_root = str(REPO_ROOT)
if repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from utils.active_space_tracking_optimizer import CASCI_Active_Root_Tracking_Optimizer

atoms = '''C          -3.75064471        0.68165389       -0.01611486
C          -2.54053453        1.28506017       -0.02511920
C          -3.51115658       -0.72936216        0.01617516
C          -2.16974693       -0.89984701        0.02515201
H          -4.70409602        1.18347797       -0.03210566
H          -4.24568095       -1.51763616        0.03218586
N          -1.55818487        0.32786460        0.00001066
H          -1.60531315       -1.81301431        0.04999777
H          -2.30897548        2.33331325       -0.04997538
C          -0.16128349        0.56490468       -0.00000662
C           0.75838878       -0.48442672       -0.01865190
C           2.11600131       -0.19081497       -0.01618676
H           0.45206714       -1.51753828       -0.03605368
N           2.61798187        1.03653436       -0.00004761
H           2.83205408       -1.00294784       -0.02941292
C           1.73933691        2.02967040        0.01614456
C           0.36087493        1.85885159        0.01862858
H           2.14739444        3.03256159        0.02937073
H          -0.26909576        2.73307591        0.03604189'''

mol = gto.M(atom=atoms, basis='ccpvdz', charge=0, spin=0, verbose=4)
mf = scf.RHF(mol)

act_list = [38, 39]
opt_obj = CASCI_Active_Root_Tracking_Optimizer(
    ncas=2,
    nelecas=(1,1),
    act_list=act_list,
    nroots=3,
    target_root=0,
    conv_tol=1e-10,
    max_cycle=200,
    act_base=1,
    )

fake_opt = as_pyscf_method(mol, opt_obj)

geom_params = {
    # MECI penalty surface can be stiff; smaller trust radius is safer.
    "trust": 0.02,
    "tmax": 0.03,

    # Start moderately loose. Refine later if needed.
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

print("\n===== Final ground state optimized geometry =====")
print(mol_opt.tostring(format="xyz"))

from utils.hessian import numerical_hessian_from_gradient_scanner, imaginary_mode_to_displaced_mols

hess4, hess2 = numerical_hessian_from_gradient_scanner(
    opt_obj,
    mol_opt,
    step=1e-3,
)

freqs_cm, _, _, _,mol_plus, mol_minus = imaginary_mode_to_displaced_mols(
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
    
