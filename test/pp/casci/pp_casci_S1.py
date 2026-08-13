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

atoms = '''C          -3.73143993        0.65188355       -0.21325648
C          -2.51817307        1.24685434       -0.32807972
C          -3.50289652       -0.69490677        0.21335648
C          -2.16127315       -0.85635017        0.32814287
H          -4.68168724        1.11637761       -0.41896743
H          -4.24671897       -1.44686974        0.41907899
N          -1.54245055        0.33052230       -0.00002689
H          -1.59175012       -1.70747153        0.65557748
H          -2.26130062        2.23819524       -0.65554408
C          -0.16244599        0.56467722       -0.00003783
C           0.74312415       -0.46184653       -0.25010711
C           2.09644794       -0.16766994       -0.23075471
H           0.41102441       -1.46229433       -0.48010942
N           2.59760033        1.03308530       -0.00002661
H           2.81355702       -0.95592807       -0.42551776
C           1.72840944        2.00130767        0.23070888
C           0.35380603        1.83246734        0.25004055
H           2.14530813        2.98201506        0.42549135
H          -0.28975431        2.66733240        0.48006407 '''

mol = gto.M(atom=atoms, basis='ccpvdz', charge=0, spin=0, verbose=4)
mf = scf.RHF(mol)

act_list = [38, 39]
opt_obj = CASCI_Active_Root_Tracking_Optimizer(
    ncas=2,
    nelecas=(1,1),
    act_list=act_list,
    nroots=3,
    target_root=1,
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

freqs_cm, _,_,_, mol_plus, mol_minus = imaginary_mode_to_displaced_mols(
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
