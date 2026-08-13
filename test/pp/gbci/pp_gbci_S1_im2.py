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

from utils.active_space_tracking_optimizer import CASCI_Active_Root_Tracking_Optimizer, GBCI_Active_Root_Tracking_Optimizer

atoms = '''C          -3.82982877        0.63275106        0.00464528
C          -2.53920964        1.24425728        0.00456754
C          -3.60199675       -0.70944046        0.00464597
C          -2.18188051       -0.86081364        0.00456981
H          -4.76594101        1.16420457        0.00514324
H          -4.31029725       -1.52002832        0.00514463
N          -1.57921306        0.32434146        0.00346821
H          -1.61378850       -1.77431135        0.00153405
H          -2.30438178        2.29404036        0.00152903
C          -0.12707750        0.57082026       -0.05000000
C           0.80402415       -0.51138841       -0.00060195
C           2.13410385       -0.20053361        0.00968282
H           0.51117904       -1.55279152       -0.00518347
N           2.66314252        1.04412675        0.00497882
H           2.85312721       -1.01340300        0.01959430
C           1.75323787        2.04461950        0.00968943
C           0.39502111        1.89959720       -0.00058958
H           2.16398591        3.04913117        0.01960323
H          -0.22481973        2.78620063       -0.00518282'''

mol = gto.M(atom=atoms, basis='ccpvdz', charge=0, spin=0, verbose=4)
mf = scf.RHF(mol)

act_list = [38, 39]
opt_obj = GBCI_Active_Root_Tracking_Optimizer(
    ncas=2,
    nelecas=(1,1),
    act_list=act_list,
    nroots=3,
    groupA = {"mol1": [0,1,2,3,4,5,6,7,8], "mol2": [9,10,11,12,13,14,15,16,17,18]},
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
    
