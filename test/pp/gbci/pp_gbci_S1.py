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

atoms = '''C          -3.74277916        0.68300914       -0.00000000
C          -2.53434166        1.29405070        0.00000000
C          -3.50313646       -0.72793393       -0.00000000
C          -2.16078760       -0.90633919       -0.00000000
H          -4.69809009        1.18130123        0.00000000
H          -4.24054134       -1.51352758       -0.00000000
N          -1.54283098        0.33045911       -0.00000000
H          -1.59821403       -1.81998326       -0.00000000
H          -2.30464184        2.34213309        0.00000000
C          -0.16584886        0.56419685        0.00000000
C           0.75378635       -0.48400386        0.00000000
C           2.10568061       -0.18658172        0.00000000
H           0.44688705       -1.51636124        0.00000000
N           2.60807481        1.03480047        0.00000000
H           2.82251096       -0.99891916        0.00000000
C           1.73094162        2.02213037       -0.00000000
C           0.35659599        1.85706993       -0.00000000
H           2.13973853        3.02543900       -0.00000000
H          -0.27361675        2.73043999       -0.00000000 '''

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

freqs_cm, mol_plus, mol_minus = imaginary_mode_to_displaced_mols(
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
    
