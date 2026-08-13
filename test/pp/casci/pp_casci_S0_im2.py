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

atoms = '''C          -3.75064410        0.68165137        0.01611813
C          -2.54053533        1.28505833        0.02511760
C          -3.51115633       -0.72936475       -0.01617191
C          -2.16974558       -0.89984849       -0.02515363
H          -4.70409509        1.18347661        0.03211517
H          -4.24568138       -1.51763776       -0.03217639
N          -1.55818454        0.32786268       -0.00001314
H          -1.60531301       -1.81301667       -0.05000223
H          -2.30897483        2.33331097        0.04997098
C          -0.16128470        0.56491177       -0.00000595
C           0.75838428       -0.48442183        0.01864290
C           2.11599977       -0.19081689        0.01619543
H           0.45205757       -1.51753092        0.03604292
N           2.61798243        1.03653100        0.00002516
H           2.83204697       -1.00295577        0.02943614
C           1.73933899        2.02966909       -0.01613597
C           0.36087756        1.85885770       -0.01863759
H           2.14740377        3.03255644       -0.02934761
H          -0.26908917        2.73308602       -0.03605263  '''

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
    
