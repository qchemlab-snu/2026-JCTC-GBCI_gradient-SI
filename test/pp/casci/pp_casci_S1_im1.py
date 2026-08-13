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


atoms = """C          -3.80228876        0.61579381       -0.02913774
C          -2.47941760        1.25831576       -0.05334364
C          -3.57182724       -0.72202723        0.03127991
C          -2.17774460       -0.89191113        0.05244420
H          -4.73718346        1.14808426       -0.05978161
H          -4.28899171       -1.52296464        0.06323977
N          -1.53470434        0.38250343       -0.00388060
H          -1.59957335       -1.79148088        0.11715595
H          -2.28777024        2.31302764       -0.11753059
C          -0.16511602        0.57721172        0.00108760
C           0.76536256       -0.50575026       -0.04054423
C           2.11140892       -0.18814914       -0.04214333
H           0.47341786       -1.54231883       -0.08662530
N           2.63322652        1.02576893       -0.00116995
H           2.82556570       -1.00525041       -0.07860386
C           1.73564772        2.03166300        0.04179246
C           0.38165070        1.89196905        0.04336317
H           2.15700832        3.02998269        0.08048656
H          -0.23418539        2.77682494        0.09452133"""

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

print("\n===== Final S1 im1 optimized geometry =====")
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
