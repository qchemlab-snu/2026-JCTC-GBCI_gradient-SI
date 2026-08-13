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


atoms = """C          -3.56908686        0.38850017       -0.72618231
C          -2.17759969        0.71497823       -0.81369582
C          -3.71087983       -0.36762545        0.39568644
C          -2.40073116       -0.48574760        0.96857388
H          -4.32778300        0.69630776       -1.42649481
H          -4.60838875       -0.80455689        0.80057895
N          -1.51714476        0.17268826        0.21848606
H          -2.07622655       -1.00061387        1.85823017
H          -1.64648870        1.29643628       -1.55061414
C          -0.05923924        0.28201197        0.48515887
C           0.78637909       -0.40952208       -0.45824355
C           2.00533781        0.12432041       -0.76333160
H           0.49993679       -1.37505496       -0.86003379
N           2.48078204        1.30420741       -0.33110595
H           2.67052448       -0.43338105       -1.41639672
C           1.64196652        1.99984282        0.45468508
C           0.39655898        1.60246120        0.84837223
H           2.01223642        2.96437426        0.79020897
H          -0.20076643        2.24175305        1.48887950"""

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

print("\n===== Final S1 optimized geometry from twisted initial structure =====")
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
