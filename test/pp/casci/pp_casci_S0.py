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

atoms = '''C       -3.7772609754      0.6647190018     -0.0000000000                 
C       -2.5473133480      1.2860706513     -0.0000000000                 
C       -3.5419182539     -0.7221696269      0.0000000000                 
C       -2.1758474284     -0.9029999721      0.0000000000                 
H       -4.7374136571      1.1646584389      0.0000000000                 
H       -4.2834016562     -1.5108615660     -0.0000000000                 
N       -1.5463455736      0.3298734774     -0.0000000000                 
H       -1.6190866486     -1.8276702023     -0.0000000000                 
H       -2.3267523378      2.3426455096      0.0000000000                 
C       -0.1372944494      0.5689770878      0.0000000000                 
C        0.7695245151     -0.4955231321     -0.0000000000                 
C        2.1269761070     -0.2096674619      0.0000000000                 
H        0.4682765225     -1.5351146814     -0.0000000000                 
N        2.6339687532      1.0392364233     -0.0000000000                 
H        2.8650309910     -1.0072103237     -0.0000000000                 
C        1.7433638531      2.0509801209      0.0000000000                 
C        0.3676011407      1.8730330034     -0.0000000000                 
H        2.1770072333      3.0473500550     -0.0000000000                 
H       -0.2597276469      2.7550531371     -0.0000000000  '''

mol = gto.M(atom=atoms, basis='ccpvdz', charge=0, spin=0, verbose=4)
mf = scf.RHF(mol)

act_list = [35, 36, 37, 38, 39, 40]
opt_obj = CASCI_Active_Root_Tracking_Optimizer(
    ncas=6,
    nelecas=(4,4),
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
    
