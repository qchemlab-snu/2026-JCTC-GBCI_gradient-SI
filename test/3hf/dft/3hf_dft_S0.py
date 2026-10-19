from pathlib import Path
import json
import sys

import numpy as np

from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hessian import (
    numerical_hessian_from_gradient_scanner,
    imaginary_mode_to_displaced_mols,
)

# 3-hydroxyflavone, input geometry (Angstrom)
atoms = '''O    -1.739428    0.678846    3.144707
   C    -1.375437    0.461027    1.993262
   C     0.062476    0.240407    1.703980
   O     0.928316    0.283115    2.781943
   C     0.438669   -0.001389    0.434219
   C     1.841594   -0.246474    0.036338
   C     2.146936   -1.313479   -0.823366
   C     3.461929   -1.549140   -1.233291
   C     4.488629   -0.716619   -0.794383
   C     4.201425    0.351958    0.051724
   C     2.886109    0.587549    0.461299
   O    -0.445801   -0.069799   -0.636618
   C    -1.780803    0.139248   -0.406555
   C    -2.634566    0.077192   -1.507554
   C    -4.006065    0.279854   -1.337018
   C    -4.524033    0.543980   -0.067300
   C    -3.671034    0.606399    1.035219
   C    -2.299678    0.403005    0.856538
   H     0.353659    0.449343    3.559517
   H     1.359849   -1.976030   -1.176896
   H     3.682734   -2.382868   -1.894550
   H     5.511857   -0.899749   -1.111524
   H     4.999843    1.004468    0.395250
   H     2.685011    1.431095    1.118079
   H    -2.234387   -0.127429   -2.497362
   H    -4.672145    0.232076   -2.195632
   H    -5.592370    0.701378    0.062773
   H    -4.073290    0.812035    2.024785'''

# ---- calculation settings -------------------------------------------------
BASIS = '6-31g*'
XC = 'b3lyp'
CHARGE = 0
SPIN = 0
UNIT = 'Angstrom'
VERBOSE = 4

SCF_CONV_TOL = 1e-10
SCF_MAX_CYCLE = 200
GRIDS_LEVEL = 3

# "numerical": central-difference Hessian from the DFT gradient scanner
#              (6*natm gradient evaluations).
# "analytic":  PySCF analytic DFT Hessian. Same information, much cheaper.
HESSIAN_MODE = 'numerical'
HESSIAN_STEP = 1e-3   # Bohr, used only when HESSIAN_MODE == "numerical"

IMAG_THRESHOLD_CM = 30.0
DISPLACE_AMPLITUDE = 0.05

OUTPUT_DIR = Path('outputs')
RUN_NAME = '3hf_dft_S0'

geom_params = {
    "trust": 0.02,
    "tmax": 0.03,

    "convergence_energy": 1e-6,
    "convergence_grms": 3e-4,
    "convergence_gmax": 4.5e-4,
    "convergence_drms": 1.2e-3,
    "convergence_dmax": 1.8e-3,
}
# ---------------------------------------------------------------------------


def build_molecule(geometry):
    return gto.M(
        atom=geometry,
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=UNIT,
        verbose=VERBOSE,
    )


def make_mf(mol):
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = GRIDS_LEVEL
    mf.conv_tol = SCF_CONV_TOL
    mf.max_cycle = SCF_MAX_CYCLE
    return mf


def run_optimization(mol):
    mf = make_mf(mol)
    mol_opt = optimize(mf, maxsteps=500, **geom_params)

    mf_opt = make_mf(mol_opt)
    e_opt = mf_opt.kernel()
    return mol_opt, mf_opt, e_opt


def run_hessian(mol_opt, mf_opt):
    """Return the (3N, 3N) Cartesian Hessian at the optimized geometry."""
    natm = mol_opt.natm

    if HESSIAN_MODE == 'numerical':
        grad_scanner = make_mf(mol_opt).nuc_grad_method().as_scanner()
        _, hess2 = numerical_hessian_from_gradient_scanner(
            grad_scanner,
            mol_opt,
            step=HESSIAN_STEP,
        )
    elif HESSIAN_MODE == 'analytic':
        hess4 = mf_opt.Hessian().kernel()
        hess2 = np.asarray(hess4).transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
        hess2 = 0.5 * (hess2 + hess2.T)
    else:
        raise ValueError(f"unknown HESSIAN_MODE: {HESSIAN_MODE}")

    return hess2


def save_result(mol_opt, e_opt, freqs_cm, imag_freq_cm, mol_plus, mol_minus, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f'{RUN_NAME}_opt.xyz').write_text(mol_opt.tostring(format='xyz'))
    if mol_plus is not None:
        (output_dir / f'{RUN_NAME}_im1.xyz').write_text(mol_plus.tostring(format='xyz'))
        (output_dir / f'{RUN_NAME}_im2.xyz').write_text(mol_minus.tostring(format='xyz'))

    result = {
        'run_name': RUN_NAME,
        'script': Path(__file__).name,
        'molecule': '3-hydroxyflavone',
        'method': f'RKS/{XC}',
        'basis': BASIS,
        'charge': CHARGE,
        'spin': SPIN,
        'unit': UNIT,
        'grids_level': GRIDS_LEVEL,
        'scf_conv_tol': SCF_CONV_TOL,
        'geom_params': geom_params,
        'hessian_mode': HESSIAN_MODE,
        'hessian_step_bohr': HESSIAN_STEP if HESSIAN_MODE == 'numerical' else None,
        'energy_hartree': float(e_opt),
        'optimized_geometry_xyz': mol_opt.tostring(format='xyz'),
        'frequencies_cm': [float(f) for f in freqs_cm],
        'n_imaginary': int(sum(1 for f in freqs_cm if f < 0)),
        'imaginary_frequency_cm': None if imag_freq_cm is None else float(imag_freq_cm),
        'minimum_confirmed': imag_freq_cm is None,
    }
    (output_dir / f'{RUN_NAME}_result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    mol = build_molecule(atoms)
    mol_opt, mf_opt, e_opt = run_optimization(mol)

    print("\n===== Final ground state optimized geometry =====")
    print(mol_opt.tostring(format='xyz'))
    print(f"Final RKS/{XC}/{BASIS} energy: {e_opt:.10f} Hartree")

    hess2 = run_hessian(mol_opt, mf_opt)

    freqs_cm, _, imag_freq_cm, _, mol_plus, mol_minus = imaginary_mode_to_displaced_mols(
        mol=mol_opt,
        hess2=hess2,
        imag_threshold_cm=IMAG_THRESHOLD_CM,
        amplitude=DISPLACE_AMPLITUDE,
        amplitude_unit='Angstrom',
    )

    print(f"\n===== Frequencies / cm^-1 ({HESSIAN_MODE} Hessian) =====")
    for i, freq in enumerate(freqs_cm):
        if freq < 0:
            print(f"mode {i:3d}: i{abs(freq):.2f}")
        else:
            print(f"mode {i:3d}:  {freq:.2f}")

    if mol_plus is None:
        print(f"\nNo imaginary frequency above {IMAG_THRESHOLD_CM} cm^-1. "
              "The optimized geometry is a minimum.")
    else:
        print(f"\nImaginary frequency found: i{abs(imag_freq_cm):.2f} cm^-1. "
              "mol_plus and mol_minus were generated.")
        print("\n===== mol_plus =====")
        print(mol_plus.tostring(format='xyz'))
        print("\n===== mol_minus =====")
        print(mol_minus.tostring(format='xyz'))

    result = save_result(
        mol_opt, e_opt, freqs_cm, imag_freq_cm, mol_plus, mol_minus, OUTPUT_DIR
    )
    print(f"\nResults saved under {OUTPUT_DIR}/ (minimum_confirmed="
          f"{result['minimum_confirmed']}).")


if __name__ == "__main__":
    main()
