from pyscf import gto, scf
from pyscf.tools import molden
from pyscf import lib

lib.num_threads(1)  # OpenMP threads for PySCF

# 원하는 basis로 바꿔 쓰면 됨
basis = "cc-pVTZ"

geometries = {
    "rhf_ccpvtz": """N        0.70394600      -0.00000300      -0.00001300
C       -0.70623000      -0.05849300      -0.03764600
H        1.21195200       0.75025000       0.48222400
H        1.27708200      -0.71446000      -0.45899200
H       -1.17965900      -0.35329900       0.89628300
H       -1.17974200       0.66848800      -0.69381700
""" }

def run_rhf_and_write_molden(name, atom):
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.charge = 1
    mol.spin = 0          # multiplicity = 2S + 1 = 1 -> spin = 2S = 0
    mol.basis = basis
    mol.verbose = 4
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    if not mf.converged:
        print(f"[WARNING] RHF did not converge for {name}")

    molden_file = f"{name}_RHF.molden"
    molden.from_scf(mf, molden_file)

    print(f"{name}")
    print(f"  RHF energy = {mf.e_tot:.12f} Ha")
    print(f"  Molden file written to: {molden_file}")


for name, atom in geometries.items():
    run_rhf_and_write_molden(name, atom)