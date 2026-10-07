from pyscf import gto, scf

atoms = '''S       -2.5678869711     -0.2820153630     -0.0000000000                 
O       -3.9257672725     -1.0696419605     -0.0000000000                 
O       -1.3123067681     -1.2242005462      0.0000000000     
'''

mol = gto.M(atom=atoms, basis='ccpvdz')
mf = scf.RHF(mol)
mf.kernel()

from pyscf.tools import molden
molden.from_scf(mf, 'so2_RHF.molden')