from pyscf.sfnoci.sfnoci import str2occ
from pyscf.sfnoci.sfnoci import group_info_list
from pyscf.sfnoci.sfnoci import SFNOCI
from pyscf import lib

from GBCI_grad import Gradients
from GBCI_grad import optimize_mo

from pyscf import scf, gto
import matplotlib.pyplot as plt
import pandas as pd

import unittest

def test_function(atom, delta = 1e-5):
    assert atom in ('H', 'F', 'Cl')
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['Li', (0,0,0)], [atom,(0,0,1.2 - delta)]]
    mol.basis = 'ccpvdz'
    mol.build()
    mol.set_common_orig([0,0,0])
    mf = scf.RHF(mol)
    mf.kernel()
    mo_coeff = mf.mo_coeff

    from pyscf.mcscf import addons
    mySFNOCI = SFNOCI(mf, 2, (1,1))
    mySFNOCI.fcisolver.conv_tol = 1e-10
    sfnoci_grad = Gradients(mySFNOCI)
    mySFNOCI.mo_coeff = mo_coeff
    mo = mo_coeff
    mo_list, po_list, group = mySFNOCI.optimize_mo(mo, debug = False)
    p = mo_list.shape[0]
    # for i in range(p):
    #     moe_list[i] = mf.mo_energy
    dmet_core_list, ov_list = mySFNOCI.get_svd_matrices(mo_list, po_list)
    dmet_act_list = mySFNOCI.get_active_dm(mo)
    h1e, ecore_list = mySFNOCI.get_h1cas(dmet_act_list , mo_list , dmet_core_list)
    eri = mySFNOCI.get_h2eff(mo)

    ncas = mySFNOCI.ncas
    nelecas = mySFNOCI.nelecas
    conf_info_list = group_info_list(ncas, nelecas, po_list, group)

    e_tot, fcivec = mySFNOCI.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                            conf_info_list, ov_list, ecore_list,
                                            ci0=None, verbose=mol.verbose)

    from pyscf.mcscf.casci import CASCI
    mycas = CASCI(mf, 2, 2)
    mycas.kernel()
    e_cas = mycas.e_tot
    
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['Li', (0,0,0)], [atom,(0,0,1.2 + delta)]]
    mol.basis = 'ccpvdz'
    mol.build()
    mol.set_common_orig([0,0,0])

    mf = scf.RHF(mol)
    mf.kernel()
    mo_coeff = mf.mo_coeff
    mySFNOCI = SFNOCI(mf, 2, (1,1))
    mySFNOCI.fcisolver.conv_tol = 1e-10
    sfnoci_grad = Gradients(mySFNOCI)
    mySFNOCI.mo_coeff = mo_coeff
    mo = mo_coeff
    mo_list, po_list, group = mySFNOCI.optimize_mo(mo, debug = False)
    p = mo_list.shape[0]

    dmet_core_list, ov_list = mySFNOCI.get_svd_matrices(mo_list, po_list)
    dmet_act_list = mySFNOCI.get_active_dm(mo)
    h1e, ecore_list = mySFNOCI.get_h1cas(dmet_act_list , mo_list , dmet_core_list)
    eri = mySFNOCI.get_h2eff(mo)
    ncas = mySFNOCI.ncas
    nelecas = mySFNOCI.nelecas
    conf_info_list = group_info_list(ncas, nelecas, po_list, group)
    e_new, fcivec = mySFNOCI.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                            conf_info_list, ov_list, ecore_list,
                                            ci0=None, verbose=mol.verbose)

    mycas = CASCI(mf, 2, 2)
    mycas.kernel()
    e_cas_new = mycas.e_tot

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = [['Li', (0,0,0)], [atom,(0,0,1.2)]]
    mol.basis = 'ccpvdz'
    mol.build()
    mol.set_common_orig([0,0,0])

    mf = scf.RHF(mol)

    mf.kernel()
    mo_coeff = mf.mo_coeff

    from pyscf.mcscf import addons
    mySFNOCI = SFNOCI(mf, 2, (1,1))
    mySFNOCI.fcisolver.conv_tol = 1e-10
    sfnoci_grad = Gradients(mySFNOCI)
    mySFNOCI.mo_coeff = mo_coeff
    mo = mo_coeff
    mo_list, moe_list, po_list, group = optimize_mo(mySFNOCI, mo, debug = False)
    p = mo_list.shape[0]
    dmet_core_list, ov_list = mySFNOCI.get_svd_matrices(mo_list, po_list)
    dmet_act_list = mySFNOCI.get_active_dm(mo)
    h1e, ecore_list = mySFNOCI.get_h1cas(dmet_act_list , mo_list , dmet_core_list)
    

    eri = mySFNOCI.get_h2eff(mo)

    ncas = mySFNOCI.ncas
    nelecas = mySFNOCI.nelecas
    conf_info_list = group_info_list(ncas, nelecas, po_list, group)

    e, fcivec = mySFNOCI.fcisolver.kernel(h1e, eri, ncas, nelecas,
                                            conf_info_list, ov_list, ecore_list,
                                            ci0=None, verbose=mol.verbose)
    
    from pyscf.tools import molden
    molden.from_mo(mol,'debug_grad.molden',mo_coeff)
    from pyscf.mcscf.casci import CASCI
    mycas = CASCI(mf, 2, 2)
    mycas.kernel()
    # e_cas = mycas.
    ci = fcivec
    moe_list = None
    de = sfnoci_grad.kernel(mo_coeff, mf.mo_energy, mo_list, moe_list, conf_info_list, dmet_core_list, ov_list, ecore_list, ci)
   
    from pyscf.grad.casci import Gradients as CASCI_Gradients
    mycas_grad = CASCI_Gradients(mycas)
    de_cas = mycas_grad.kernel()
    ANG2BOHR = 1.0 / lib.param.BOHR

    print("GBCI analytic gradient :")
    print(de[1][2])
    print("CASCI analytic gradient :")
    print(de_cas[1][2])

    gbci_num_grad = (e_new - e_tot)/(2*delta * ANG2BOHR)
    casci_num_grad = (e_cas_new - e_cas)/(2*delta * ANG2BOHR)
    print("GBCI numerical gradient :")
    print(gbci_num_grad)
    print("CASCI numerical gradient :")
    print(casci_num_grad)

    return de[1][2], de_cas[1][2], gbci_num_grad, casci_num_grad

class KnownValues(unittest.TestCase):
    def test_gbci_grad_LiH(self):
        gbci_an, _, gbci_num, _ = test_function('H')
        #value from a not-yet-debugged version
        self.assertAlmostEqual(gbci_an, -0.1138835547, 6)
        self.assertAlmostEqual(gbci_num, -0.11384433427508966, 6)

    def test_gbci_grad_LiF(self):
        gbci_an, _, gbci_num, _ = test_function('F')
        #value from a not-yet-debugged version
        self.assertAlmostEqual(gbci_an, -0.4164488739, 6)
        self.assertAlmostEqual(gbci_num, -0.4164489295031934, 6)

    def test_gbci_grad_LiCl(self):
        gbci_an, _,  gbci_num, _ = test_function('Cl')
        #value from a not-yet-debugged version
        self.assertAlmostEqual(gbci_an, -1.0641237911, 6)
        self.assertAlmostEqual(gbci_num, -1.0642374166125366, 6)
 
if __name__ == "__main__":
    print("Full Tests for GBCI")
    unittest.main()