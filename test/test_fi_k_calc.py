import logging
import unittest
import functools

import numpy as np
import numpy.testing as npt

import scipy.sparse as sps

import qutip

from qfi_local_noise import (
    DitstringSuperpositionKet,
    DitstringActionHamiltonian,
    QutipReducedOperatorCalculator,
    LocalKrausOperators,
    DitstringSuperpositionKetReducedPsiXiCalculator,

    compute_Fisher_information,

    compute_Eve_operators_leqk,
    compute_pinched_DF_lower_bound,
)




class Test_compute_Eve_operators_leqk(unittest.TestCase):

    def test_simple(self):

        # simple Bell state (|↑↑⟩ + |↓↓⟩)/√2
        dsk = DitstringSuperpositionKet(
            2, 2,
            np.array([1, 1])/np.sqrt(2),
            np.array([[0,0],
                      [1,1,]])
        )

        # Z+Z Hamiltonian
        daH = DitstringActionHamiltonian(
            dsk,
            np.array([2, -2]),
        )

        # Local Kraus operators -- say dephainsg
        p = 0.08
        lko = LocalKrausOperators(
            [np.sqrt(1-p/2)*qutip.qeye(2),
             np.sqrt(p/2)*qutip.sigmaz(),]
        )

        # full setting --
        dskrpxc = DitstringSuperpositionKetReducedPsiXiCalculator(
            dsk, daH, lko
        )

        (Nhatpsi, Nhatxipsi), _ = compute_Eve_operators_leqk(
            dskrpxc, lko,
            k=1,
        )

        print(f"{Nhatpsi=}\n{Nhatxipsi=}\n")

        # check -->
        E0, E1 = lko.local_kraus_operators[0], lko.local_kraus_operators[1]
        E00, E01, E10, E11 = qutip.tensor([E0,E0]), qutip.tensor([E0,E1]), \
            qutip.tensor([E1,E0]), qutip.tensor([E1,E1])
        print(f"{E00=}")
        H = qutip.tensor(qutip.sigmaz(), qutip.qeye(2)) + \
            qutip.tensor(qutip.qeye(2), qutip.sigmaz())
        psi, xi = (
            qutip.Qobj(dsk.to_sparse(), dims=[[2,2],[1,1]]), \
            H * qutip.Qobj(dsk.to_sparse(), dims=[[2,2],[1,1]])
        )
        print(f"{psi=}\n{xi=}")
        npt.assert_almost_equal( (xi.dag()*psi).full().item(), 0.0 )
        check_Nhatxipsi = np.array([
            [ (psi.dag()*E00.dag()*E00*xi).full().item(),
              (psi.dag()*E01.dag()*E00*xi).full().item(),
              (psi.dag()*E10.dag()*E00*xi).full().item(), ],
            [ (psi.dag()*E00.dag()*E01*xi).full().item(),
              (psi.dag()*E01.dag()*E01*xi).full().item(),
              (psi.dag()*E10.dag()*E01*xi).full().item(), ],
            [ (psi.dag()*E00.dag()*E10*xi).full().item(),
              (psi.dag()*E01.dag()*E10*xi).full().item(),
              (psi.dag()*E10.dag()*E10*xi).full().item(), ],
        ])
        check_Nhatpsi = np.array([
            [ (psi.dag()*E00.dag()*E00*psi).full().item(),
              (psi.dag()*E01.dag()*E00*psi).full().item(),
              (psi.dag()*E10.dag()*E00*psi).full().item(), ],
            [ (psi.dag()*E00.dag()*E01*psi).full().item(),
              (psi.dag()*E01.dag()*E01*psi).full().item(),
              (psi.dag()*E10.dag()*E01*psi).full().item(), ],
            [ (psi.dag()*E00.dag()*E10*psi).full().item(),
              (psi.dag()*E01.dag()*E10*psi).full().item(),
              (psi.dag()*E10.dag()*E10*psi).full().item(), ],
        ])

        npt.assert_almost_equal(Nhatxipsi, check_Nhatxipsi)
        npt.assert_almost_equal(Nhatpsi, check_Nhatpsi)
        


    def test_more(self):

        psi = qutip.Qobj(np.sqrt(np.array([[1/3], [1/6], [1/6], [1/3]])),
                         dims=[[2,2],[1,1]])
        H = qutip.Qobj(np.diag([-2, 0, 0, 2]), dims=[[2,2],[2,2]])

        print(f"{psi=}")

        xi = H * psi
        xi -= (psi.dag() * xi).tr() * psi   # .tr() to make it a scalar

        # Local Kraus operators
        p = 0.5
        lko = LocalKrausOperators([
            #    np.sqrt(1-p/2) * qutip.qeye(2),
            #    np.sqrt(p/2) * qutip.sigmaz()
            qutip.Qobj( np.array([[np.sqrt(1-p), 0],
                                  [0           , 1]]) ),
            qutip.Qobj( np.array([[0         , 0],
                                  [np.sqrt(p), 0]]) ),
        ])

        rpxc = QutipReducedOperatorCalculator(
            2, 2,
            [ psi * psi.dag(),  xi * psi.dag() ],
            lko
        )

        (Nhatpsi, Nhatxipsi), _ = compute_Eve_operators_leqk(
            rpxc, lko,
            k=1,
        )

        print(f"{Nhatpsi=}\n{Nhatxipsi=}\n")

        # check -->
        E0, E1 = lko.local_kraus_operators[0], lko.local_kraus_operators[1]
        E00, E01, E10, E11 = qutip.tensor([E0,E0]), qutip.tensor([E0,E1]), \
            qutip.tensor([E1,E0]), qutip.tensor([E1,E1])
        print(f"{E00=}")
        print(f"{psi=}\n{xi=}")
        npt.assert_almost_equal( (xi.dag()*psi).full().item(), 0.0 )
        check_Nhatxipsi = np.array([
            [ (psi.dag()*E00.dag()*E00*xi).full().item(),
              (psi.dag()*E01.dag()*E00*xi).full().item(),
              (psi.dag()*E10.dag()*E00*xi).full().item(), ],
            [ (psi.dag()*E00.dag()*E01*xi).full().item(),
              (psi.dag()*E01.dag()*E01*xi).full().item(),
              (psi.dag()*E10.dag()*E01*xi).full().item(), ],
            [ (psi.dag()*E00.dag()*E10*xi).full().item(),
              (psi.dag()*E01.dag()*E10*xi).full().item(),
              (psi.dag()*E10.dag()*E10*xi).full().item(), ],
        ])
        check_Nhatpsi = np.array([
            [ (psi.dag()*E00.dag()*E00*psi).full().item(),
              (psi.dag()*E01.dag()*E00*psi).full().item(),
              (psi.dag()*E10.dag()*E00*psi).full().item(), ],
            [ (psi.dag()*E00.dag()*E01*psi).full().item(),
              (psi.dag()*E01.dag()*E01*psi).full().item(),
              (psi.dag()*E10.dag()*E01*psi).full().item(), ],
            [ (psi.dag()*E00.dag()*E10*psi).full().item(),
              (psi.dag()*E01.dag()*E10*psi).full().item(),
              (psi.dag()*E10.dag()*E10*psi).full().item(), ],
        ])
        print(f"{check_Nhatpsi=}\n{check_Nhatxipsi=}")

        npt.assert_almost_equal(Nhatpsi, check_Nhatpsi)
        npt.assert_almost_equal(Nhatxipsi, check_Nhatxipsi)
        


    def test_consistency_with_pinched_lower_bound(self):

        # GHZ state (|↑↑↑↑⟩ + |↓↓↓↓⟩)/√2
        dsk = DitstringSuperpositionKet(
            4, 2,
            np.array([1, 1])/np.sqrt(2),
            np.array([[0,0,0,0],
                      [1,1,1,1,]])
        )

        # Z+Z+Z+Z Hamiltonian
        daH = DitstringActionHamiltonian(
            dsk,
            np.array([4, -4]),
        )

        # Local Kraus operators -- amplitude damping
        p = 0.05
        lko = LocalKrausOperators(
            [qutip.Qobj([[np.sqrt(1-p),0],[0,1]]),
             qutip.Qobj([[0,0],[np.sqrt(p),0]])]
        )

        # full setting --
        dskrpxc = DitstringSuperpositionKetReducedPsiXiCalculator(
            dsk, daH, lko
        )

        (Nhatpsi, Nhatxipsi), _ = compute_Eve_operators_leqk(
            dskrpxc, lko,
            k=2,
        )
        
        print(f"{Nhatpsi=}\n{Nhatxipsi=}\n")

        # now dephase operators in the computational basis
        rho_deph_E = np.diag(np.diag(Nhatpsi))
        D_deph_E = np.diag(np.diag(Nhatxipsi))
        D_deph_E = D_deph_E + D_deph_E.conj().T

        DFlb_deph_lb, _ = compute_Fisher_information(rho_deph_E, D_deph_E)

        # compute the bound --
        DFlb_check, _ = compute_pinched_DF_lower_bound(dskrpxc, lko, k=2)
        
        npt.assert_almost_equal(DFlb_deph_lb, DFlb_check)










class Test_compute_pinched_DF_lower_bound(unittest.TestCase):
    def test_simple(self):

        # GHZ state (|↑↑↑↑⟩ + |↓↓↓↓⟩)/√2
        dsk = DitstringSuperpositionKet(
            4, 2,
            np.array([1, 1])/np.sqrt(2),
            np.array([[0,0,0,0],
                      [1,1,1,1,]])
        )

        # Z+Z+Z+Z Hamiltonian
        daH = DitstringActionHamiltonian(
            dsk,
            np.array([4, -4]),
        )

        # Local Kraus operators -- amplitude damping
        p = 0.05
        lko = LocalKrausOperators(
            [qutip.Qobj([[np.sqrt(1-p),0],[0,1]]),
             qutip.Qobj([[0,0],[np.sqrt(p),0]])]
        )

        # full setting --
        dskrpxc = DitstringSuperpositionKetReducedPsiXiCalculator(
            dsk, daH, lko
        )

        # compute the bound --
        DFlb, _ = compute_pinched_DF_lower_bound(dskrpxc, lko, k=2)

        variance4 = 4 * daH.variance()

        FBob = variance4 - DFlb

        print(f"{DFlb=}")
        print(f"{variance4=}")
        print(f"{FBob=}")
        # From earlier runs, we expect:
        # DFlb=6.527206287523119
        # variance4=63.999999999999986
        # FBob=57.472793712476864

        npt.assert_almost_equal(
            [DFlb, variance4, FBob],
            [ 6.527206287523119, 64.0, 57.472793712476864 ]
        )
              





if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('qfi_local_noise._states_desc.validate').setLevel(logging.INFO)
    unittest.main()
