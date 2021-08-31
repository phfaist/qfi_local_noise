import logging
import unittest
import functools

import numpy as np
import numpy.testing as npt

import scipy.sparse as sps
import scipy.special as spa

import qutip

from qfi_local_noise import (
    LocalKrausOperators,
    DitstringSuperpositionKet,
    SymmetricKet,
    SymmetricSuperpositionKet,
    SymmetricOperator,
    SymmetricEnsembleKetBras,
    SymmetricReducedOperatorsCalculator,
    SymmetricTraceWithKrausPairStringCalculator,
    get_symm_state_basis_ditstrings,
)

# test internal function(s):
# from qfi_local_noise._symm_states_desc import (
#     reduced_k_symmetric_hp_hpp,
# )


# class Test_reduced_k_symmetric_hp_hpp(unittest.TestCase):
#     def test_big_ints(self):
#         res = reduced_k_symmetric_hp_hpp(10000, 1500, 800, 800)
#         print("RESULT --> ", res)
#         print("sum of all elments --> ", res.sum())


                                         

class Test_get_symm_state_comp_basis_ditstrings(unittest.TestCase):

    def test_simple(self):

        xns = get_symm_state_basis_ditstrings(4, [0,1,2])
        #print(xns)

        npt.assert_almost_equal(
            xns,
            np.array([
                [0,2,3,3],
                [0,3,2,3],
                [0,3,3,2],
                [2,0,3,3],
                [2,3,0,3],
                [2,3,3,0],
                [3,0,2,3],
                [3,0,3,2],
                [3,2,0,3],
                [3,2,3,0],
                [3,3,0,2],
                [3,3,2,0],
            ])
        )



class TestSymmetricKet(unittest.TestCase):

    def test_simple_0(self):

        symmket = SymmetricSuperpositionKet(
            4, 3, # 4 qutrits
            np.array([0, 0, 1]),
            np.array([
                [3,0], # 3 qutrits in the "1" state
                [1,2], # 1 qutrit in the "1" state, d28 in the "2" state
                [0,0], # ground state
            ]))

        #print(symmket.to_ditstring_superposition_ket())

        psi = symmket.to_ditstring_superposition_ket().to_sparse().toarray()
        #print(f"{psi=}")

        gnd = np.zeros(shape=(3**4,1))
        gnd[0] = 1

        npt.assert_almost_equal(
            psi,
            gnd
        )

    def test_simple_1(self):

        symmket = SymmetricSuperpositionKet(
            4, 3, # 4 qutrits
            np.array([1, 0, 0]),
            np.array([
                [3,0], # 3 qutrits in the "1" state
                [1,2], # 1 qutrit in the "1" state, d28 in the "2" state
                [0,0], # ground state
            ]))

        print(symmket.to_ditstring_superposition_ket())

        psi = symmket.to_ditstring_superposition_ket().to_sparse().toarray()
        #print(f"{psi=}")

        st = np.zeros(shape=(3**4,1))
        st[3**2+3**1+3**0] = 1
        st[3**3+3**1+3**0] = 1
        st[3**3+3**2+3**0] = 1
        st[3**3+3**2+3**1] = 1
        st /= np.sqrt(4)

        npt.assert_almost_equal(
            psi,
            st
        )









class TestSymmetricEnsembleKetBras(unittest.TestCase):
    def test_simple(self):

        coeffs = [0.5, 1]
        kets = np.array([
            [0, 1, 2, 4],
            [1, 2, 3, 5]
        ])
        
        op = SymmetricEnsembleKetBras(3, 2, coeffs, kets)

        result = op.overlap(SymmetricOperator(3, 2, np.diag(np.arange(4))))

        npt.assert_almost_equal(
            result,
            coeffs[0]*kets[0,:].conj().T @ np.diag(np.arange(4)) @ kets[0,:] +
            coeffs[1]*kets[1,:].conj().T @ np.diag(np.arange(4)) @ kets[1,:]
        )





class TestSymmetricReducedOperatorsCalculator(unittest.TestCase):
    def test_simple(self):
        n = 4
        symm_ops = [
            SymmetricOperator(n, 2, np.diag(np.arange(n+1))),
            SymmetricOperator(n, 2, np.arange((n+1)**2).reshape(n+1,n+1)),
            SymmetricOperator(
                n, 2,
                np.arange((n+1)**2).reshape(n+1,n+1)
                + 1j*np.arange(100,100+(n+1)**2).reshape(n+1,n+1),
            )
        ]

        q_ops = [
            qutip.Qobj(S.to_sparse(), dims=[[2]*n,[2]*n])
            for S in symm_ops
        ]

        lko = LocalKrausOperators([
            np.sqrt(np.array([[0.8, 0], [0, 0.5]])),
            np.sqrt(np.array([[0, 0.5], [0.2, 0]])),
        ])
        E0dagE0 = lko.local_kraus_operators[0].dag()*lko.local_kraus_operators[0]

        sroc = SymmetricReducedOperatorsCalculator(n, 2, symm_ops, lko)

        for which_sites in [(0,), (0,1,), (1,2,), (1,2,3)]:
            #print(f"{which_sites=}")
            red_ops = sroc.reduced_operators(which_sites)
            
            Kr = qutip.tensor(*[
                qutip.qeye(2) if i in which_sites else E0dagE0
                for i in range(n)
            ])

            #print(f"{Kr=}")

            for S, Q, R in zip(symm_ops, q_ops, red_ops):
                ared = R.to_sparse().toarray()
                bred = (Kr*Q).ptrace(which_sites).full()
                npt.assert_almost_equal(ared, bred)

    def test_large_system(self):
        n = 1000

        # GHZ on n qubits
        op1 = np.zeros((n+1,n+1,), dtype=complex)
        op1[0,0] = op1[n,n] = 1/2
        op1[n,0] = 1j/2
        op1[0,n] = -1j/2

        # |+>^{\otimes n}
        twopn = 2**n
        op2 = np.array([
            [ np.sqrt(spa.comb(n, k, exact=True)/twopn) ]
            for k in range(n+1)
        ])
        op2 = op2 @ op2.T
        
        symm_ops = [
            SymmetricOperator( n, 2, op1 ),
            SymmetricOperator( n, 2, op2 ),
        ]

        lko = LocalKrausOperators([
            np.sqrt(np.array([[1, 0], [0, 1j]])),
            #np.sqrt(np.array([[0, 0.5*1j], [0.2, 0]])),
        ])
        E0dagE0 = lko.local_kraus_operators[0].dag()*lko.local_kraus_operators[0]

        sroc = SymmetricReducedOperatorsCalculator(n, 2, symm_ops, lko)

        which_sites = (0,1,2,)
        
        #print(f"{which_sites=}")
        red_ops = sroc.reduced_operators(which_sites)

        #print("RESULT --> ", red_ops[0].get_symm_sparse())

        tgt0 = np.zeros((4,4,))
        tgt0[0,0] = tgt0[3,3] = 0.5
        npt.assert_almost_equal( red_ops[0].get_symm_sparse().toarray(), tgt0 )

        # |+><+|^{\otimes k} is a matrix uniformly filled with 1/(2^k) entries
        tgt1 = np.ones((8,8,))/8
        npt.assert_almost_equal( red_ops[1].to_sparse().toarray(), tgt1 )
                


class TestSymmetricTraceWithKrausPairStringCalculator(unittest.TestCase):

    def test_simple_with_GHZ(self):
        
        GHZi = SymmetricOperator(
            3, 2,
            np.array([
                [  1, 0, 0, 1j ],
                [  0, 0, 0, 0 ],
                [  0, 0, 0, 0 ],
                [ 1j, 0, 0, 1 ]
            ])/2
        )

        # lko = LocalKrausOperators([
        #     np.array([[1, 0],
        #               [0, 0.8]]),
        #     np.array([[0, 0.6],
        #               [0, 0]]),
        # ])
        lko = LocalKrausOperators([
            np.array([[1, 0],
                      [0, 0]]),
            np.array([[0, 0],
                      [0, 1]]),
        ])

        ocalc = SymmetricTraceWithKrausPairStringCalculator(lko, [(1,1), (1,1), (1,1)])
        
        the_trace = ocalc.calculate_trace(GHZi);
        print(f"{the_trace = }")

        GHZi_d = np.zeros((8,8), dtype=complex)
        GHZi_d[0,0] = 1
        GHZi_d[7,0] = 1j
        GHZi_d[0,7] = 1j
        GHZi_d[7,7] = 1
        GHZi_d /= 2
        GHZi_qobj = qutip.Qobj(GHZi_d, dims=[[2,2,2],[2,2,2]])

        EdE = lko.local_kraus_operators[1].dag() * lko.local_kraus_operators[1]
        EdE3 = qutip.tensor([EdE, EdE, EdE])

        npt.assert_almost_equal(
            the_trace,
            qutip.expect(GHZi_qobj, EdE3)
        )



    def test_more_compare_dense(self):
        
        A = np.arange(16).reshape(4,4)
        rhoA = A.conj().T @ A
        rhoA = rhoA / np.trace(rhoA)

        rhosymmA = SymmetricOperator(3, 2, rhoA)

        lko = LocalKrausOperators([
            np.array([[1, 0],
                      [0, np.sqrt(0.7)]]),
            np.array([[0, np.sqrt(0.3)],
                      [0, 0]]),
        ])
        E0, E1 = lko.local_kraus_operators

        ocalc = SymmetricTraceWithKrausPairStringCalculator(lko, [(0,1), (0,1), (1,0)])
        
        the_trace = ocalc.calculate_trace(rhosymmA);
        print(f"{the_trace = }")

        rhoA_full = qutip.Qobj(rhosymmA.to_sparse(), dims=[[2]*3,[2]*3])

        KRR = qutip.tensor([E1.dag()*E0, E1.dag()*E0, E0.dag()*E1])

        the_trace_check = (KRR * rhoA_full).tr()

        npt.assert_almost_equal(
            the_trace,
            the_trace_check,
        )


    def test_simple_compare_dense(self):
        
        psiA = np.sqrt(np.array([1/3,1/3,1/3,0]))[:,np.newaxis]
        rhoA = psiA @ psiA.conj().T
        print(f"{rhoA=}")

        rhosymmA = SymmetricOperator(3, 2, rhoA)

        lko = LocalKrausOperators([
            np.array([[1, 0],
                      [0, 0.8]]),
            np.array([[0, 0.6],
                      [0, 0]]),
        ])
        E0, E1 = lko.local_kraus_operators

        ocalc = SymmetricTraceWithKrausPairStringCalculator(lko, [(0,1), (0,1), (1,0)])
        
        the_trace = ocalc.calculate_trace(rhosymmA);
        print(f"{the_trace = }")

        rhoA_full = qutip.Qobj(rhosymmA.to_sparse(), dims=[[2]*3,[2]*3])
        print(f"{rhoA_full=}")

        KRR = qutip.tensor([E1.dag()*E0, E1.dag()*E0, E0.dag()*E1])

        the_trace_check = (KRR * rhoA_full).tr()

        npt.assert_almost_equal(
            the_trace,
            the_trace_check,
        )










if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
