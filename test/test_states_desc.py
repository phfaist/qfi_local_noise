import logging
import unittest
import functools
import itertools

import numpy as np
import numpy.testing as npt

import scipy.sparse as sps

import qutip

from qfi_local_noise import (
    DitstringSuperpositionKet,
    EnsembleKetBras,
    DitstringActionHamiltonian,
    LocalKrausOperators,
)


class TestDitstringSuperpositionKet(unittest.TestCase):
    def test_simple(self):
        
        ket = DitstringSuperpositionKet(4, 2,
                                        np.array([0.5, 0.5, 0.5, 0.5]),
                                        np.array([[1,0,0,0],
                                                  [0,1,0,0],
                                                  [0,1,1,1],
                                                  [1,1,0,0]],
                                                 dtype=int))

        self.assertTrue(np.all(ket.psi_xp == np.array([0.25, 0.25, 0.25, 0.25])))




class TestEnsembleKetBras(unittest.TestCase):
    def test_simple_expectation_dense(self):

        coeffs = np.array( [10, 20, 30] )
        kets = np.array([ [1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0] ])

        ekb = EnsembleKetBras(1, 4, coeffs, kets)

        op = np.array([[ 1, 0, 0, 0 ],
                       [ 0, 2j, 0, 0 ],
                       [ 0, 0, 3, 0 ],
                       [ 0, 0, 0, 0 ]])
        
        npt.assert_almost_equal( ekb.real_overlap(op),
                                 10+30*3 )
        
    # def test_simple_expectation_qutip(self):

    #     coeffs = np.array( [10, 20, 30] )
    #     kets = np.array([ [1, 0, 0, 0],
    #                       [0, 1, 0, 0],
    #                       [0, 0, 1, 0] ])

    #     ekb = EnsembleKetBras(1, 4, coeffs, kets)

    #     op = qutip.Qobj(
    #         np.array([[ 1, 0, 0, 0 ],
    #                   [ 0, 2j, 0, 0 ],
    #                   [ 0, 0, 3, 0 ],
    #                   [ 0, 0, 0, 0 ]]))
        
    #     npt.assert_almost_equal( ekb.real_overlap(op),
    #                              10+30*3 )
        

    def test_more_expectation_dense(self):

        coeffs = np.array( [0.01, 0.09, 0.9] )
        kets = np.array([ [0, 1, 0, 0],
                          [0.5,0.5j,0.5,0.5],
                          [1,1j,0,0]/np.sqrt(2),
                        ])

        ekb = EnsembleKetBras(1, 4, coeffs, kets)

        op = np.array([[ 0, 1, 0, 0 ],
                       [ 0, 0, 2j, 0 ],
                       [ 1, 0, 0, 0 ],
                       [ 0, 0, 0, 0 ]])
        
        npt.assert_almost_equal( ekb.real_overlap(op),
                                 np.einsum('I,Ia,ab,Ib', coeffs, kets.conj(),
                                           op, kets).real )

    def test_more_expectation_sparse(self):

        coeffs = np.array( [0.01, 0.09, 0.9] )
        kets = sps.csr_matrix(
            np.array([ [0, 1, 0, 0],
                       [0.5,0.5j,0.5,0.5],
                       [1,1j,0,0]/np.sqrt(2),
                      ])
        )

        # sparse kets
        ekb_s = EnsembleKetBras(1, 4, coeffs, kets)

        op = sps.csr_matrix(
            np.array([[ 0, 1, 0, 0 ],
                      [ 0, 0, 2j, 0 ],
                      [ 1, 0, 0, 0 ],
                      [ 0, 0, 0, 0 ]])
        )
        
        # sparse operator
        npt.assert_almost_equal(
            ekb_s.real_overlap(op),
            np.einsum('I,Ia,ab,Ib', coeffs, kets.toarray().conj(), op.toarray(),
                      kets.toarray()).real
        )

        # dense operator
        npt.assert_almost_equal(
            ekb_s.real_overlap(op.toarray()),
            np.einsum('I,Ia,ab,Ib', coeffs, kets.toarray().conj(), op.toarray(),
                      kets.toarray()).real
        )

        # dense kets, sparse operator
        ekb_d = EnsembleKetBras(1, 4, coeffs, kets.toarray())

        op = sps.csr_matrix(
            np.array([[ 0, 1, 0, 0 ],
                      [ 0, 0, 2j, 0 ],
                      [ 1, 0, 0, 0 ],
                      [ 0, 0, 0, 0 ]])
        )
        
        npt.assert_almost_equal(
            ekb_d.real_overlap(op),
            np.einsum('I,Ia,ab,Ib', coeffs, kets.toarray().conj(), op.toarray(),
                      kets.toarray()).real
        )

    def test_dense_overlap_with_tensor_product(self):

        coeffs = np.array( [0.01, 0.09, 0.9] )
        kets = np.array([ [0, 1, 0, 0, 0, 0, 0, 0],
                       [0.5,0.5j,0,-0.5, 0, 0, 0.5, 0],
                       [1,0,0,1,0,-1j,0,0]/np.sqrt(3),
                      ])
        bras = np.array([ [0, -1, 0, 0, 0, 0, 0, 0],
                       [0.5,-0.5j,0,-0.5, 0, 0, -0.5, 0],
                       [1,0,0,1j,0,0,0,0]/np.sqrt(2),
                      ])

        # sparse kets/bras ensemble
        ekb_s = EnsembleKetBras(3, 2, coeffs, kets, bras)

        # compute the overlap with the tensor product of three single-qubit
        # operators
        O1, O2, O3 = qutip.sigmaz(), 2j*qutip.sigmam()+qutip.sigmay(), qutip.sigmaz()-1j*qutip.sigmax()+qutip.sigmay()
        ops = np.array([O.full() for O in (O1, O2, O3)])

        #print(f"{ops=}")

        Oqutip = qutip.tensor(O1, O2, O3)

        #print(f" {bras[1,:]=}  {kets[1,:].T=}")

        #print(f"Term-by-term:  {coeffs[0]*bras[0,:].dot(Oqutip.data.dot(kets[0,:].T))=}\n"
        #      f"               {coeffs[1]*bras[1,:].dot(Oqutip.data.dot(kets[1,:].T))=}\n"
        #      f"               {coeffs[2]*bras[2,:].dot(Oqutip.data.dot(kets[2,:].T))=}")

        # overlap with adjoint (HS inner product)
        ovlp = ekb_s.overlap_with_tensor_product_operator(ops)

        #  -- test against sparse calculation
        ovlp_check = ( ekb_s.to_sparse().conj().T.dot(Oqutip.data) ).diagonal().sum()
        npt.assert_almost_equal(ovlp, ovlp_check)

        #  -- test against own .overlap() method
        ovlp_check = ekb_s.overlap(Oqutip)
        npt.assert_almost_equal(ovlp, ovlp_check)

        # overlap withOUT adjoint (simple trace(AB))
        ovlp = ekb_s.overlap_with_tensor_product_operator(ops, use_adjoint=False)

        ovlp_check = ekb_s.roverlap(Oqutip)
        npt.assert_almost_equal(ovlp, ovlp_check)

    def test_simple_overlap_with_tensor_product(self):

        coeffs = np.array( [1.0, 0.0, 0.0] )
        kets = np.array([
            [0, 1, 0, 0,],
            np.array([0, 1, 1, 0,],)/np.sqrt(2),
            np.array([0, 1, 0, 1,],)/np.sqrt(2),
        ])
        bras = np.array([ 
            np.array([0, 1, 1, 0,],)/np.sqrt(2),
            np.array([0, 1, 0, 0,]),
            np.array([0, 1, 0, 1,],)/np.sqrt(2),
        ])

        # sparse kets/bras ensemble
        ekb_s = EnsembleKetBras(2, 2, coeffs, kets, bras)

        # compute the overlap with the tensor product of three single-qubit
        # operators
        O1, O2 = qutip.qeye(2), qutip.sigmaz()
        ops = np.array([O.full() for O in (O1, O2)])
        #print(f"{ops=}")

        Oqutip = qutip.tensor(O1, O2)
        #print(f"{Oqutip=}")

        # overlap with adjoint (HS inner product)
        ovlp = ekb_s.overlap_with_tensor_product_operator(ops)

        #  -- test against sparse calculation
        #print(f"{ekb_s.to_sparse().conj().T.toarray()=}")
        ovlp_check = ( ekb_s.to_sparse().conj().T.dot(Oqutip.data) ).diagonal().sum()
        npt.assert_almost_equal(ovlp, ovlp_check)

        #  -- test against own .overlap() method
        ovlp_check = ekb_s.overlap(Oqutip)
        npt.assert_almost_equal(ovlp, ovlp_check)

        # overlap withOUT adjoint (simple trace(AB))
        ovlp = ekb_s.overlap_with_tensor_product_operator(ops, use_adjoint=False)

        ovlp_check = ekb_s.roverlap(Oqutip)
        npt.assert_almost_equal(ovlp, ovlp_check)


    def test_overlap_with_tensor_product(self):

        coeffs = np.array( [0.01, 0.09, 0.9] )
        kets = sps.csr_matrix(
            np.array([ [0, 1, 0, 0, 0, 0, 0, 0],
                       [0.5,0.5j,0,-0.5, 0, 0, 1, 0],
                       [1,0,0,1,0,-1j,0,0]/np.sqrt(3),
                      ])
        )
        bras = sps.csr_matrix(
            np.array([ np.array([0, -1, 0, 1, 0, 0, 0, 0])/np.sqrt(2),
                       [0.5,-0.5j,0,-0.5, 0, 0, -1, 0],
                       np.array([1,0,1j,0,0,0,0,0])/np.sqrt(2),
                      ])
        )

        # sparse kets/bras ensemble
        ekb_s = EnsembleKetBras(3, 2, coeffs, kets, bras)

        # compute the overlap with the tensor product of three single-qubit
        # operators
        O1, O2, O3 = qutip.sigmaz(), 2j*qutip.sigmam()+qutip.sigmay(), qutip.sigmaz()-1j*qutip.sigmax()+qutip.sigmay()
        ops = np.array([O.full() for O in (O1, O2, O3)])

        # overlap with adjoint (HS inner product)
        ovlp = ekb_s.overlap_with_tensor_product_operator(ops)

        #  -- test against sparse calculation
        Oqutip = qutip.tensor(O1, O2, O3)
        ovlp_check = ( ekb_s.to_sparse().conj().T.dot(Oqutip.data) ).diagonal().sum()
        npt.assert_almost_equal(ovlp, ovlp_check)

        #  -- test against own .overlap() method
        Oqutip = qutip.tensor(O1, O2, O3)
        ovlp_check = ekb_s.overlap(Oqutip)
        npt.assert_almost_equal(ovlp, ovlp_check)

        # overlap withOUT adjoint (simple trace(AB))
        ovlp = ekb_s.overlap_with_tensor_product_operator(ops, use_adjoint=False)

        Oqutip = qutip.tensor(O1, O2, O3)
        ovlp_check = ekb_s.roverlap(Oqutip)
        npt.assert_almost_equal(ovlp, ovlp_check)


    def test_to_sparse(self):

        coeffs = np.array( [0.01, 0.09, 0.9] )
        kets = sps.csr_matrix(
            np.array([ [0, 1, 0, 0],
                       [0.5,0.5j,0.5,0.5],
                       [1,1j,0,0]/np.sqrt(2),
                      ])
        )
        bras = sps.csr_matrix(
            np.array([ 
                       [0.5,0.5j,0.5,0.5],
                       [1,1j,0,0]/np.sqrt(2),
                       [0, 1, 0, 0],
                      ])
        )

        # sparse kets
        ekb_s = EnsembleKetBras(1, 4, coeffs, kets, bras)

        npt.assert_almost_equal(
            ekb_s.to_sparse().toarray(),
            np.einsum('I,Ia,Ib->ab', coeffs, kets.toarray(), bras.toarray())
        )



class TestDitstringActionHamiltonian(unittest.TestCase):

    def test_simple(self):
        
        # note 1**2 + 7**2 + 5**2 + 4**2 + 3**2  == 100
        psi_x = np.array([-0.1j, 0.7, 0.5j, 0.4j, -0.3])
        xn = np.array([[0,0,0,0],
                       [1,0,0,0],
                       [0,1,0,0],
                       [0,1,1,1],
                       [1,1,0,0]],
                      dtype=int)
        xn_basis_vectors = [
            qutip.tensor(qutip.basis(2,0),qutip.basis(2,0),qutip.basis(2,0),qutip.basis(2,0)),
            qutip.tensor(qutip.basis(2,1),qutip.basis(2,0),qutip.basis(2,0),qutip.basis(2,0)),
            qutip.tensor(qutip.basis(2,0),qutip.basis(2,1),qutip.basis(2,0),qutip.basis(2,0)),
            qutip.tensor(qutip.basis(2,0),qutip.basis(2,1),qutip.basis(2,1),qutip.basis(2,1)),
            qutip.tensor(qutip.basis(2,1),qutip.basis(2,1),qutip.basis(2,0),qutip.basis(2,0)),
        ]

        psi_vector = functools.reduce(
            lambda a, b: a+b,
            [psi_x[x]*xn_basis_vectors[x] for x in range(len(psi_x))]
        )
        
        dsk = DitstringSuperpositionKet(
            4, 2,
            psi_x,
            xn
        )

        en = np.array([0, 1, 10, 20, 30])

        H_operator = functools.reduce(
            lambda a, b: a+b,
            [en[x]*xn_basis_vectors[x]*xn_basis_vectors[x].dag() for x in range(len(en))]
        )

        daH = DitstringActionHamiltonian(dsk, en) # yn=None

        self.assertTrue(daH.diagonal_action)

        # check average energy
        npt.assert_almost_equal(
            daH.average_H(),
            (psi_vector.dag() * H_operator * psi_vector).full().item()
        )
        # check correction of en
        npt.assert_almost_equal(
            daH.en,
            en - daH.average_H()*np.ones_like(en)
        )
        # check variance
        npt.assert_almost_equal(
            daH.variance(),
            (psi_vector.dag() * H_operator * H_operator * psi_vector).full().item()
            -  daH.average_H() *  daH.average_H()
        )
        
        
    def test_with_yn(self):
        
        # note 1**2 + 7**2 + 5**2 + 4**2 + 3**2  == 100
        psi_x = np.array([-0.1j, 0.7, 0.5j, 0.4j, -0.3])
        xn = np.array([[0,0,0,0],
                       [1,0,0,0],
                       [0,1,0,0],
                       [0,1,1,1],
                       [1,1,0,0]],
                      dtype=int)
        xn_basis_vectors = [
            qutip.tensor(qutip.basis(2,0),qutip.basis(2,0),qutip.basis(2,0),qutip.basis(2,0)),
            qutip.tensor(qutip.basis(2,1),qutip.basis(2,0),qutip.basis(2,0),qutip.basis(2,0)),
            qutip.tensor(qutip.basis(2,0),qutip.basis(2,1),qutip.basis(2,0),qutip.basis(2,0)),
            qutip.tensor(qutip.basis(2,0),qutip.basis(2,1),qutip.basis(2,1),qutip.basis(2,1)),
            qutip.tensor(qutip.basis(2,1),qutip.basis(2,1),qutip.basis(2,0),qutip.basis(2,0)),
        ]

        psi_vector = functools.reduce(
            lambda a, b: a+b,
            [psi_x[x]*xn_basis_vectors[x] for x in range(len(psi_x))]
        )
        
        dsk = DitstringSuperpositionKet(
            4, 2,
            psi_x,
            xn
        )

        en = np.array([
            [3.2,  0],
            [10, -2],
            [10, -11],
            [20, 21j],
            [30+30j, 30-30j]
        ])
        yn = np.array(
            [
                [[0,0,0,0],[0,0,0,1]],
                [[0,1,0,0],[1,1,1,0]],
                [[1,0,0,0],[0,0,1,0]],
                [[0,1,1,0],[1,1,1,1]],
                [[0,0,1,1],[1,0,1,0]],
            ],
            dtype=int
        )

        daH = DitstringActionHamiltonian(dsk, en, yn)


        yn_basis_vectors = [
            [
                qutip.tensor(*[qutip.basis(2,xi) for xi in yn[x,m]])
                for m in range(en.shape[1])
            ]
            for x in range(en.shape[0])
        ]

        H_operator = functools.reduce(
            lambda a, b: a+b,
            [en[x,m]*yn_basis_vectors[x][m]*xn_basis_vectors[x].dag()
             for x in range(en.shape[0])
             for m in range(en.shape[1])]
        )
        print(f"{psi_vector=}")
        print(f"{H_operator=}")

        for x in range(en.shape[0]):
            for m in range(en.shape[1]):
                for xp in range(en.shape[0]):
                    term = (
                        xn_basis_vectors[xp].dag() * en[x,m] * yn_basis_vectors[x][m]
                    ).full().item()
                    if np.absolute(term) > 1e-6:
                        print("  ({}, {},{}) ->  {}".format(xp,x,m, term))

        check_avg_H = (psi_vector.dag()*H_operator*psi_vector).full().item()

        # test the test!  Make sure the example gives a real average energy
        # value, because we didn't ensure that H was Hermitian --
        assert np.isreal(check_avg_H)

        # check that diagonal_action was set to False
        self.assertFalse(daH.diagonal_action)

        # check average energy
        npt.assert_almost_equal(
            daH.average_H(),
            (psi_vector.dag() * H_operator * psi_vector).full().item()
        )
        # check correction of en -- now it's a little more complicated, so check
        # it by converting to an operator
        new_yn_basis_vectors = [
            [
                qutip.tensor(*[qutip.basis(2,xi) for xi in daH.yn[x,m]])
                for m in range(daH.en.shape[1])
            ]
            for x in range(daH.en.shape[0])
        ]
        shifted_H_operator = functools.reduce(
            lambda a, b: a+b,
            [daH.en[x,m]*new_yn_basis_vectors[x][m]*xn_basis_vectors[x].dag()
                     for x in range(daH.en.shape[0])
                     for m in range(daH.en.shape[1])]
        )
        P_subspace_xns = np.array([
            xn_basis_vectors[x].full()
            for x in range(xn.shape[0])
        ]).T
        npt.assert_almost_equal(
            (H_operator - shifted_H_operator).full() @ P_subspace_xns,
            daH.average_H()*np.eye(2**4) @ P_subspace_xns
        )
        # check variance
        npt.assert_almost_equal(
            daH.variance(),
            (psi_vector.dag() * shifted_H_operator.dag()
             * shifted_H_operator * psi_vector).full().item()
        )
        

    def test_from_ditstring_action(self):

        n = 10
        k = 2
        omega = 2.0
        J = 0.0

        xn = np.vstack([ np.zeros((1,n), dtype=int),
                         np.ones((1,n), dtype=int) ])
        psi_x = np.sqrt(np.array([1/2, 1/2]))        
        dsk = DitstringSuperpositionKet(n, 2, psi_x, xn)

        def ditstring_action(Xn):
            print(Xn)
            return (omega/2) * np.array([1,-1], dtype=int)[Xn] .sum() + \
                (J/2) * np.logical_xor(Xn[:-1], Xn[1:]) .sum()

        daH = DitstringActionHamiltonian.from_ditstring_action(
            dsk,
            ditstring_action
        )

        # * On-site Z terms
        en = (omega/2) * np.array([1,-1], dtype=int)[xn] .sum(axis=1)
        # * Domain wall terms
        en += (J/2) * np.logical_xor(xn[:,:-1], xn[:,1:]) .sum(axis=1)
        
        npt.assert_almost_equal(daH.en, en[:,np.newaxis])
        npt.assert_almost_equal(daH.yn, xn[:,np.newaxis,:])
#




class TestLocalKrausOperators(unittest.TestCase):
    def test_compl_kraus(self):
        
        lko = LocalKrausOperators([
            qutip.Qobj([[1, 0], [0, np.sqrt(0.5)]]),
            qutip.Qobj([[0, 0], [0, np.sqrt(0.25)]]),
            qutip.Qobj([[0, np.sqrt(0.25)], [0, 0]]),
        ])

        N1c = lko.get_complementary_kraus()
        self.assertEqual(len(N1c), 2) # given by local dimension == 2
        
        for ell, k, kp in itertools.product(*[range(2), range(3), range(2)]):
            npt.assert_almost_equal(
                N1c[ell].data[k,kp],
                lko.local_kraus_operators[k][ell,kp]
            )










if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
