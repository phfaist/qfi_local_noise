import sys
import logging

import itertools
import unittest

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import scipy as sp
import scipy.sparse as sps

import qutip

import qfi_local_noise




logger = None



class TestCaseCompareDitstringWithDense(unittest.TestCase):
    def _run_comparison_test(self, q_psi, q_H, dsk, daH, lko, *, k):
        
        n = dsk.n
        localdim = dsk.localdim
        assert q_psi.dims == [[localdim]*n, [1]*n]
        assert q_H.dims == [[localdim]*n, [localdim]*n]

        rpxc = qfi_local_noise.DitstringSuperpositionKetReducedPsiXiCalculator(
            dsk,
            daH,
            lko
        )

        avg_H_psi = daH.average_H()
        F_Alice = 4*daH.variance()

        # First, compute the "pinched bound"
        DFlb_pinched, _ = qfi_local_noise.compute_pinched_DF_lower_bound(rpxc, lko, k=k)

        # Then compute the "coherent" version of the bound
        (Nhatpsi, Nhatxipsi), _ = qfi_local_noise.compute_Eve_operators_leqk(rpxc, lko, k=k)
        DFlb, _ = qfi_local_noise.compute_Fisher_information(Nhatpsi, Nhatxipsi+Nhatxipsi.conj().T)


        npt.assert_almost_equal(
            avg_H_psi,
            (q_psi.dag() * q_H * q_psi).tr() # .tr() is to make scalar from Qobj
        )

        Hbar = q_H - avg_H_psi * qutip.tensor(*[qutip.qeye(localdim) for _ in range(n)])

        npt.assert_almost_equal(
            F_Alice,
            4 * (q_psi.dag() * Hbar * Hbar * q_psi).tr() # .tr() is to make scalar from Qobj
        )

        # Now, compute the same objects but using the qutip dense objects.
        Ncomplfull = lko.get_full_complementary_super_n(n)
        
        rho_E = qutip.vector_to_operator(
            Ncomplfull * qutip.operator_to_vector(q_psi * q_psi.dag())
        ).full()
        D_E = qutip.vector_to_operator(
            Ncomplfull * qutip.operator_to_vector(Hbar * q_psi * q_psi.dag()
                                                  + q_psi * q_psi.dag() * Hbar)
        ).full()

        # the approximation that we make with k < n is to set to zero all
        # rows/columns of the output of the complementary channel, for each k,kp
        # that is associated with a Kraus operator of weight > k.
        #
        # This implementation assumes we're dealing with bits on Bob's end
        assert lko.localdim_out == 2
        for kk in range(2**n):
            # count number of bits in kk.  If strictly greater than k, then set
            # rho and D's k-th row and column to zero.
            w = bin(kk).count('1')
            if w > k:
                rho_E[kk,:] = 0
                rho_E[:,kk] = 0
                D_E[kk,:] = 0
                D_E[:,kk] = 0


        # compute pinched version
        rho_deph_E = np.diag(np.diag(rho_E))
        D_deph_E = np.diag(np.diag(D_E))

        DFlb_pinched_check, _ = qfi_local_noise.compute_Fisher_information(rho_deph_E, D_deph_E)

        npt.assert_almost_equal(DFlb_pinched, DFlb_pinched_check)


        # compute full version
        DFlb_check, _ = qfi_local_noise.compute_Fisher_information(rho_E, D_E)

        npt.assert_almost_equal(DFlb, DFlb_check)



    def test_GHZ_3qubits_k2(self):

        n = 3
        k = 2
        omega = 2.0
        J = 0.0

        xn = np.vstack([ np.zeros((1,n), dtype=int),
                         np.ones((1,n), dtype=int) ])
        psi_x = np.sqrt(np.array([1/2, 1/2]))        
        dsk = qfi_local_noise.DitstringSuperpositionKet(n, 2, psi_x, xn)

        def ditstring_action(Xn):
            #print(Xn)
            return (omega/2) * np.array([1,-1], dtype=int)[Xn] .sum() + \
                (J/2) * np.logical_xor(Xn[:-1], Xn[1:]) .sum()

        daH = qfi_local_noise.DitstringActionHamiltonian.from_ditstring_action(
            dsk,
            ditstring_action
        )

        psi = np.zeros( (2**3,) )
        psi[0] = 1/np.sqrt(2)
        psi[-1] = 1/np.sqrt(2)
        q_psi = qutip.Qobj(psi, dims=[[2]*3,[1]*3])

        H = np.diag( [ditstring_action([a,b,c])
                      for a,b,c in itertools.product(*[range(2),range(2),range(2)])] )
        q_H = qutip.Qobj(H, dims=[[2]*3,[2]*3])

        print(q_psi)
        print(q_H)

        p = 0.5
        lko = qfi_local_noise.LocalKrausOperators([
            #    np.sqrt(1-p/2) * qutip.qeye(2),
            #    np.sqrt(p/2) * qutip.sigmaz()
            qutip.Qobj( np.array([[np.sqrt(1-p), 0],
                                  [0           , 1]]) ),
            qutip.Qobj( np.array([[0         , 0],
                                  [np.sqrt(p), 0]]) ),
        ])

        self._run_comparison_test(q_psi, q_H, dsk, daH, lko, k=2)




class TestCaseCompareSymmetricWithDense(unittest.TestCase):
    def _run_comparison_test(self, q_psi, q_H, sk, soH, lko, *, k):
        
        n = sk.n
        localdim = sk.localdim
        assert q_psi.dims == [[localdim]*n, [1]*n]
        assert q_H.dims == [[localdim]*n, [localdim]*n]

        psi_op = sk.to_density_operator()
        xipsi_op = soH.operator_apply( psi_op )

        # todo: might need a shift in general
        assert np.absolute( xipsi_op.q_operator_exc.tr() ) <= 1e-6

        rpxc = qfi_local_noise.SymmetricReducedOperatorsCalculator(
            n, localdim,
            [ psi_op, xipsi_op,],
            lko
        )

        avg_H_psi = soH.real_overlap(sk)
        var_H_psi = (soH.operator_apply(soH)).real_overlap(sk) - avg_H_psi**2
        F_Alice = 4 * var_H_psi

        print(f"{avg_H_psi=}  {var_H_psi=}  {F_Alice=}")

        owkpsc = qfi_local_noise.SymmetricTraceWithKrausPairStringCalculator

        # First, compute the "pinched bound"
        DFlb_pinched, _ = qfi_local_noise.compute_pinched_DF_lower_bound(
            rpxc,
            lko,
            owkpsc=owkpsc,
            iter_sites_with_multiplier=
                lambda: qfi_local_noise.iter_symm_sites_with_multiplier(n,k),
            k=k)


        # Then compute the "coherent" version of the bound
        (Nhatpsi, Nhatxipsi), _ = qfi_local_noise.compute_Eve_operators_leqk(
            rpxc, lko,
            owkpsc=owkpsc,
            k=k
        )

        print(f"{Nhatpsi=}\n{Nhatxipsi+Nhatxipsi.conj().T=}")

        DFlb, _ = qfi_local_noise.compute_Fisher_information(
            Nhatpsi,
            Nhatxipsi+Nhatxipsi.conj().T
        )

        npt.assert_almost_equal(
            avg_H_psi,
            (q_psi.dag() * q_H * q_psi).tr() # .tr() is to make scalar from Qobj
        )

        Hbar = q_H - avg_H_psi * qutip.tensor(*[qutip.qeye(localdim) for _ in range(n)])

        npt.assert_almost_equal(
            F_Alice,
            4 * (q_psi.dag() * Hbar * Hbar * q_psi).tr() # .tr() is to make scalar from Qobj
        )

        # Now, compute the same objects but using the qutip dense objects.
        Ncomplfull = lko.get_full_complementary_super_n(n)
        
        rho_E = qutip.vector_to_operator(
            Ncomplfull * qutip.operator_to_vector(q_psi * q_psi.dag())
        ).full()
        D_E = qutip.vector_to_operator(
            Ncomplfull * qutip.operator_to_vector(Hbar * q_psi * q_psi.dag()
                                                  + q_psi * q_psi.dag() * Hbar)
        ).full()

        # the approximation that we make with k < n is to set to zero all
        # rows/columns of the output of the complementary channel, for each k,kp
        # that is associated with a Kraus operator of weight > k.
        #
        # This implementation assumes we're dealing with bits on Bob's end
        assert lko.localdim_out == 2
        for kk in range(2**n):
            # count number of bits in kk.  If strictly greater than k, then set
            # rho and D's k-th row and column to zero.
            w = bin(kk).count('1')
            if w > k:
                rho_E[kk,:] = 0
                rho_E[:,kk] = 0
                D_E[kk,:] = 0
                D_E[:,kk] = 0

        print(f"{rho_E=}\n{D_E=}")

        # compute pinched version
        rho_deph_E = np.diag(np.diag(rho_E))
        D_deph_E = np.diag(np.diag(D_E))

        DFlb_pinched_check, _ = qfi_local_noise.compute_Fisher_information(rho_deph_E, D_deph_E)

        npt.assert_almost_equal(DFlb_pinched, DFlb_pinched_check)


        # compute full version
        DFlb_check, _ = qfi_local_noise.compute_Fisher_information(rho_E, D_E)

        npt.assert_almost_equal(DFlb, DFlb_check)


    def test_GHZ_3qubits_k2(self):

        n = 3
        k = 2
        omega = 2.0

        exc = np.vstack([ 0, n ])
        psi_x = np.sqrt(np.array([1/2, 1/2]))        
        sk = qfi_local_noise.SymmetricSuperpositionKet(n, 2, psi_x, exc).to_symmetric_ket()

        oper_exc = np.diag((omega/2)*(2*np.arange(n+1)-n))
        soH = qfi_local_noise.SymmetricOperator(n, 2, oper_exc)


        psi = np.zeros( (2**n,) )
        psi[0] = 1/np.sqrt(2)
        psi[-1] = 1/np.sqrt(2)
        q_psi = qutip.Qobj(psi, dims=[[2]*n,[1]*n])

        def ditstring_action(Xn):
            return (omega/2) * np.array([-1,1], dtype=int)[np.array(Xn,dtype=int)] .sum()
        H = np.diag( [ ditstring_action(X)
                      for X in itertools.product(*[range(2) for _ in range(n)]) ] )
        q_H = qutip.Qobj(H, dims=[[2]*n,[2]*n])

        print(q_psi)
        print(q_H)

        p = 0.5
        lko = qfi_local_noise.LocalKrausOperators([
            #    np.sqrt(1-p/2) * qutip.qeye(2),
            #    np.sqrt(p/2) * qutip.sigmaz()
            qutip.Qobj( np.array([[np.sqrt(1-p), 0],
                                  [0           , 1]]) ),
            qutip.Qobj( np.array([[0         , 0],
                                  [np.sqrt(p), 0]]) ),
        ])

        self._run_comparison_test(q_psi, q_H, sk, soH, lko, k=k)


    def test_unifsymm_3qubits_k2(self):

        n = 2
        k = 1
        omega = 2.0

        psi_x = np.ones(shape=(n+1,))/np.sqrt(n+1)
        sk = qfi_local_noise.SymmetricKet(n, 2, psi_x)
        print(f"{sk.q_psi_exc=}")

        oper_exc = np.diag((omega/2)*(2*np.arange(n+1)-n))
        soH = qfi_local_noise.SymmetricOperator(n, 2, oper_exc)
        print(f"{soH.q_operator_exc=}")

        psi = sk.to_sparse()
        q_psi = qutip.Qobj(psi, dims=[[2]*n,[1]*n])
        print(f"{q_psi=}  {q_psi.norm()=}")

        def ditstring_action(Xn):
            #print(f"{Xn=}")
            return (omega/2) * np.array([-1,1], dtype=int)[np.array(Xn,dtype=int)] .sum()
        H = np.diag( [ ditstring_action(X)
                      for X in itertools.product(*[range(2) for _ in range(n)])] )
        q_H = qutip.Qobj(H, dims=[[2]*n,[2]*n])

        # we could also use:
        #
        # q_H = qutip.Qobj(soH.to_sparse(), dims=[[2]*n,[2]*n])
        #
        # which would give the Hamiltonian projected onto the symmetric subspace.

        print(f"{q_H=}")

        p = 0.5
        lko = qfi_local_noise.LocalKrausOperators([
            #    np.sqrt(1-p/2) * qutip.qeye(2),
            #    np.sqrt(p/2) * qutip.sigmaz()
            qutip.Qobj( np.array([[np.sqrt(1-p), 0],
                                  [0           , 1]]) ),
            qutip.Qobj( np.array([[0         , 0],
                                  [np.sqrt(p), 0]]) ),
        ])

        self._run_comparison_test(q_psi, q_H, sk, soH, lko, k=k)







if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('qfi_local_noise._states_desc.validate').setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    unittest.main()
