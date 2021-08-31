import logging
import unittest
import functools

import numpy as np
import numpy.testing as npt

import scipy.sparse as sps

import qutip

from qfi_local_noise import (
    # symmetric stuff
    LocalKrausOperators,
    SymmetricKet,
    SymmetricOperator,
    SymmetricReducedOperatorsCalculator,
    SymmetricTraceWithKrausPairStringCalculator,
    # iter sites related
    iter_symm_sites_with_multiplier,
    # computing the FI bounds
    compute_pinched_DF_lower_bound,
    # optimization related
    OptimizationStateAnsatz,
    SymmetricRealStateAnsatz,
    SymmetricFullStateAnsatz,
    SymmetricExponentialTailsStateAnsatz,
    StateOptimizer,
)


def compute_fisher_information_symm_pinched(soH, lko, sk, *, k):

    n = sk.n
    localdim = sk.localdim

    avg_H_psi = soH.real_overlap(sk)
    var_H_psi = (soH.operator_apply(soH)).real_overlap(sk) - avg_H_psi**2
    F_Alice = 4 * var_H_psi

    psi_op = sk.to_density_operator()

    H_op_sp = soH.get_symm_sparse()
    psi_op_sp = psi_op.get_symm_sparse()

    xipsi_op = SymmetricOperator(
        n, localdim,
        H_op_sp @ psi_op_sp  -  avg_H_psi * psi_op_sp
    )

    rpxc = SymmetricReducedOperatorsCalculator(
        n, localdim,
        [ psi_op, xipsi_op,],
        lko
    )

    owkpsc = SymmetricTraceWithKrausPairStringCalculator

    # First, compute the "pinched bound"
    DFlb_pinched, _ = compute_pinched_DF_lower_bound(
        rpxc,
        lko,
        owkpsc=owkpsc,
        iter_sites_with_multiplier=
            lambda: iter_symm_sites_with_multiplier(n,k),
        k=k)

    return F_Alice - DFlb_pinched



class fn_state_optimizer_pinched_optim_shots:
    def __init__(self, soH, lko, k):
        self.soH, self.lko, self.k = soH, lko, k
    def __call__(self, sk):
        return compute_fisher_information_symm_pinched(
            self.soH, self.lko, sk, k=self.k
        )


class TestStateOptimizer(unittest.TestCase):

    def test_pinched_optim_shots_mp(self):

        n = 5
        soH = SymmetricOperator(n, 2, np.diag(2*np.arange(n+1)-n))

        p = 0.05
        lko = LocalKrausOperators([
            #    np.sqrt(1-p/2) * qutip.qeye(2),
            #    np.sqrt(p/2) * qutip.sigmaz()
            qutip.Qobj( np.array([[np.sqrt(1-p), 0],
                                  [0           , 1]]) ),
            qutip.Qobj( np.array([[0         , 0],
                                  [np.sqrt(p), 0]]) ),
        ])

        k = 3

        ansatz = SymmetricExponentialTailsStateAnsatz(n, 2)
        opt = StateOptimizer(ansatz, fn_state_optimizer_pinched_optim_shots(soH, lko, k))
        result = opt.optimize_shots()

        self.assertGreaterEqual(result.F_Bob, 85.0)


    def test_pinched_optim_shots_serial(self):

        n = 5
        soH = SymmetricOperator(n, 2, np.diag(2*np.arange(n+1)-n))

        p = 0.05
        lko = LocalKrausOperators([
            #    np.sqrt(1-p/2) * qutip.qeye(2),
            #    np.sqrt(p/2) * qutip.sigmaz()
            qutip.Qobj( np.array([[np.sqrt(1-p), 0],
                                  [0           , 1]]) ),
            qutip.Qobj( np.array([[0         , 0],
                                  [np.sqrt(p), 0]]) ),
        ])

        k = 3

        def fn(sk):
            return compute_fisher_information_symm_pinched(
                soH, lko, sk, k=k
            )

        ansatz = SymmetricExponentialTailsStateAnsatz(n, 2)
        opt = StateOptimizer(ansatz, fn)
        result = opt.optimize_shots(parallel=False)

        self.assertGreaterEqual(result.F_Bob, 85.0)



    def test_pinched_optim_basinhopping(self):

        n = 5
        soH = SymmetricOperator(n, 2, np.diag(2*np.arange(n+1)-n))

        p = 0.05
        lko = LocalKrausOperators([
            #    np.sqrt(1-p/2) * qutip.qeye(2),
            #    np.sqrt(p/2) * qutip.sigmaz()
            qutip.Qobj( np.array([[np.sqrt(1-p), 0],
                                  [0           , 1]]) ),
            qutip.Qobj( np.array([[0         , 0],
                                  [np.sqrt(p), 0]]) ),
        ])

        k = 3

        def fn(sk):
            return compute_fisher_information_symm_pinched(
                soH, lko, sk, k=k
            )

        ansatz = SymmetricExponentialTailsStateAnsatz(n, 2)
        opt = StateOptimizer(ansatz, fn)
        result = opt.optimize_basinhopping(basinhopping_kwargs=dict(niter=10))

        self.assertGreaterEqual(result.F_Bob, 85.0)






if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('qfi_local_noise._states_desc.validate').setLevel(logging.INFO)
    logging.getLogger('qfi_local_noise._symm_states_desc.validate').setLevel(logging.INFO)
    logging.getLogger('qfi_local_noise._optimize_state.validate').setLevel(logging.INFO)
    logger = logging.getLogger(__name__)
    unittest.main()
