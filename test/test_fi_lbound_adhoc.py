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
    LocalKrausOperators,
    DitstringSuperpositionKetReducedPsiXiCalculator,
    #
    compute_Eve_operators_leqk,
    #
    compute_Fisher_information,
    #
    ExpandableSelectionOfDitstrings,
    FI_lbound_projditstrings,
)




class Test_FI_lbound_projditstrings(unittest.TestCase):

    def test_simple(self):

        # simple pure state with some complex phases (|↑↑⟩ + i|↓↓⟩)/√2
        dsk = DitstringSuperpositionKet(
            2, 2,
            np.array([1, 1j])/np.sqrt(2),
            np.array([[0,0],
                      [1,1,]])
        )

        # Z+Z Hamiltonian
        daH = DitstringActionHamiltonian(
            dsk,
            np.array([2, -2]),
        )
        F_Alice = 4 * daH.variance()

        # Local Kraus operators -- say dephainsg
        p = 0.08
        lko = LocalKrausOperators(
            [np.sqrt(1-p/2)*qutip.qeye(2),
             np.sqrt(p/2)*qutip.sigmaz(),]
        )

        esd = ExpandableSelectionOfDitstrings(dsk.n, dsk.localdim, [
            [0,0],
            [0,1],
            [1,0],
            [1,1],
        ])

        # compute our brutal lower bound -- except here we can use a full basis
        # of the entire space, so it's the exact value, not only a lower bound.
        Flb, _ = FI_lbound_projditstrings(dsk, daH, lko, esd=esd)
        #print(f"{Flb=}")

        # full setting --
        dskrpxc = DitstringSuperpositionKetReducedPsiXiCalculator(
            dsk, daH, lko
        )
        # compute full state on Eve's side
        (Nhatpsi, Nhatxipsi), _ = compute_Eve_operators_leqk(
            dskrpxc, lko,
            k=2,
        )

        # and Eve's corresponding Fisher information
        DF, _ = compute_Fisher_information(Nhatpsi, Nhatxipsi+Nhatxipsi.conj().T)
        #print(f"{DF=}")

        npt.assert_almost_equal(Flb, F_Alice - DF)

        

    def test_more_complex(self):

        # Some 8-qubit state written as a superposition of some ditstrings
        dsk = DitstringSuperpositionKet(
            8, 2,
            # coefficients -- real part and a phase
            np.sqrt(np.array([0.5, 0.3, 0.1, 0.1]))*np.exp(1j*np.pi*np.array([0.1,0.9,-0.8,-0.01])),
            # the basis ditstrings that are involved in the expression of this state
            np.array([[0,0,1,0,0,1,0,0],
                      [1,0,1,0,1,1,0,1],
                      [1,0,1,1,1,1,1,1],
                      [1,1,1,0,0,0,1,0]])
        )

        omega1, omega2 = 2.0, 1.0

        def H_action_on_ditstring(X):
            # Say that H is a string of ω₁σ_X 's on the four two qubits, and a
            # string of ω₂*σ_Z 's on the remaining qubits.
            #
            # E.g.:  [0 1 0 0 0 1 1]  ->   2ω₁+2ω₂ : [1 0 0 0 0 1 1]
            e2 = omega2*np.sum(np.array([1, -1])[X[2:]])
            e = [ 2*omega1 + e2 ]
            Y = [ [ 1-X[0], 1-X[2], ] + list(X[2:]) ]
            return e, Y

        # Z+Z Hamiltonian
        daH = DitstringActionHamiltonian.from_ditstring_action(dsk, H_action_on_ditstring)
        F_Alice = 4 * daH.variance()

        esd = (
            ExpandableSelectionOfDitstrings(dsk.n, dsk.localdim)
            .added_dsk_and_daH(dsk, daH)
            .added_random_nearby_ditstrings(16, up_to_hamming_distance=3)
        )

        for p in np.linspace(1e-3, 0.999, 5):

            # Local Kraus operators -- say dephasing
            lko = LocalKrausOperators(
                [np.sqrt(1-p/2)*qutip.qeye(2),
                 np.sqrt(p/2)*qutip.sigmaz(),]
            )

            N1_full = lko.get_full_super_n(dsk.n)

            # compute our brutal lower bound -- except here we can use a full basis
            # of the entire space, so it's the exact value, not only a lower bound.
            Flb, _ = FI_lbound_projditstrings(dsk, daH, lko, esd=esd)
            #print(f"{p=} {Flb=}")

            # compute the full Fisher information.
            psi_s = dsk.to_sparse()
            xi_s = daH.get_xi_ditstringket().to_sparse()

            #print(f"Got sparse psi & xi")

            rho_B = qutip.vector_to_operator(
                N1_full * qutip.operator_to_vector(
                    qutip.Qobj( psi_s*psi_s.conj().T, dims=[[2]*8,[2]*8] ) )
            )
            Nxipsi_B = qutip.vector_to_operator(
                N1_full * qutip.operator_to_vector(
                    qutip.Qobj( -1j*xi_s*psi_s.conj().T, dims=[[2]*8,[2]*8] ) )
            )
            D_B = (Nxipsi_B + Nxipsi_B.dag()).full()

            #print(f"Got rho_B & D_B")

            # and Eve's corresponding Fisher information
            F_true, _ = compute_Fisher_information(rho_B, D_B)
            #print(f"{p=} {F_true=}")

            self.assertLess( Flb, F_true + 1e-6*Flb ) # include tolerance

        






if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('qfi_local_noise._states_desc.validate').setLevel(logging.INFO)
    unittest.main()
