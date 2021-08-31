import logging
import unittest
import functools

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

import scipy.sparse as sps

import qutip

from qfi_local_noise import (
    compute_Fisher_information,
)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    #logging.getLogger('qfi_local_noise._states_desc.validate').setLevel(logging.INFO)

logger = logging.getLogger(__name__)



class TestMethod_splslvlyap(unittest.TestCase):

    def _run_test_for(self, rho, D):
        
        print("Hi!")

        FI, vals = compute_Fisher_information(rho, D, method='splslvlyap')

        R = vals['R']

        logger.debug(f"{R=}")

        # R must be Hermitian
        npt.assert_almost_equal( R.conj().T, R )

        # R must satisfy the sld equation
        npt.assert_almost_equal( R @ rho  +  rho @ R,  2*D )

        # FI value must be tr(R D) == tr(rho R^2)
        npt.assert_almost_equal(FI, np.trace(R @ D))
        npt.assert_almost_equal(FI, np.trace(rho @ R @ R))


    def test_pure(self):

        ket_r = np.array([0.5, 0.3,  0.05, -0.03])
        ket_i = np.array([  0, 0.1, -0.02j,    0])

        ket = (ket_r + ket_i*1j)[:,np.newaxis]
        ket /= npl.norm(ket)
        logger.debug(f"{ket=}")
        assert np.absolute(npl.norm(ket)-1) < 1e-6

        rho = ket @ ket.conj().T

        logger.debug(f"{rho=}")

        # some random Hermitian matrix
        H = np.array([
            [ 1,  90, -3,  4j ],
            [ 90,  0, -1j,  3 ],
            [ -3, 1j,  0,  -1 ],
            [ -4j, 3, -1,  10 ],
        ])
        assert npl.norm(H - H.conj().T) < 1e-6

        logger.debug(f"{H=}")

        D = -1j*(H @ rho - rho @ H)

        logger.debug(f"{D=}")

        self._run_test_for(rho, D)

        
    def test_mixed(self):

        # some random complex matrix
        A = np.array([
            [ 1, 5, 3j-5, 6, 1],
            [ -3, 0,  0, 2, 0],
            [ 0,  1j, 0, 0, 0],
            [ 0,  1j, 0, 31, 0],
            [ 0,  0, 1-1j, 0, 0],
        ])
        # form a density matrix out of it
        rho = (A @ A.conj().T)
        rho /= np.trace(rho)
        assert np.absolute(np.trace(rho) - 1) < 1e-6

        logger.debug(f"{rho=}")

        # some random Hermitian matrix
        H = np.array([
            [ 1,  90, -3,  4j, 0 ],
            [ 90,  0, -1j,  3, 0 ],
            [ -3, 1j,  0,  -1, 3 ],
            [ -4j, 3, -1,  10, 0 ],
            [  0,  0,  3,  0, -1 ],
        ])
        assert npl.norm(H - H.conj().T) < 1e-6

        logger.debug(f"{H=}")

        D = -1j*(H @ rho - rho @ H)

        logger.debug(f"{D=}")

        self._run_test_for(rho, D)

        





if __name__ == '__main__':
    unittest.main()
