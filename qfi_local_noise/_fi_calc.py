import logging
logger = logging.getLogger(__name__)

import itertools
import functools

import numpy as np
import numpy.linalg as npl

import scipy as sp
import scipy.linalg as spl
import scipy.special as spa
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.optimize as spo

import qutip


# Special logger for all the _validate calls, so that it can be silenced easily.
# The logger instance will be found by name in the global scope of this module
# by the _validate() command itself.
from .validate import validate_logger





def compute_Fisher_information(rho, D, *, method='splslvlyap', **kwargs):
    r"""
    Compute the Fisher information associated with `rho` and `D`.
    """

    if method == 'splslvlyap':
        return compute_Fisher_information_splslvlyap(rho, D, **kwargs)

    raise ValueError("Invalid method: {!r}".format(method))



def compute_Fisher_information_splslvlyap(rho, D, *, asserts_on=True, tol_check=1e-6):
    
    R = spl.solve_continuous_lyapunov(rho, 2*D)

    #if asserts_on:
    #    assert npl.norm(R.conj().T - R, 2) < tol_check

    # enforce R to be Hermitian, remains a solution of the Lyapunov equation
    R = (R + R.conj().T)/2

    #value = np.trace(rhoE @ R @ R).real
    # By the property of the SLD we have that  value == np.trace(D @ R) -> 
    value = np.sum( np.multiply(D.conj(), R) ).real

    return value, {'value': value, 'R': R}
