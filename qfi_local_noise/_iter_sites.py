import logging
logger = logging.getLogger(__name__)

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



# tools for iteration over choices of sites


def _add_and_carry_in_place(state, base, idx=-1):
    #print(f"{state=}, {base=}, {idx=}, {state.size=}")
    need_carry = True
    while need_carry and (idx < state.size and idx >= -state.size):
        need_carry = False
        state[idx] += 1
        if state[idx] >= base:
            state[idx] = 0
            idx -= 1
            if not (idx == -1 or idx < -state.size):
                # no overflow, we can continue our while loop
                need_carry = True
    #print(f"new state is {state=}")


def iter_strings_k(n, k, m):
    """
    Iterate over ditstrings of length `n` where at most `k` elements are
    nonzero.  Each nonzero element ranges from `1` to `m`.  Strings are emitted
    in lexicographical order.
    """
    # initial state -- all zeros
    state = np.zeros((n,), dtype=int)

    if k == 0:
        # that was it (!)
        return

    while True:
        #print(f"next state is {state=}")
        yield state

        # Update to next state.  Idea is to count and carry as usual, except if
        # there are already k nonzeros in which case we count and carry by
        # ignoring all the trailing zeros.  This is the algorithm described here
        # - https://stackoverflow.com/a/10458380/1694896 - adapted from bits to
        # base-m "mits"
        if np.count_nonzero(state) < k:
            _add_and_carry_in_place(state, m)
            continue

        # there are k nonzeros already, find first nonzero from least
        # significant end.  See https://stackoverflow.com/a/52911347/1694896
        last_nonzero = np.max(np.nonzero(state))
        # and increment that one
        _add_and_carry_in_place(state, m, last_nonzero)
        if not np.any(state):
            # end of iteration reached, as we've gone back to the all-zero
            # state.
            return

# test with: [ np.copy(x) for x in lbln.iter_strings_k(n=10, k=3, m=3)]


# different strategy: iterate over same terms, but by choice of sites on which
# the terms act (not lexicographically)
def iter_leqk_sites(n, k):
    # use iter_strings_k() to iterate over all bitstrings with
    # at most k bits set -- this will fix our choice of sites.
    for choice_sites in iter_strings_k(n, k, 2):
        yield tuple(*np.nonzero(choice_sites))

def _iter_iid_kraus_nz_index_string_on_k_sites(k, m):
    # iterate over all combinations of nonzero Kraus indexes on the given sites.
    # Easy, this is basically the cartesian product.

    if k == 0:
        yield np.array([], dtype=int)
        return

    # we will simply iterate over all combinations with a count-and-carry
    # strategy.  The array idx will count items from 0...m-1 for convenience;
    # we'll be consistently returning idx+ones(k) so that returned indices are
    # in the range 1...m.
    idx = np.zeros((k,), dtype=int)
    ones_offset = np.ones((k,), dtype=int)
    while True:
        yield idx + ones_offset
        _add_and_carry_in_place(idx, m-1)
        if not np.any(idx):
            # if all zero, this means we overflowed and completed the iteration
            return



# ------------------------------------------------


# iterator to use with translation-invariant states with periodic boundary
# conditions with period K.  Here, we iterate over choices of k sites modulo a
# permutation that acts

def iter_symm_sites_with_multiplier_2(n, k):
    # compute binomial coefficient using recurrence formula for constant n
    binom_coeff = 1
    for i in range(k+1):
        yield np.arange(i), binom_coeff
        binom_coeff = binom_coeff * (n - i) // (i + 1)



# ------------------------------------------------


# iterate: the first site, the first two sites, the first three sites, etc.  For
# use with symmetric states & operators.  A suitable multiplying factor is
# provided.

def iter_symm_sites_with_multiplier(n, k):
    # compute binomial coefficient using recurrence formula for constant n
    binom_coeff = 1
    for i in range(k+1):
        yield np.arange(i), binom_coeff
        binom_coeff = binom_coeff * (n - i) // (i + 1)
