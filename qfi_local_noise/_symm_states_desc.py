import itertools
import functools
import logging
logger = logging.getLogger(__name__)

import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import scipy as sp
import scipy.linalg as spl
import scipy.special as spa
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.optimize as spo

import qutip

from ._util import abs2, multinomial_coefficient, iter_spmatrix, _validate

from ._states_desc import (
    KetDesc,
    OperatorDesc,
    DitstringSuperpositionKet,
    EnsembleKetBras,
    TraceWithKrausPairStringCalculator,
    ReducedOperatorsCalculator,
)


# Special logger for all the _validate calls, so that it can be silenced easily.
# The logger instance will be found by name in the global scope of this module
# by the _validate() command itself.
from .validate import validate_logger






### PhF---Never mind this for now, we'll stick to embedding the basis states
### into {0...n}^(localdim-1) for now where everything is much simpler.
#
# # Simple access to the basis structure of the symmetric subspace.  What is the
# # excitation structure of the j-th basis vector of the symmetric subspace?
# #
# # For localdim=4, the basis is enumerated as follows (ground state populations
# # are omitted; we write |n₁,n₂,n₃⟩ where nᵢ with i=0,1,2,3 are the populations):
# #
# #   |0,0,0⟩, |0,0,1⟩, |0,0,2⟩, ...,          |0,0,n⟩,     (n+1)
# #   |0,1,0⟩, |0,1,1⟩, |0,1,2⟩, ..., |0,1,n-1⟩,            (n)
# #   ...
# #   |0,n-1,0⟩, |0,n-1,1⟩                                  (2)
# #   |0,n,0⟩                                               (1)
# #   |1,0,0⟩, |1,0,1⟩, |1,0,2⟩, ..., |1,0,n-1⟩,            (n)
# #
# def symm_basis_exc_to_j(n, localdim, exc):
#     .....?
#
# def symm_basis_j_to_exc(n, localdim, j):
#     .....?






def _symm_basisvec_to_full_ket(n, localdim, p):
    _validate(r""" $localdim == 2 """)

    _validate(r""" $p >= 0  and  $p <= $n """)

    Xn_start = [0] * (n-p)  +  [1] * p
    # e.g.  Xn = 0 0 0 0 0 0 0 1 1 1 1

    data, rows = [], []

    pow2s = 2**np.arange(n)[::-1]
    pow2n = 2**n

    for Xn in _next_permutation(Xn_start):
        data.append( 1 )
        rows.append( np.array(Xn,dtype=int).dot(pow2s) ) # row in std computational basis ordering

    rows = np.array(rows, dtype=int)
    data = np.array(data) / np.sqrt(len(data))
    cols = np.zeros( shape=(len(rows),), dtype=int)
    return sps.coo_matrix((data, (rows, cols)), shape=(pow2n,1)).tocsr()


class SymmetricBasisVecToFullKetCache:
    def __init__(self, n, localdim):
        self.n = n
        self.localdim = localdim
        self.symm_full_kets = {}

    def precompute_all(self):
        for p in range(self.n+1):
            self._compute_and_store(p)

    def get(self, p):
        if p not in self.symm_full_kets:
            self._compute_and_store(p)
        return self.symm_full_kets[p]

    def _compute_and_store(self, p):
        self.symm_full_kets[p] = \
            _symm_basisvec_to_full_ket(self.n, self.localdim, p)





class SymmetricKet(KetDesc):
    def __init__(self, n, localdim, psi_exc):
        super().__init__(n, localdim)

        self.ntypdim = self.localdim-1
        self.nexcdim = self.n+1

        if localdim == 2 and isinstance(psi_exc, np.ndarray) and len(psi_exc.shape) == 1:
            psi_exc = psi_exc[:,np.newaxis]

        _validate(r""" list($psi_exc .shape)
                       == [$self.nexcdim**$self.ntypdim,1] """)

        self.psi_exc = psi_exc
        self.q_psi_exc = qutip.Qobj(
            self.psi_exc,
            dims=[[self.nexcdim]*self.ntypdim,[1]*self.ntypdim]
        )

        self.symmetric_basis_vec_to_full_ket_cache = None

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.n, self.localdim, self.q_psi_exc)

    def norm(self):
        return self.q_psi_exc.norm()

    def normalized_by(self, div_factor):
        return SymmetricKet(self.n, self.localdim, self.q_psi_exc/div_factor)

    def to_density_operator(self):
        return SymmetricOperator(self.n, self.localdim, self.q_psi_exc * self.q_psi_exc.dag())

    def get_symm_sparse(self):
        return self.q_psi_exc.data

    def to_symmetric_superposition_ket(self):
        # find excitations on which we have a nonzero coefficient
        rows, cols = self.q_psi_exc.data.nonzero()
        psi_x = []
        excitations = []
        for r in rows:
            psi_x.append(self.q_psi_exc.data[r, 0])
            excitations.append([r])
        return SymmetricSuperpositionKet(self.n, self.localdim, psi_x, excitations)


    def set_symmetric_basis_vec_to_full_ket_cache(self, cache_object):
        _validate(r""" isinstance($cache_object, SymmetricBasisVecToFullKetCache) """)
        self.symmetric_basis_vec_to_full_ket_cache = cache_object

    def to_sparse(self):
        if self.symmetric_basis_vec_to_full_ket_cache is None:
            self.symmetric_basis_vec_to_full_ket_cache = \
                SymmetricBasisVecToFullKetCache(self.n, self.localdim)
            
        cache = self.symmetric_basis_vec_to_full_ket_cache
        result = 0

        for p, pp, val in iter_spmatrix(self.q_psi_exc.data):
            ket = cache.get(p)
            assert pp == 0
            
            result = result + val * ket

        return result





class SymmetricOperator(OperatorDesc):
    def __init__(self, n, localdim, operator_exc):
        super().__init__(n, localdim)

        self.ntypdim = self.localdim-1
        self.nexcdim = self.n+1

        _validate(r""" list($operator_exc .shape)
                       == [$self.nexcdim**$self.ntypdim,$self.nexcdim**$self.ntypdim] """)

        self.operator_exc = operator_exc
        self.q_operator_exc = qutip.Qobj(
            self.operator_exc,
            dims=[[self.nexcdim]*self.ntypdim,[self.nexcdim]*self.ntypdim]
        )

        self.real_overlap = self._real_overlap_via_overlap

        self.symmetric_basis_vec_to_full_ket_cache = None

    def operator_apply(self, other):
        _validate(r""" isinstance($other, SymmetricOperator) """)
        _validate(r""" $other.n == $self.n  and  $other.localdim == $self.localdim """)
        return SymmetricOperator(self.n, self.localdim,
                                 self.q_operator_exc * other.q_operator_exc)

    def _symm_expect(self, other):
        _validate(r""" isinstance($other, (SymmetricKet, SymmetricOperator)) """)
        _validate(r""" $other.n == $self.n  and  $other.localdim == $self.localdim """)
        if isinstance(other, SymmetricOperator):
            return qutip.expect(self.q_operator_exc, other.q_operator_exc)
        elif isinstance(other, SymmetricKet):
            return qutip.expect(self.q_operator_exc, other.q_psi_exc)
        raise ValueError(f"I don't know how to deal with this {other=}")

    def get_symm_sparse(self):
        return self.q_operator_exc.data

    def set_symmetric_basis_vec_to_full_ket_cache(self, cache_object):
        _validate(r""" isinstance($cache_object, SymmetricBasisVecToFullKetCache) """)
        self.symmetric_basis_vec_to_full_ket_cache = cache_object

    def to_sparse(self):
        if self.symmetric_basis_vec_to_full_ket_cache is None:
            self.symmetric_basis_vec_to_full_ket_cache = \
                SymmetricBasisVecToFullKetCache(self.n, self.localdim)
            
        cache = self.symmetric_basis_vec_to_full_ket_cache
        result = 0

        for p, pp, val in iter_spmatrix(self.q_operator_exc.data):
            ket = cache.get(p)
            bra = cache.get(pp).conj().T
            
            result = result + val * ket @ bra

        return result

    def roverlap(self, A):
        if isinstance(A, SymmetricKet):
            return (A.q_psi_exc.data @ self.q_operator_exc.data @ A.q_psi_exc.data).sum()

        if not isinstance(A, SymmetricOperator):
            self._roverlap_via_sparse(A)

        # overlap with SymmetricOperator
        _validate(r""" $self .n == $A .n and $self .localdim == $A .localdim """)

        return self.q_operator_exc.data.T.multiply(A.q_operator_exc.data).sum()

    def overlap(self, A):
        if isinstance(A, SymmetricKet):
            return (A.q_psi_exc.data.conj().T @ self.q_operator_exc.data.conj().T
                    @ A.q_psi_exc.data).sum()

        if not isinstance(A, SymmetricOperator):
            self._overlap_via_sparse(A)

        # overlap with SymmetricOperator
        _validate(r""" $self .n == $A .n and $self .localdim == $A .localdim """)

        return self.q_operator_exc.data.conj().multiply(A.q_operator_exc.data).sum()
        




class SymmetricEnsembleKetBras(OperatorDesc):
    r"""
    Same as `EnsembleKetBras`, but now kets and bras are specified in the
    symmetric subspace as for `SymmetricOperator`.
    """
    def __init__(self, n, localdim, coeffs, symm_kets, symm_bras=None):
        super().__init__(n, localdim)

        # only for qubits for now
        _validate(r""" $localdim == 2 """)
        
        # use an internal `EnsembleKetBras` instance for a "single" q-(n+1)-dit
        # which represents the symmetric subspace.
        self.ensemble = EnsembleKetBras(1, n+1, coeffs, symm_kets, symm_bras)

    def to_symmetric_operator(self):
        return SymmetricOperator(self.n, self.localdim, self.ensemble.to_sparse())

    def overlap(self, A):
        _validate(r""" isinstance(A, SymmetricOperator) """)
        return self.ensemble.overlap(A.get_symm_sparse())

    def roverlap(self, A):
        _validate(r""" isinstance(A, SymmetricOperator) """)
        return self.ensemble.roverlap(A.get_symm_sparse())

    def real_overlap(self, A):
        _validate(r""" isinstance(A, SymmetricOperator) """)
        return self.ensemble.real_overlap(A.get_symm_sparse())







# ------------------------------------------------




class SymmetricSuperpositionKet(KetDesc):
    def __init__(self, n, localdim, psi_x, excitations):
        r"""
        Describe a state living in the symmetric subspace of `n` q-`localdim`-its
        
        - `psi_x` -- 1D numpy array storing the coefficients for the different
           excitation types.

        - `excitations` is a 2D numpy array of integer type. Here
          `excitations[x][a]` is the number of excitations of (non-zero) type
          `a` (with `a`=0,...,`localdim-2`) sites that are in the local basis
          state i.  We ignore excitations of basis type `0` because they can be
          inferred from the remaining excitations.

          Each "excitation specification" `excitations[x]` corresponds to the
          basis vector::

            ∑ₓ  psi_x[x] |hₙ[excitations[x]]⟩

          Where `|hₙ[exc]⟩` is the symmetrized version of a state with `exc[a]`
          qudits in the basis state `1+a` and `n - ∑ₐ exc[a]` qudits in the
          ground state.
        """
        super().__init__(n, localdim)


        self.psi_x = np.array(psi_x)
        self.excitations = np.array(excitations)

        if self.localdim == 2 and len(self.excitations.shape) == 1:
            self.excitations = self.excitations[:,np.newaxis]

        # checks that all the excitation specifications are unique --
        _validate(r""" len(set([tuple(e) for e in self.excitations]))
                       == len(self.excitations) """)

        self.num_psi_x = self.psi_x.shape[0]

        _validate('$self.psi_x.shape == ($self.num_psi_x, )')

        _validate('$self.excitations.shape == ($self.num_psi_x, $self.localdim-1)')
        _validate('np.issubdtype($self.excitations.dtype, np.integer)')

    def norm(self):
        return np.sqrt((self.psi_x.conj()*self.psi_x).sum())

    def normalized_by(self, div_factor):
        return SymmetricSuperpositionKet(self.n, self.localdim,
                                         self.psi_x/div_factor, self.excitations)

    def to_symmetric_ket(self):
        # only implemented for qubits
        _validate(r""" $self.localdim == 2 """)
        psi_exc = np.zeros(shape=(self.n+1,))
        psi_exc[self.excitations[:,0]] = self.psi_x
        return SymmetricKet(self.n, 2, psi_exc)
        

    def to_ditstring_superposition_ket(self):
        
        psi_x = np.zeros(shape=(0,), dtype=complex)
        xn = np.zeros(shape=(0, self.n), dtype=int)

        for j_exc, (coeff_exc, exc) in enumerate(zip(self.psi_x, self.excitations)):
            # for each excitation structure, prepare the associated basis states
            # with all their permutations

            these_xn = get_symm_state_basis_ditstrings(self.n, exc)
            num_xn = these_xn.shape[0]

            #print(psi_x)
            unif_coeff_exc_by_xns = (coeff_exc/np.sqrt(num_xn))*np.ones((num_xn,))
            #print(unif_coeff_exc_by_xns)
            psi_x = np.concatenate([psi_x, unif_coeff_exc_by_xns])
            xn = np.concatenate([xn, these_xn], axis=0)

        return DitstringSuperpositionKet(self.n, self.localdim, psi_x, xn)





class SymmetricTraceWithKrausPairStringCalculator(TraceWithKrausPairStringCalculator):
    def __init__(self, local_kraus_operators, kraus_pair_string):
        super().__init__(local_kraus_operators, kraus_pair_string)

        _validate(r""" $self.local_kraus_operators.localdim == 2 """)

        self.k = self.kraus_pair_string.shape[0] # number of kraus operator pairs

        if not self.kraus_pair_string.size:

            self.EdE = None
            self.calculate_trace = self._calculate_trace_singleop

        elif np.all(self.kraus_pair_string == self.kraus_pair_string[0,:]):

            # it's the same operator E_{k'}^† E_k on each site -- we have a
            # special routine for that
            Ekp = self.local_kraus_operators.local_kraus_operators[
                self.kraus_pair_string[0,0]
            ]
            Ek = self.local_kraus_operators.local_kraus_operators[
                self.kraus_pair_string[0,1]
            ]
            self.EdE = (Ekp.dag() * Ek).full()

            self.calculate_trace = self._calculate_trace_singleop

        else:

            # prepare the list of distinct operators that appear in the sandwich
            Eks = self.local_kraus_operators.local_kraus_operators

            ops_pairs = []
            ops_count = []
            found_kraus_pairs = {}
            for l, r in self.kraus_pair_string:
                if (l,r) in found_kraus_pairs:
                    j = found_kraus_pairs[(l,r)]
                    ops_count[j] += 1
                    continue
                # add new unique kraus operator pair
                found_kraus_pairs[ (l,r) ] = len(ops_pairs)
                ops_pairs.append( (l,r) )
                ops_count.append( 1 )

            self.ops = np.array([
                (Eks[l].dag()*Eks[r]).full()
                for l, r in ops_pairs
            ])
            self.ops_count = np.array(ops_count, dtype=int)

            assert np.sum(ops_count) == self.k

            self.calculate_trace = self._calculate_trace_general



    def _calculate_trace_singleop(self, symm_operator_desc):
        _validate(r""" isinstance($symm_operator_desc, SymmetricOperator) """)
        _validate(r""" $symm_operator_desc.n == len($self.kraus_pair_string) """)
        _validate(r""" $symm_operator_desc.localdim == 2 """)
        _validate(r""" len($self.kraus_pair_string) == 0  or
                       np.all($self.kraus_pair_string == $self.kraus_pair_string[0,:]) """)

        #
        # We need to compute
        #
        # ⟨ hᵏ_{r'} | A^⊗k | hᵏᵣ ⟩
        #
        # = (k choose r)⁻½  (k choose r')⁻½
        #     ∑_{|x⃗|=r,|x⃗'|=r'}  ⟨0|A|0⟩^#(0,0) ⟨0|A|1⟩^#(0,1) ⟨1|A|0⟩^#(1,0) ⟨1|A|1⟩^#(1,1)
        #
        # where #(a,b) is the number of sites i where x⃗[i] == a and x⃗'[i] == b.
        #
        # So we need to count the # of pairs of bitstrings with given
        # constraints #(0,0)=:m_00, #(0,1)=:m_01, #(1,0)=:m_10, #(1,1)=:m_11,
        # along with m_11+m_10 = r and m_11+m_01 = r'.
        #
        # Strategy:
        #
        #   - For fixed m_00, m_01, m_10, m_11, there are (n -multinomial- m_00,
        #     m_01, m_10, m_11) choices of pairs of bitstrings with the given
        #     frequencies.
        #
        #   - Sum over choices of m_11=0...min(r,r'), and sub-sum over
        #     m_01=0...(r-m_11) and m_10=0...(r'-m_11).  In each term, we have
        #     m_00 = k-m_11-m_01-m_10.
        #
        #
        # To compute the multinomial coefficient as we're performing the
        # iterations.  We write the multinomial as a product of binomials:
        #
        #  (    k                )     (  k   ) ( k-m_11 ) ( k-m_11-m_10 )
        #  ( m_00 m_01 m_10 m_11 )  =  ( m_11 ) (  m_10  ) (    m_01     )
        #
        # The first binomial is computed using the recursion relations as we
        # iterate through m_11.  The other two are computed with spa.comb().

        k = self.k

        result_trace = 0.0

        EdE = self.EdE

        #print(f"{EdE=}")

        # iterate over matrix elements of symm_operator_desc
        A = symm_operator_desc.get_symm_sparse()

        #print(f"""{A=}\n{A.toarray()=}""")

        #rows, cols = A.nonzero()
        #for r,rp in zip(rows, cols):
        for r, rp, Arrp in iter_spmatrix(A):

            #print(f"({r=},{rp=})")

            this_r_rp_term = 0.0

            # Bounds on m_11 are given by the constraints:
            #
            #   m_11 + m_10 == r    ~~>    m_11 <= r   &  m_10 = r - m_11
            #   m_11 + m_01 == r'   ~~>    m_11 <= r'  &  m_01 = r' - m_11
            #
            #   \sum m_ij == k      ~~>    m_00 = k - m_11 - m_01 - m_10
            #   m_00 >= 0  ~~>  k >= m_11 + m_01 + m_10 = r + r' - m_11
            #              ~~>  m_11 >= r + r' - k
            #
            binom_k_m11 = 1
            for m_11 in range(max(0, r+rp-k), min(r,rp)+1):

                m_10 = r - m_11
                m_01 = rp - m_11
                m_00 = k - m_11 - m_10 - m_01

                binom_coeff = (binom_k_m11
                               * spa.comb(k-m_11, m_10, exact=True)
                               * spa.comb(k-m_11-m_10, m_01, exact=True))

                # add this term
                this_r_rp_term += binom_coeff * (
                    #EdE[0,0]**m_00 * EdE[0,1]**m_01 * EdE[1,0]**m_10 * EdE[1,1]**m_11
                    (EdE[0,0]**m_00 if m_00 else 1) * 
                    (EdE[0,1]**m_01 if m_01 else 1) * 
                    (EdE[1,0]**m_10 if m_10 else 1) * 
                    (EdE[1,1]**m_11 if m_11 else 1)
                )
            
                #print(f"{m_11=} {m_10=} {m_01=} {m_00=}; {binom_coeff=}; {this_r_rp_term=}")

                # update binomial coefficient
                binom_k_m11 = binom_k_m11 * (k - m_11) // (m_11 + 1)
                #print(f"updated {binom_k_m11=}")


            kcr  = spa.comb(k, r, exact=True)
            kcrp = spa.comb(k, rp, exact=True)

            # float() to avoid weird errors that 'int' doesn't have member name 'sqrt'
            result_trace += Arrp * this_r_rp_term / np.sqrt(float(kcr*kcrp))

        return result_trace



    def _calculate_trace_general(self, symm_operator_desc):
        _validate(r""" isinstance($symm_operator_desc, SymmetricOperator) """)
        _validate(r""" $symm_operator_desc.n == len($self.kraus_pair_string) """)
        _validate(r""" $symm_operator_desc.localdim == 2 """)

        # implementation depends on the fact that np.power(0., 0) == 1 and not
        # any NaN stuff
        _validate(r""" np.power(0., 0) == 1 """)

        k = self.k

        ops = self.ops
        mjs = self.ops_count
        num_ops = ops.shape[0]

        #print(f"""{mjs=}\n{ops=}""")

        result_trace = 0.0

        # iterate over matrix elements of symm_operator_desc
        A = symm_operator_desc.get_symm_sparse()

        #print(f"""{A=}\n{A.toarray()=}""")

        use_exact_comb = True

        # Now, we need to iterate over all possible combinations of
        # the numbers { m_(j,a,ap) }_(j,a,ap).  Here
        #
        #   m_(j,a,ap) = # of sites i within the block j with bit pair
        #   values (x_i = a, x'_i = ap)
        #

        ### a bit slow, presumably because of recursion?  We could try to unfold this function.
        def iter_partitions_of_r(r, mjs):
            for r0 in range( max(0, r - np.sum(mjs[1:])), min(r,mjs[0])+1 ):
                if len(mjs) == 1:
                    yield (r0,)
                    return
                for rrest in iter_partitions_of_r(r-r0,mjs[1:]):
                    yield (r0,)+rrest
                
        def iter_mab(rj, rjp, mj):
            for m11 in range( max(0,rj+rjp-mj), min(rj,rjp)+1 ):
                m10 = rj - m11
                m01 = rjp - m11
                m00 = mj - m01 - m10 - m11
                yield [[ m00, m01 ], [ m10, m11 ]]

        for r, rp, Arrp in iter_spmatrix(A):

            #print(f"({r=},{rp=}) -> {Arrp=}")

            this_r_rp_term = 0

            for rjs, rjps in itertools.product(*[ iter_partitions_of_r(r, mjs),
                                                  iter_partitions_of_r(rp, mjs), ]):
                for m_jab in itertools.product(*[
                        iter_mab(rjs[j], rjps[j], mjs[j])
                        for j in range(num_ops)
                ]):
                    m_jab = np.array(m_jab, dtype=int)
                    #print(f"{m_jab=}")

                    # compute this term
                    this_term = np.prod(np.power(ops, m_jab))
                    #print(f"    [{this_term=} [{np.power(ops, m_jab)=}]")
                    for j in range(num_ops):
                        m_coeff = multinomial_coefficient(m_jab[j,:,:].flatten(),
                                                          exact=use_exact_comb)
                        #print(f"        [ × {m_coeff=}   -- associated with {j=}]")
                        this_term *= m_coeff

                    this_r_rp_term += this_term
            #

            kcr  = spa.comb(k, r, exact=use_exact_comb)
            kcrp = spa.comb(k, rp, exact=use_exact_comb)

            # float() to avoid weird errors that 'int' doesn't have member name 'sqrt'
            this_r_rp_term_withfactors = Arrp * this_r_rp_term / np.sqrt(float(kcr * kcrp))

            #print(f"  ({r=},{rp=}) {Arrp=} {kcr=} {kcrp=}  ->  {this_r_rp_term_withfactors=}")

            result_trace += this_r_rp_term_withfactors

        return result_trace



        










class SymmetricReducedOperatorsCalculator(ReducedOperatorsCalculator):
    def __init__(self, n, localdim, symm_operators, local_kraus_operators):
        super().__init__(n, localdim, len(symm_operators))

        # Implementation only works for qubits (at least for now)
        _validate(r""" $self.localdim == 2 """)

        self.symm_operators = symm_operators

        _validate(r""" np.all([isinstance(s, SymmetricOperator) for s in $self.symm_operators]) """)

        E0 = local_kraus_operators.local_kraus_operators[0]
        E0dagE0 = E0.dag() * E0

        _validate(r""" npl.norm(np.diag(np.diag(E0dagE0.full())) - E0dagE0.full(), 2) <= 1e-6 """)
        self.E0dagE0_diag = np.diag(E0dagE0.full())

        # predetermine which (row,column) pairs of the symm_operators have
        # nonzero entries.  We will have to compute an associated reduced
        # operator for each of those (row/col) pairs
        self.rowcols = frozenset(
            (row,col)
            for S in symm_operators
            for (row,col) in np.array(S.q_operator_exc.data.nonzero(),dtype=int).T
        )

    def reduced_operators(self, which_sites):
        # of course, the only relevant thing about which_sites is the how many
        # sites, since we're symmetric over all sites
        n = self.n
        k = len(which_sites)

        E0dagE0_diag = self.E0dagE0_diag

        # compute n-choose-k now for use later
        n_nk_binom = spa.comb(n, n-k, exact=True)
        n_nk_binom_2 = n_nk_binom*n_nk_binom

        #print(f"{which_sites=}  {E0dagE0_diag=}")

        red_Xi = [
            np.zeros(shape=(k+1,k+1,), dtype=complex)
            for _ in self.symm_operators
        ]

        for p,pp in self.rowcols:

            diffpp = pp-p
            diffp = np.absolute(diffpp)
            minppp = min(p, pp)
            # We can straight away skip pairs (p,p') where |p'-p| > k:
            if diffp > k:
                continue

            # the reduced operator on k sites of |hₚ⟩⟨hₚₚ| is a matrix with the
            # elements `red_p_pp_dat` placed on the `red_p_pp_off`-th diagonal.
            # The `red_p_pp_off` is an offset from the main diagonal, like
            # sps.diags: ==0 for main diagonal, >0 upper tri part, <0 for lower
            # tri part.
            #
            red_p_pp_dat, red_p_pp_off = \
                reduced_k_symmetric_hp_hpp(
                    n, k, p, pp,
                    n_nk_binom=n_nk_binom,
                    n_nk_binom_2=n_nk_binom_2,
                    mk_sps_diag_fn=lambda dat,off: (dat,off)
                )
            
            # compute ⟨hⁿ⁻ᵏ_q|E₀^† E₀|hⁿ⁻ᵏ_q⟩ with q ranging in the range such
            # that q = max(p-k,pp-k,0)...min(p,pp,n-k)
            qmin = max(p-k, pp-k, 0)
            qmax = min(minppp, n-k)
            q_vals = np.arange( qmin, qmax+1 )
            E0_mult_vals = E0dagE0_diag[1]**q_vals * E0dagE0_diag[0]**(n-k-q_vals)
            #print(f"{k=} ({p=},{pp=})  {red_p_pp_dat=}  {red_p_pp_off=}  {q_vals=}  {E0_mult_vals=}")
            diag_dat = np.array(red_p_pp_dat[:], dtype=complex)
            diag_dat[minppp-q_vals] *= E0_mult_vals
            #print(f"{diag_dat=}")

            for i,s in enumerate(self.symm_operators):

                value = s.q_operator_exc.data[p,pp]

                if red_p_pp_off >= 0: # upper tri part diagonal
                    idx = (range(len(red_p_pp_dat)),range(red_p_pp_off,k+1))
                else: # red_p_pp_off < 0  --- lower tri part diagonal
                    idx = (range(-red_p_pp_off,k+1),range(len(red_p_pp_dat)))

                red_Xi[i][idx] += value * diag_dat

        #print(f"done -> {red_Xi=}")

        return tuple([
            SymmetricOperator(k, self.localdim, Xi)
            for Xi in red_Xi
        ])




















# ------------------------------------------------------------------------------





def get_symm_state_basis_ditstrings(n, exc):
    r"""
    Returns `xn`, the ditstrings that form the basis state assocaited with the
    symmetric state with excitation structure `exc`.

    - `exc[a]` is the number of qudits that are excited to the `1+a`-th basis
      state; the remaining qudits are in the ground state `0`
    """
    if not np.array(exc).size:
        return np.zeros(n, dtype=int)

    Xn_start = [0] * (n-np.sum(exc))  +  functools.reduce(
            lambda a, b: a+b,
            [ [1+j] * exc[j]
              for j in range(len(exc)) ]
        )
    # e.g.  Xn = 0 0 0 0 0 0 0 1 1 1 1 2 2 3 3 3 3 3 3

    all_xn = []
    for Xn in _next_permutation(Xn_start):
        all_xn.append(Xn)

    return np.array(all_xn, dtype=int)
        



def _cmp(a, b):
    return (a > b) - (a < b) 

# see https://stackoverflow.com/a/4250183/1694896 and
# http://blog.bjrn.se/2008/04/lexicographic-permutations-using.html
def _next_permutation(seq, pred=_cmp):
    """Like C++ std::next_permutation() but implemented as
    generator. Yields copies of seq."""
    def reverse(seq, start, end):
        # seq = seq[:start] + reversed(seq[start:end]) + \
        #       seq[end:]
        end -= 1
        if end <= start:
            return
        while True:
            seq[start], seq[end] = seq[end], seq[start]
            if start == end or start+1 == end:
                return
            start += 1
            end -= 1
    if not len(seq):
        return #raise StopIteration
    try:
        seq[0]
    except TypeError:
        raise TypeError("seq must allow random access.")
    first = 0
    last = len(seq)
    seq = seq[:]
    # Yield input sequence as the STL version is often
    # used inside do {} while.
    yield seq[:]
    if last == 1:
        return #raise StopIteration
    while True:
        next = last - 1
        while True:
            # Step 1.
            next1 = next
            next -= 1
            if pred(seq[next], seq[next1]) < 0:
                # Step 2.
                mid = last - 1
                while not (pred(seq[next], seq[mid]) < 0):
                    mid -= 1
                seq[next], seq[mid] = seq[mid], seq[next]
                # Step 3.
                reverse(seq, next1, last)
                # Change to yield references to get rid of
                # (at worst) |seq|! copy operations.
                yield seq[:]
                break
            if next == first:
                return #raise StopIteration
    return #raise StopIteration            






# ------------------------------------------------------------------------------







def reduced_k_symmetric_hp_hpp(n, k, p, pp, *,
                               n_nk_binom=None, n_nk_binom_2=None, mk_sps_diag_fn=None):

    # no numpy types, so that we can work with Python's integer arithmetic for large system sizes
    n = int(n)
    k = int(k)
    p = int(p)
    pp = int(pp)

    minppp = min(p, pp)

    qmin = max(p-k, pp-k, 0)
    qmax = min(minppp, n-k)

    diffp = np.absolute(pp-p)

    #if n > 100:
    #    need_careful_int_arithmetic = True

    if mk_sps_diag_fn is None:
        mk_sps_diag_fn = lambda data, offset: sps.diags(data, offset, shape=(k+1,k+1))

    if diffp > k:
        # we can directly return the all zero matrix in this case because
        # this situation is excluded
        #return np.zeros(shape=(k+1,k+1))
        return mk_sps_diag_fn([], k)

    if n_nk_binom_2 is None:
        n_nk_binom = spa.comb(n, n-k, exact=True)
        n_nk_binom_2 = n_nk_binom*n_nk_binom
    else:
        n_nk_binom_2 = int(n_nk_binom_2)
        n_nk_binom = int(n_nk_binom)

    # the matrix we are forming consists of only the (pp-p)-th off-diagonal.
    # Prepare the diagonal entries first.  The full matrix is (k+1)×(k+1)
    # and the x-th off-diagonal has k+1-|x| entries.
    theoffdiags = np.zeros(shape=(k + 1 - diffp))

    # iterate over binomial coefficients using recurrence relations
    # (cf. Drafts&Calculations Vol. XI 8.12.2019)
    binom = lambda N, K: spa.comb(N, K, exact=True)
    pq_comb = binom(p, qmin)
    ppq_comb = binom(pp, qmin)
    bin2_comb = binom(n-p, n-k-qmin)
    bin2p_comb = binom(n-pp, n-k-qmin)

    #print(f"{pq_comb=} {type(pq_comb)=}, {ppq_comb=} {type(ppq_comb)=}, {bin2_comb=}, {type(bin2_comb)=}, {bin2p_comb=}, {type(bin2p_comb)=}")
    #print(f"{type(n_nk_binom_2)=}")

    for q in range(qmin, qmax+1):
        #print("theoffdiags=",theoffdiags, ", minppp-q=", minppp-q)
        #if 1: #not  need_careful_int_arithmetic:
        # for small enough system sizes I think this is good enough
        theoffdiags[minppp-q] = np.sqrt(
            pq_comb * ppq_comb * bin2_comb * bin2p_comb / n_nk_binom_2  # NOT int division //
        )
        #else:
        #    # we gotta really care about overflows
        #    intpart, rem = divmod(int(pq_comb) * int(ppq_comb) * int(bin2_comb) * int(bin2p_comb), n_nk_binom)
        #    intpart2, rem2 = divmod(intpart, n_nk_binom)
        #    theoffdiags[minppp-q] = np.sqrt( float(intpart2) + float(rem2/n_nk_binom) + float(rem/n_nk_binom_2) )
        # update binom coefficients
        pq_comb = pq_comb * (p-q) // (q+1)
        ppq_comb = ppq_comb * (pp-q) // (q+1)
        bin2_comb = bin2_comb * (n-k-q) // (k+q+1-p)
        bin2p_comb = bin2p_comb * (n-k-q) // (k+q+1-pp)
        #print(f"{type(pq_comb)=}, {type(ppq_comb)=}, {type(bin2_comb)=}, {type(bin2p_comb)=}")

    # create the resulting matrix
    return mk_sps_diag_fn(theoffdiags, pp-p)
#
