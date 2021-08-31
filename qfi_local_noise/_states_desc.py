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

from ._util import abs2, _validate

# Special logger for all the _validate calls, so that it can be silenced easily.
# The logger instance will be found by name in the global scope of this module
# by the _validate() command itself.
from .validate import validate_logger


class KetDesc:
    def __init__(self, n, localdim):
        super().__init__()
        self.n = n
        self.localdim = localdim

    def norm(self):
        raise RuntimeError("Method isn't reimplemented by subclass")

    def normalized_by(self, div_factor):
        raise RuntimeError("Method isn't reimplemented by subclass")

    def to_sparse(self):
        raise RuntimeError("Method isn't reimplemented by subclass")

#

class OperatorDesc:
    def __init__(self, n, localdim):
        super().__init__()
        self.n = n
        self.localdim = localdim

    def to_sparse(self):
        r"""
        Computes the real part of the Hilbert-Schmidt inner product of the current
        operator with A but without taking any daggers::

          Re tr(O A^\dagger) = Re tr(O^\dagger A)
        """
        raise RuntimeError("Method isn't reimplemented by subclass")

    def overlap(self, A):
        r"""
        Computes the complex-valued Hilbert-Schmidt inner product of the current
        operator with A, where the current operator is the one with the dagger::

          returned value = tr(O^\dagger A)

        Subclasses can implement this method for a choice of different
        possibilities of what `A` can be.  E.g. `EnsembleKetBras` accept numpy
        arrays and scipy.sparse objects.

        We don't automatically provide default implementation, because we don't
        want a typo or another dumb thing to make an inefficient implementation
        silently kick in.  If subclasses reimplement `to_sparse()` and want to
        implement overlap() by first computing a sparse representation,
        they can use in their constructors the instruction::

            self.overlap = self._overlap_via_sparse
        """
        raise RuntimeError("Method isn't reimplemented by subclass")

    def real_overlap(self, A):
        r"""
        Computes the real part of the Hilbert-Schmidt inner product of the current
        operator with A but without taking any daggers::

          Re tr(O A^\dagger) = Re tr(O^\dagger A)


        We don't automatically provide default implementation, because we don't
        want a typo or another dumb thing to make an inefficient implementation
        silently kick in.  If subclasses reimplement `to_sparse()` and want to
        implement real_overlap() by first computing a sparse representation,
        they can use in their constructors the instruction::

            self.real_overlap = self._real_overlap_via_sparse
        """
        raise RuntimeError("Method isn't reimplemented by subclass")

    def roverlap(self, A):
        r"""
        Computes the complex value resulting from taking the trace of the product of
        this operator with `A`.  No daggers are applied::

          returned value = tr(O A)

        Subclasses can implement this method for a choice of different
        possibilities of what `A` can be.  E.g. `EnsembleKetBras` accept numpy
        arrays and scipy.sparse objects.

        We don't automatically provide default implementation, because we don't
        want a typo or another dumb thing to make an inefficient implementation
        silently kick in.  If subclasses reimplement `to_sparse()` and want to
        implement overlap() by first computing a sparse representation,
        they can use in their constructors the instruction::

            self.overlap = self._overlap_via_sparse
        """
        raise RuntimeError("Method isn't reimplemented by subclass")


    def _overlap_via_sparse(self, A):
        return self.to_sparse().conj().multiply(A).sum()

    def _roverlap_via_sparse(self, A):
        return self.to_sparse().T.multiply(A).sum()

    def _real_overlap_via_overlap(self, A):
        return self.overlap(A).real

    def _real_overlap_via_sparse(self, A):
        return self.to_sparse().conj().multiply(A).real.sum()



class DenseOperatorDesc(OperatorDesc):
    def __init__(self, n, localdim, O):
        super().__init__(n, localdim)
        self.O = np.array(O)

    def to_sparse(self):
        raise sps.csr_matrix(self.O)

    def roverlap(self, A):
        if sps.issparse(A):
            return A.multiply(self.O.T).sum()
        return np.multiply(self.O.T, A).sum()

    def overlap(self, A):
        if sps.issparse(A):
            return A.multiply(self.O.conj()).sum()
        return np.multiply(self.O.conj(), A).sum()

    def real_overlap(self, A):
        if sps.issparse(A):
            return A.multiply(self.O.conj()).real.sum()
        return np.multiply(self.O.conj(), A).real.sum()







class PsiXiDesc(OperatorDesc):
    r"""
    Base class to specify the action of a Hamiltonian on a pure state.

    - `psi_ket_desc` shoule be a `KetDesc`.

    Subclasses should store useful information in a way that's useful to them.
    They should compute the (not necessarily public/documented) attributes
    `avg_H_psi` and `var_H_psi`.
    """
    def __init__(self, psi_ket_desc):
        super().__init__(psi_ket_desc.n, psi_ket_desc.localdim)

        self.psi_ket_desc = psi_ket_desc

        # these should be initialized by inherited classes
        self.avg_H_psi = None
        self.var_H_psi = None

    def average_H(self):
        return self.avg_H_psi

    def variance(self):
        # returns ⟨H²⟩_ψ - ⟨H⟩_ψ²
        return self.var_H_psi




class ReducedOperatorsCalculator:
    def __init__(self, n, localdim, num_operators):
        super().__init__()
        self.n = n
        self.localdim = localdim
        self.num_operators = num_operators

    def __call__(self, which_sites):
        return self.reduced_operators(which_sites)

    def reduced_operators(self, which_sites):
        r"""
        Return a tuple `(A_k, B_k, ...)` describing the reduced operators of the
        given operators for which we are to compute the reduced operators, on
        the sites indexed by `which_sites`.  Here `which_sites` is a list or
        tuple of site indices.

        The attribute `num_operators` of the current object provides the
        number of operators for which we are simultaneously computing the
        reduced operator.

        Each `A_k`, `B_k`, etc. should be `OperatorDesc` instances.  For
        instance, they could be `EnsembleKetBras` instances.
        """
        raise RuntimeError("Should have been implemented by subclass")
    




# ------------------------------------------------------------------------------


class QutipKet(KetDesc):
    def __init__(self, ket, n, localdim):
        super().__init__(n, localdim)
        self.ket = ket

        _validate(r""" $self.ket.dims == [ [$self.localdim] * $self.n, [1]*$self.n ] """)

    def norm(self):
        return self.ket.norm()

    def normalized_by(self, div_factor):
        return QutipKet(self.ket/div_factor, self.n, self.localdim)

    def to_sparse(self):
        return self.ket.data



class QutipOperator(OperatorDesc):
    def __init__(self, n, localdim, operator):
        super().__init__(n, localdim)
        self.operator = operator

        _validate(r""" $self.operator .dims ==
                       [ [$self.localdim] * $self.n, [$self.localdim] * $self.n ] """)

        self.real_overlap = self._real_overlap_via_overlap

    def __repr__(self):
        return "{}({}, {}, <{}>)".format(self.__class__.__name__, self.n,
                                       self.localdim, self.operator)

    def to_sparse(self):
        return self.operator.data

    def overlap(self, A):
        if isinstance(A, qutip.Qobj):
            A = A.data
        return self.operator.data.conj().multiply(A).sum()

    def roverlap(self, A):
        if isinstance(A, qutip.Qobj):
            A = A.data
        return self.operator.data.T.multiply(A).sum()



class QutipReducedOperatorCalculator(ReducedOperatorsCalculator):
    def __init__(self, n, localdim, q_operators, local_kraus_operators):
        super().__init__(n, localdim, len(q_operators))
        self.q_operators = q_operators
        self.local_kraus_operators = local_kraus_operators
        E0 = self.local_kraus_operators.local_kraus_operators[0]
        self.E0dagE0 = E0.dag() * E0

    def reduced_operators(self, which_sites):
        Kr = qutip.tensor(*[
            qutip.qeye(self.localdim) if i in which_sites else self.E0dagE0
            for i in range(self.n)
        ])
        #print(f"{which_sites=}\n{Kr=}")
        if len(which_sites) == 0:
            return [
                QutipOperator(1, 1, qutip.Qobj([[(Kr*q).tr()]]))
                for q in self.q_operators
            ]
        if len(which_sites) == self.n:
            # no partial trace to take
            return [QutipOperator(self.n, self.localdim, q) for q in self.q_operators]
        return [
            QutipOperator(len(which_sites), self.localdim,
                          (Kr*q).ptrace(which_sites))
            for q in self.q_operators
        ]



# ------------------------------------------------------------------------------


class DitstringSuperpositionKet(KetDesc):
    def __init__(self, n, localdim, psi_x, xn, *, tol_check=1e-6, normalized=True):
        super().__init__(n, localdim)

        # verify that all ditstrings xn[j,:] are unique:
        _validate(r""" len(set([tuple(X) for X in xn])) == len(xn) """)

        self.psi_x = np.array(psi_x)
        self.xn = np.array(xn)

        self.num_x = xn.shape[0]

        self.tol_check = tol_check

        self.psi_xp = abs2(psi_x)

        self.normalized = normalized

        _validate(r""" list($self.psi_x .shape) == [ $self.num_x ] """)
        if self.normalized:
            _validate(r""" np.absolute( np.sum($self.psi_xp) - 1 ) < $self.tol_check """)

        _validate(r""" np.issubdtype($self.xn .dtype, np.integer) """)
        _validate(r""" list($self.xn .shape) == [ $self.num_x, $self.n ] """)


    def norm(self):
        return np.sqrt((self.psi_x.conj()*self.psi_x).sum())
        
    def normalized_by(self, div_factor):
        return DitstringSuperpositionKet(self.n, self.localdim, self.psi_x/div_factor,
                                         self.xn, tol_check=self.tol_check)

    def to_sparse(self):
        comp_basis_idxs = np.flip(self.localdim**np.arange(self.n)).dot(self.xn.T)
        return sps.coo_matrix(
            # (data, (row, col))
            (self.psi_x, (comp_basis_idxs, np.zeros_like(comp_basis_idxs))),
            shape=(self.localdim**self.n,1),
        ).tocsr()

#

class EnsembleKetBras(OperatorDesc):
    r"""
    A n-body operator described as a sum of coeff|ket⟩⟨bra| terms.
    
    - `coeffs` is a dense 1d numpy array of complex or real coefficients

    - `kets` is a 2-dimensional dense or sparse array such that `kets[x,:]` are
      the components of the `x`-th ket in the ensemble

    - `bras` is a 2-dimensional dense or sparse array such that `bras[x,:]` are
      the components of the `x`-th bra in the ensemble.  (We assume that any
      complex conjugation stemming from transforming a ket into a bra has
      already been applied.)

      If `bras` is `None`, then we use the complex-conjugated version of `kets`
      as bras.

    The operator is given explicitly as::

        O[I,J] = ∑_x coeffs[x]*kets[x,I]*bras[x,J]

    Expectation values of an operator `O` will be computed as follows (or
    equivalent)::

        sum_j p[j] * bras[j].dot( O.dot(kets[j]) )
    """
    def __init__(self, n, localdim, coeffs, kets, bras=None):
        super().__init__(n, localdim)

        self.coeffs = np.array(coeffs)
        self.kets = kets
        self.bras = bras
        
        # make kets, bras numpy arrays in case they aren't dense or sparse arrays
        if not sps.issparse(self.kets):
            self.kets = np.array(self.kets)
        if not self.kets.shape:
            self.kets = self.kets.reshape( (self.kets.size,1) )

        if self.bras is not None and not sps.issparse(self.bras):
            self.bras = np.array(self.bras)
        if self.bras is not None and not self.bras.shape:
            self.bras = self.bras.reshape( (self.bras.size,1) )

        self.num_ketbras = self.kets.shape[0]

        self.dim = self.localdim**self.n
        
        _validate(r""" list($self.kets .shape)
                       == [$self.num_ketbras,$self.dim] """)
        _validate(r""" $self.bras is None  or  list($self.bras .shape)
                       == list($self.kets.shape) """)

    def __repr__(self):
        return "{}({}, {}, {}, {}, {})".format(self.__class__.__name__,
                                               self.n, self.localdim, self.coeffs, self.kets, self.bras)


    def roverlap(self, other):
        return self.overlap(other, _use_adjoint=False)

    def overlap(self, other, *, _use_adjoint=True):
        coeffs, kets, bras = self.coeffs, self.kets, self.bras
        if bras is None:
            bras = kets.conj()

        if isinstance(other, qutip.Qobj):
            _validate(r""" list($other .dims)
                           == [ [$self.localdim]*$self.n, [$self.localdim]*$self.n ] """)
            other = other.data

        if not hasattr(other, 'shape'):
            # simple scalar
            other = np.array(other)[np.newaxis,np.newaxis] # 2-D 1x1 matrix

        _validate(r""" list($other .shape) == [ $self.dim, $self.dim ] """)

        if self.n == 0:
            kets = np.array(kets) # so that kets.T works if kets is a scalar

        def maybe_apply_conjT(X):
            if _use_adjoint:
                return X.conj().T
            return X

        # don't forget to take adjoint of `self` for the Hilbert-Schmidt inner
        # product.  Here instead of conjugating ourselves, we're conjugating
        # `other`, and then at the end we'll take the conjugate of the result [
        # exploiting tr(O^\dagger A) = (tr(O A^\dagger))^* ]
        if not sps.issparse(other) and sps.issparse(kets):
            # dense other and sparse kets -- need to convert other to sparse first
            opkets = (maybe_apply_conjT(sps.csr_matrix(other)).dot(kets.T)).T
        else:
            # op&kets:  sparse&sparse OR sparse&dense OR dense&dense
            #     ===   NOT( dense & sparse )
            #
            # These are all cases where other.dot(..) works:
            opkets = (maybe_apply_conjT(other).dot(kets.T)).T

        if sps.issparse(bras):
            # sparse bras, sparse|dense opkets
            brasopkets = bras.multiply(opkets)
        elif sps.issparse(opkets):
            # dense bras, sparse opkets
            brasopkets = opkets.multiply(bras)
        else:
            # dense bras & opkets
            brasopkets = np.multiply(bras, opkets)

        value = np.dot( coeffs, brasopkets.sum(axis=1) ).item()
        if _use_adjoint:
            value = value.conjugate()

        return value

    def real_overlap(self, operator):
        return self.overlap(operator).real


    def overlap_with_tensor_product_operator(self, ops, use_adjoint=True):
        """
        `ops` is numpy array such that `ops[j,a,b]` is the `[a,b]` matrix element of
        the operator that acts on the `j`-th site.
        """
        n = self.n
        localdim = self.localdim
        num_ketbras = self.num_ketbras

        totaldim = localdim**n

        _validate(r""" $ops .shape == ($n, $localdim, $localdim) """)
        
        if n == 0:
            # special case
            _validate(r""" $num_ketbras == 1 """)
            coeffs, kets, bras = self.coeffs, self.kets, self.bras
            if bras is None:
                bras = np.array(kets).conj()
            if use_adjoint:
                coeffs, kets, bras = np.array(coeffs).conj(), np.array(bras).conj(), np.array(kets).conj()
            return np.array(coeffs * kets * bras).item()

        # I want to compute
        #
        #   ∑ⱼ cⱼ ⟨ϕⱼ| (⊗ᵢAᵢ) |ψⱼ⟩  =  ∑ cⱼ ⟨ϕⱼ|x⃗⟩ ∏ᵢ⟨xᵢ|Aᵢ|xᵢ'⟩ ⟨xᵢ'|ψⱼ⟩
        #
        # So the idea is to iterate over the nonzeros of |ϕⱼ⟩ and |ψⱼ⟩ and do
        # the contraction.

        coeffs, kets, bras = self.coeffs, self.kets, self.bras
        if bras is None:
            bras = kets.conj()
        if use_adjoint:
            coeffs, kets, bras = coeffs.conj(), bras.conj(), kets.conj()

        result = 0
        for j in range(num_ketbras):

            #print(f"{j=}  {coeffs[j]=}  {kets[j,:]=}  {bras[j,:]=}")
            
            if sps.issparse(kets):
                _, kets_x_j  = kets[j,:].nonzero()
            else:
                kets_x_j, = kets[j,:].nonzero()
            if sps.issparse(bras):
                _, bras_x_j  = bras[j,:].nonzero()
            else:
                bras_x_j,  = bras[j,:].nonzero()

            # bras_x_j[m] is a basis vector # in 0...localdim**n-1.  Convert
            # that into a ditstring-> [see
            # https://stackoverflow.com/a/22227898/1694896, modified for any
            # local dimension basis]
            bras_u = (
                (bras_x_j[:,np.newaxis] // (localdim**np.arange(n)[::-1].reshape((1,n))))
            ) % localdim
            kets_u = (
                (kets_x_j[:,np.newaxis] // (localdim**np.arange(n)[::-1].reshape((1,n))))
            ) % localdim

            #print(f"    {bras_x_j=}  {bras_u=}\n    {kets_x_j=}  {kets_u=}\n")

            opsinner = np.prod( ops[range(n), bras_u[:,np.newaxis,:], kets_u[np.newaxis,:,:]], axis=2 )

            #print(f"    {opsinner=}")

            bras_j_ops = bras[j,bras_x_j].dot( opsinner )

            #print(f"    {bras_j_ops=}")

            # instread of bras_j_ops.dot(kets[j,kets_x_j].T), we use
            # kets[j,kets_x_j].T.dot(bras_j_ops.T) so we use the sparse matrix's
            # .dot() method.  (In case the operatoration with numpy array made
            # `bras_j_ops` a dense array.)
            #
            #this_term = coeffs[j] * bras_j_ops.dot(kets[j,kets_x_j].T)
            this_term = coeffs[j] * kets[j,kets_x_j].dot(bras_j_ops.T)
            #print(f"{this_term=}")
            result = result + this_term.item()

        return result


    def to_sparse(self):
        # A[a,b] = \sum_I coeffs[I] kets[I,a] bras[I,b]

        kets = self.kets
        if not sps.issparse(kets):
            kets = sps.csr_matrix(kets)
        bras = self.bras
        if bras is None:
            bras = kets.conj()
        elif not sps.issparse(bras):
            bras = sps.csr_matrix(bras)

        # (  (kets.T <a,I>)
        #    .multiply( coeffs[np.newaxis,:] <1,I> )  <a,I>  ).dot( bras <I,b> )   <a,b>
        return kets.T.multiply(self.coeffs[np.newaxis,:]).dot(bras)



# tools for specifying the action of a Hamiltonian on a DitstringSuperpositionKet --

class DitstringActionHamiltonian(PsiXiDesc):
    r"""
    Two possibilities:

    - If `yn=None`, then `en[x]` is the (real) eigenenergy associated with the
      ditstring `ditstringket.xn[x]`
    
    - Otherwise, `en[x,m]` is the complex amplitude with which the ditstring
      `xn[x,:]` gets mapped onto the ditstring `yn[x,m,:]` (for m in a given
      range).  I.e., `yn[x,m,:]` are ditstrings indexed by `m` onto which the
      corresponding ditstring `xn[x]` is mapped to with corresponding complex
      coefficients `en[x,m]`.
    """
    def __init__(self, ditstringket, en, yn=None, *, tol_check=1e-6):
        super().__init__(ditstringket)

        # also as field 'ditstringket', not only 'psi_ket_desc'
        self.ditstringket = ditstringket

        _validate(r""" isinstance($self.ditstringket, DitstringSuperpositionKet) """)

        self.en = en
        self.yn = yn

        self.tol_check = tol_check

        self.raw_en = en
        self.raw_yn = yn

        if self.yn is None:

            self.diagonal_action = True
            self.m = None

            _validate(r""" list($self.en .shape) == [ $self.ditstringket.num_x ] """)

            # compute the necessary shift (by identity operator) to construct H̄.
            self.avg_H_psi = np.sum(self.ditstringket.psi_xp * self.en)
            self.en = self.en - self.avg_H_psi #*np.ones_like(self.en)
            self.var_H_psi = np.sum(self.ditstringket.psi_xp * self.en**2)

        else:

            self.diagonal_action = False
            self.m = self.en.shape[1]

            _validate(r""" list($self.en .shape) == [ $self.ditstringket.num_x, $self.m ] """)
            _validate(r""" list($self.yn .shape)
                           == [ $self.ditstringket.num_x, $self.m, $self.n ] """)


            # compute the necessary shift (by identity operator) to construct H̄.
            # u[x,i,:] is the ditstring xn[i,:] translated into a basis vector 0->[1,0], 1->[0,1]
            u = np.eye(self.localdim)[self.ditstringket.xn,:]
            # same for v[x,m,i,:]
            v = np.eye(self.localdim)[self.yn,:]

            # xnym[X,x,m] = ∏_i ⟨xn[X,i]|yn[x,m,i]⟩
            xnym = np.prod(np.einsum('Xil,xmil->Xxmi', u, v), axis=3)
            # -> avg_H_psi = ∑_{x,X,m} psi_x[X].conj() psi_x[x] en[x,m] xnym[X,x,m]
            self.avg_H_psi = np.einsum(
                'X,x,xm,Xxm->',
                self.ditstringket.psi_x.conj(), self.ditstringket.psi_x,
                self.en, xnym
            )
            if np.absolute(self.avg_H_psi) > self.tol_check:
                # need to substract avgH*identity from H... add another yn
                # vector to each image with -avgH*<orig ditstring>
                #
                # yn[xp,m,i] ->
                self.yn = np.concatenate( (self.yn, self.ditstringket.xn[:,np.newaxis,:],),
                                          axis=1)
                # en[xp,m] ->
                self.en = np.concatenate( (self.en,
                                           -self.avg_H_psi*np.ones((self.ditstringket.num_x,1)),),
                                          axis=1)
                # recompute the temporary v data ->
                v = np.eye(self.localdim)[self.yn,:]

            # from this point on we can assume that ⟨H⟩ = 0 and H|ψ> = H̄|ψ⟩
            xpmpEExm = np.prod(np.einsum('XMil,xmil->XMxmi', v, v), axis=4)
            self.var_H_psi = np.einsum(
                'X,x,XM,xm,XMxm->',
                self.ditstringket.psi_x.conj(), self.ditstringket.psi_x,
                self.en.conj(), self.en, xpmpEExm
            )

        if not np.isreal(self.avg_H_psi):
            logger.warning("The specified Hamiltonian does not seem to be Hermitian: avg_H_psi={}"
                           .format(self.avg_H_psi))


    @classmethod
    def from_ditstring_action(Cls, ditstringket, action_fn, **kwargs):
        r"""
        Construct this object from a function `action_fn` that takes a single
        ditstring as input and outputs either a single scalar (the ditstring's
        energy) or a tuple (energies, ditstrings) that describe how the input
        ditstring is mapped.
        """
        
        en_yn_list = []
        max_m = 0
        for xn in ditstringket.xn:
            res = action_fn(xn)
            try:
                en, yn = res
            except TypeError:
                # only single value returned, it's the energy
                en, yn = [res], [xn]

            en = np.array(en)
            yn = np.array(yn)
            if len(en.shape) == 0:
                en = en[np.newaxis]
            if len(yn.shape) == 1:
                yn = yn[np.newaxis,:]

            if en.shape[0] > max_m:
                max_m = en.shape[0]

            _validate(r""" len($en .shape) == 1 """)
            _validate(r""" list($yn .shape) == [$en.shape[0],$ditstringket.n] """)

            en_yn_list.append( (en,yn) )

        num_x = ditstringket.num_x

        # we collected all en's & yn's, now reshape them into proper array
        en = np.zeros( (num_x, max_m,) )
        yn = np.zeros( (num_x, max_m, ditstringket.n), dtype=int )

        for j, en_yn_res in enumerate(en_yn_list):
            this_en, this_yn = en_yn_res
            en[j,:len(this_en)] = this_en
            yn[j,:len(this_en),:] = this_yn

        return Cls(ditstringket, en, yn, **kwargs)


    def get_xi_ditstringket(self):
        r"""
        Construct a `DitstringSuperpositionKet` instance that describes the ket
        :math:`\bar{H}|\psi\rangle`.
        """

        if self.diagonal_action:
            # easy case! simply multiply all coefficients.
            xi_x = np.multiply(self.ditstringket.psi_x, self.en)
            return DitstringSuperpositionKet(self.n, self.localdim,
                                             xi_x, self.ditstringket.xn,
                                             normalized=False)

        # the ditstrings and coefficients are:
        #    xi_xn[(x,m),:] = yn[x,m,:]
        #    xi_x[(x,m)] = psi_x[x] en[x, m]
        xi_x = np.multiply(self.ditstringket.psi_x[:,np.newaxis], self.en).reshape(
            (self.ditstringket.num_x*self.m,),
            order='C'
        )
        #   (x,m,i) -> permute -> (i, x, m) -> reshape -> (i, (xm)) -> permute -> ((xm), i)
        xi_xn = np.transpose(
            np.reshape(
                np.transpose(self.yn, [2, 0, 1]),
                (self.n, self.ditstringket.num_x*self.m,),
                order='C'
            ),
            [1, 0]
        )
        return DitstringSuperpositionKet(self.n, self.localdim, xi_x, xi_xn, normalized=False)
        



# tools for noise model --

class LocalKrausOperators:
    def __init__(self, local_kraus_operators, *,
                 input_local_unitary=None, output_local_unitary=None, tol_check=1e-6):
        """
        - Kraus operators are qubit operators given as qutip.Qobj instances

        - `input_local_unitary` and `output_local_unitary` add a rotation of the
          input and the output of the channel, by transforming each local Kraus
          operator individually.
        """
        super().__init__()

        _validate(r""" len($local_kraus_operators) > 0 """)

        self.raw_local_kraus_operators = [ qutip.Qobj(E) for E in local_kraus_operators ]
        self.localdim_in = self.raw_local_kraus_operators[0].dims[0][0]
        self.localdim_out = self.raw_local_kraus_operators[0].dims[1][0]

        _validate(r""" np.all( [E.dims == [[$self.localdim_in], [$self.localdim_out]]
                                for E in $self.raw_local_kraus_operators] ) """)

        if self.localdim_in == self.localdim_out:
            self.localdim = self.localdim_in
        else:
            self.localdim = None

        self.num_local_kraus_operators = len(self.raw_local_kraus_operators)

        # assert that the Kraus operators define a valid q. channel
        sumEkdagEk = functools.reduce(
            lambda a, b: a + b,
            (E.dag()*E for E in self.raw_local_kraus_operators)
        )
        _validate(r""" ($sumEkdagEk - qutip.qeye($self.localdim_in)).norm() < $tol_check """)


        self.input_local_unitary = input_local_unitary
        self.output_local_unitary = output_local_unitary

        if self.input_local_unitary is not None:
            self.input_local_unitary = qutip.Qobj(self.input_local_unitary)
            _validate(r""" ($self.input_local_unitary*self.input_local_unitary.dag()
                            - qutip.qeye($self.localdim_in)).norm() < $tol_check """)
        if self.output_local_unitary is not None:
            self.output_local_unitary = qutip.Qobj(self.output_local_unitary)
            _validate(r""" ($self.output_local_unitary*self.output_local_unitary.dag()
                            - qutip.qeye($self.localdim_out)).norm() < $tol_check """)

        if self.input_local_unitary is not None and self.output_local_unitary is not None:
            # rotate input & output
            self.local_kraus_operators = \
                [ self.output_local_unitary*E*self.input_local_unitary
                  for E in self.raw_local_kraus_operators ]
        elif self.input_local_unitary is not None:
            # rotate input & output
            self.local_kraus_operators = \
                [ E*self.input_local_unitary  for E in self.raw_local_kraus_operators ]
        elif self.output_local_unitary is not None:
            # rotate input & output
            self.local_kraus_operators = \
                [ self.output_local_unitary*E  for E in self.raw_local_kraus_operators ]
        else:
            self.local_kraus_operators = self.raw_local_kraus_operators
 
    def __repr__(self):
        args = repr(self.local_kraus_operators)
        if self.input_local_unitary:
            args += ', input_local_unitary={!r}'.format(self.input_local_unitary)
        if self.output_local_unitary:
            args += ', output_local_unitary={!r}'.format(self.output_local_unitary)
        return "{}({})".format(self.__class__.__name__, args)

    def get_super(self):
        return super_from_kraus(self.local_kraus_operators)

    def get_full_super_n(self, n):
        N1_super = self.get_super()
        return qutip.super_tensor(*[ N1_super for _ in range(n) ])

    def get_complementary_kraus(self):
        Ekdata = np.array([ E.full() for E in self.local_kraus_operators ])
        # --> Ekdata[k,ℓ,m].  Here: k iterates over the Kraus operators, ℓ
        # iterates over output system, m iterates over input system.
        
        # complementary Kraus operators Ê_ℓ satisfy ⟨k|Ê_ℓ|m⟩ = ⟨ℓ|Eₖ|m⟩
        # -------> simply swap the axes
        Ecdata = np.swapaxes(Ekdata, 0, 1)
        return [ qutip.Qobj(Ec) for Ec in Ecdata ]

    def get_complementary_super(self):
        return super_from_kraus( self.get_complementary_kraus() )

    def get_full_complementary_super_n(self, n):
        N1c_super = self.get_complementary_super()
        return qutip.super_tensor(*[ N1c_super for _ in range(n) ])



def super_from_kraus(Eks):
    return functools.reduce(
        lambda a,b: a+b,
        [ qutip.sprepost(E, E.dag()) for E in Eks ]
    )



class TraceWithKrausPairStringCalculator:
    def __init__(self, local_kraus_operators, kraus_pair_string):
        super().__init__()
        self.local_kraus_operators = local_kraus_operators
        self.localdim = self.local_kraus_operators.localdim
        self.kraus_pair_string = np.array(kraus_pair_string, dtype=int)


    def calculate_trace(self, operator_desc):
        raise RuntimeError("This method is not implemented by subclass")
        

class QutipTraceWithKrausPairStringCalculator(TraceWithKrausPairStringCalculator):
    def __init__(self, local_kraus_operators, kraus_pair_string):
        super().__init__(local_kraus_operators, kraus_pair_string)
        
        if not len(self.kraus_pair_string):
            self.EkpdagEk = 1
            self.EkpdagEk_per_site = []
            self.EkpdagEk_ops = np.zeros(shape=(0,self.localdim,self.localdim))
        else:
            Ek = self.local_kraus_operators.local_kraus_operators

            self.EkpdagEk_per_site = [
                Ek[ r ].dag() * Ek[ l ]
                for (l, r) in self.kraus_pair_string
            ]
            self.EkpdagEk_ops = np.array([ EdE.full() for EdE in self.EkpdagEk_per_site])
            self.EkpdagEk = qutip.tensor(*self.EkpdagEk_per_site)

        #print(f"{kraus_pair_string=} {self.EkpdagEk=}")


    def calculate_trace(self, operator_desc):
        if isinstance(operator_desc, EnsembleKetBras):
            return self._calculate_trace_ensketbras(operator_desc)

        # default implementation:
        value = operator_desc.roverlap(self.EkpdagEk)
        #print(f"calculate_trace(): {operator_desc=}  {self.EkpdagEk=}  -> {value=}")
        return value

    def _calculate_trace_ensketbras(self, ensketbras):
        return ensketbras.overlap_with_tensor_product_operator(self.EkpdagEk_ops, use_adjoint=False)










class DitstringSuperpositionKetReducedPsiXiCalculator(ReducedOperatorsCalculator):
    r"""
    Compute the marginals trₙ₋ₖ((E₀^†E₀)ⁿ⁻ᵏ|ψ⟩⟨ψ|) and
    trₙ₋ₖ((E₀^†E₀)ⁿ⁻ᵏ|ξ⟩⟨ψ|).

    The first Kraus operator of `local_kraus_operators` is used for E0
    premultiplication.
    """
    
    def __init__(self, ditstringket, ditstringactionH, local_kraus_operators):

        super().__init__(ditstringket.n, ditstringket.localdim, num_operators=2)

        _validate(r""" isinstance($ditstringket, DitstringSuperpositionKet) """)
        _validate(r""" isinstance($ditstringactionH, DitstringActionHamiltonian) """)

        self.ditstringket = ditstringket
        self.ditstringactionH = ditstringactionH
        self.local_kraus_operators = local_kraus_operators

        _validate(r""" $self.ditstringket is $self.ditstringactionH.ditstringket """)

        self.n = self.ditstringket.n
        self.localdim = self.ditstringket.localdim

        self.premultiply_E0 = qutip.Qobj(self.local_kraus_operators.local_kraus_operators[0])

        _validate(r""" $self.premultiply_E0 .dims == [[$self.localdim],[$self.localdim]] """)

        self.E0dagE0 = ( self.premultiply_E0.dag() * self.premultiply_E0 ).full()


    def reduced_operators(self, which_sites):
        r"""
        Return a tuple `(psi_ensemble, xipsi_ensemble)`, where each ensemble object
        is a `EnsembleKetBras` object instance associated with
        tr_{A\which_sites}[|ψ⟩⟨ψ|] and tr_{A\which_sites}[|ξ⟩⟨ψ|], respectively.
        """
        return self._calc_psi_xipsi_redop_general(which_sites)


    def _calc_psi_xipsi_redop_general(self, which_sites):
        # general reduced state, might include cross-terms of xn's because they
        # don't differ on the sites we are tracing out.

        n = self.n
        localdim = self.localdim

        psi_x = self.ditstringket.psi_x
        xn = self.ditstringket.xn
        num_x = self.ditstringket.num_x
        num_m = self.ditstringactionH.m
        if num_m is None:
            num_m = 1
        en = self.ditstringactionH.en
        yn = self.ditstringactionH.yn

        numsites = len(which_sites)

        xnk = xn[:,which_sites]
        xnkc = np.delete(xn,which_sites,1)
        if self.ditstringactionH.diagonal_action:
            yn   = xn
            ynk  = xnk[:,np.newaxis,:]
            ynkc = xnkc[:,np.newaxis,:]
        else:
            # en[x,m], yn[x,m,i]
            ynk  = yn[:,:,which_sites]
            ynkc = np.delete(yn,which_sites,2)

        # compute the coefficients that we get by tracing out part of the
        # ditstrings, sandwiching in an E0dagE0 operator.
        #
        # e0_coeffs_psi[xp,x] = ⟨xnkc[xp]| (E0dagE0)⊗ⁿ⁻ᵏ |xnkc[x]⟩
        #
        # e0_coeffs_Hbarpsi[xp,x,m] = ⟨xnkc[xp]| (E0dagE0)⊗ⁿ⁻ᵏ |ynkc[x,m]⟩

        # u[x,i,:] is the ditstring xnkc[i,:] translated into a basis vector 0->[1,0], 1->[0,1]
        u = np.eye(localdim)[xnkc,:]

        # x,X = which ditstring; i = which site; l,L = indicies within local site dimension
        e0_coeffs_psi = np.prod(
            np.einsum('Xil,lL,xiL->Xxi', u, self.E0dagE0, u).real,
            axis=2 # product over i
        ) # indices 'Xx' left ["X" here stands for "x-prime"]

        #print(f"{e0_coeffs_psi=}")

        # do the same for ynk
        if self.ditstringactionH.diagonal_action:
            en = en[:,np.newaxis]
            e0_coeffs_xi = e0_coeffs_psi[:,:,np.newaxis]
        else:
            v = np.eye(localdim)[ynkc,:]  # v[x,m,i,l]
            e0_coeffs_xi = np.prod(
                np.einsum('Xil,lL,xmiL->Xxmi', u, self.E0dagE0, v),
                axis=3 # product over i
            ) # indices 'Xx' left ["X" here stands for "x-prime"]
        
        # a this point, the desired marginals are:
        #
        # psi_k = ∑_{x,xp} psi_x[xp].conj() psi_x[x] e0_coeffs_psi[xp,x] |xnk[x]⟩⟨xnk[xp]|
        #
        # xi_k = ∑_{x,y} psi_x[xp].conj() psi_x[x] en[x,m] e0_coeffs_xi[xp,x,m] |ynk[x,m]⟩⟨xnk[xp]|
        #

        # if numsites == 0, this means the full trace was requested.
        if numsites == 0:
            psi_ens = EnsembleKetBras(
                0, localdim,
                np.einsum('X,x,Xx->', psi_x.conj(), psi_x, e0_coeffs_psi),
                1
            )
            xi_coeff = np.einsum('X,x,xm,Xxm->', psi_x.conj(), psi_x, en, e0_coeffs_xi)
            xi_ens = EnsembleKetBras(
                0, localdim,
                xi_coeff,
                1
            )
            return ( psi_ens, xi_ens )

        # The summation that will be left open (for the ensemble) is over xp.
        # The bras will simply be xnk[xp,:] in both cases.  The kets will be
        # given by contracting everything that's before; maybe we can leave out
        # a real value for the "probabilities" in the case of psi_k.

        kdim = localdim**numsites
        # xnk[x,:] is ditstring, xn_idxs[x,i] is the basis vector associated with dit value xnk[i]
        xn_idxs = np.dot(xnk, localdim**np.arange(numsites))
        xn_kets = sps.coo_matrix(
            # (data, (row, col))
            (np.ones(num_x), (np.arange(num_x), xn_idxs)),
            shape=(num_x,kdim)
        ).tocsr()
        if self.ditstringactionH.diagonal_action:
            yn_idxs = xn_idxs
            yn_kets = xn_kets
        else:
            # same for ynk[x,m,i] -> yn_idxs[x,m]
            yn_idxs = np.dot(ynk, localdim**np.arange(numsites))

            # think: yn_kets = zeros(num_x*num_m, kdim); and then:  yn_kets[(x,m),I] = 1
            yn_kets = sps.coo_matrix(
                # (data, (row, col))
                (np.ones(num_x*num_m),
                 (np.arange(num_x*num_m), yn_idxs.reshape((num_x*num_m,), order='C'))),
                shape=(num_x*num_m,kdim)
            ).tocsr()

        #print(f"{psi_x.conj()=}\n{psi_x=}\n{yn_e=}\n{e0_coeffs_Hbarpsi=}\n{xn_kets=}")

        psi_kets = (
            sps.csr_matrix( np.einsum('X,x,Xx->Xx', psi_x.conj(), psi_x, e0_coeffs_psi) )
            .dot( xn_kets )
        )
        xi_kets = (
            sps.csr_matrix(
                np.einsum('X,x,xm,Xxm->Xxm', psi_x.conj(), psi_x, en, e0_coeffs_xi)
                .reshape( (num_x, num_x*num_m,), order='C' )
            ).dot( yn_kets ) )

        # use xn_kets as bras for both.  They are real, no need for conj()
        xn_bras = xn_kets

        return (
            EnsembleKetBras(numsites, localdim, np.ones(num_x), psi_kets, xn_bras),
            EnsembleKetBras(numsites, localdim, np.ones(num_x), xi_kets, xn_bras),
        )



