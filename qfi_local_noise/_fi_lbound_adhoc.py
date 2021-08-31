import logging
logger = logging.getLogger(__name__)

import itertools
import functools

import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import scipy as sp
import scipy.linalg as spl
import scipy.sparse as sps

import qutip

from ._util import (
    _validate
)

from ._iter_sites import (
    iter_leqk_sites,
    iter_strings_k,
    _iter_iid_kraus_nz_index_string_on_k_sites,
)

from ._states_desc import (
    DitstringSuperpositionKet,
    DitstringActionHamiltonian,
    LocalKrausOperators,
)

from ._fi_calc import (
    compute_Fisher_information
)


# Special logger for all the _validate calls, so that it can be silenced easily.
# The logger instance will be found by name in the global scope of this module
# by the _validate() command itself.
from .validate import validate_logger





def _make_unique_ditstrings(xn):
    dstuples = sorted(set([ tuple(X) for X in xn ]))
    return np.array(dstuples, dtype=int)


class ExpandableSelectionOfDitstrings:
    r"""
    A collection of ditstrings associated with computational basis states that
    we use to express states and operators in terms of.  This class provides
    methods to expand this collection to include new ditstrings that are close
    in Hamming distance to a ditstring that is already in the collection.
    """
    def __init__(self, n, localdim, sxn=None):
        self.n = n
        self.localdim = localdim

        if sxn is not None:
            self.sxn = np.array(sxn, dtype=int)
        else:
            self.sxn = np.zeros(shape=(0,self.n), dtype=int)

        _validate(r""" $self.sxn.shape[1] == $self.n """)


    def added_dsk_and_daH(self, dsk, daH):
        new_sxn = np.vstack([
            self.sxn,
            dsk.xn,
        ])
        if not daH.diagonal_action:
            for ynm in daH.yn:
                new_sxn = np.vstack([
                    new_sxn,
                    ynm,
                ])
        # make sure the ditstrings are unique
        new_sxn = _make_unique_ditstrings(new_sxn)
        return self.__class__(self.n, self.localdim, new_sxn)

    def added_all_nearby_ditstrings(self, up_to_hamming_distance=5):
        add_ditstrings = []
        for base_ditstring in self.sxn:
            for offset_ditstring in \
                iter_strings_k(self.n, up_to_hamming_distance, self.localdim):
                #
                add_ditstrings.append( (base_ditstring + offset_ditstring) % self.localdim )
        # original strings are also included when iter_strings_k yields the element [0,0,0...,0]
        new_sxn = _make_unique_ditstrings(add_ditstrings)
        return self.__class__(self.n, self.localdim, new_sxn)

    def added_random_nearby_ditstrings(self, num_random=1024, up_to_hamming_distance=5):
        r"""
        Add random ditstrings that are up to Hamming distance XXX of a ditstring
        that we're already considering:
        """
        add_ditstrings = []
        for rndi in range(num_random):
            # pick which ditstring to tweak
            x = npr.randint(self.sxn.shape[0])
            # number of sites to tweak
            ni = npr.randint(1, up_to_hamming_distance+1)
            # pick which sites (well, if there are collisions, too bad)
            sites = npr.randint(self.n, size=ni)
            # tweak those sites
            this_ditstring = np.array(self.sxn[x], dtype=int)
            this_ditstring[sites] = ( this_ditstring[sites] + npr.randint(1,self.localdim,size=ni) ) % self.localdim
            # record this ditstring to add
            #print(f"{this_ditstring=}, {sites=}")
            add_ditstrings.append(this_ditstring)
            
        new_sxn = _make_unique_ditstrings(np.vstack([
            self.sxn,
            *add_ditstrings
        ]))
        return self.__class__(self.n, self.localdim, new_sxn)




def _get_full_tensor_from_qutip_super(N1_map):
    """
    Transforms a 'super' type Qobj to a tensor-rank-4 ndarray that can be used
    for `_contract_noise_ditstrings`, for instance.
    """
    assert N1_map.type == 'super'
    localdim = N1_map.dims[0][0][0]
    # axes are ( (ll'), (pp') ) as returned by qutip (l,p=bra-like, l',p'=ket-like).  So we need to 
    N1_map_full = N1_map.full()
    N1_map_full = np.reshape(N1_map_full, (localdim*localdim, localdim, localdim), 'C') # -> ((ll'), p, p')
    N1_map_full = np.transpose(N1_map_full, (1, 2, 0)) # -> (p, p', (ll'))
    N1_map_full = np.reshape(N1_map_full, (localdim, localdim, localdim, localdim,), 'C') # -> (p, p', l, l')
    N1_map_full = np.transpose(N1_map_full, (2, 3, 0, 1)) # -> (l, l', p, p')
    return N1_map_full


def FI_lbound_projditstrings(dsk, daH, lko, esd=None):
    r"""
    Simplistic lower bound obtained by projecting the operators onto a subspace
    spanned by the given selection of computational basis states given as
    ditstrings.
    """

    _validate(r""" isinstance($dsk, DitstringSuperpositionKet) """)
    _validate(r""" isinstance($daH, DitstringActionHamiltonian) """)
    _validate(r""" isinstance($lko, LocalKrausOperators) """)
    _validate(r""" isinstance($esd, ExpandableSelectionOfDitstrings) """)

    _validate(r""" $dsk is $daH.ditstringket """)
    _validate(r""" $dsk.n == $daH.n """)
    _validate(r""" $dsk.localdim == $daH.localdim """)

    _validate(r""" $dsk.localdim == $lko.localdim """)
    
    if esd is None:
        esd = ExpandableSelectionOfDitstrings(dsk.n, dsk.localdim).added_dsk_and_daH(dsk, daH)
        
    _validate(r""" $esd.n == $dsk.n """)
    _validate(r""" $esd.localdim == $dsk.localdim """)


    n = dsk.n
    localdim = dsk.localdim

    sxn = esd.sxn

    #print(f"{sxn=}")

    # H̄|ψ⟩ — given via Exn
    xi_dsk = daH.get_xi_ditstringket()

    #
    # Compute all relevant super-matrix elements in this basis
    #
    usxn = np.eye(localdim)[sxn,:]

    uket_psi_xn = np.eye(localdim)[dsk.xn,:]
    uket_xi_xn = np.eye(localdim)[xi_dsk.xn,:]

    N1_map_full = _get_full_tensor_from_qutip_super(lko.get_super())
    #print(f"{N1_map_full=}")

    # we want to compute:
    #
    # [NOTE: We're using QUTIP convention on vectorizing operators, where  |ψ,ξ⟩⟩  ↔  |ξ⟩⟨ψ|
    #
    # ⟨⟨sxn[x],sxn[x']| 𝒩₁^⊗ⁿ |ψ,ψ⟩⟩
    #
    # BB[x,x'] = ∑_{z,z'} {{ ∏_{i} ∑_{l,l',p,p'} usxn[x,i,l] usxn[x',i,l'] 𝒩₁[l,l'←p,p'] uket_psi_xn[z,i,p] uket_psi_xn[z',i,p'] }} ket_psi_x[z].conj() ket_psi_x[z']
    BB_overlaps = np.prod(
        np.einsum('xil,XiL,lLpP,zip,ZiP->ixXzZ', usxn, usxn, N1_map_full, uket_psi_xn, uket_psi_xn),
        axis=0
    )
    BB = np.einsum('xXzZ,z,Z->xX', BB_overlaps, dsk.psi_x.conj(), dsk.psi_x)

    logger.debug("Computed B overlaps, now computing A overlaps…")

    # we want to compute:
    #
    # ⟨⟨sxn[x],sxn[x']| 𝒩₁^⊗ⁿ (-i|ψ,H̄ψ⟩⟩)
    #
    # AA1[x,x'] = -1j ∑_{z,z'} {{ ∏_{i} ∑_{l,l',p,p'} usxn[x,i,l] usxn[x',i,l'] 𝒩₁[l,l'←p,p'] uket_psi_xn[z,i,p] uket_Hpsi_xn[z',i,p'] }} ket_psi_x[z].conj() ket_Hpsi_x[z']
    AA1_overlaps = np.prod(
        np.einsum('xil,XiL,lLpP,zip,ZiP->ixXzZ', usxn, usxn, N1_map_full, uket_psi_xn, uket_xi_xn),
        axis=0
    )
    AA1 = np.einsum('xXzZ,z,Z->xX', AA1_overlaps, dsk.psi_x.conj(), -1j*xi_dsk.psi_x)
    AA = AA1 + AA1.conj().transpose()

    #print(f"{BB=}\n{AA=}")

    # alright, now we simply have to compute the Fisher information
    #  F(BB ; AA)   [  = max_S 4 tr(S AA) - 4 tr(S² BB)  ]

    logger.debug("Computed restricted A & B matrices, solving for the Fisher information … ")

    F, variables = compute_Fisher_information(BB, AA)

    variables['rho_proj'] = BB
    variables['N_mi_xipsi_proj'] = AA1

    return F, variables


