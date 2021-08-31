import logging
logger = logging.getLogger(__name__)

import itertools
import functools

import numpy as np
import numpy.linalg as npl

import scipy as sp
import scipy.linalg as spl

import qutip


from ._iter_sites import (
    iter_leqk_sites,
    _iter_iid_kraus_nz_index_string_on_k_sites,
)

from ._states_desc import (
    QutipTraceWithKrausPairStringCalculator,
)


# Special logger for all the _validate calls, so that it can be silenced easily.
# The logger instance will be found by name in the global scope of this module
# by the _validate() command itself.
from .validate import validate_logger






def compute_Eve_operators_leqk(
        reduced_operators_calculator, lko, owkpsc=None,
        *,
        k,
        tol_check=1e-6,
        init_op=None
):
    """
    Compute the operators 𝒩̂ⁿ₍ₖ₎(X_i) for a set of operators { X_i }, and where
    𝒩̂ⁿ₍ₖ₎ is the channel defined by

      𝒩̂ⁿ₍K₎(X) = ∑_{k,k'} √(J_k J_{k'}) tr(E_{k'}^† E_{k} X) |k⟩⟨k'|

    where the sum ranges over all combinations of at most k sites (or the
    combinations provided by an iterator) and where J_k = 1 (or a multiplicity
    provided by an iterator that exploits symmetry).

    - `reduced_operators_calculator` is an object that can compute the reduced
      operator on given sites of the given set of operators { X_i }.

      This instance is expected to be a `ReducedOperatorsCalculator` subclass.

    - `lko` is a `LocalKrausOperators` object instance

    The basis of the E system in which the operators are represented is
    determined by the order of the iteration over all subsets of sites, and then
    all combinations of nontrivial Kraus operators over those sites.  The same
    basis is used for all operators.
    """

    n = reduced_operators_calculator.n
    localdim = reduced_operators_calculator.localdim

    #print(f"compute_Eve_operators_leqk()")

    if owkpsc is None:
        owkpsc = QutipTraceWithKrausPairStringCalculator

    #iter_sites_with_multiplier = \
    #    lambda: ((sites, 1) for sites in iter_leqk_sites(n, k))

    def iter_nzEk_on_sites(sites):
        return _iter_iid_kraus_nz_index_string_on_k_sites(
            len(sites),
            lko.num_local_kraus_operators
        )

    # The "left" iterations correspond to the kets of the output of the
    # complementary channel (k usually in my notation) whereas the "right"
    # indices correspond to k' in my notation.  The E_{k'} 's are the ones taken
    # to be dagger'ed.

    # compute the structure of Eve by running through the iterators, and sort
    # out the terms to compute by their "merged sites subset"
    lrsites_by_mergedsites = {}
    Ekblocks = []
    for j_l_sites, l_sites in enumerate(iter_leqk_sites(n,k)):
        for j_r_sites, r_sites in enumerate(iter_leqk_sites(n,k)):

            if j_l_sites == 0:
                # initialize Ekblocks on the first time the iteration of the inner
                # loop is performed (i.e., on the first iteration of the outer loop)
                Ekblocks.append({'sites': r_sites,
                                 'dim': sum(1 for _ in iter_nzEk_on_sites(r_sites))})

            merged_sites = tuple(sorted( set(l_sites) | set(r_sites) ))

            # # which systems are part of l_sites in the merged sites ---
            # # l_site_idx[s] is either the index in l_sites where merged_sites[s]
            # # appears, or None of the site doesn't appear in l_sites.
            # l_sites_idx = [ next((i for i in range(len(l_sites)) if l_sites[i] == s), None)
            #                for s in merged_sites ]
            # # same for r_sites
            # r_sites_idx = [ next((i for i in range(len(r_sites)) if r_sites[i] == s), None)
            #                for s in merged_sites ]

            # reverse of {l|r}_sites_idx. List of indices in merged_sites where
            # the l_sites are.
            l_sites_widx = [ next((i for i,m in enumerate(merged_sites) if m == s),)
                             for s in l_sites ]
            r_sites_widx = [ next((i for i,m in enumerate(merged_sites) if m == s),)
                             for s in r_sites ]

            if merged_sites not in lrsites_by_mergedsites:
                lrsites_by_mergedsites[merged_sites] = []

            lrsites_by_mergedsites[merged_sites].append({
                'j_l_sites': j_l_sites,
                'l_sites': l_sites,
                #'l_multiplier': l_multiplier,
                #'l_sites_idx': l_sites_idx,
                'l_sites_widx': np.array(l_sites_widx, dtype=int),
                'j_r_sites': j_r_sites,
                'r_sites': r_sites,
                #'r_multiplier': r_multiplier,
                #'r_sites_idx': r_sites_idx,
                'r_sites_widx': np.array(r_sites_widx, dtype=int),
            })

    Ekblocks_dims = np.array([x['dim'] for x in Ekblocks], dtype=int)
    # compute the cumulative sums, making sure first element is zero
    Ekblocks_offsets = np.cumsum(np.hstack([[0],Ekblocks_dims[:-1]]))
    dE = np.sum(Ekblocks_dims)

    #logger.debug(f"{dE=}  {Ekblocks_dims=}  {Ekblocks_offsets=}")
    #print(f"{dE=} {Ekblocks_dims=} {Ekblocks_offsets=}\n{lrsites_by_mergedsites=}")

    # now we're turning to computing the reduced operators

    if init_op is None:
        init_op = lambda dE: np.empty( shape=(dE,dE,), dtype=complex )

    E0 = lko.local_kraus_operators[0]

    Nhat_Xi = [ init_op(dE) for _ in range(reduced_operators_calculator.num_operators) ]

    for merged_sites, msilist in lrsites_by_mergedsites.items():

        # compute reduced state on relevant sites (2*k)
        reduced_Xi_merged_sites = reduced_operators_calculator(merged_sites)

        #print(f"{merged_sites=}\n{reduced_Xi_merged_sites=}")

        kraus_pairs_string = np.zeros(shape=(len(merged_sites),2), dtype=int)

        for msi in msilist:

            # msi contains the fields msi['j_l_sites'], msi['l_sites'],
            # msi['l_multiplier'], msi['l_sites_widx'], and same for 'r'

            j_l_offset = Ekblocks_offsets[msi['j_l_sites']]
            j_r_offset = Ekblocks_offsets[msi['j_r_sites']]

            l_sites, r_sites = msi['l_sites'], msi['r_sites']
            l_sites_widx, r_sites_widx = msi['l_sites_widx'], msi['r_sites_widx']

            #l_multiplier, r_multiplier = msi['l_multiplier'], msi['r_multiplier']
            #term_multiplier = np.sqrt(l_multiplier * r_multiplier)

            #print(f"  {msi=}")

            # now iterate over pairs of Kraus operator choices that are supported on
            # these sites
            for ji_l, l_Ek_idx in enumerate(iter_nzEk_on_sites(l_sites)):

                # this_Ek_l = qutip.tensor(*[
                #     lko.local_kraus_operators[ l_Ek_idx[l_site_idx] ]
                #     if l_site_idx is not None else lko.local_kraus_operators[0]
                #     for l_site_idx in msi['l_sites_idx']
                # ])

                for ji_r, r_Ek_idx in enumerate(iter_nzEk_on_sites(r_sites)):

                    # compute the Kraus operator pair restricted on these sites
                    # this_Ekdag_r = qutip.tensor(*[
                    #     lko.local_kraus_operators[ r_Ek_idx[r_site_idx] ]
                    #     if r_site_idx is not None else lko.local_kraus_operators[0]
                    #     for r_site_idx in msi['r_sites_idx']
                    # ])

                    kraus_pairs_string[:,:] = 0
                    kraus_pairs_string[l_sites_widx,0] = l_Ek_idx
                    kraus_pairs_string[r_sites_widx,1] = r_Ek_idx

                    #print(f"    {l_Ek_idx=}  {r_Ek_idx=}  {kraus_pairs_string=}")

                    okpsc_instance = owkpsc(lko, kraus_pairs_string)

                    for i in range(reduced_operators_calculator.num_operators):

                        term_value = okpsc_instance.calculate_trace(
                            reduced_Xi_merged_sites[i]
                        )
                        # term_value = reduced_Xi_merged_sites[i].overlap(this_EkpdagEk)

                        #print(f"      {i=}[{j_l_offset=},{ji_l=};{j_r_offset=},{ji_r=}] --> {term_value=}")

                        Nhat_Xi[i][j_l_offset+ji_l][j_r_offset+ji_r] = term_value
                        #  ...  * term_multiplier


    info_dic = {
        'lrsites_by_mergedsites': lrsites_by_mergedsites,
        'Ekblocks': Ekblocks
    }

    #print(f"compute_Eve_operators_leqk() done!")

    return Nhat_Xi, info_dic
    
    



def compute_pinched_DF_lower_bound(
        reduced_psi_xipsi_calculator, lko, owkpsc=None,
        *,
        k,
        iter_sites_with_multiplier=None,
        tol_check=1e-6,
):
    """
    Compute the "pinched" version of our lower bound on the Fisher information
    loss (which gives an upper bound on Bob's Fisher information).

    - `reduced_psi_xipsi_calculator` is an object that can compute the reduced
      state on given sites of the operators |ψ⟩⟨ψ| and |ξ⟩⟨ψ|.

      This instance is expected to be a `ReducedOperatorsCalculator` subclass
      with `num_operators==2`.

    - `lko` is a `LocalKrausOperators` object instance

    - If `iter_sites_with_multiplier` is specified, it can be set to a
      function (e.g. generating function) that yields `(sites, multiplier)` that
      is meant to iterate over all combinations of at most k sites each with
      multiplier 1.  Specify a custom generator here if you want to exploit some
      symmetry.  For instance, for a permutationally-invariant state and
      Hamiltonian, you could specify a generating function that only iterates
      over the site sets with multipliers: ``((1,), n) , ((1,2,), n-choose-2) ,
      ((1,2,3,), n-choose-3) , ... , ((1,2,...,k,), n-choose-k)`` where the
      multiplier corresponds to the number of times to account for that term.
    """

    Glb = 0.0

    n = reduced_psi_xipsi_calculator.n
    localdim = reduced_psi_xipsi_calculator.localdim

    if owkpsc is None:
        owkpsc = QutipTraceWithKrausPairStringCalculator

    if iter_sites_with_multiplier is None:
        iter_sites_with_multiplier = \
            lambda: ((sites, 1) for sites in iter_leqk_sites(n, k))

    # iterate over choices of k sites
    for sites, multiplier in iter_sites_with_multiplier():

        # compute relevant reduced states
        psi_k, xipsi_k = reduced_psi_xipsi_calculator(sites)

        iter_Ek_on_sites = _iter_iid_kraus_nz_index_string_on_k_sites(
                len(sites),
            lko.num_local_kraus_operators
        )
        for Ek_idx in iter_Ek_on_sites:
            # if len(sites) == 0, we get a single iteration with an empty list here

            #thisEkdagEk = lko.get_Kraus_op_from_ditstring(Ek_idx, compute_EkdagEk=True)
            okpsc_instance = owkpsc(lko, tuple(zip(Ek_idx,Ek_idx)))

            # compute this term in our lower bound:
            x = 2 * okpsc_instance.calculate_trace(xipsi_k).real
            y = okpsc_instance.calculate_trace(psi_k).real

            # x = 2 * xipsi_k.real_overlap(thisEkdagEk)
            # y = psi_k.real_overlap(thisEkdagEk)

            assert np.isreal(x)
            assert np.isreal(y)

            if np.abs(y) > tol_check * np.abs(x):
                #print(f"{sites=},{Ek_idx=}: {x=}, {y=}")
                Glb += multiplier * (x ** 2) / y
    
    return Glb, {}
    



