import os
import os.path
import re

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




from ._states_desc import (
    KetDesc,
    OperatorDesc,
    QutipKet,
    QutipOperator,
    QutipReducedOperatorCalculator,
    QutipTraceWithKrausPairStringCalculator,
    DitstringSuperpositionKet,
    EnsembleKetBras,
    DitstringActionHamiltonian,
    LocalKrausOperators,
    DitstringSuperpositionKetReducedPsiXiCalculator,
)


from ._symm_states_desc import (
    SymmetricKet,
    SymmetricOperator,
    SymmetricSuperpositionKet,
    SymmetricEnsembleKetBras,
    SymmetricTraceWithKrausPairStringCalculator,
    SymmetricReducedOperatorsCalculator,
    get_symm_state_basis_ditstrings,
)


from ._iter_sites import (
    iter_strings_k,
    iter_leqk_sites,
    iter_symm_sites_with_multiplier,
)


from ._fi_calc import (
    compute_Fisher_information,
)

from ._optimize_state import (
    OptimizationStateAnsatz,
    SymmetricRealStateAnsatz,
    SymmetricFullStateAnsatz,
    SymmetricExponentialTailsStateAnsatz,
    SymmetricFFTModesStateAnsatz,
    SymmetricExponentialPeaksStateAnsatz,
    StateOptimizer,
)

from ._fi_k_calc import (
    compute_Eve_operators_leqk,
    compute_pinched_DF_lower_bound,
)

from ._fi_lbound_adhoc import (
    ExpandableSelectionOfDitstrings,
    FI_lbound_projditstrings,
)
