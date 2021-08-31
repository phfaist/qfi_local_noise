`qfi_local_noise` — Quantum Fisher information for many-body systems exposed to local noise
===========================================================================================

This code base was used to produce the numerical data in [Faist, Woods, Albert,
Renes, Eisert, Preskill, <in preparation>].

Simple example usage:

.. code-block:: python
    
    from qfi_local_noise import (
        DitstringSuperpositionKet,
        DitstringActionHamiltonian,
        # QutipReducedOperatorCalculator,
        LocalKrausOperators,
        DitstringSuperpositionKetReducedPsiXiCalculator,
        #
        #compute_Fisher_information,
        #
        #compute_Eve_operators_leqk,
        compute_pinched_DF_lower_bound,
    )
    
    
    # GHZ state (|↑↑↑↑⟩ + |↓↓↓↓⟩)/√2
    dsk = DitstringSuperpositionKet(
        4, 2,
        np.array([1, 1])/np.sqrt(2),
        np.array([[0,0,0,0],
                  [1,1,1,1,]])
    )

    # Z+Z+Z+Z Hamiltonian
    daH = DitstringActionHamiltonian(
        dsk,
        np.array([4, -4]),
    )

    # Local Kraus operators -- amplitude damping
    p = 0.05
    lko = LocalKrausOperators(
        [qutip.Qobj([[np.sqrt(1-p),0],[0,1]]),
         qutip.Qobj([[0,0],[np.sqrt(p),0]])]
    )

    # full setting --
    dskrpxc = DitstringSuperpositionKetReducedPsiXiCalculator(
        dsk, daH, lko
    )

    # compute the bound --
    DFlb, _ = compute_pinched_DF_lower_bound(dskrpxc, lko, k=2)

    F_Alice = 4 * daH.variance()

    F_Bob_ub = F_Alice - DFlb

    print(f"Upper bound on Bob's quantum Fisher information w.r.t. time = {F_Bob_ub:.4g}")



