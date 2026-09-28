# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the package
import altar
# my base
from .HMC import HMC


# declaration
class MALA(HMC, family="altar.samplers.mala"):
    """
    The Metropolis-adjusted Langevin algorithm: {HMC} with a single leapfrog step per proposal,
    x' = x + (ε²/2) M⁻¹ ∇log π(x) + ε M^{-1/2} z, followed by the Metropolis-Hastings
    acceptance, with a step size steered to its optimal acceptance rate, 0.574

    My implementation is a same-named class in {altar.bayesian.samplers.native} or
    {altar.bayesian.samplers.cuda}, the {HMC} one with a different target acceptance
    """

    # user configurable state
    leapfrog_steps = altar.properties.int(default=1)
    leapfrog_steps.doc = "the number of leapfrog substeps per proposal; 1 for MALA proper"


# end of file
