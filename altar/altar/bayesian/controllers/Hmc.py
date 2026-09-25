# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar
# my superclass
from .Annealer import Annealer
# my sampler default
from ..samplers.HMC import HMC


# my declaration
class Hmc(Annealer, family="altar.controllers.hmc"):
    """
    Hamiltonian Monte Carlo with no annealing ladder: {Annealer} with its sampler pinned to
    the leapfrog-based {HMC} sampler instead of {Metropolis}. {Annealer}'s default
    {ConstantTemperature} scheduler holds beta fixed at 1, so the entire {job.steps} budget of
    trajectories is spent sampling the actual posterior, not an intermediate beta ladder.
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=HMC)
    sampler.doc = "the sampler of the posterior distribution"


# end of file
