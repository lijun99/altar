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
    {ConstantTemperature} scheduler holds beta fixed at 1, so the entire budget of trajectories
    is spent sampling the actual posterior, not an intermediate beta ladder. The chains walk
    {rounds} times, {job.steps} trajectories each; between walks, the mass matrix adapts to the
    samples, and, during the scheduler's {burnin}, the outlier chains are replaced.
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=HMC)
    sampler.doc = "the sampler of the posterior distribution"

    rounds = altar.properties.int(default=1)
    rounds.validators = altar.constraints.isGreaterEqual(value=1)
    rounds.doc = "the walks of {job.steps} trajectories; the mass matrix adapts to the samples " \
                 "between them, and the scheduler replaces outlier chains during its burn-in"


    # implementation details
    def continuing(self, worker, iteration, tolerance):
        """
        Whether to walk the chains again, after {iteration} walks: for {rounds} walks
        """
        return iteration < self.rounds


# end of file
