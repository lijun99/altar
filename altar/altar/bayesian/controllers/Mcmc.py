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
# my sampler and scheduler defaults
from ..samplers.Metropolis import Metropolis
from ..schedulers.ConstantTemperature import ConstantTemperature


# my declaration
class Mcmc(Annealer, family="altar.controllers.mcmc"):
    """
    Markov chain Monte Carlo with no annealing ladder: {Annealer} with its sampler pinned to
    {Metropolis} and its scheduler to {ConstantTemperature}, which holds beta fixed, at 1 by
    default. The chains walk {rounds} times, {job.steps} Metropolis steps each, and the
    proposal adapts to the samples between walks: its covariance, and its scaling from the
    acceptance rate
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=Metropolis)
    sampler.doc = "the sampler of the posterior distribution"

    scheduler = altar.bayesian.scheduler(default=ConstantTemperature)
    scheduler.doc = "keeps beta fixed; see {ConstantTemperature.beta_start}"

    rounds = altar.properties.int(default=16)
    rounds.doc = "the walks of {job.steps} Metropolis steps; the proposal adapts to the samples " \
                 "between them"


    # implementation details
    def continuing(self, worker, iteration, tolerance):
        """
        Whether to walk the chains again, after {iteration} walks: for {rounds} walks
        """
        return iteration < self.rounds


# end of file
