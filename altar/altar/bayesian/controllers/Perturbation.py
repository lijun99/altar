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
# my parts
from ..samplers.Perturbation import Perturbation as sampler
from ..schedulers.ConstantTemperature import ConstantTemperature


# declaration
class Perturbation(Annealer, family="altar.controllers.perturbation"):
    """
    Exact sampling of linear-gaussian posteriors: {Annealer} at β = 1, with its sampler pinned to
    {Perturbation}. Without bounds, one walk draws exact, independent samples; with bounds, the
    chains walk {rounds} times, {job.steps} exact trajectories each. The log evidence comes with
    the samples.
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=sampler)
    sampler.doc = "the sampler of the posterior distribution"

    scheduler = altar.bayesian.scheduler(default=ConstantTemperature)
    scheduler.doc = "keeps beta fixed at 1"

    rounds = altar.properties.int(default=1)
    rounds.validators = altar.constraints.isGreaterEqual(value=1)
    rounds.doc = "the walks; one is enough without bounds"


    # implementation details
    def continuing(self, worker, iteration, tolerance):
        """
        Whether to walk the chains again, after {iteration} walks
        """
        return iteration - self.start_iteration < self.rounds


# end of file
