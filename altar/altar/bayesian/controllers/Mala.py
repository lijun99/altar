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
from ..samplers.MALA import MALA


# my declaration
class Mala(Annealer, family="altar.controllers.mala"):
    """
    The Metropolis-adjusted Langevin algorithm with no annealing ladder: {Annealer} with its
    sampler pinned to {MALA} instead of {Metropolis}. {Annealer}'s default {ConstantTemperature}
    scheduler holds beta fixed at 1, so the entire {job.steps} budget of proposals is spent
    sampling the actual posterior, not an intermediate beta ladder.
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=MALA)
    sampler.doc = "the sampler of the posterior distribution"


# end of file
