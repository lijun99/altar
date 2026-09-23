# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2026 parasim inc
# (c) 2010-2026 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar
# my superclass
from .Catmip import Catmip
# my sampler default
from ..samplers.HMC import HMC


# my declaration
class CatmipHmc(Catmip, family="altar.controllers.catmiphmc"):
    """
    CATMIP annealing (the COV coefficient-of-variation beta schedule) with the sampler pinned
    to the leapfrog-based {HMC} sampler instead of {Metropolis}
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=HMC)
    sampler.doc = "the sampler of the posterior distribution"


# end of file
