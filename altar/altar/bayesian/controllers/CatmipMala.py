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
from .Catmip import Catmip
# my sampler default
from ..samplers.MALA import MALA


# my declaration
class CatmipMala(Catmip, family="altar.controllers.catmip_mala"):
    """
    CATMIP annealing (the COV coefficient-of-variation beta schedule) with the sampler pinned
    to the Metropolis-adjusted Langevin {MALA} sampler instead of {Metropolis}
    """

    # user configurable state
    sampler = altar.bayesian.sampler(default=MALA)
    sampler.doc = "the sampler of the posterior distribution"


# end of file
