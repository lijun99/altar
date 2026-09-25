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
# my scheduler default
from ..schedulers.COV import COV


# my declaration
class Catmip(Annealer, family="altar.controllers.catmip"):
    """
    The CATMIP controller: simulated annealing driven by the COV (coefficient-of-variation)
    schedule of Ching[2007], as used by Minson[2013]. This is {Annealer} with its scheduler
    pinned to {COV} instead of the generic default.
    """

    # user configurable state
    scheduler = altar.bayesian.scheduler(default=COV)
    scheduler.doc = "the generator of the annealing schedule"


# end of file
