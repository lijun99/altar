# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from .StepCounter import StepCounter as stepcounter

@altar.foundry(
    implements=stepcounter,
    tip="run a fixed number of MC steps per β step")
def fixedsteps():
    from .StepCounter import FixedSteps
    __doc__ = FixedSteps.__doc__
    return FixedSteps

@altar.foundry(
    implements=stepcounter,
    tip="run MC steps in blocks until the chain has decorrelated from where it started")
def decorrelating():
    from .StepCounter import DecorrelatingSteps
    __doc__ = DecorrelatingSteps.__doc__
    return DecorrelatingSteps

# end of file
