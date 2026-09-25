# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from .LangevinScheduler import LangevinScheduler as langevinscheduler

@altar.foundry(
    implements=langevinscheduler,
    tip="a Langevin scheduler with power-law decay")
def powerdecay():
    from .PowerDecay import PowerDecay
    __doc__ = PowerDecay.__doc__
    return PowerDecay

@altar.foundry(
    implements=langevinscheduler,
    tip="a Langevin scheduler with exponential decay")
def expdecay():
    from .ExpDecay import ExpDecay
    __doc__ = ExpDecay.__doc__
    return ExpDecay

# end of file
