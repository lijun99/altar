# -*- python -*-
# -*- coding: utf-8 -*-

import altar
from .Scheduler import Scheduler as scheduler

@altar.foundry(
    implements=scheduler,
    tip="a scheduler that keeps temperature fixed at 1 (beta=1)")
def constanttemperature():
    from .ConstantTemperature import ConstantTemperature
    __doc__ = ConstantTemperature.__doc__
    return ConstantTemperature

@altar.foundry(
    implements=scheduler,
    tip="a Bayesian scheduler based on the COV algorithm")
def cov():
    from .COV import COV
    __doc__ = COV.__doc__
    return COV

# end of file
