# -*- python -*-
# -*- coding: utf-8 -*-

import altar
from .StepSizer import StepSizer as stepsizer

@altar.foundry(
    implements=stepsizer,
    tip="step size regulator with a fixed value")
def fixedstep():
    from .StepSizer import FixedStepSize
    __doc__ = FixedStepSize.__doc__
    return FixedStepSize

@altar.foundry(
    implements=stepsizer,
    tip="step size regulator with linear acceptance feedback a + b * r")
def linearrate():
    from .StepSizer import LinearRate
    __doc__ = LinearRate.__doc__
    return LinearRate

@altar.foundry(
    implements=stepsizer,
    tip="step size regulator targeting an acceptance rate")
def targetedrate():
    from .StepSizer import TargetedRate
    __doc__ = TargetedRate.__doc__
    return TargetedRate

@altar.foundry(
    implements=stepsizer,
    tip="dual-averaging step size regulator")
def dual():
    from .StepSizer import DualAveragingStepSize
    __doc__ = DualAveragingStepSize.__doc__
    return DualAveragingStepSize

# end of file
