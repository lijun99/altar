# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# the package
import altar
# and the protocols
from altar.bayesian.Controller import Controller as controller
from altar.bayesian.Sampler import Sampler as sampler
from altar.bayesian.Scheduler import Scheduler as scheduler



# implementations
@altar.foundry(implements=sampler, tip="the Metropolis algorithm as a Bayesian sampler")
def metropolis():
    # grab the factory
    from .cudaMetropolis import cudaMetropolis as metropolis
    # attach its docstring
    __doc__ = metropolis.__doc__
    # and return it
    return metropolis


# implementations
@altar.foundry(implements=sampler, tip="the Metropolis sampler with targeted correlation")
def metropolisvaryingsteps():
    # grab the factory
    from .cudaMetropolisVaryingSteps import cudaMetropolisVaryingSteps as metropolisvarysteps
    # attach its docstring
    __doc__ = metropolisvaryingsteps.__doc__
    # and return it
    return metropolisvaryingsteps

# implementations
@altar.foundry(implements=sampler, tip="the Metropolis sampler with a targeted acceptance rate")
def adaptivemetropolis():
    # grab the factory
    from .cudaAdaptiveMetropolis import cudaAdaptiveMetropolis as adaptivemetropolis
    # attach its docstring
    __doc__ = adaptivemetropolis.__doc__
    # and return it
    return adaptivemetropolis


# end of file
