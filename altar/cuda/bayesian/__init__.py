# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
The cuda sampler/state implementations now live under {altar.bayesian.samplers.cuda}/
{altar.bayesian.states.cuda}, mirroring {altar.distributions.cuda}'s shape; the foundries
below exist only for {.pfg} files that still spell out the explicit
"altar.cuda.bayesian.metropolis"/... path, and resolve to the very same classes
{altar.bayesian.metropolis}/... do.
"""

# the package
import altar
# and the protocols
from altar.bayesian.controllers.Controller import Controller as controller
from altar.bayesian.samplers.Sampler import Sampler as sampler
from altar.bayesian.schedulers.Scheduler import Scheduler as scheduler


# implementations
@altar.foundry(implements=sampler, tip="the Metropolis algorithm as a Bayesian sampler")
def metropolis():
    # grab the factory
    from altar.bayesian.samplers.cuda.Metropolis import Metropolis as metropolis
    # attach its docstring
    __doc__ = metropolis.__doc__
    # and return it
    return metropolis


@altar.foundry(implements=sampler, tip="the Metropolis sampler with targeted correlation")
def metropolisvaryingsteps():
    # grab the factory
    from altar.bayesian.samplers.cuda.MetropolisVaryingSteps import MetropolisVaryingSteps as metropolisvaryingsteps
    # attach its docstring
    __doc__ = metropolisvaryingsteps.__doc__
    # and return it
    return metropolisvaryingsteps


@altar.foundry(implements=sampler, tip="the Metropolis sampler with a targeted acceptance rate")
def adaptivemetropolis():
    # grab the factory
    from altar.bayesian.samplers.cuda.AdaptiveMetropolis import AdaptiveMetropolis as adaptivemetropolis
    # attach its docstring
    __doc__ = adaptivemetropolis.__doc__
    # and return it
    return adaptivemetropolis


# end of file
