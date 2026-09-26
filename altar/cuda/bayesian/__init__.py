# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
{altar.bayesian.metropolis} now picks its cpu/cuda implementation itself, at {initialize}
time (see {altar.bayesian.samplers.Metropolis._makeImpl}), so it is safe to use regardless of
backend. The foundry below exists only for {.pfg} files that still spell out the explicit
"altar.cuda.bayesian.metropolis" path, and is a pure alias for {altar.bayesian.metropolis}.
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
    from altar.bayesian.samplers.Metropolis import Metropolis as metropolis
    # attach its docstring
    __doc__ = metropolis.__doc__
    # and return it
    return metropolis


# end of file
