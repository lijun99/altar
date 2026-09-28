# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar

# publish the protocol for probability distributions
from altar.distributions import Distribution as distribution

# implementations
@altar.foundry(implements=distribution, tip="the moment magnitude prior for slips")
def moment():
    # grab the factory
    from .Moment import Moment as moment
    # attach its docstring
    __doc__ = moment.__doc__
    # and return it
    return moment

# implementations
@altar.foundry(implements=altar.models.model, tip="static inversion model, built on the linear model")
def static():
    # grab the factory
    from .Static import Static as static
    # attach its docstring
    __doc__ = static.__doc__
    # and return it
    return static

# implementations
@altar.foundry(implements=altar.models.model, tip="kinematic inversion model, cuda only")
def kinematic():
    # grab the factory
    from .Kinematic import Kinematic as kinematic
    # attach its docstring
    __doc__ = kinematic.__doc__
    # and return it
    return kinematic

# end of file
