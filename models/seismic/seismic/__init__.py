# -*- python -*-
# -*- coding: utf-8 -*-
#
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# the package
import altar

# publish the protocol for probability distributions
from altar.distributions import Distribution as distribution

# implementations
@altar.foundry(implements=distribution, tip="the Moment Magnitude distribution")
def moment():
    # grab the factory
    from .Moment import Moment as moment
    # attach its docstring
    __doc__ = moment.__doc__
    # and return it
    return moment

# implementations
@altar.foundry(implements=altar.models.model, tip="static inversion model")
def static():
    # grab the factory
    from .Static import Static as static
    # attach its docstring
    __doc__ = static.__doc__
    # and return it
    return static

# implementations
@altar.foundry(implements=altar.models.model, tip="static inversion model with Cp")
def staticCp():
    # grab the factory
    from .StaticCp import StaticCp as staticCp
    # attach its docstring
    __doc__ = staticCp.__doc__
    # and return it
    return staticCp

# end of file
