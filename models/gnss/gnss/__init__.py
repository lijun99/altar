# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar

# the protocol
from altar.models.Model import Model as model


# implementations
@altar.foundry(implements=model, tip="the trajectory model of GNSS position time series")
def trajectory():
    # grab the factory
    from .Trajectory import Trajectory as trajectory
    # attach its docstring
    __doc__ = trajectory.__doc__
    # and return it
    return trajectory


@altar.foundry(implements=model,
               tip="time-dependent slip on a fault mesh from GNSS position time series")
def slip():
    # grab the factory
    from .Slip import Slip as slip
    # attach its docstring
    __doc__ = slip.__doc__
    # and return it
    return slip


# the spatially pooled sparsity prior on a group of basis functions
from altar.distributions import distribution


@altar.foundry(implements=distribution,
               tip="a sparsity prior on a group of basis functions, pooled over neighboring stations")
def pooled():
    # grab the factory
    from .Pooled import Pooled as pooled
    # attach its docstring
    __doc__ = pooled.__doc__
    # and return it
    return pooled


# end of file
