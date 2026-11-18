# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the package
import altar


# publish the protocol for norms
from .DataObs import DataObs as data

# implementations
@altar.foundry(implements=data, tip="the data with L2 norm")
def datal2():
    # grab the factory
    from .DataL2 import DataL2 as datal2
    # attach its docstring
    __doc__ = datal2.__doc__
    # and return it
    return datal2


@altar.foundry(implements=data, tip="the data with L2 norm and a covariance separable in space and time")
def datakron():
    # grab the factory
    from .DataKron import DataKron as datakron
    # attach its docstring
    __doc__ = datakron.__doc__
    # and return it
    return datakron


# end of file
