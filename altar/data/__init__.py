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


# publish the protocol for norms
from .DataObs import DataObs as data

from .DataL2 import DataL2 

# implementations
@altar.foundry(implements=data, tip="the data with L2 norm")
def datal2():
    # grab the factory
    from .DataL2 import DataL2 as datal2
    # attach its docstring
    __doc__ = datal2.__doc__
    # and return it
    return datal2


# end of file
