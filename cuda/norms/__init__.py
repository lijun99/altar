# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu



# the package
import altar
import altar.cuda


# publish the protocol for norms
from altar.norms.Norm import Norm as norm


# implementations
@altar.foundry(implements=norm, tip="the cudaL2 norm")
def l2():
    # grab the factory
    from .cudaL2 import cudaL2 as l2
    # attach its docstring
    __doc__ = l2.__doc__
    # and return it
    return l2


# end of file
