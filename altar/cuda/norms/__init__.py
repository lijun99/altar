# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu



# the package
import altar
import altar.cuda


# publish the protocol for norms
from altar.norms.Norm import Norm as norm


# {L2} is a unified component now: it picks its cpu/cuda implementation internally, on first
# use; this foundry exists only for {.pfg} files that still spell out the explicit
# "altar.cuda.norms.l2" path, and resolves to the very same class {altar.norms.l2} does
@altar.foundry(implements=norm, tip="the L2 norm")
def l2():
    # grab the factory
    from altar.norms.L2 import L2 as l2
    # attach its docstring
    __doc__ = l2.__doc__
    # and return it
    return l2


# end of file
