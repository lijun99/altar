# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from altar.simulations.Monitor import Monitor as monitor

@altar.foundry(
    implements=monitor,
    tip="a monitor that times the various simulation phases")
def profiler():
    from .Profiler import Profiler
    __doc__ = Profiler.__doc__
    return Profiler

# end of file
