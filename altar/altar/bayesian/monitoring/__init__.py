# -*- python -*-
# -*- coding: utf-8 -*-

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
