# -*- python -*-
# -*- coding: utf-8 -*-

import altar
from altar.simulations.Archiver import Archiver as archiver

@altar.foundry(
    implements=archiver,
    tip="an archiver to record the results and progress")
def recorder():
    from .Recorder import Recorder
    __doc__ = Recorder.__doc__
    return Recorder

# end of file
