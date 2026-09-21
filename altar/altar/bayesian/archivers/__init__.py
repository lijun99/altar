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

@altar.foundry(
    implements=archiver,
    tip="an archiver that records the results and progress to HDF5 files")
def h5recorder():
    from .H5Recorder import H5Recorder
    __doc__ = H5Recorder.__doc__
    return H5Recorder

# end of file
