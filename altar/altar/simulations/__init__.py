# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# get the package
import altar

# the protocols
from .Archiver import Archiver as archiver
from .Dispatcher import Dispatcher as dispatcher
from .Monitor import Monitor as monitor
from .Run import Run as run
from .RNG import RNG as rng


# the implementations
@altar.foundry(implements=rng, tip="the default random number generator")
def numpy():
    # grab the factory
    from .NumpyRNG import NumpyRNG as numpy
    # attach its docstring
    __doc__ = numpy.__doc__
    # and return it
    return numpy


@altar.foundry(implements=run, tip="the default job parameter specification")
def job():
    # grab the factory
    from .Job import Job as job
    # attach its docstring
    __doc__ = job.__doc__
    # and return it
    return job


@altar.foundry(implements=monitor, tip="simple monitor that uses journal channels")
def reporter():
    # grab the factory
    from .Reporter import Reporter as reporter
    # attach its docstring
    __doc__ = reporter.__doc__
    # and return it
    return reporter


# end of file
