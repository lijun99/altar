# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# the package
import altar


# publish the protocol for probability distributions
from .Distribution import Distribution as distribution


# implementations
@altar.foundry(implements=distribution, tip="the uniform probability distribution")
def uniform():
    # grab the factory
    if altar.backends.active() == "cuda":
        try:
            from altar.cuda.distributions.cudaUniform import cudaUniform as uniform
        except ImportError:
            from .Uniform import Uniform as uniform
    else:
        from .Uniform import Uniform as uniform
    # attach its docstring
    __doc__ = uniform.__doc__
    # and return it
    return uniform


@altar.foundry(implements=distribution, tip="the gaussian probability distribution")
def gaussian():
    # grab the factory
    if altar.backends.active() == "cuda":
        try:
            from altar.cuda.distributions.cudaGaussian import cudaGaussian as gaussian
        except ImportError:
            from .Gaussian import Gaussian as gaussian
    else:
        from .Gaussian import Gaussian as gaussian
    # attach its docstring
    __doc__ = gaussian.__doc__
    # and return it
    return gaussian


@altar.foundry(implements=distribution, tip="the unit gaussian probability distribution")
def ugaussian():
    # grab the factory
    from .UnitGaussian import UnitGaussian as ugaussian
    # attach its docstring
    __doc__ = ugaussian.__doc__
    # and return it
    return ugaussian


# end of file
