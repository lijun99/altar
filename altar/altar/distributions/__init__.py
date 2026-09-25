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


# implementations; each is a single component that picks its cpu/cuda backend internally, at
# {initialize} time, so there is no backend switch here any more -- see {Base._makeImpl}
@altar.foundry(implements=distribution, tip="the uniform probability distribution")
def uniform():
    from .Uniform import Uniform as uniform
    return uniform


@altar.foundry(implements=distribution, tip="the gaussian probability distribution")
def gaussian():
    from .Gaussian import Gaussian as gaussian
    return gaussian


@altar.foundry(implements=distribution, tip="the unit gaussian probability distribution")
def ugaussian():
    from .UnitGaussian import UnitGaussian as ugaussian
    return ugaussian


@altar.foundry(
    implements=distribution,
    tip="the uniform probability distribution over (0, 1), excluding 0")
def positiveuniform():
    from .PositiveUniform import PositiveUniform as positiveuniform
    return positiveuniform


@altar.foundry(
    implements=distribution, tip="the gaussian probability distribution, truncated to a finite support")
def tgaussian():
    from .TGaussian import TGaussian as tgaussian
    return tgaussian


# end of file
