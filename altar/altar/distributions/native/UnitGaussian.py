# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import math
import typing
import numpy

if typing.TYPE_CHECKING:
    from altar.shells.Application import Application
    from altar.simulations.NumpyRNG import NumpyRNG

# and my base class
from .Base import Base as base


# the declaration
class UnitGaussian(base):
    """
    The cpu implementation of the unit Gaussian probability distribution (σ = 1)
    """


    def initialize(self, rng: NumpyRNG, application: Application | None = None) -> typing.Self:
        """
        Initialize with the given random number generator
        """
        # hold on to the generator
        return super().initialize(rng=rng, application=application)


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray,
               batch: int | None = None) -> numpy.ndarray:
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # all samples are valid, so there is nothing to do
        return mask


    def draw(self, shape: tuple[int, ...]) -> numpy.ndarray:
        """
        An array of {shape} with values drawn from me
        """
        return self.rng.standard_normal(size=shape)


    def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        The log density of each entry of {x}
        """
        return -0.5 * x * x - 0.5 * math.log(2 * math.pi)


# end of file
