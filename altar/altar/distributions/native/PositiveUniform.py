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
class PositiveUniform(base):
    """
    The cpu implementation of the uniform probability distribution over (0, 1), excluding 0
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
        # every sample the generator itself produces is already in (0, 1), so there is
        # nothing to check
        return mask


    def draw(self, shape: tuple[int, ...]) -> numpy.ndarray:
        """
        An array of {shape} with values drawn uniformly from (0, 1), never 0
        """
        x = self.rng.random(size=shape)
        # redraw the zeros, however unlikely
        while (zero := x == 0).any():
            x[zero] = self.rng.random(size=int(zero.sum()))
        return x


    def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        The log density of each entry of {x}: 0 on (0, 1), -inf elsewhere
        """
        return numpy.where((x > 0) & (x < 1), 0.0, -numpy.inf)


# end of file
