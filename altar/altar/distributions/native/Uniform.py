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
class Uniform(base):
    """
    The cpu implementation of the uniform probability distribution
    """


    def initialize(self, rng: NumpyRNG, application: Application | None = None) -> typing.Self:
        """
        Initialize with the given random number generator
        """
        # hold on to the generator
        super().initialize(rng=rng, application=application)
        # set up my transform, if reparameterizing
        self._initialize_transform(application=application)
        # all done
        return self


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray,
               batch: int | None = None) -> numpy.ndarray:
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones;
        {theta} is physical, reparameterized or not
        """
        # mark the samples with a parameter outside my support
        return self.outside(theta=theta, mask=mask, support=self.support)


    def draw(self, shape: tuple[int, ...]) -> numpy.ndarray:
        """
        An array of {shape} with values drawn uniformly from my support
        """
        low, high = self.support
        return self.rng.uniform(low, high, size=shape)


    def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        The log density of each entry of {x}: -log(high - low) on [low, high)
        """
        low, high = self.support
        return numpy.where((x >= low) & (x < high), -math.log(high - low), -numpy.inf)


    def prior_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta; when reparameterized,
        this is exactly the transform's jacobian-gradient, since a uniform prior's
        physical-space gradient is always zero
        """
        g = self.restrict(theta=gradient)
        if self.reparameterize:
            θ = self.restrict(theta=theta)
            self.transform.jacobian_gradient(theta=θ, gradient=g, batch=batch)
        else:
            g[...] = 0
        return self


# end of file
