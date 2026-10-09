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

# and my base class
from .Base import Base as base


# the declaration
class SoftUniform(base):
    """
    The cpu implementation of the uniform distribution with logistic edges
    """


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray,
               batch: int | None = None) -> numpy.ndarray:
        """
        Nothing to check: I am positive everywhere
        """
        return mask


    def draw(self, shape: tuple[int, ...]) -> numpy.ndarray:
        """
        An array of {shape} with values drawn uniformly from my support
        """
        low, high = self.support
        return self.rng.uniform(low, high, size=shape)


    def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        The log density of each entry of {x}:
        -log(b - a) + log(1 - exp(-k (b - a))) - softplus(-k (x - a)) - softplus(k (x - b))
        """
        low, high = self.support
        k = self.sharpness
        constant = -math.log(high - low) + math.log1p(-math.exp(-k * (high - low)))
        return constant - numpy.logaddexp(0, -k * (x - low)) - numpy.logaddexp(0, k * (x - high))


    def prior_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta =
        k (sigmoid(-k (x - a)) - sigmoid(k (x - b)))
        """
        low, high = self.support
        k = self.sharpness
        x = self.restrict(theta=theta)
        g = self.restrict(theta=gradient)
        g[...] = 0.5 * k * (numpy.tanh(-0.5 * k * (x - low)) - numpy.tanh(0.5 * k * (x - high)))
        return self


    # private data, set by the shim before {initialize} runs
    support: tuple[float, float]
    sharpness: float


# end of file
