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
class TGaussian(base):
    """
    The cpu implementation of the Gaussian probability distribution, truncated to a finite
    support
    """


    def initialize(self, rng: NumpyRNG, application: Application | None = None) -> typing.Self:
        """
        Initialize with the given random number generator
        """
        # cache 1/sigma^2, needed by {prior_gradient}
        self.sigma_invsqr = 1 / (self.sigma * self.sigma)
        # the mass the truncation retains
        low, high = self.support
        if not low < high:
            raise ValueError(f"tgaussian: empty support {self.support}")
        self.mass = Φ((high - self.mean) / self.sigma) - Φ((low - self.mean) / self.sigma)
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
        An array of {shape} with values drawn from me: gaussian draws within my support, or,
        when the support holds little of the gaussian, the inverse of my cdf at uniform draws
        """
        low, high = self.support
        x = numpy.empty(shape)
        missing = numpy.ones(shape, dtype=bool)
        # rejection, while it accepts often enough
        for _ in range(16 if self.mass > 0.05 else 0):
            n = int(missing.sum())
            if n == 0:
                return x
            y = self.rng.normal(self.mean, self.sigma, size=n)
            keep = (y >= low) & (y <= high)
            idx = numpy.flatnonzero(missing)[keep]
            x.flat[idx] = y[keep]
            missing.flat[idx] = False
        # the rest by inverting the cdf, by bisection
        n = int(missing.sum())
        if n:
            erf = numpy.vectorize(math.erf, otypes=[float])
            cdf = lambda v: 0.5 * (1 + erf((v - self.mean) / (self.sigma * math.sqrt(2))))
            p = cdf(numpy.full(n, low)) + self.rng.random(size=n) * self.mass
            a, b = numpy.full(n, float(low)), numpy.full(n, float(high))
            for _ in range(64):
                mid = 0.5 * (a + b)
                below = cdf(mid) < p
                a, b = numpy.where(below, mid, a), numpy.where(below, b, mid)
            x[missing] = 0.5 * (a + b)
        return x


    def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        The log density of each entry of {x}: the gaussian's, renormalized by the mass the
        truncation retains, and -inf outside my support
        """
        low, high = self.support
        u = (x - self.mean) / self.sigma
        logp = -0.5 * u * u - math.log(math.sqrt(2 * math.pi) * self.sigma * self.mass)
        return numpy.where((x >= low) & (x <= high), logp, -numpy.inf)


    def prior_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta, elementwise, for the
        samples in {theta}. {gradient} has the same shape as {theta}.

        Truncation only rescales the normalization constant, so within the support the log
        density has the same shape as the untruncated Gaussian's:
        d\log P(\theta)/d\theta = (mean - theta) / sigma^2, elementwise.
        """
        # grab the portion of the sample and gradient that are mine
        θ = self.restrict(theta=theta)
        g = self.restrict(theta=gradient)
        # and fill it
        g[...] = (self.mean - θ) * self.sigma_invsqr
        # reparameterized: chain the physical-space gradient into sampling space
        if self.reparameterize:
            self.transform.chain_gradient(theta=θ, gradient=g, batch=batch)
        # all done
        return self


    # private data, set by the shim before {initialize} runs
    mean: float
    sigma: float
    # set by {initialize}
    sigma_invsqr: float
    mass: float


def Φ(z: float) -> float:
    """
    The standard normal cdf
    """
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# end of file
