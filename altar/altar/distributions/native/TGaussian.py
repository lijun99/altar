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
import math
import numpy
# get the package
import altar

# and my base class
from .Base import Base as base


# the declaration
class TGaussian(base):
    """
    The cpu implementation of the Gaussian probability distribution, truncated to a finite
    support
    """


    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator
        """
        # cache 1/sigma^2, needed by {prior_gradient}
        self.sigma_invsqr = 1 / (self.sigma * self.sigma)
        # set up my pdf
        self.pdf = altar.pdf.tgaussian(
            rng=rng.rng, mean=self.mean, sigma=self.sigma, support=self.support)
        # set up my transform, if reparameterizing
        self._initialize_transform(application=application)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones;
        {theta} is physical, reparameterized or not
        """
        # mark the samples with a parameter outside my support
        return self.outside(theta=theta, mask=mask, support=self.support)


    def log_density(self, x):
        """
        The log density of each entry of {x}: the gaussian's, renormalized by the mass the
        truncation retains, and -inf outside my support
        """
        low, high = self.support
        Φ = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
        mass = Φ((high - self.mean) / self.sigma) - Φ((low - self.mean) / self.sigma)
        u = (x - self.mean) / self.sigma
        logp = -0.5 * u * u - math.log(math.sqrt(2 * math.pi) * self.sigma * mass)
        return numpy.where((x >= low) & (x <= high), logp, -numpy.inf)


    def prior_gradient(self, theta, gradient, batch=None):
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
        numpy.asarray(g)[:] = (self.mean - numpy.asarray(θ)) * self.sigma_invsqr

        # reparameterized: chain the physical-space gradient into sampling space
        if self.reparameterize:
            self.transform.chain_gradient(theta=θ, gradient=g, batch=batch)

        # all done
        return self


    # private data, set by the shim before {initialize} runs
    mean = None
    sigma = None
    support = None
    # set by {initialize}
    sigma_invsqr = None


# end of file
