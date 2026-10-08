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
class Gaussian(base):
    """
    The cpu implementation of the Gaussian probability distribution
    """


    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator
        """
        # cache 1/sigma^2, needed by {prior_gradient}
        self.sigma_invsqr = 1 / (self.sigma * self.sigma)
        # set up my pdf
        self.pdf = altar.pdf.gaussian(rng=rng.rng, mean=self.mean, sigma=self.sigma)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # all samples are valid, so there is nothing to do
        return mask


    def log_density(self, x):
        """
        The log density of each entry of {x}
        """
        u = (x - self.mean) / self.sigma
        return -0.5 * u * u - math.log(math.sqrt(2 * math.pi) * self.sigma)


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta, elementwise, for the
        samples in {theta}. {gradient} has the same shape as {theta}.

        For a Gaussian, d\log P(\theta)/d\theta = (mean - theta) / sigma^2, elementwise.
        """
        # grab the portion of the sample and gradient that are mine
        θ = self.restrict(theta=theta)
        g = self.restrict(theta=gradient)
        # and fill it
        numpy.asarray(g)[:] = (self.mean - numpy.asarray(θ)) * self.sigma_invsqr

        # all done
        return self


    # private data, set by the shim before {initialize} runs
    mean = None
    sigma = None
    # set by {initialize}
    sigma_invsqr = None


# end of file
