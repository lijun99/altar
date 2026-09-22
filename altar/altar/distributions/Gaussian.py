# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# get the package
import altar

# get the protocol
from . import distribution
# and my base class
from .Base import Base as base


# the declaration
class Gaussian(base, family="altar.distributions.gaussian"):
    """
    The Gaussian probability distribution
    """


    # user configurable state
    mean = altar.properties.float(default=0)
    mean.doc = "the mean value of the distribution"

    sigma = altar.properties.float(default=1)
    sigma.doc = "the standard deviation of the distribution"

    sigma_invsqr = None

    # protocol obligations
    @altar.export
    def initialize(self, rng):
        """
        Initialize with the given random number generator
        """
        # set up sigma^2
        sigma = self.sigma
        self.sigma_invsqr = 1/(sigma*sigma)

        # set up my pdf
        self.pdf = altar.pdf.gaussian(rng=rng.rng, mean=self.mean, sigma=self.sigma)
        # all done
        return self


    @altar.export
    def verify(self, theta, mask):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # all samples are valid, so there is nothing to do
        return mask

    @altar.export
    def prior_gradient(self, theta, gradient):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta, elementwise, for the
        samples in {theta}. {gradient} has the same shape as {theta}.

        For a Gaussian, d\log P(\theta)/d\theta = (mean - theta) / sigma^2, elementwise.
        """
        # grab the portion of the sample and gradient that are mine
        θ = self.restrict(theta=theta)
        g = self.restrict(theta=gradient)

        # find out how many samples in the set
        samples = θ.rows
        # and how many parameters belong to me
        parameters = θ.columns

        # go through the samples in θ
        for sample in range(samples):
            # and every parameter in this sample
            for parameter in range(parameters):
                g[sample, parameter] = (self.mean - θ[sample, parameter]) * self.sigma_invsqr

        # all done
        return self


# end of file
