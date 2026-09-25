# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

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
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # unpack my support
        low, high = self.support
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)

        # find out how many samples in the set
        samples = θ.rows
        # and how many parameters belong to me
        parameters = θ.columns

        # go through the samples in θ
        for sample in range(samples):
            # and the parameters in this sample
            for parameter in range(parameters):
                # if the parameter lies outside my support
                if not (low <= θ[sample,parameter] <= high):
                    # mark the entire sample as invalid
                    mask[sample] += 1
                    # and skip checking the rest of the parameters
                    break

        # all done; return the rejection map
        return mask


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


    # private data, set by the shim before {initialize} runs
    mean = None
    sigma = None
    support = None
    # set by {initialize}
    sigma_invsqr = None


# end of file
