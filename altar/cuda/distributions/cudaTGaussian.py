# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# get the package
import altar
import altar.cuda.ext.cudaaltar as libcudaaltar

# get the base
from .cudaDistribution import cudaDistribution

# the declaration
class cudaTGaussian(cudaDistribution, family="altar.cuda.distributions.tgaussian"):
    """
    The cuda gaussian probability distribution
    """

    # user configurable state
    mean = altar.properties.float(default=0.0)
    mean.doc = "the mean value"
    sigma = altar.properties.float(default=1.0)
    sigma.doc = " the standard deviation"
    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the truncated gaussian distribution"

    def cu_initialize(self, application):
        """
        cuda initialize distribution
        """
        # super class process
        super().cu_initialize(application=application)

        # compute the normalized support Phi(a) = 1/2(1+erf((a-mean)/(sqrt(2)*sigma))
        from math import erf, sqrt
        sqrt2sigma = sqrt(2.0)*self.sigma
        Phi = lambda x : 0.5+0.5*erf((x-self.mean)/sqrt2sigma)
        low, high = self.support
        self.support_normalized = (Phi(low), Phi(high))
        # all done
        return self

    def cu_init_sample(self, theta, batch):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """

        # call cuda c extension
        libcudaaltar.cudaTGaussian_sample(theta.data, batch, self.idx_range, (self.mean, self.sigma), self.support_normalized)
        # and return
        return self

    def cu_verify(self, theta, mask, batch):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        Arguments:
            theta cuArray (samples x total_parameters)
        """

        # call cuda c extension
        libcudaaltar.cudaRanged_verify(theta.data, mask.data, batch, self.idx_range, self.support)
        # return the invalidity flags
        return mask

    def cu_eval_prior(self, theta, prior, batch):
        """
        Fill my portion of {likelihood} with the likelihoods of the samples in {theta}
        """
        # call extension
        libcudaaltar.cudaTGaussian_logpdf(theta.data, prior.data, batch, self.idx_range,
                                          (self.mean, self.sigma), self.support_normalized)
        # all done
        return self

    # local variables
    support_normalized = None
# end of file
