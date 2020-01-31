# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# get the package
import altar
import altar.ext.cudaaltar as libcudaaltar

# get the base
from .cudaDistribution import cudaDistribution

# the declaration
class cudaTGaussian(cudaDistribution, family="altar.cuda.distributions.gaussian"):
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

    def cuInitSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        batch = theta.shape[0]
        # call cuda c extension
        libcudaaltar.cudaTGaussian_sample(theta.data, batch, self.idx_range, (self.mean, self.sigma), self.support)
        # and return
        return self

    def cuVerify(self, theta, mask):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        Arguments:
            theta cuArray (samples x total_parameters)
        """
        # number of samples to be processed
        batch = theta.shape[0]
        # call cuda c extension
        libcudaaltar.cudaRanged_verify(theta.data, mask.data, batch, self.idx_range, self.support)
        # return the invalidity flags
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill my portion of {likelihood} with the likelihoods of the samples in {theta}
        """
        # call extension
        libcudaaltar.cudaTGaussian_logpdf(theta.data, prior.data, batch, self.idx_range,
                                          (self.mean, self.sigma), self.support)
        # all done
        return self

    # local variables

# end of file
