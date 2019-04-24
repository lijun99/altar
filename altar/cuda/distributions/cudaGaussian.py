# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2019 parasim inc
# (c) 2010-2019 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# get the package
import altar
import altar.ext.cudaaltar as libcudaaltar

# get the base
from .cudaDistribution import cudaDistribution

# the declaration
class cudaGaussian(cudaDistribution, family="altar.cuda.distributions.gaussian"):
    """
    The cuda gaussian probability distribution
    """

    # user configurable state
    mean = altar.properties.float(default=0.0)
    mean.doc = "the mean value"
    sigma = altar.properties.float(default=1.0)
    sigma.doc = " the standard deviation"

    def cuInitSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        batch = theta.shape[0]
        # call cuda c extension
        libcudaaltar.cudaGaussian_sample(theta.data, batch, self.idx_range, (self.mean, self.sigma))
        # and return
        return self

    def cuVerify(self, theta, mask):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        Arguments:
            theta cuArray (samples x total_parameters)   
        """
        # all samples are valid!
        # all done; return the rejection map
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill my portion of {likelihood} with the likelihoods of the samples in {theta}
        """
        # call extension
        libcudaaltar.cudaGaussian_logpdf(theta.data, prior.data, batch, self.idx_range, (self.mean, self.sigma))
        # all done
        return self
        
    # local variables

# end of file
