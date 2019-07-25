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
class cudaUniform(cudaDistribution, family="altar.cuda.distributions.uniform"):
    """
    The cuda uniform probability distribution
    """

    # user configurable state
    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the prior distribution"


    def cuInitSample(self, theta):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # number of samples to be processed
        batch = theta.shape[0]
        # call cuda c extension
        libcudaaltar.cudaUniform_sample(theta.data, batch, self.idx_range, self.support)

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

        # all done; return the rejection map
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill my portion of {likelihood} with the likelihoods of the samples in {theta}
        """
        # call cuda c extension
        libcudaaltar.cudaUniform_logpdf(theta.data, prior.data, batch, self.idx_range, self.support)

        # all done
        return self

    # local variables

# end of file
