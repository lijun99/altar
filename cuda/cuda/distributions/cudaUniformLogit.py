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
class cudaUniformLogit(cudaDistribution, family="altar.cuda.distributions.uniformlogit"):
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
        # call cuda c extension to initialize uniform distributed samples
        libcudaaltar.cudaUniformLogit_sample(theta.data, batch, self.idx_range, self.support)
        theta.print()

        # and return
        return self

    def cuVerify(self, theta, mask):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        Arguments:
            theta cuArray (samples x total_parameters)
        """
        # now unbounded, nothing to do
        # all done; return the rejection map
        return mask

    def cuEvalPrior(self, theta, prior, batch):
        """
        Fill my portion of {likelihood} with the likelihoods of the samples in {theta}
        """
        # call cuda c extension
        libcudaaltar.cudaUniformLogit_logpdf(theta.data, prior.data, batch, self.idx_range, self.support)

        # all done
        return self

    def cuTransform(self, theta, batch):
        """
        Transform {theta} from (-Infty, Infty) to ranged with inverse logit function
        """

        libcudaaltar.cudaUniformLogit_inverse(theta.data, batch, self.idx_range, self.support)

        return self

    # local variables

# end of file
