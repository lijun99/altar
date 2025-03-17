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


    def cuInitSample(self, theta, batch):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # call cuda c extension to initialize uniform distributed samples
        libcudaaltar.cudaLogistic_sample(theta.data, batch, self.idx_range)

        # and return
        return self

    def cuVerify(self, theta, mask, batch):
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
        libcudaaltar.cudaLogistic_logpdf(theta.data, prior.data, batch, self.idx_range)

        # all done
        return self

    def cuEvalPriorPhysical(self, theta, prior, batch):
        """
        Fill my portion of {prior} with the prior probabilities of the physical samples in {theta}
        """
        # use the cudaUniform_logpdf for physical parameters
        libcudaaltar.cudaUniform_logpdf(theta.data, prior.data, batch, self.idx_range, self.support)

        # all done
        return self

    def cuToPhysical(self, theta, batch):
        """
        Transform {theta} from (-Infty, Infty) to physical ranged parameters with inverse logit function
        """

        libcudaaltar.cudaUniformLogit_tophysical(theta.data, batch, self.idx_range, self.support)
        # all done
        return self

    def cuToSampling(self, theta, batch):
        """
        Transform {theta} from physical ranged parameters to sampling unbounded parameters with logit function
        """

        libcudaaltar.cudaUniformLogit_tosampling(theta.data, batch, self.idx_range, self.support)
        # all done
        return self

    def cuPriorGradient(self, theta, prior, batch, index=None):
        """
        Fill my portion of {prior} with the gradient of d\log P(\theta)/d\theta_{index}
        """
        # call extension
        libcudaaltar.cudaLogistic_logpdfgradient(theta.data, prior.data, batch, self.idx_range)
        return self

    # local variables

# end of file
