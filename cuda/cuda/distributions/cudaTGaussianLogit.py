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
class cudaTGaussianLogit(cudaDistribution, family="altar.cuda.distributions.tgaussianlogit"):
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

    def cuInitialize(self, application):
        """
        cuda initialize distribution
        """
        # super class process
        super().cuInitialize(application=application)

        # compute the normalized support Phi(a) = 1/2(1+erf((a-mean)/(sqrt(2)*sigma))
        from math import erf, sqrt
        sqrt2sigma = sqrt(2.0)*self.sigma
        Phi = lambda x : 0.5+0.5*erf((x-self.mean)/sqrt2sigma)
        low, high = self.support
        self.support_normalized = (Phi(low), Phi(high))
        # all done
        return self

    def cuInitSample(self, theta, batch):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """

        # call cuda c extension
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

        # all valid, simply return
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
        Fill my portion of {likelihood} with the likelihoods of the samples in physical {theta}
        """
        # call extension
        libcudaaltar.cudaTGaussian_logpdf(theta.data, prior.data, batch, self.idx_range,
                                          (self.mean, self.sigma), self.support_normalized)
        # all done
        return self

    def cuToPhysical(self, theta, batch):
        """
        Transform {theta} from (-Infty, Infty) to physical ranged parameters with inverse logit function
        """

        libcudaaltar.cudaTGaussianLogit_tophysical(theta.data, batch, self.idx_range, (self.mean, self.sigma), self.support_normalized)
        # all done
        return self

    def cuToSampling(self, theta, batch):
        """
        Transform {theta} from physical ranged parameters to sampling unbounded parameters with logit function
        """

        libcudaaltar.cudaTGaussianLogit_tosampling(theta.data, batch, self.idx_range, (self.mean, self.sigma), self.support_normalized)
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
    support_normalized = None
# end of file
