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
    The cuda uniform probability distribution with logit transformation.
    Samples in sampling space: (-∞, ∞)
    Parameters in physical space: [a, b], where a = support[0], b = support[1]
    Transformation: physical = a + (b-a) * sigmoid(sampling)
                    sampling = logit((physical - a)/(b - a)) 
    """

    # user configurable state
    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the prior distribution in physical space"

    # enable reparameterization
    has_reparametrization = True


    def cu_init_sample(self, theta, batch):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # call cuda c extension to initialize uniform distributed samples
        libcudaaltar.cudaLogistic_sample(theta.data, batch, self.idx_range)

        # and return
        return self

    def cu_verify(self, theta, mask, batch):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        Arguments:
            theta cuArray (samples x total_parameters)
        """
        # now unbounded, nothing to do
        # all done; return the rejection map
        return mask

    def cu_eval_prior(self, theta, prior, batch):
        """
        Fill my portion of {prior} with the prior probabilities of the samples in {theta}
        theta should be in physical space [support[0], support[1]]
        """
        # evaluate uniform prior in physical space
        libcudaaltar.cudaUniform_logpdf(theta.data, prior.data, batch, self.idx_range, self.support)
        return self

    def cu_eval_jacobian(self, theta, jacobian, batch):
        """
        Evaluate the log of the Jacobian determinant for the logit transformation.
        theta should be in sampling space (-∞, ∞)
        """
        # compute log|d(physical)/d(sampling)| = log(sigmoid'(x)) + log(support_width)
        libcudaaltar.cudaLogistic_logpdf(theta.data, jacobian.data, batch, self.idx_range)
        return self

    def cu_to_physical(self, phi=None, theta=None, batch=None):
        """
        Transform parameters from sampling space {phi} (-inf, inf) to physical space {theta}
        [support[0], support[1]] using the inverse logit (sigmoid) function.
        """
        if theta is None or batch is None:
            raise TypeError("cu_to_physical requires theta and batch")
        if phi is None:
            phi = theta
        libcudaaltar.cudaUniformLogit_tophysical(
            phi.data, theta.data, batch, self.idx_range, self.support
        )
        return self

    def cu_to_sampling(self, theta=None, phi=None, batch=None):
        """
        Transform parameters from physical space [support[0], support[1]] to sampling space (-inf, inf)
        using the logit function.
        """
        if theta is None or batch is None:
            raise TypeError("cu_to_sampling requires theta and batch")
        if phi is None:
            phi = theta
        libcudaaltar.cudaUniformLogit_tosampling(
            theta.data, phi.data, batch, self.idx_range, self.support
        )
        return self

    def cu_prior_gradient(self, theta, gradient, batch, index=None):
        """
        Compute the gradient of log prior with respect to physical parameters.
        For uniform distribution, this is zero everywhere except at boundaries.

        Parameters:
            theta: parameters in physical space [support[0], support[1]]
            gradient: target array for the gradient
            batch: number of samples
            index: optional, specific parameter index
        """
        # for uniform distribution, gradient is 0 except at boundaries
        return self

    def cu_jacobian_gradient(self, theta, gradient, batch, index=None):
        """
        Compute the gradient of the log Jacobian determinant.

        Parameters:
            theta: parameters in sampling space (-∞, ∞)
            gradient: target array for the gradient
            batch: number of samples
            index: optional, specific parameter index
        """
        # compute d/dx log(sigmoid'(x))
        libcudaaltar.cudaUniformLogit_logpdfgradient(
            theta.data, gradient.data, batch, self.idx_range, self.support
        )
        return self

    # local variables

# end of file
