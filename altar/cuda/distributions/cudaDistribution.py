# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

# externals
import math
# get the package
import altar
import altar.cuda

# get the base
from altar.distributions.Base import Base


# the declaration
class cudaDistribution(Base, family="altar.distributions.cudadistribution"):
    """
    The base class for probability distributions
    """

    # user configurable state from its cpu superclass
    parameters = altar.properties.int()
    parameters.doc = "the number of model parameters that belong to me"

    offset = altar.properties.int(default=0)
    offset.doc = "the starting point of my parameters in the overall model state"

    # class members
    has_reparametrization = False  # whether this distribution uses parameter transformation

    # configuration
    @altar.export
    def initialize(self, rng):
        """
        Initialize with the given random number generator
        """
        # will recommend a framework change to use application instead of rng
        # some distribution might need info from application
        # e.g, cascaded need worker id
        # so, use cu_initialize instead
        return self

    @altar.export
    def verify(self, theta, mask):
        # to satisfy component requirement
        # use cu_verify instead
        return self

    # cuda methods
    def cu_initialize(self, application):
        """
        cuda specific initialization
        """
        # basic initialization
        self.idx_range = (self.offset, self.offset + self.parameters)
        self.device = application.controller.worker.device
        self.precision = application.job.gpuprecision

        return self

    def cu_init_sample(self, theta, batch):
        """
        cuda process to initialize random samples
        """
        return self

    def cu_verify(self, theta, mask, batch):
        """
        cuda process to verify the validity of samples
        """
        return mask

    def cu_eval_prior(self, theta, prior, batch):
        """
        Evaluate the prior P(θ) in physical parameter space.
        For distributions without reparameterization, θ is in the original space.
        For distributions with reparameterization, θ should be transformed to physical space first.
        """
        # Default implementation: flat prior
        return prior

    def cu_eval_prior_with_physical(self, theta, prior, batch):
        """
        Add any prior contributions that depend on physical parameters.
        """
        # Default implementation: no extra contributions
        return self

    def cu_eval_prior_physical(self, theta, prior, batch):
        """
        Evaluate the prior using parameters already in physical space.
        """
        if not self.has_reparametrization:
            return self.cu_eval_prior(theta=theta, prior=prior, batch=batch)
        raise NotImplementedError(f"Distribution {self.pyre_name}: cu_eval_prior_physical not implemented")

    def cu_eval_jacobian(self, theta, jacobian, batch):
        """
        Evaluate the log of the Jacobian determinant for the transformation from sampling to physical space.
        Only relevant when has_reparametrization is True.

        For distributions without reparameterization, this does nothing (jacobian = 0).
        For distributions with reparameterization, this should compute log|dθ_physical/dθ_sampling|.
        """
        if not self.has_reparametrization:
            return jacobian
        raise NotImplementedError(f"Distribution {self.pyre_name}: cu_eval_jacobian not implemented")

    def cu_jacobian_gradient(self, theta, gradient, batch, index=None):
        """
        Compute the gradient of the log Jacobian determinant with respect to parameters.
        Only relevant when has_reparametrization is True.

        Parameters:
            theta: parameters in sampling space
            gradient: target array for the gradient
            batch: number of samples
            index: optional, specific parameter index to compute gradient for

        For distributions without reparameterization, this does nothing (gradient = 0).
        For distributions with reparameterization, this should compute
        ∂/∂θ_sampling log|dθ_physical/dθ_sampling|
        """
        if not self.has_reparametrization:
            return gradient
        raise NotImplementedError(f"Distribution {self.pyre_name}: cu_jacobian_gradient not implemented")

    def cu_to_physical(self, theta_sampling=None, theta_physical=None, batch=None, theta=None):
        """
        Transform parameters from sampling space to physical space.
        theta_physical = T(theta_sampling) where T is the transformation function.

        Parameters:
            theta_sampling: source parameters in sampling space
            theta_physical: target array for physical parameters
            batch: number of samples to transform
            theta: in-place transform when provided (sampling -> physical)

        For distributions without reparameterization, this is a no-op when called in-place;
        callers should copy the full parameter set before invoking per-pset transforms.
        When separate arrays are provided, this copies theta_sampling to theta_physical.
        For distributions with reparameterization, derived classes must implement
        the specific transformation.
        """
        if theta is not None:
            if theta_sampling is not None or theta_physical is not None:
                raise TypeError("cu_to_physical: use theta or theta_sampling/theta_physical, not both")
            theta_sampling = theta
            theta_physical = theta
        if theta_sampling is None or theta_physical is None or batch is None:
            raise TypeError("cu_to_physical requires theta (in-place) or theta_sampling/theta_physical and batch")
        if not self.has_reparametrization:
            if theta_physical is not theta_sampling:
                theta_physical.copy(theta_sampling)
            return self
        raise NotImplementedError(f"Distribution {self.pyre_name}: cu_to_physical not implemented")

    def cu_to_sampling(self, theta_physical=None, theta_sampling=None, batch=None, theta=None):
        """
        Transform parameters from physical space to sampling space.
        theta_sampling = T^(-1)(theta_physical) where T^(-1) is the inverse transformation.

        Parameters:
            theta_physical: source parameters in physical space
            theta_sampling: target array for sampling parameters
            batch: number of samples to transform
            theta: in-place transform when provided (physical -> sampling)

        For distributions without reparameterization, this is a no-op when called in-place;
        callers should copy the full parameter set before invoking per-pset transforms.
        When separate arrays are provided, this copies theta_physical to theta_sampling.
        For distributions with reparameterization, derived classes must implement
        the specific inverse transformation.
        """
        if theta is not None:
            if theta_physical is not None or theta_sampling is not None:
                raise TypeError("cu_to_sampling: use theta or theta_physical/theta_sampling, not both")
            theta_physical = theta
            theta_sampling = theta
        if theta_physical is None or theta_sampling is None or batch is None:
            raise TypeError("cu_to_sampling requires theta (in-place) or theta_physical/theta_sampling and batch")
        if not self.has_reparametrization:
            if theta_sampling is not theta_physical:
                theta_sampling.copy(theta_physical)
            return self
        raise NotImplementedError(f"Distribution {self.pyre_name}: cu_to_sampling not implemented")

    def cu_prior_gradient(self, theta, grad_prior, batch, index=None):
        """
        Compute the gradient of log prior with respect to parameters
        For transformed parameters, this should include the transformation contribution
        """
        # Default implementation: flat prior (gradient = 0)
        return self

    def cu_jacobian_gradient(self, theta, gradient, batch, index=None):
        """
        Compute the gradient of the log Jacobian determinant with respect to parameters.
        Only relevant when has_reparametrization is True.

        Parameters:
            theta: parameters in sampling space
            gradient: target array for the gradient
            batch: number of samples
            index: optional, specific parameter index to compute gradient for

        For distributions without reparameterization, this does nothing (gradient = 0).
        For distributions with reparameterization, this should compute
        ∂/∂θ_sampling log|dθ_physical/dθ_sampling|
        """
        # Default implementation: gradient = 0
        return self

    def update(self, **kwargs):
        """
        update distribution parameters if needed
        """
        # default, do nothing
        return self


    # private data
    device = None
    idx_range = None
    precision = None

# end of file
