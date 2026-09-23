# -*- python -*-
# -*- coding: utf-8 -*-
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# lijun zhu <ljzhu@caltech.edu>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

import altar
# my base
from .BayesianState import BayesianState


class HMCState(BayesianState):
    """
    The scratch state a {Hmc} sampler uses while running one leapfrog trajectory: my sample
    matrix, likelihoods, momentum, and the gradients of the log prior/data/posterior
    likelihoods with respect to theta. Built and discarded per trajectory by the sampler; the
    persistent per-β state remains a plain {CoolingStep}, exactly as it does for {Metropolis}.
    """

    # momentum
    momentum = None        # (samples x parameters) matrix

    # gradient information
    grad_prior = None      # (samples x parameters) matrix
    grad_data = None       # (samples x parameters) matrix
    grad_posterior = None  # (samples x parameters) matrix


    @classmethod
    def alloc(cls, samples, parameters):
        theta = altar.matrix(shape=(samples, parameters)).zero()
        prior, data, posterior = cls._alloc_likelihoods(samples)
        momentum = altar.matrix(shape=(samples, parameters)).zero()
        grad_prior = altar.matrix(shape=(samples, parameters)).zero()
        grad_data = altar.matrix(shape=(samples, parameters)).zero()
        grad_posterior = altar.matrix(shape=(samples, parameters)).zero()
        return cls(beta=0, theta=theta, likelihoods=(prior, data, posterior),
                   momentum=momentum, gradients=(grad_prior, grad_data, grad_posterior))

    def clone(self):
        beta = self.beta
        theta = self.theta.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()
        momentum = self.momentum.clone()
        gradients = self.grad_prior.clone(), self.grad_data.clone(), self.grad_posterior.clone()
        return type(self)(beta=beta, theta=theta, likelihoods=likelihoods,
                          momentum=momentum, gradients=gradients)

    def compute_posterior(self):
        # the shared prior + beta*data computation
        super().compute_posterior()
        # plus the posterior gradient: grad_posterior = grad_prior + beta * grad_data;
        # {daxpy} is vector-only, so scale a clone rather than accumulate in place
        self.grad_posterior.copy(self.grad_prior)
        scaled = self.grad_data.clone()
        scaled.scale(self.beta)
        self.grad_posterior += scaled
        return self

    def __init__(self, beta, theta, likelihoods, momentum=None, gradients=None, **kwds):
        # chain up (skip BayesianState.__init__ so we control the extra fields below)
        super(BayesianState, self).__init__(**kwds)
        self.beta = beta
        self.theta = theta
        self.prior, self.data, self.posterior = likelihoods
        dof = self.parameters
        self.momentum = momentum if momentum is not None else altar.matrix(shape=(self.samples, dof)).zero()
        if gradients is not None:
            self.grad_prior, self.grad_data, self.grad_posterior = gradients
        else:
            self.grad_prior = altar.matrix(shape=(self.samples, dof)).zero()
            self.grad_data = altar.matrix(shape=(self.samples, dof)).zero()
            self.grad_posterior = altar.matrix(shape=(self.samples, dof)).zero()
        return

# end of file
