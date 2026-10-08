# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import typing
import numpy
# my base
from .BayesianState import BayesianState, Likelihoods

# the prior, data and posterior gradients of a step
Gradients = tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]


class HMCState(BayesianState):
    """
    The scratch state a {Hmc} sampler uses while running one leapfrog trajectory: my sample
    matrix, likelihoods, momentum, and the gradients of the log prior/data/posterior
    likelihoods with respect to theta. Built and discarded per trajectory by the sampler; the
    persistent per-β state remains a plain {CoolingStep}, exactly as it does for {Metropolis}.
    """

    # momentum, (samples x parameters)
    momentum: numpy.ndarray

    # gradient information, (samples x parameters)
    grad_prior: numpy.ndarray
    grad_data: numpy.ndarray
    grad_posterior: numpy.ndarray

    # reparameterization, set by the sampler when the model reparameterizes
    phi: numpy.ndarray | None = None          # (samples x parameters), theta in sampling space
    Jacobian: numpy.ndarray | None = None     # (samples x parameters), d(theta)/d(phi)
    log_jacobian: numpy.ndarray | None = None # (samples,), log|d(theta)/d(phi)|


    @classmethod
    def alloc(cls, samples: int, parameters: int, dtype: str = "float64") -> typing.Self:
        theta = numpy.zeros((samples, parameters), dtype=dtype)
        prior, data, posterior = cls._alloc_likelihoods(samples)
        momentum = numpy.zeros_like(theta)
        gradients = tuple(numpy.zeros_like(theta) for _ in range(3))
        return cls(beta=0, theta=theta, likelihoods=(prior, data, posterior),
                   momentum=momentum, gradients=gradients)

    def clone(self) -> typing.Self:
        likelihoods = self.prior.copy(), self.data.copy(), self.posterior.copy()
        gradients = self.grad_prior.copy(), self.grad_data.copy(), self.grad_posterior.copy()
        clone = type(self)(beta=self.beta, theta=self.theta.copy(), likelihoods=likelihoods,
                           momentum=self.momentum.copy(), gradients=gradients)
        # and the reparameterization state, when there is one
        for name in ("phi", "Jacobian", "log_jacobian"):
            value = getattr(self, name)
            if value is not None:
                setattr(clone, name, value.copy())
        return clone

    def compute_posterior(self) -> typing.Self:
        # the shared prior + beta*data computation
        super().compute_posterior()
        # plus the posterior gradient
        self.grad_posterior[...] = self.grad_prior + self.beta * self.grad_data
        return self

    def __init__(self, beta: float, theta: numpy.ndarray, likelihoods: Likelihoods,
                 momentum: numpy.ndarray | None = None, gradients: Gradients | None = None,
                 **kwds) -> None:
        # chain up (skip BayesianState.__init__ so we control the extra fields below)
        super(BayesianState, self).__init__(**kwds)
        self.beta = beta
        self.theta = theta
        self.prior, self.data, self.posterior = likelihoods
        # the momentum and the gradients, in the precision of the samples
        self.momentum = momentum if momentum is not None else numpy.zeros_like(theta)
        if gradients is None:
            gradients = tuple(numpy.zeros_like(theta) for _ in range(3))
        self.grad_prior, self.grad_data, self.grad_posterior = gradients
        return

# end of file
