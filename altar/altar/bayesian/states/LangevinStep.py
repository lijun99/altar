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
import math
import typing
import numpy
# my base
from .BayesianState import BayesianState, Likelihoods

if typing.TYPE_CHECKING:
    import h5py
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.models.Bayesian import Bayesian
    from altar.simulations.Archiver import Archiver


class LangevinStep(BayesianState):
    """
    Encapsulation of the SGLD (Stochastic Gradient Langevin Dynamics) state of the
    calculation, including gradients and the per-step sampling rate.

    Extends {BayesianState} with the (samples x parameters) gradient matrices of the prior
    and data log-likelihoods, the current sampling rate {epsilon_t}, and the update

        theta(t+1) = theta(t) + epsilon_t/2 (grad_prior + grad_data) + N(0, epsilon_t)
    """

    # gradient information, (samples x parameters)
    grad_prior: numpy.ndarray
    grad_data: numpy.ndarray

    # the current sampling rate, set by the controller before each walk
    epsilon_t: float | None = None

    # reparameterization: the chains move {theta_sampling}, {theta} follows in physical space
    has_reparametrization: bool = False
    theta_sampling: numpy.ndarray          # (samples x parameters); {theta} itself unless reparameterized
    jacobian: numpy.ndarray | None = None  # (samples,), log|d(theta)/d(theta_sampling)|
    Jacobian: numpy.ndarray | None = None  # (samples x parameters), d(theta)/d(theta_sampling)


    @classmethod
    def allocate(cls, annealer: Annealer) -> typing.Self:
        model = annealer.model
        return cls.alloc(samples=model.job.chains, parameters=model.parameters,
                         has_reparametrization=getattr(model, "has_reparametrization", False))

    @classmethod
    def alloc(cls, samples: int, parameters: int,
              has_reparametrization: bool = False) -> typing.Self:
        theta = numpy.zeros((samples, parameters))
        prior, data, posterior = cls._alloc_likelihoods(samples)
        gradients = numpy.zeros((samples, parameters)), numpy.zeros((samples, parameters))
        return cls(beta=1, theta=theta, likelihoods=(prior, data, posterior),
                   gradients=gradients, has_reparametrization=has_reparametrization)

    def clone(self) -> typing.Self:
        likelihoods = self.prior.copy(), self.data.copy(), self.posterior.copy()
        gradients = self.grad_prior.copy(), self.grad_data.copy()
        clone = type(self)(beta=self.beta, theta=self.theta.copy(), likelihoods=likelihoods,
                           gradients=gradients, has_reparametrization=self.has_reparametrization)
        if self.has_reparametrization:
            clone.theta_sampling[...] = self.theta_sampling
            clone.jacobian[...] = self.jacobian
        clone.epsilon_t = self.epsilon_t
        return clone

    def _on_start(self, annealer: Annealer) -> None:
        # the log-jacobian of the initial samples, then the gradients, which depend on the
        # likelihoods computed by {start} just before this hook runs
        if self.has_reparametrization:
            self.jacobian[...] = 0
            annealer.model.eval_prior_with_physical(step=self, likelihood=self.jacobian)
        self.compute_gradients(controller=annealer)
        return

    def compute_gradients(self, controller: Annealer) -> typing.Self:
        """
        The gradients of the log prior and data likelihood w.r.t. {theta_sampling}: the prior
        gradient of a reparameterized prior already is, the data one needs the chain rule
        """
        model = controller.model
        model.gradient(controller=controller, step=self, batch=self.samples)
        if self.has_reparametrization:
            self.Jacobian[...] = 1.0
            model.eval_jacobian(step=self, batch=self.samples)
            self.grad_data *= self.Jacobian
        return self

    def refresh_physical(self, model: Bayesian) -> typing.Self:
        """
        Rebuild {theta} and {jacobian} from {theta_sampling}, after an update
        """
        if self.has_reparametrization:
            self.theta[...] = self.theta_sampling
            model.to_physical(theta=self.theta)
            self.jacobian[...] = 0
            model.eval_prior_with_physical(step=self, likelihood=self.jacobian)
        return self

    def updateTheta(self, rng: numpy.random.Generator) -> typing.Self:
        """
        Update theta(t+1) = theta(t) + epsilon_t/2 (grad_prior + grad_data) + eta_t,
        where eta_t ~ N(0, epsilon_t), in sampling space when reparameterized, with the noise
        drawn from {rng}
        """
        eta_t = math.sqrt(self.epsilon_t) * rng.standard_normal(size=self.theta_sampling.shape)
        drift = 0.5 * self.epsilon_t * (self.grad_prior + self.grad_data)
        self.theta_sampling += drift + eta_t
        return self

    def __init__(self, beta: float, theta: numpy.ndarray, likelihoods: Likelihoods,
                 gradients: tuple[numpy.ndarray, numpy.ndarray] | None = None,
                 has_reparametrization: bool = False, **kwds) -> None:
        # chain up (skip BayesianState.__init__ so we control gradient defaulting below)
        super(BayesianState, self).__init__(**kwds)
        self.beta = beta
        self.theta = theta
        self.prior, self.data, self.posterior = likelihoods
        self.has_reparametrization = has_reparametrization
        if has_reparametrization:
            self.theta_sampling = theta.copy()
            self.jacobian = numpy.zeros(theta.shape[0])
            self.Jacobian = numpy.ones(theta.shape)
        else:
            self.theta_sampling = theta
        if gradients is None:
            gradients = numpy.zeros(theta.shape), numpy.zeros(theta.shape)
        self.grad_prior, self.grad_data = gradients
        return

    def _extra_record(self, archiver: Archiver) -> None:
        """
        Record the gradients and the current sampling rate alongside the base data
        """
        archiver.write("Controller/epsilon_t", self.epsilon_t)
        archiver.write("Gradients/prior",      self.grad_prior)
        archiver.write("Gradients/likelihood", self.grad_data)

    def _extra_save_hdf5(self, f: h5py.File) -> None:
        """
        Persist the gradients and epsilon_t into their own hdf5 groups
        """
        controllergrp = f.create_group('Controller')
        controllergrp.create_dataset('epsilon_t', data=numpy.asarray(self.epsilon_t))
        gradientsgrp = f.create_group('Gradients')
        gradientsgrp.create_dataset('prior',      data=self.grad_prior)
        gradientsgrp.create_dataset('likelihood', data=self.grad_data)

# end of file
