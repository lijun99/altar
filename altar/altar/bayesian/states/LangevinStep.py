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
import math
# the package
import altar
# my base
from .BayesianState import BayesianState


class LangevinStep(BayesianState):
    """
    Encapsulation of the SGLD (Stochastic Gradient Langevin Dynamics) state of the
    calculation, including gradients and the per-step sampling rate.

    Extends {BayesianState} with the (samples x parameters) gradient matrices of the prior
    and data log-likelihoods, the current sampling rate {epsilon_t}, and the update

        theta(t+1) = theta(t) + epsilon_t/2 (grad_prior + grad_data) + N(0, epsilon_t)
    """

    # gradient information
    grad_prior = None  # (samples x parameters) matrix
    grad_data = None   # (samples x parameters) matrix

    # the current sampling rate, set by the controller before each walk
    epsilon_t = None

    # reparameterization: the chains move {theta_sampling}, {theta} follows in physical space
    has_reparametrization = False
    theta_sampling = None  # (samples x parameters) matrix; {theta} itself unless reparameterized
    jacobian = None        # (samples) vector, log|d(theta)/d(theta_sampling)|
    Jacobian = None        # (samples x parameters) matrix, d(theta)/d(theta_sampling)


    @classmethod
    def allocate(cls, annealer):
        model = annealer.model
        return cls.alloc(samples=model.job.chains, parameters=model.parameters,
                         has_reparametrization=getattr(model, "has_reparametrization", False))

    @classmethod
    def alloc(cls, samples, parameters, has_reparametrization=False):
        theta = altar.matrix(shape=(samples, parameters)).zero()
        prior, data, posterior = cls._alloc_likelihoods(samples)
        grad_prior = altar.matrix(shape=(samples, parameters)).zero()
        grad_data = altar.matrix(shape=(samples, parameters)).zero()
        return cls(beta=1, theta=theta, likelihoods=(prior, data, posterior),
                   gradients=(grad_prior, grad_data), has_reparametrization=has_reparametrization)

    def clone(self):
        beta = self.beta
        theta = self.theta.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()
        gradients = self.grad_prior.clone(), self.grad_data.clone()
        clone = type(self)(beta=beta, theta=theta, likelihoods=likelihoods, gradients=gradients,
                           has_reparametrization=self.has_reparametrization)
        if self.has_reparametrization:
            clone.theta_sampling.copy(self.theta_sampling)
            clone.jacobian.copy(self.jacobian)
        clone.epsilon_t = self.epsilon_t
        return clone

    def _on_start(self, annealer):
        # the log-jacobian of the initial samples, then the gradients, which depend on the
        # likelihoods computed by {start} just before this hook runs
        if self.has_reparametrization:
            self.jacobian.zero()
            annealer.model.eval_prior_with_physical(step=self, likelihood=self.jacobian)
        self.compute_gradients(controller=annealer)
        return

    def compute_gradients(self, controller):
        """
        The gradients of the log prior and data likelihood w.r.t. {theta_sampling}: the prior
        gradient of a reparameterized prior already is, the data one needs the chain rule
        """
        model = controller.model
        model.gradient(controller=controller, step=self, batch=self.samples)
        if self.has_reparametrization:
            self.Jacobian.fill(1.0)
            model.eval_jacobian(step=self, batch=self.samples)
            self.grad_data.ndarray()[:] *= self.Jacobian.ndarray()
        return self

    def refresh_physical(self, model):
        """
        Rebuild {theta} and {jacobian} from {theta_sampling}, after an update
        """
        if self.has_reparametrization:
            self.theta.copy(self.theta_sampling)
            model.to_physical(theta=self.theta)
            self.jacobian.zero()
            model.eval_prior_with_physical(step=self, likelihood=self.jacobian)
        return self

    def updateTheta(self, uninormal):
        """
        Update theta(t+1) = theta(t) + epsilon_t/2 (grad_prior + grad_data) + eta_t,
        where eta_t ~ N(0, epsilon_t), in sampling space when reparameterized. {uninormal} is
        a unit-normal pdf (altar.pdf.ugaussian) used to draw the noise.
        """
        # draw standard normal noise, then scale to N(0, epsilon_t): sigma = sqrt(epsilon_t)
        eta_t = altar.matrix(shape=self.theta.shape).random(pdf=uninormal)
        eta_t.scale(math.sqrt(self.epsilon_t))

        # combine the gradients: 0.5 * epsilon_t * (grad_prior + grad_data)
        drift = self.grad_prior.clone()
        drift += self.grad_data
        drift.scale(0.5 * self.epsilon_t)

        # theta += drift + eta_t
        self.theta_sampling += drift
        self.theta_sampling += eta_t

        # all done
        return self

    def __init__(self, beta, theta, likelihoods, gradients=None, has_reparametrization=False, **kwds):
        # chain up (skip BayesianState.__init__ so we control gradient defaulting below)
        super(BayesianState, self).__init__(**kwds)
        self.beta = beta
        self.theta = theta
        self.prior, self.data, self.posterior = likelihoods
        dof = self.parameters
        self.has_reparametrization = has_reparametrization
        if has_reparametrization:
            self.theta_sampling = theta.clone()
            self.jacobian = altar.vector(shape=self.samples).zero()
            self.Jacobian = altar.matrix(shape=(self.samples, dof))
        else:
            self.theta_sampling = theta
        if gradients is not None:
            self.grad_prior, self.grad_data = gradients
        else:
            self.grad_prior = altar.matrix(shape=(self.samples, dof)).zero()
            self.grad_data = altar.matrix(shape=(self.samples, dof)).zero()
        return

    def _extra_record(self, archiver):
        """
        Record the gradients and the current sampling rate alongside the base data
        """
        archiver.write("Controller/epsilon_t", self.epsilon_t)
        archiver.write("Gradients/prior",      self.grad_prior)
        archiver.write("Gradients/likelihood", self.grad_data)

    def _extra_save_hdf5(self, f):
        """
        Persist the gradients and epsilon_t into their own hdf5 groups
        """
        import numpy
        controllergrp = f.create_group('Controller')
        controllergrp.create_dataset('epsilon_t', data=numpy.asarray(self.epsilon_t))
        gradientsgrp = f.create_group('Gradients')
        gradientsgrp.create_dataset('prior',      data=self.grad_prior.ndarray())
        gradientsgrp.create_dataset('likelihood', data=self.grad_data.ndarray())

# end of file
