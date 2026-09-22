# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# lijun zhu <ljzhu@caltech.edu>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
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


    @classmethod
    def alloc(cls, samples, parameters):
        theta = altar.matrix(shape=(samples, parameters)).zero()
        prior, data, posterior = cls._alloc_likelihoods(samples)
        grad_prior = altar.matrix(shape=(samples, parameters)).zero()
        grad_data = altar.matrix(shape=(samples, parameters)).zero()
        return cls(beta=1, theta=theta, likelihoods=(prior, data, posterior),
                   gradients=(grad_prior, grad_data))

    def clone(self):
        beta = self.beta
        theta = self.theta.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()
        gradients = self.grad_prior.clone(), self.grad_data.clone()
        clone = type(self)(beta=beta, theta=theta, likelihoods=likelihoods, gradients=gradients)
        clone.epsilon_t = self.epsilon_t
        return clone

    def _on_start(self, annealer):
        # gradients depend on the likelihoods computed by {start} just before this hook runs
        annealer.model.gradient(controller=annealer, step=self, batch=self.samples)
        return

    def updateTheta(self, uninormal):
        """
        Update theta(t+1) = theta(t) + epsilon_t/2 (grad_prior + grad_data) + eta_t,
        where eta_t ~ N(0, epsilon_t). {uninormal} is a unit-normal pdf (altar.pdf.ugaussian)
        used to draw the noise.
        """
        # draw standard normal noise, then scale to N(0, epsilon_t): sigma = sqrt(epsilon_t)
        eta_t = altar.matrix(shape=self.theta.shape).random(pdf=uninormal)
        eta_t.scale(math.sqrt(self.epsilon_t))

        # combine the gradients: 0.5 * epsilon_t * (grad_prior + grad_data)
        drift = self.grad_prior.clone()
        drift += self.grad_data
        drift.scale(0.5 * self.epsilon_t)

        # theta += drift + eta_t
        self.theta += drift
        self.theta += eta_t

        # all done
        return self

    def __init__(self, beta, theta, likelihoods, gradients=None, **kwds):
        # chain up (skip BayesianState.__init__ so we control gradient defaulting below)
        super(BayesianState, self).__init__(**kwds)
        self.beta = beta
        self.theta = theta
        self.prior, self.data, self.posterior = likelihoods
        dof = self.parameters
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
