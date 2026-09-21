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

import altar
# my base
from .BayesianState import BayesianState


class HMCState(BayesianState):
    """
    Encapsulation of the HMC state of the calculation, including gradients

    Extends {BayesianState} with the (samples x parameters) gradient matrices
    of the prior, data, and posterior log-likelihoods.
    """

    # gradient information
    grad_prior = None     # (samples x parameters) matrix
    grad_data = None      # (samples x parameters) matrix
    grad_posterior = None # (samples x parameters) matrix


    @classmethod
    def alloc(cls, samples, parameters):
        theta = altar.matrix(shape=(samples, parameters)).zero()
        prior, data, posterior = cls._alloc_likelihoods(samples)
        grad_prior = altar.matrix(shape=(samples, parameters)).zero()
        grad_data = altar.matrix(shape=(samples, parameters)).zero()
        grad_posterior = altar.matrix(shape=(samples, parameters)).zero()
        return cls(beta=0, theta=theta, likelihoods=(prior, data, posterior),
                   gradients=(grad_prior, grad_data, grad_posterior))

    def clone(self):
        beta = self.beta
        theta = self.theta.clone()
        likelihoods = self.prior.clone(), self.data.clone(), self.posterior.clone()
        gradients = self.grad_prior.clone(), self.grad_data.clone(), self.grad_posterior.clone()
        return type(self)(beta=beta, theta=theta, likelihoods=likelihoods, gradients=gradients)

    def compute_posterior(self):
        # the shared prior + beta*data computation
        super().compute_posterior()
        # plus the posterior gradient: grad_posterior = grad_prior + beta * grad_data
        self.grad_posterior.copy(self.grad_prior)
        altar.blas.daxpy(self.beta, self.grad_data, self.grad_posterior)
        return self

    def _on_start(self, annealer):
        # gradients depend on the likelihoods computed by {start} just before this hook runs
        annealer.model.gradients(annealer=annealer, step=self)
        return

    def __init__(self, beta, theta, likelihoods, gradients=None, **kwds):
        # chain up (skip BayesianState.__init__ so we control gradient defaulting below)
        super(BayesianState, self).__init__(**kwds)
        self.beta = beta
        self.theta = theta
        self.prior, self.data, self.posterior = likelihoods
        dof = self.parameters
        if gradients is not None:
            self.grad_prior, self.grad_data, self.grad_posterior = gradients
        else:
            self.grad_prior = altar.matrix(shape=(self.samples, dof)).zero()
            self.grad_data = altar.matrix(shape=(self.samples, dof)).zero()
            self.grad_posterior = altar.matrix(shape=(self.samples, dof)).zero()
        return

    def _extra_record(self, archiver):
        """
        Record the gradients alongside the base likelihood/parameter-set datasets
        """
        archiver.write("Gradients/prior",      self.grad_prior)
        archiver.write("Gradients/likelihood", self.grad_data)
        archiver.write("Gradients/posterior",  self.grad_posterior)

    def _extra_save_hdf5(self, f):
        """
        Persist the gradients into their own "Gradients" hdf5 group
        """
        gradientsgrp = f.create_group('Gradients')
        gradientsgrp.create_dataset('prior',      data=self.grad_prior.ndarray())
        gradientsgrp.create_dataset('likelihood', data=self.grad_data.ndarray())
        gradientsgrp.create_dataset('posterior',  data=self.grad_posterior.ndarray())

# end of file
