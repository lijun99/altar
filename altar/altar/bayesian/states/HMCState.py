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

class HMCState:
    """
    Encapsulation of the HMC state of the calculation, including gradients
    """

    # public data
    theta = None     # (samples x parameters) matrix
    prior = None     # (samples) vector with logs of the prior
    data = None      # (samples) vector with logs of the data likelihoods
    posterior = None # (samples) vector with logs of the posterior

    # gradient information
    grad_prior = None     # (samples x parameters) matrix
    grad_data = None      # (samples x parameters) matrix
    grad_posterior = None # (samples x parameters) matrix

    @property
    def samples(self):
        return self.theta.rows

    @property
    def parameters(self):
        return self.theta.columns

    @classmethod
    def start(cls, annealer):
        model = annealer.model
        step = cls.alloc(samples=model.job.chains, parameters=model.parameters)
        model.initialize_sample(step=step)
        model.likelihoods(annealer=annealer, step=step)
        model.gradients(annealer=annealer, step=step)  # <-- assumes model provides gradients
        step.prior.print()
        return step

    @classmethod
    def allocate(cls, annealer):
        model = annealer.model
        step = cls.alloc(samples=model.job.chains, parameters=model.parameters)
        return step

    @classmethod
    def alloc(cls, samples, parameters):
        theta = altar.matrix(shape=(samples, parameters)).zero()
        prior = altar.vector(shape=samples).zero()
        data = altar.vector(shape=samples).zero()
        posterior = altar.vector(shape=samples).zero()
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
        self.posterior.copy(self.prior)
        altar.blas.daxpy(self.beta, self.data, self.posterior)
        # Compute posterior gradient: grad_posterior = grad_prior + beta * grad_data
        self.grad_posterior.copy(self.grad_prior)
        altar.blas.daxpy(self.beta, self.grad_data, self.grad_posterior)
        return self

    def statistics(self):
        θ = self.theta
        self.mean, self.sd = θ.mean_sd(axis=0)
        return self

    weights = None  # a (samples) vector of importance weights w_i ∝ exp(Δβ · data_i); set by scheduler

    def __init__(self, beta, theta, likelihoods, gradients=None, **kwds):
        super().__init__(**kwds)
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

    def record(self, archiver):
        """
        Record me using the provided {archiver}.
        """
        psets = getattr(archiver, "psets", None) or {}

        archiver.write("Annealer/beta", self.beta)

        # importance weights (set by scheduler; may be None at beta=0)
        if self.weights is not None:
            archiver.write("Annealer/weights", self.weights)

        # parameter sets
        if len(psets) == 0:
            archiver.write("ParameterSets/theta", self.theta)
        else:
            theta = self.theta.ndarray()
            for name, pset in psets.items():
                archiver.write(f"ParameterSets/{name}",
                               theta[:, pset.offset:pset.offset+pset.count])

        # bayesian quantities
        archiver.write("Bayesian/prior",      self.prior)
        archiver.write("Bayesian/likelihood", self.data)
        archiver.write("Bayesian/posterior",  self.posterior)

        # gradients
        archiver.write("Gradients/prior",      self.grad_prior)
        archiver.write("Gradients/likelihood", self.grad_data)
        archiver.write("Gradients/posterior",  self.grad_posterior)

        # all done
        return self

# end of file
