# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu, Codex

"""GPU state container for Hamiltonian Monte Carlo"""

import altar
import altar.cuda


class cudaHMCState:
    """Encapsulation of an HMC state on the GPU"""

    beta = 1.0
    eta = 0.01
    reparameterization = False

    theta = None
    phi = None
    Jacobian = None
    momentum = None

    prior = None
    data = None
    posterior = None
    U = None
    H = None

    prior_gradient = None
    data_gradient = None
    U_gradient = None

    report_seq = 0

    def __init__(self, *, beta, eta, theta, momentum, likelihoods,
                 gradients, potential, reparameterization=False,
                 phi=None, jacobian=None, **kwds):
        super().__init__(**kwds)
        self.beta = beta
        self.eta = eta
        self.theta = theta
        self.momentum = momentum
        self.prior, self.data, self.posterior = likelihoods
        self.prior_gradient, self.data_gradient, self.U_gradient = gradients
        self.U, self.H = potential
        self.reparameterization = reparameterization
        if reparameterization:
            self.phi = phi if phi is not None else theta.clone()
            if jacobian is not None:
                self.Jacobian = jacobian
            else:
                self.Jacobian = altar.cuda.matrix(shape=(self.samples, self.parameters),
                                                  dtype=theta.dtype).fill(1.0)
        else:
            self.phi = theta
            self.Jacobian = None

    @property
    def samples(self):
        return self.theta.shape[0]

    @property
    def parameters(self):
        return self.theta.shape[1]

    @classmethod
    def start(cls, controller):
        model = controller.model
        samples = model.job.chains
        precision = model.job.gpuprecision
        reparameterization = getattr(model, "reparameterization", False)
        return cls.alloc(samples=samples, parameters=model.parameters,
                         dtype=precision, reparameterization=reparameterization)

    @classmethod
    def alloc(cls, *, samples, parameters, dtype, reparameterization=False):
        theta = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        momentum = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        if reparameterization:
            phi = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
            jacobian = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).fill(1.0)
        else:
            phi = theta
            jacobian = None
        prior = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        data = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        posterior = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        U = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        H = altar.cuda.vector(shape=samples, dtype=dtype).zero()
        grad_prior = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        grad_data = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        grad_U = altar.cuda.matrix(shape=(samples, parameters), dtype=dtype).zero()
        return cls(beta=1.0, eta=0.01, theta=theta, momentum=momentum,
                   likelihoods=(prior, data, posterior), gradients=(grad_prior, grad_data, grad_U),
                   potential=(U, H), reparameterization=reparameterization,
                   phi=phi, jacobian=jacobian)

    def clone(self):
        theta = self.theta.clone()
        momentum = self.momentum.clone()
        likelihoods = (self.prior.clone(), self.data.clone(), self.posterior.clone())
        gradients = (self.prior_gradient.clone(), self.data_gradient.clone(), self.U_gradient.clone())
        potential = (self.U.clone(), self.H.clone())
        phi = self.phi.clone() if self.reparameterization else theta
        jacobian = self.Jacobian.clone() if self.Jacobian is not None else None
        return type(self)(beta=self.beta, eta=self.eta, theta=theta, momentum=momentum,
                          likelihoods=likelihoods, gradients=gradients, potential=potential,
                          reparameterization=self.reparameterization, phi=phi, jacobian=jacobian)

    def compute_posterior(self):
        self.posterior.copy(self.prior)
        altar.cuda.cublas.axpy(alpha=self.beta, x=self.data, y=self.posterior, batch=self.samples)
        return self

    def copy_from_cpu(self, state):
        self.beta = state.beta
        self.theta.copy_from_host(source=state.theta)
        if self.reparameterization and hasattr(state, "theta_sampling"):
            self.phi.copy_from_host(source=state.theta_sampling)
            if getattr(state, "jacobian", None) is not None and self.Jacobian is not None:
                self.Jacobian.copy_from_host(source=state.jacobian)
        self.prior.copy_from_host(source=state.prior)
        self.data.copy_from_host(source=state.data)
        self.posterior.copy_from_host(source=state.posterior)
        return self

    def copy_to_cpu(self, state):
        state.beta = self.beta
        self.theta.copy_to_host(target=state.theta)
        if self.reparameterization and getattr(state, "theta_sampling", None) is not None:
            self.phi.copy_to_host(target=state.theta_sampling)
            if self.Jacobian is not None and getattr(state, "jacobian", None) is not None:
                self.Jacobian.copy_to_host(target=state.jacobian)
        self.prior.copy_to_host(target=state.prior)
        self.data.copy_to_host(target=state.data)
        self.posterior.copy_to_host(target=state.posterior)
        return self

    def report(self, controller):
        self.print(channel=controller.info)
        controller.model.likelihoods(annealer=controller, step=self)
        self.save_hdf5(path="hmc_results", iteration=getattr(self, "report_seq", 0))
        self.report_seq = getattr(self, "report_seq", 0) + 1
        return self

    def print(self, channel, indent=" " * 2):
        θ = self.theta
        channel.line("state")
        channel.line(f"{indent}beta: {self.beta}")
        channel.line(f"{indent}η: {self.eta}")
        channel.line(f"{indent}θ: ({θ.rows} samples) x ({θ.cols} parameters)")
        mean, sd = θ.mean_sd()
        channel.line(f"{indent}parameters (mean, sd):")
        parameters = self.parameters
        if parameters <= 25:
            for i in range(parameters):
                channel.line(f"{indent} ({mean[i]}, {sd[i]})")
        else:
            for i in range(20):
                channel.line(f"{indent} ({mean[i]}, {sd[i]})")
            channel.line(f"{indent} ... ...")
            for i in range(parameters - 5, parameters):
                channel.line(f"{indent} ({mean[i]}, {sd[i]})")
        channel.log()
        return channel

    def save_hdf5(self, path=None, iteration=None, psets=None):
        import os
        import h5py
        import numpy
        str_iteration = 'final' if iteration is None else str(iteration).zfill(3)
        if path is not None:
            str_path = path.path if isinstance(path, altar.primitives.path) else path
            if not os.path.exists(str_path):
                os.makedirs(str_path)
        else:
            str_path = '.'
        filename = os.path.join(str_path, f"state_{str_iteration}.h5")
        f = h5py.File(filename, 'w')
        annealergrp = f.create_group('Controller')
        annealergrp.create_dataset('beta', data=numpy.asarray(self.beta))
        psetsgrp = f.create_group('ParameterSets')
        theta_host = self.theta.copy_to_host(type="numpy")
        if psets is None or len(psets) == 0:
            psetsgrp.create_dataset('theta', data=theta_host)
        else:
            for name, pset in psets.items():
                psetsgrp.create_dataset(name, data=theta_host[:, pset.offset:pset.offset+pset.count])
        bayesiangrp = f.create_group('Bayesian')
        bayesiangrp.create_dataset('prior', data=self.prior.copy_to_host(type="numpy"))
        bayesiangrp.create_dataset('likelihood', data=self.data.copy_to_host(type="numpy"))
        bayesiangrp.create_dataset('posterior', data=self.posterior.copy_to_host(type="numpy"))
        f.close()
        return self

# end of file
