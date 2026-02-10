# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

"""Hamiltonian Monte Carlo sampler that drives CUDA leapfrog kernels."""

import numpy

import altar
import altar.cuda
from altar.cuda import cublas, libcudaaltar

from altar.bayesian.samplers.Sampler import Sampler
from altar.bayesian.stepsizers.StepSizer import StepSizer

from .HMCState import cudaHMCState


class HMCSampler(altar.component, family="altar.samplers.hmc", implements=Sampler):
    """Hamiltonian Monte Carlo sampler using a leapfrog integrator."""

    step_adjuster = StepSizer()
    step_adjuster.doc = "component that adapts the step size based on acceptance ratios"

    def __init__(self, model, **kwds):
        super().__init__(**kwds)
        self.model = model
        self.proposal_state = None
        self._current_eta = None

    def initialize(self, controller):
        samples = controller.job.chains
        parameters = self.model.parameters
        dtype = controller.job.gpuprecision
        reparameterization = getattr(self.model, 'reparameterization', False)
        self.proposal_state = cudaHMCState.alloc(
            samples=samples, parameters=parameters, dtype=dtype,
            reparameterization=reparameterization
        )
        self._current_eta = self.step_adjuster.initialize(self.proposal_state.eta)
        self._set_step_size(self._current_eta)
        return self

    def sample(self, controller, step):
        """
        Advance all chains using one or more leapfrog trajectories.

        Returns the total number of accepted proposals across all trajectories.
        """
        state = self.proposal_state
        state.beta = step.beta

        eta = getattr(step, 'hmc_step_size', None)
        if eta is None:
            eta = state.eta
        self._set_step_size(self._clamp_step_size(eta))

        leapfrog_repeats = int(getattr(step, 'hmc_leapfrog_steps', 1))
        leapfrog_substeps = int(getattr(step, 'hmc_leapfrog_substeps', 1))

        accepted_total = 0
        for _ in range(leapfrog_repeats):
            self._copy_state_from_step(step)
            accepted = self._trajectory(controller, leapfrog_substeps)
            self._copy_accepted_to_step(step)
            self._update_step_size(accepted=accepted, attempts=state.samples)
            accepted_total += accepted

        return accepted_total

    def _set_step_size(self, eta):
        eta = float(eta)
        self._current_eta = eta
        self.proposal_state.eta = eta
        if self.step_adjuster is not None and hasattr(self.step_adjuster, 'step_size'):
            self.step_adjuster.step_size = eta
        return eta

    def _clamp_step_size(self, eta):
        eta = float(eta)
        adjuster = self.step_adjuster
        min_eta = getattr(adjuster, 'min_step_size', 1e-12)
        max_eta = getattr(adjuster, 'max_step_size', float('inf'))
        return max(min_eta, min(max_eta, eta))

    def _trajectory(self, controller, leapfrog_substeps):
        state = self.proposal_state
        old_state = state.clone()

        def compute_potential_and_gradients():
            try:
                self.model.evaluateLikelihoods(state)
            except AttributeError:
                self.model.likelihoods(controller, state)
            try:
                self.model.evaluateGradients(state)
            except AttributeError:
                self.model.gradients(controller, state)
            jacobian = state.Jacobian.data if state.Jacobian is not None else None
            libcudaaltar.cudaHMC_compute_potential_and_gradient(
                state.prior.data,
                state.data.data,
                state.prior_gradient.data,
                state.data_gradient.data,
                state.U.data,
                state.U_gradient.data,
                jacobian,
                state.samples,
                state.parameters,
                state.beta,
                int(state.reparameterization)
            )

        # compute initial energy
        compute_potential_and_gradients()
        kinetic_old = altar.cuda.vector(shape=state.samples, dtype=state.theta.dtype).zero()
        libcudaaltar.cudaHMC_kinetic_energy(
            state.momentum.data,
            kinetic_old.data,
            state.samples,
            state.parameters
        )
        potential_old = state.U.clone()
        h_old = potential_old.clone()
        cublas.axpy(alpha=1.0, x=kinetic_old, y=h_old, batch=state.samples)

        # leapfrog updates
        eta = state.eta
        half_step = 0.5 * eta
        self._update_momentum(half_step)
        for k in range(leapfrog_substeps):
            self._update_position()
            compute_potential_and_gradients()
            if k < leapfrog_substeps - 1:
                self._update_momentum(eta)
        self._update_momentum(half_step)

        # recompute energies at the proposal
        compute_potential_and_gradients()
        kinetic_new = altar.cuda.vector(shape=state.samples, dtype=state.theta.dtype).zero()
        libcudaaltar.cudaHMC_kinetic_energy(
            state.momentum.data,
            kinetic_new.data,
            state.samples,
            state.parameters
        )
        potential_new = state.U.clone()
        h_new = potential_new.clone()
        cublas.axpy(alpha=1.0, x=kinetic_new, y=h_new, batch=state.samples)

        delta_h = h_new.clone()
        cublas.axpy(alpha=-1.0, x=h_old, y=delta_h, batch=state.samples)

        # Metropolis-Hastings decision
        mask_dev = altar.cuda.vector(shape=state.samples, dtype='int32').zero()
        libcudaaltar.cudaHMC_metropolis_hastings(delta_h.data, mask_dev.data, state.samples)
        accepted = int(numpy.count_nonzero(mask_dev.copy_to_host(type="numpy")))

        # restore rejected proposals
        libcudaaltar.cudaHMC_restore_rejected(
            state.theta.data,
            old_state.theta.data,
            state.momentum.data,
            old_state.momentum.data,
            mask_dev.data,
            state.samples,
            state.parameters
        )
        if state.reparameterization:
            libcudaaltar.cudaHMC_restore_rejected_matrix(
                state.phi.data,
                old_state.phi.data,
                mask_dev.data,
                state.samples,
                state.parameters
            )
            if state.Jacobian is not None and old_state.Jacobian is not None:
                libcudaaltar.cudaHMC_restore_rejected_matrix(
                    state.Jacobian.data,
                    old_state.Jacobian.data,
                    mask_dev.data,
                    state.samples,
                    state.parameters
                )

        # update energies/gradients for the final (accepted/rejected) state
        compute_potential_and_gradients()
        libcudaaltar.cudaHMC_kinetic_energy(
            state.momentum.data,
            state.H.data,
            state.samples,
            state.parameters
        )
        cublas.axpy(alpha=1.0, x=state.U, y=state.H, batch=state.samples)

        return accepted

    def _update_position(self):
        if self.proposal_state.reparameterization:
            libcudaaltar.cudaHMC_update_position(
                self.proposal_state.phi.data,
                self.proposal_state.momentum.data,
                self.proposal_state.samples,
                self.proposal_state.parameters,
                self.proposal_state.eta
            )
            self._transform_phi_to_theta()
            return
        libcudaaltar.cudaHMC_update_position(
            self.proposal_state.theta.data,
            self.proposal_state.momentum.data,
            self.proposal_state.samples,
            self.proposal_state.parameters,
            self.proposal_state.eta
        )

    def _transform_phi_to_theta(self):
        try:
            self.model.transformToPhysical(self.proposal_state)
        except AttributeError:
            raise NotImplementedError("model.transformToPhysical is required for reparameterized HMC")

    def _update_momentum(self, step_size):
        libcudaaltar.cudaHMC_update_momentum(
            self.proposal_state.momentum.data,
            self.proposal_state.U_gradient.data,
            self.proposal_state.samples,
            self.proposal_state.parameters,
            -step_size
        )

    def _update_step_size(self, *, accepted, attempts):
        if self.step_adjuster is None:
            return self.proposal_state.eta

        ratio = (accepted / attempts) if attempts else 0.0
        adjuster = self.step_adjuster

        if hasattr(adjuster, 'record'):
            eta = adjuster.record(value=self.proposal_state.eta, accepted=accepted, attempts=attempts)
        else:
            try:
                eta = adjuster.adjust(attempts=attempts, accepted=accepted)
            except TypeError:
                eta = adjuster.adjust(value=self.proposal_state.eta, ratio=ratio, attempts=attempts, accepted=accepted)

        return self._set_step_size(self._clamp_step_size(eta))

    def _copy_state_from_step(self, step):
        self.proposal_state.theta.copy_from_host(source=step.theta)
        if getattr(step, "momentum", None) is not None:
            self.proposal_state.momentum.copy_from_host(source=step.momentum)
        else:
            libcudaaltar.cudaHMC_sample_momentum(
                self.proposal_state.momentum.data,
                self.proposal_state.samples,
                self.proposal_state.parameters
            )
        if self.proposal_state.reparameterization:
            if getattr(step, 'theta_sampling', None) is not None:
                self.proposal_state.phi.copy_from_host(source=step.theta_sampling)
            if getattr(step, 'jacobian', None) is not None and self.proposal_state.Jacobian is not None:
                self.proposal_state.Jacobian.copy_from_host(source=step.jacobian)

    def _copy_accepted_to_step(self, step):
        self.proposal_state.theta.copy_to_host(target=step.theta)
        if getattr(step, "momentum", None) is not None:
            self.proposal_state.momentum.copy_to_host(target=step.momentum)
        if self.proposal_state.reparameterization and getattr(step, 'theta_sampling', None) is not None:
            self.proposal_state.phi.copy_to_host(target=step.theta_sampling)
            if getattr(step, 'jacobian', None) is not None and self.proposal_state.Jacobian is not None:
                self.proposal_state.Jacobian.copy_to_host(target=step.jacobian)
        for attr in ("prior", "data", "posterior", "U", "H"):
            target = getattr(step, attr, None)
            source = getattr(self.proposal_state, attr, None)
            if target is not None and source is not None:
                source.copy_to_host(target=target)
        for grad_attr in ("prior_gradient", "data_gradient", "U_gradient"):
            target = getattr(step, grad_attr, None)
            source = getattr(self.proposal_state, grad_attr, None)
            if target is not None and source is not None:
                source.copy_to_host(target=target)

# end of file
