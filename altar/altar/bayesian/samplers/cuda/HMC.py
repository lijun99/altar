# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

"""Hamiltonian Monte Carlo sampler that drives CUDA leapfrog kernels."""

from collections import namedtuple

import numpy

import altar
import altar.cuda
from altar.cuda import cublas, libcudaaltar

from altar.bayesian.samplers.Sampler import Sampler
from altar.bayesian.stepsizers.StepSizer import StepSizer

from altar.bayesian.states.cuda.HMCState import HMCState

# acceptance statistics container, the same shape {Metropolis}/{HMC} (cpu) use; hmc has no
# notion of an invalid (out of support) candidate, so the middle field is always 0
Statistics = namedtuple('Statistics', ['accepted', 'invalid', 'rejected'])


class HMC(altar.component, family="altar.samplers.hmc", implements=Sampler):
    """
    Hamiltonian Monte Carlo sampler using a leapfrog integrator, driving the cuda
    {altar.cuda.libcudaaltar.leapfrog} kernels. The cuda counterpart of
    {altar.bayesian.samplers.HMC}; see that class for the algorithm itself.
    """

    step_adjuster = StepSizer()
    step_adjuster.doc = "component that adapts the step size based on acceptance ratios"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        self.info = application.info
        model = application.model
        samples = application.job.chains
        parameters = model.parameters
        dtype = application.job.gpuprecision
        reparameterization = getattr(model, 'reparameterization', False)
        self.proposal_state = HMCState.alloc(
            samples=samples, parameters=parameters, dtype=dtype,
            reparameterization=reparameterization
        )
        self._current_eta = self.step_adjuster.initialize(self.proposal_state.eta)
        self._set_step_size(self._current_eta)
        # all done
        return self


    @altar.export
    def sample_posterior(self, annealer, step):
        """
        Sample the posterior distribution
        """
        # grab the dispatcher
        dispatcher = annealer.dispatcher
        # notify we have started sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_start, controller=annealer)
        # walk the chains; statistics stored on self
        self._walk(annealer=annealer, step=step)
        # notify we are done sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_finish, controller=annealer)
        # all done
        return self.statistics


    @altar.export
    def update(self, annealer, statistics):
        """
        Notification that a β step is complete; a no-op, since the step size is already
        adjusted after every trajectory in {_walk}/{_update_step_size} -- the same rationale
        as {altar.bayesian.samplers.HMC.update}
        """
        return


    def _walk(self, annealer, step):
        """
        Advance all chains using one or more leapfrog trajectories
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
        dispatcher = annealer.dispatcher
        for _ in range(leapfrog_repeats):
            dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)
            self._copy_state_from_step(step)
            accepted = self._trajectory(annealer, leapfrog_substeps)
            self._copy_accepted_to_step(step)
            self._update_step_size(accepted=accepted, attempts=state.samples)
            accepted_total += accepted
            dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)

        attempts = state.samples * leapfrog_repeats
        self.statistics = Statistics(accepted_total, 0, attempts - accepted_total)
        # all done
        return


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

    def _trajectory(self, annealer, leapfrog_substeps):
        state = self.proposal_state
        old_state = state.clone()
        leapfrog = libcudaaltar.leapfrog
        model = annealer.model

        def compute_potential_and_gradients():
            try:
                model.evaluateLikelihoods(state)
            except AttributeError:
                model.likelihoods(annealer, state)
            try:
                model.evaluateGradients(state)
            except AttributeError:
                model.gradients(annealer, state)
            jacobian = state.Jacobian.grid if state.Jacobian is not None else None
            leapfrog.cudaLeapfrog_computePotentialAndGradient(
                state.prior.grid,
                state.data.grid,
                state.prior_gradient.grid,
                state.data_gradient.grid,
                state.U.grid,
                state.U_gradient.grid,
                state.beta,
                jacobian,
            )

        # compute initial energy
        compute_potential_and_gradients()
        kinetic_old = altar.cuda.vector(shape=state.samples, dtype=state.theta.dtype).zero()
        leapfrog.cudaLeapfrog_kineticEnergy(state.momentum.grid, kinetic_old.grid)
        potential_old = state.U.clone()
        h_old = potential_old.clone()
        cublas.axpy(alpha=1.0, x=kinetic_old, y=h_old, batch=state.samples)

        # leapfrog updates
        eta = state.eta
        half_step = 0.5 * eta
        self._update_momentum(half_step)
        for k in range(leapfrog_substeps):
            self._update_position(annealer)
            compute_potential_and_gradients()
            if k < leapfrog_substeps - 1:
                self._update_momentum(eta)
        self._update_momentum(half_step)

        # recompute energies at the proposal
        compute_potential_and_gradients()
        kinetic_new = altar.cuda.vector(shape=state.samples, dtype=state.theta.dtype).zero()
        leapfrog.cudaLeapfrog_kineticEnergy(state.momentum.grid, kinetic_new.grid)
        potential_new = state.U.clone()
        h_new = potential_new.clone()
        cublas.axpy(alpha=1.0, x=kinetic_new, y=h_new, batch=state.samples)

        delta_h = h_new.clone()
        cublas.axpy(alpha=-1.0, x=h_old, y=delta_h, batch=state.samples)

        # Metropolis-Hastings decision
        mask_dev = altar.cuda.vector(shape=state.samples, dtype='int32').zero()
        leapfrog.cudaLeapfrog_metropolis(delta_h.grid, mask_dev.grid)
        accepted = int(numpy.count_nonzero(mask_dev.copy_to_host(type="numpy")))

        # restore rejected proposals
        leapfrog.cudaLeapfrog_restoreRejected(
            state.theta.grid,
            old_state.theta.grid,
            state.momentum.grid,
            old_state.momentum.grid,
            mask_dev.grid,
        )
        if state.reparameterization:
            leapfrog.cudaLeapfrog_restoreMatrix(
                state.phi.grid,
                old_state.phi.grid,
                mask_dev.grid,
            )
            if state.Jacobian is not None and old_state.Jacobian is not None:
                leapfrog.cudaLeapfrog_restoreMatrix(
                    state.Jacobian.grid,
                    old_state.Jacobian.grid,
                    mask_dev.grid,
                )

        # update energies/gradients for the final (accepted/rejected) state
        compute_potential_and_gradients()
        leapfrog.cudaLeapfrog_kineticEnergy(state.momentum.grid, state.H.grid)
        cublas.axpy(alpha=1.0, x=state.U, y=state.H, batch=state.samples)

        return accepted

    def _update_position(self, annealer):
        if self.proposal_state.reparameterization:
            libcudaaltar.leapfrog.cudaLeapfrog_updatePosition(
                self.proposal_state.phi.grid,
                self.proposal_state.momentum.grid,
                self.proposal_state.eta
            )
            self._transform_phi_to_theta(annealer)
            return
        libcudaaltar.leapfrog.cudaLeapfrog_updatePosition(
            self.proposal_state.theta.grid,
            self.proposal_state.momentum.grid,
            self.proposal_state.eta
        )

    def _transform_phi_to_theta(self, annealer):
        model = annealer.model
        try:
            model.transformToPhysical(self.proposal_state)
        except AttributeError:
            raise NotImplementedError("model.transformToPhysical is required for reparameterized HMC")

    def _update_momentum(self, step_size):
        libcudaaltar.leapfrog.cudaLeapfrog_updateMomentum(
            self.proposal_state.momentum.grid,
            self.proposal_state.U_gradient.grid,
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
            libcudaaltar.leapfrog.cudaLeapfrog_sampleMomentum(self.proposal_state.momentum.grid)
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


    # private data
    proposal_state = None   # my {HMCState} scratch state, allocated once, in {initialize}
    info = None             # the application info channel
    statistics = None       # (accepted, invalid, rejected) from the last {_walk} call
    _current_eta = None      # the current leapfrog step size


# end of file
