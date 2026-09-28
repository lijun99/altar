# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""Hamiltonian Monte Carlo sampler that drives CUDA leapfrog kernels."""

from collections import namedtuple

import numpy

import altar
import altar.cuda
from altar.cuda import cublas, libcudaaltar

from altar.bayesian.states.cuda.HMCState import HMCState

# acceptance statistics container, the same shape {Metropolis}/{HMC} (cpu) use; hmc has no
# notion of an invalid (out of support) candidate, so the middle field is always 0
Statistics = namedtuple('Statistics', ['accepted', 'invalid', 'rejected'])


class HMC:
    """
    The cuda implementation of Hamiltonian Monte Carlo, using a leapfrog integrator driving
    the cuda {altar.cuda.libcudaaltar.leapfrog} kernels. See
    {altar.bayesian.samplers.HMC}, the pyre component a {.pfg} actually selects, which picks
    me (or my cpu counterpart) once, at {initialize} time; and that class's cpu
    implementation, {altar.bayesian.samplers.native.HMC}, for the algorithm itself.
    """

    # protocol-shaped obligations (called by the shim, not pyre-dispatched directly)
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        self.info = application.info
        self.samples = application.job.chains
        self.dtype = application.job.gpuprecision
        # the number of trajectories per call to {sample_posterior}, matching cpu
        # {altar.bayesian.samplers.native.HMC}'s own {self.steps}
        self.steps = application.job.steps
        # HMC's theoretically-optimal acceptance rate (Neal/Betancourt; Stan's NUTS defaults
        # to 0.8), unless the user picked a target explicitly
        if getattr(self.stepsizer, "target", None) is None:
            self.stepsizer.target = 0.7
        # all done
        return self


    def _allocate(self, model):
        """
        Allocate my scratch state on first use: the model, initialized after me, only knows its
        parameter count and whether it reparameterizes once its parameter sets are laid out
        """
        self.proposal_state = HMCState.alloc(
            samples=self.samples, parameters=model.parameters, dtype=self.dtype,
            reparameterization=getattr(model, 'reparameterization', False)
        )
        self.step_size = self.stepsizer.initialize(self.proposal_state.eta)
        self._set_step_size(self.step_size)
        return self


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


    def update(self, annealer, statistics):
        """
        Notification that a β step is complete; a no-op, since the step size is already
        adjusted after every trajectory in {_walk}/{_update_step_size} -- the same rationale
        as {altar.bayesian.samplers.HMC.update}
        """
        return


    def _walk(self, annealer, step):
        """
        Advance all chains using one or more leapfrog trajectories: {self.steps} trajectories
        (matching cpu {altar.bayesian.samplers.native.HMC}'s own outer loop), each with
        {self.leapfrog_steps} leapfrog substeps
        """
        if self.proposal_state is None:
            self._allocate(model=annealer.model)
        state = self.proposal_state
        state.beta = step.beta

        self._set_step_size(self._clamp_step_size(self.step_size))

        leapfrog_repeats = self.steps
        leapfrog_substeps = self.leapfrog_steps

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
        self.step_size = eta
        self.proposal_state.eta = eta
        if self.stepsizer is not None and hasattr(self.stepsizer, 'step_size'):
            self.stepsizer.step_size = eta
        return eta

    def _clamp_step_size(self, eta):
        eta = float(eta)
        adjuster = self.stepsizer
        min_eta = getattr(adjuster, 'min_step_size', 1e-12)
        max_eta = getattr(adjuster, 'max_step_size', float('inf'))
        return max(min_eta, min(max_eta, eta))

    def _trajectory(self, annealer, leapfrog_substeps):
        state = self.proposal_state
        old_state = state.clone()
        leapfrog = libcudaaltar.leapfrog
        model = annealer.model

        def compute_potential_and_gradients():
            # the real model protocol (matching {altar.bayesian.samplers.native.HMC}'s own
            # {_evaluate}); {evaluateLikelihoods}/{evaluateGradients}/{gradients} (plural)
            # never existed on any real model -- this branch was dead code, only ever
            # exercised by a test double built to match it

            # {eval_prior} accumulates into whatever it's handed (so multiple parameter sets
            # can each add their contribution); this is called many times per trajectory
            # (once per leapfrog substep, plus both endpoints), so it must be re-zeroed
            # before every call, exactly as {altar.bayesian.samplers.native.HMC._evaluate}
            # does for its own candidate -- without this, {state.prior} silently accumulated
            # across calls, inflating the potential and making every trajectory reject
            state.prior.zero()
            model.likelihoods(annealer=annealer, step=state)
            # log|J| in its own buffer so {prior}/{posterior} stay physical-space densities
            if state.reparameterization:
                state.log_jacobian.zero()
                model.eval_prior_with_physical(
                    step=state, likelihood=state.log_jacobian, batch=state.samples)
            model.gradient(controller=annealer, step=state, batch=state.samples)
            # refresh {state.Jacobian} (d(physical)/d(sampling)) for the leapfrog kernel's
            # reparameterized path below; stale otherwise (allocated once, at a constant 1)
            if state.reparameterization:
                model.eval_jacobian(step=state, batch=state.samples)
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
            # sampling-space potential: U -= log|J| (its gradient is already in prior_gradient)
            if state.reparameterization:
                cublas.axpy(alpha=-1.0, x=state.log_jacobian, y=state.U, batch=state.samples)

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
        if self.stepsizer is None:
            return self.proposal_state.eta

        ratio = (accepted / attempts) if attempts else 0.0
        adjuster = self.stepsizer

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
        # {Jacobian}/{log_jacobian} are recomputed at the start of every trajectory; no copy in
        if self.proposal_state.reparameterization:
            if getattr(step, 'theta_sampling', None) is not None:
                self.proposal_state.phi.copy_from_host(source=step.theta_sampling)

    def _copy_accepted_to_step(self, step):
        self.proposal_state.theta.copy_to_host(target=step.theta)
        if getattr(step, "momentum", None) is not None:
            self.proposal_state.momentum.copy_to_host(target=step.momentum)
        if self.proposal_state.reparameterization and getattr(step, 'theta_sampling', None) is not None:
            self.proposal_state.phi.copy_to_host(target=step.theta_sampling)
            if getattr(step, 'jacobian', None) is not None:
                self.proposal_state.log_jacobian.copy_to_host(target=step.jacobian)
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


    # private data; the component-typed attributes and scalar traits are set by the shim's
    # initialize() before it calls mine (see {altar.bayesian.samplers.HMC._makeImpl})
    stepsizer = None         # the step size regulator
    leapfrog_steps = 10      # the number of leapfrog substeps per trajectory

    steps = 1               # the number of trajectories per call to {sample_posterior};
                            # filled in from {application.job.steps} in {initialize}
    proposal_state = None   # my {HMCState} scratch state, allocated once, on first use
    samples = None          # the number of chains, for {_allocate}
    dtype = None            # the gpu precision, for {_allocate}
    info = None             # the application info channel
    statistics = None       # (accepted, invalid, rejected) from the last {_walk} call
    step_size = None      # the current leapfrog step size


# end of file
