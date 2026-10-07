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
from altar.bayesian.statistics import weighted_variance

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
            self.stepsizer.target = self.target_acceptance
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
        Advance all chains using {self.steps} leapfrog trajectories (matching cpu
        {altar.bayesian.samplers.native.HMC}'s own outer loop), each with {self.leapfrog_steps}
        leapfrog substeps; the potential and its gradient carry over from one trajectory to the
        next, so they are evaluated once per substep, plus once per walk
        """
        if self.proposal_state is None:
            self._allocate(model=annealer.model)
        state = self.proposal_state
        state.beta = step.beta

        self._set_step_size(self._clamp_step_size(self.step_size))

        # the mass matrix, from the population
        self._set_mass(step)
        # the chains, and their potential and its gradient, at the start of the walk
        self._copy_state_from_step(step)
        self._potential_and_gradients(annealer)

        # the states the chains keep as they walk, into {step}, when pooling
        pool = getattr(annealer.worker, "pool", None)
        keep = lambda offset: self._copy_accepted_to_step(step, offset=offset)
        if pool is not None:
            pool.begin()

        accepted_total = 0
        dispatcher = annealer.dispatcher
        for trajectory in range(self.steps):
            dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)
            self._update_mass(trajectory)
            libcudaaltar.leapfrog.cudaLeapfrog_sampleMomentum(state.momentum.grid)
            # p ~ N(0, M), M = diag(1/var)
            if self._sd is not None:
                state.momentum *= self._inv_sd
            accepted = self._trajectory(annealer, self.leapfrog_steps)
            self._update_step_size(accepted=accepted, attempts=state.samples)
            accepted_total += accepted
            dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)
            if pool is not None:
                pool.advance(keep)
        if pool is not None:
            pool.end(keep)
        else:
            self._copy_accepted_to_step(step)

        attempts = state.samples * self.steps
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

    def _potential_and_gradients(self, annealer):
        """
        The likelihoods, the potential U and its gradient at the current chains
        """
        state = self.proposal_state
        model = annealer.model
        # {eval_prior} accumulates into whatever it's handed, so re-zero before every call
        state.prior.zero()
        model.likelihoods(annealer=annealer, step=state)
        # log|J| in its own buffer so {prior}/{posterior} stay physical-space densities
        if state.reparameterization:
            state.log_jacobian.zero()
            model.eval_prior_with_physical(
                step=state, likelihood=state.log_jacobian, batch=state.samples)
        model.gradient(controller=annealer, step=state, batch=state.samples)
        # d(physical)/d(sampling), for the chain rule on the data gradient in the kernel
        if state.reparameterization:
            model.eval_jacobian(step=state, batch=state.samples)
        jacobian = state.Jacobian.grid if state.Jacobian is not None else None
        libcudaaltar.leapfrog.cudaLeapfrog_computePotentialAndGradient(
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
        return

    def _trajectory(self, annealer, leapfrog_substeps):
        """
        One trajectory from the current chains, whose potential and gradient are up to date,
        ending in the Metropolis-Hastings decision; the rejected chains get their whole starting
        state back, so the potential and gradient stay up to date for the next trajectory
        """
        state = self.proposal_state
        old_state = state.clone()
        leapfrog = libcudaaltar.leapfrog

        # the energy at the start
        kinetic_old = altar.cuda.vector(shape=state.samples, dtype=state.theta.dtype).zero()
        self._kinetic(kinetic_old)
        h_old = state.U.clone()
        cublas.axpy(alpha=1.0, x=kinetic_old, y=h_old, batch=state.samples)

        # leapfrog updates; the last one leaves the potential and gradient at the proposal
        eta = state.eta
        half_step = 0.5 * eta
        self._update_momentum(half_step)
        for k in range(leapfrog_substeps):
            self._update_position(annealer)
            self._potential_and_gradients(annealer)
            if k < leapfrog_substeps - 1:
                self._update_momentum(eta)
        self._update_momentum(half_step)

        # the energy at the proposal
        kinetic_new = altar.cuda.vector(shape=state.samples, dtype=state.theta.dtype).zero()
        self._kinetic(kinetic_new)
        h_new = state.U.clone()
        cublas.axpy(alpha=1.0, x=kinetic_new, y=h_new, batch=state.samples)

        delta_h = h_new.clone()
        cublas.axpy(alpha=-1.0, x=h_old, y=delta_h, batch=state.samples)

        # Metropolis-Hastings decision
        mask_dev = altar.cuda.vector(shape=state.samples, dtype='int32').zero()
        leapfrog.cudaLeapfrog_metropolis(delta_h.grid, mask_dev.grid)
        accepted = int(numpy.count_nonzero(mask_dev.copy_to_host(type="numpy")))

        # restore the rejected chains
        leapfrog.cudaLeapfrog_restoreRejected(
            state.theta.grid,
            old_state.theta.grid,
            state.momentum.grid,
            old_state.momentum.grid,
            mask_dev.grid,
        )
        for name in ("prior_gradient", "data_gradient", "U_gradient"):
            leapfrog.cudaLeapfrog_restoreMatrix(
                getattr(state, name).grid, getattr(old_state, name).grid, mask_dev.grid)
        for name in ("prior", "data", "posterior", "U"):
            leapfrog.cudaLeapfrog_restoreVector(
                getattr(state, name).grid, getattr(old_state, name).grid, mask_dev.grid)
        if state.reparameterization:
            leapfrog.cudaLeapfrog_restoreMatrix(state.phi.grid, old_state.phi.grid, mask_dev.grid)
            leapfrog.cudaLeapfrog_restoreMatrix(
                state.Jacobian.grid, old_state.Jacobian.grid, mask_dev.grid)
            leapfrog.cudaLeapfrog_restoreVector(
                state.log_jacobian.grid, old_state.log_jacobian.grid, mask_dev.grid)

        # the total energy of the final state
        self._kinetic(state.H)
        cublas.axpy(alpha=1.0, x=state.U, y=state.H, batch=state.samples)

        return accepted

    def _update_position(self, annealer):
        # the velocity, M^-1 p
        velocity = self._scale(self._var)
        if self.proposal_state.reparameterization:
            libcudaaltar.leapfrog.cudaLeapfrog_updatePosition(
                self.proposal_state.phi.grid,
                velocity.grid,
                self.proposal_state.eta
            )
            self._transform_phi_to_theta(annealer)
            return
        libcudaaltar.leapfrog.cudaLeapfrog_updatePosition(
            self.proposal_state.theta.grid,
            velocity.grid,
            self.proposal_state.eta
        )

    def _kinetic(self, energy):
        """
        Fill {energy} with p^T M^-1 p / 2 of each chain
        """
        libcudaaltar.leapfrog.cudaLeapfrog_kineticEnergy(self._scale(self._sd).grid, energy.grid)

    def _scale(self, factor):
        """
        The momenta, scaled cell by cell by {factor}, or themselves for a unit mass
        """
        momentum = self.proposal_state.momentum
        if factor is None:
            return momentum
        self._scaled.copy(momentum)
        self._scaled *= factor
        return self._scaled

    def _set_mass(self, step):
        """
        The diagonal mass matrix at the start of the walk, M = diag(1/var), from the variance of
        each parameter in sampling space over the population: its weighted samples before
        resampling, when the scheduler kept them; a unit mass unless {adapt_mass_matrix}
        """
        self._sd = self._var = self._inv_sd = None
        if not self.adapt_mass_matrix:
            return
        weighted = getattr(step, "weighted_theta", None)
        weights = getattr(step, "weights", None)
        if weighted is not None and weights is not None:
            self._estimate_mass(numpy.asarray(weighted), numpy.asarray(weights))
            return
        θ = step.theta_sampling if self.proposal_state.reparameterization else step.theta
        self._estimate_mass(numpy.asarray(θ))
        return

    def _update_mass(self, trajectory):
        """
        Re-estimate the mass matrix from the chains themselves, every {mass_update_interval}
        trajectories of the walk, as {altar.bayesian.samplers.native.HMC} does
        """
        interval = self.mass_update_interval
        if self._sd is None or not interval or trajectory == 0 or trajectory % interval:
            return
        state = self.proposal_state
        self._estimate_mass(numpy.asarray(state.phi if state.reparameterization else state.theta))
        return

    def _estimate_mass(self, θ, w=None):
        """
        The mass matrix from the variance of the rows of {θ}, with weights {w}
        """
        state = self.proposal_state
        if w is None:
            w = numpy.ones(θ.shape[0])
        var = numpy.clip(weighted_variance(θ, w), self.min_variance, self.max_variance)
        # broadcast to the chains, for the cell by cell products on the device
        def rows(v):
            return altar.cuda.matrix(source=numpy.tile(v, (state.samples, 1)), dtype=self.dtype)
        self._var, self._sd, self._inv_sd = rows(var), rows(numpy.sqrt(var)), rows(1 / numpy.sqrt(var))
        if self._scaled is None:
            self._scaled = altar.cuda.matrix(shape=state.momentum.shape, dtype=self.dtype)
        return

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
        # the chains start from the first rows of the population
        rows = slice(0, self.proposal_state.samples)
        self.proposal_state.theta.copy_from_host(source=numpy.asarray(step.theta)[rows])
        # {Jacobian}/{log_jacobian} are recomputed at the start of the walk; no copy in
        if self.proposal_state.reparameterization:
            if getattr(step, 'theta_sampling', None) is not None:
                self.proposal_state.phi.copy_from_host(source=numpy.asarray(step.theta_sampling)[rows])

    def _copy_accepted_to_step(self, step, offset=0):
        # into the rows of the population from {offset} on
        rows = slice(offset, offset + self.proposal_state.samples)
        def put(source, target):
            source.copy_to_host(target=numpy.asarray(target)[rows])
        put(self.proposal_state.theta, step.theta)
        if getattr(step, "momentum", None) is not None:
            put(self.proposal_state.momentum, step.momentum)
        if self.proposal_state.reparameterization and getattr(step, 'theta_sampling', None) is not None:
            put(self.proposal_state.phi, step.theta_sampling)
            if getattr(step, 'jacobian', None) is not None:
                put(self.proposal_state.log_jacobian, step.jacobian)
        for attr in ("prior", "data", "posterior", "U", "H"):
            target = getattr(step, attr, None)
            source = getattr(self.proposal_state, attr, None)
            if target is not None and source is not None:
                put(source, target)
        for grad_attr in ("prior_gradient", "data_gradient", "U_gradient"):
            target = getattr(step, grad_attr, None)
            source = getattr(self.proposal_state, grad_attr, None)
            if target is not None and source is not None:
                put(source, target)


    # private data; the component-typed attributes and scalar traits are set by the shim's
    # initialize() before it calls mine (see {altar.bayesian.samplers.HMC._makeImpl})
    target_acceptance = 0.7  # the default acceptance rate the stepsizer steers to
    stepsizer = None         # the step size regulator
    leapfrog_steps = 10      # the number of leapfrog substeps per trajectory

    steps = 1               # the number of trajectories per call to {sample_posterior};
                            # filled in from {application.job.steps} in {initialize}
    proposal_state = None   # my {HMCState} scratch state, allocated once, on first use
    adapt_mass_matrix = True # whether to scale the momenta by the population's variances
    mass_update_interval = 20 # trajectories between estimates of the mass matrix in a walk
    min_variance = 1e-8     # the bounds of the variances of the mass matrix
    max_variance = 1e8
    _var = None             # the variances of the mass matrix, and their square roots and
    _sd = None              # inverses, broadcast to the chains; none for a unit mass
    _inv_sd = None
    _scaled = None          # scratch, for the scaled momenta
    samples = None          # the number of chains, for {_allocate}
    dtype = None            # the gpu precision, for {_allocate}
    info = None             # the application info channel
    statistics = None       # (accepted, invalid, rejected) from the last {_walk} call
    step_size = None      # the current leapfrog step size


# end of file
