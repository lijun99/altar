# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import math
import numpy
from collections import namedtuple
# the package
import altar
# my scratch state
from altar.bayesian.states.HMCState import HMCState

# acceptance statistics container
Statistics = namedtuple('Statistics', ['accepted', 'invalid', 'rejected'])


# declaration
class HMC:
    """
    The cpu implementation of Hamiltonian Monte Carlo: propose a candidate by simulating
    (leapfrog discretized) Hamiltonian dynamics from a freshly sampled momentum, then
    accept/reject with the usual Metropolis-Hastings criterion in energy space. Requires the
    model to implement {gradient}, and, like SGLD, only supports unbounded priors (see
    {BayesianL2.verify_unbounded_priors}, which {model.gradient} itself calls).

    See {altar.bayesian.samplers.HMC}, the pyre component a {.pfg} actually selects, which
    picks me (or my cuda counterpart) once, at {initialize} time.

    The mass matrix is a diagonal approximation to the inverse posterior covariance
    (M^-1 = diag(variance)), periodically re-estimated from the current population's spread,
    mirroring {GaussianProposal}'s covariance estimate for {Metropolis}; under mpi, that
    estimate -- like the step size -- is pooled across every rank's chains, not just the
    calling rank's own shard.
    """

    # types
    HMCState = HMCState


    # protocol-shaped obligations (called by the shim, not pyre-dispatched directly)
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # grab the info channel
        self.info = application.info
        # pull the chain length (number of trajectories per outer step) from the job spec
        self.steps = application.job.steps
        # get the capsule of the random number generator
        rng = application.rng.rng

        # HMC's theoretically-optimal acceptance rate (Neal/Betancourt; Stan's NUTS defaults
        # to 0.8), unless the user picked a target explicitly
        if getattr(self.stepsizer, "target", None) is None:
            self.stepsizer.target = 0.7
        # initialize the step size regulator and record the initial step size
        self.step_size = self.stepsizer.initialize(value=self.step_size)

        # the distribution used to draw momentum, one N(0,1) per (sample, parameter)
        self.uninormal = altar.pdf.ugaussian(rng=rng)
        # the distribution for the Metropolis-Hastings acceptance draws; strictly positive so
        # {log(dice)} never sees zero
        self.uniform = altar.pdf.uniform_pos(rng=rng)

        # if i am one of several mpi ranks, my controller's worker is an {MPIAnnealing} with a
        # {communicator}; grab it (None on a single-process run) so per-trajectory step size
        # adaptation and the mass matrix estimate can pool statistics across every rank's
        # chains, instead of each rank adapting off its own shard alone
        worker = getattr(application.controller, 'worker', None)
        self.communicator = getattr(worker, 'communicator', None)

        # the diagonal mass matrix, as M^-1 (the per-parameter variance); {walk_chains}
        # supplies the first real estimate before its first trajectory
        self.mass_variance = None
        # all done
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
        self.walk_chains(annealer=annealer, step=step)
        # notify we are done sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_finish, controller=annealer)
        # all done
        return self.statistics


    def update(self, annealer, statistics):
        """
        Notification that a β step is complete; a no-op since {step_size} is already adjusted
        after every trajectory in {walk_chains} -- CATMIP's β schedule can reach β=1 in just a
        handful of outer iterations (see {COV}), which is nowhere near enough adaptation
        opportunities for a fixed-in-advance leapfrog step, so this adapts on the much shorter
        per-trajectory timescale instead, unlike {Metropolis}
        """
        return


    def walk_chains(self, annealer, step):
        """
        Run one Hamiltonian trajectory per configured chain step, each ending in a
        Metropolis-Hastings accept/reject decision
        """
        # get the model
        model = annealer.model
        # and the event dispatcher
        dispatcher = annealer.dispatcher

        # unpack what i need from the cooling step; these are never mutated directly -- only
        # accepted rows get copied back into them, exactly as {Metropolis} does
        β = step.beta
        θ = step.theta
        prior = step.prior
        data = step.data
        posterior = step.posterior
        samples = step.samples
        parameters = step.parameters
        log = math.log

        # reset the accept/reject counters; hmc has no notion of an invalid (out of support)
        # candidate -- gradient-based samplers only support unbounded priors to begin with
        accepted = rejected = 0

        # a vector with random numbers for the Metropolis-Hastings acceptance
        dice = altar.vector(shape=samples)

        # step all chains together, one full trajectory per iteration
        for trajectory in range(self.steps):
            # (re)estimate the mass matrix from the current population every
            # {mass_update_interval} trajectories, always including the very first
            if self.adapt_mass_matrix and trajectory % self.mass_update_interval == 0:
                self.mass_variance = self._estimate_mass_variance(theta=θ, parameters=parameters)
            variance = self.mass_variance if self.mass_variance is not None else numpy.ones(parameters)
            precision_sqrt = 1.0 / numpy.sqrt(variance)

            # notify we are advancing the chains
            dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)

            # seed a candidate from the current state; {prior}/{data}/{posterior} start out
            # zeroed, not copied from {step} -- {eval_prior} accumulates into whatever it's
            # handed (so that multiple psets can each add their contribution), so every
            # evaluation below re-zeroes them first, exactly as {Metropolis} does
            candidate = self.HMCState(
                beta=β, theta=θ.clone(),
                likelihoods=(altar.vector(shape=samples), altar.vector(shape=samples),
                             altar.vector(shape=samples)))
            # draw fresh momentum for this trajectory, scaled by the mass matrix:
            # p ~ N(0, M), M = diag(1/variance)
            candidate.momentum.random(pdf=self.uninormal)
            candidate.momentum.ndarray()[:] *= precision_sqrt
            # the kinetic energy of the freshly drawn momentum, one scalar per chain
            ke_old = self._kinetic(candidate.momentum, variance)

            # the gradient of the log posterior at the trajectory's starting point
            self._evaluate(annealer=annealer, model=model, candidate=candidate, samples=samples)

            # leapfrog integration: a half momentum step, {leapfrog_steps} full position
            # steps (each followed by a fresh gradient evaluation and, except on the last
            # substep, a full momentum step), and a final half momentum step; the momentum
            # update never involves the mass matrix (it only depends on ∇U), only the
            # position update does, through M^-1 = diag(variance)
            ε = self.step_size
            half = 0.5 * ε
            self._axpy(half, candidate.grad_posterior, candidate.momentum)
            for k in range(self.leapfrog_steps):
                self._update_position(candidate.theta, candidate.momentum, ε, variance)
                self._evaluate(annealer=annealer, model=model, candidate=candidate, samples=samples)
                if k < self.leapfrog_steps - 1:
                    self._axpy(ε, candidate.grad_posterior, candidate.momentum)
            self._axpy(half, candidate.grad_posterior, candidate.momentum)

            # the kinetic energy at the end of the trajectory
            ke_new = self._kinetic(candidate.momentum, variance)

            # randomize the Metropolis-Hastings acceptance vector
            dice.random(self.uniform)

            # notify we are starting accepting samples
            dispatcher.notify(event=dispatcher.accept_start, controller=annealer)

            # accept/reject: go through all the samples; with U = -posterior and
            # H = U + KE, accept when log(dice) < -ΔH, i.e. when
            # log(dice) < (posterior_new - posterior_old) - (KE_new - KE_old)
            trajectory_accepted = 0
            for sample in range(samples):
                Δ = (candidate.posterior[sample] - posterior[sample]) - (ke_new[sample] - ke_old[sample])
                if log(dice[sample]) > Δ:
                    # rejected: {θ}, {prior}, {data}, {posterior} already hold the right values
                    rejected += 1
                    continue
                # otherwise, accept: copy the candidate sample and its likelihoods
                accepted += 1
                trajectory_accepted += 1
                θ.setRow(sample, candidate.theta.getRow(sample))
                prior[sample] = candidate.prior[sample]
                data[sample] = candidate.data[sample]
                posterior[sample] = candidate.posterior[sample]

            # notify we are done accepting samples
            dispatcher.notify(event=dispatcher.accept_finish, controller=annealer)

            # adjust the step size now, on the per-trajectory timescale -- see {update}'s
            # docstring for why this can't wait for the once-per-β-step external call; pool
            # the attempt/accept counts across mpi ranks first, so every rank ends up with the
            # same {step_size} instead of drifting apart on its own local shard
            attempts, trajectory_accepted_total = samples, trajectory_accepted
            if self.communicator is not None:
                attempts = int(self.communicator.sum(attempts))
                trajectory_accepted_total = int(self.communicator.sum(trajectory_accepted_total))
            self.step_size = self.stepsizer.adjust(attempts=attempts, accepted=trajectory_accepted_total)

            # notify we are done advancing the chains
            dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)

        # store statistics on self for access by update() and the controller
        self.statistics = Statistics(accepted, 0, rejected)
        # all done
        return


    # implementation details
    def _estimate_mass_variance(self, theta, parameters):
        """
        Estimate the diagonal mass matrix M^-1 (the per-parameter variance) from the current
        population in {theta}, pooled across every mpi rank's chains when running in
        parallel, not just the calling rank's own shard
        """
        arr = theta.ndarray()
        n = arr.shape[0]
        sum_x = arr.sum(axis=0)
        sum_x2 = (arr ** 2).sum(axis=0)

        if self.communicator is not None:
            n = int(self.communicator.sum(n))
            sum_x = numpy.array([self.communicator.sum(float(v)) for v in sum_x])
            sum_x2 = numpy.array([self.communicator.sum(float(v)) for v in sum_x2])

        mean = sum_x / n
        variance = sum_x2 / n - mean ** 2
        # guard against a degenerate (collapsed or blown-up) population
        return numpy.clip(variance, self.min_variance, self.max_variance)


    def _update_position(self, theta, momentum, epsilon, variance):
        """
        theta += epsilon * M^-1 * momentum, where M^-1 = diag(variance); done through
        {.ndarray()} since it's a per-column broadcast, which gsl has no single call for
        """
        theta.ndarray()[:] += epsilon * variance[numpy.newaxis, :] * momentum.ndarray()
        return theta


    def _evaluate(self, annealer, model, candidate, samples):
        """
        Evaluate the (prior, data, posterior) likelihoods and their gradients at
        {candidate.theta}, and combine them into {candidate.grad_posterior}
        """
        # {eval_prior} accumulates into whatever it's handed, so it can add up the
        # contributions of multiple parameter sets; re-zero before every evaluation, exactly
        # as {Metropolis.walk_chains} does for its own candidate
        candidate.prior.zero()
        model.likelihoods(annealer=annealer, step=candidate)
        model.gradient(controller=annealer, step=candidate, batch=samples)
        candidate.compute_posterior()
        return candidate


    def _axpy(self, alpha, x, y):
        """
        y += alpha*x, for (samples x parameters) matrices; {altar.blas.daxpy} is vector-only,
        so scale a clone of {x} and accumulate in place instead
        """
        scaled = x.clone()
        scaled.scale(alpha)
        y += scaled
        return y


    def _kinetic(self, momentum, variance):
        """
        The kinetic energy 0.5 * momentum^T M^-1 momentum = 0.5 * sum(momentum**2 * variance)
        of each chain, as a plain numpy array; M^-1 = diag(variance) is the same mass matrix
        used to scale the momentum draw and the leapfrog position update
        """
        return 0.5 * numpy.sum((momentum.ndarray() ** 2) * variance[numpy.newaxis, :], axis=1)


    # private data; the component-typed attributes and scalar traits are set by the shim's
    # initialize() before it calls mine (see {altar.bayesian.samplers.HMC._makeImpl})
    stepsizer = None       # the step size regulator
    leapfrog_steps = 10    # the number of leapfrog substeps per trajectory
    step_size = 0.01       # the leapfrog step size epsilon; adapted after each trajectory
    adapt_mass_matrix = True
    mass_update_interval = 20
    min_variance = 1e-8
    max_variance = 1e8

    steps = 1              # the number of trajectories per call to {sample_posterior}
    info = None            # the application info channel
    uninormal = None       # the distribution used to draw momentum
    uniform = None         # the distribution of the Metropolis-Hastings acceptance draws
    statistics = None      # (accepted, invalid, rejected) from the last walk_chains call
    communicator = None    # the mpi communicator, if running in parallel; else None
    mass_variance = None   # the current diagonal mass matrix M^-1, as a numpy array


# end of file
