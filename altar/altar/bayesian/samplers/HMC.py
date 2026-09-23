# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2026 parasim inc
# (c) 2010-2026 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu


# externals
import math
from collections import namedtuple
# the package
import altar
# my protocol
from .Sampler import Sampler as sampler
# my scratch state
from ..states.HMCState import HMCState

# acceptance statistics container
Statistics = namedtuple('Statistics', ['accepted', 'invalid', 'rejected'])


# declaration
class HMC(altar.component, family="altar.samplers.hmc", implements=sampler):
    """
    Hamiltonian Monte Carlo: propose a candidate by simulating (unit mass, leapfrog
    discretized) Hamiltonian dynamics from a freshly sampled momentum, then accept/reject with
    the usual Metropolis-Hastings criterion in energy space. Requires the model to implement
    {gradient}, and, like SGLD, only supports unbounded priors (see
    {BayesianL2.verify_unbounded_priors}, which {model.gradient} itself calls).
    """

    # types
    HMCState = HMCState

    # user configurable state
    leapfrog_steps = altar.properties.int(default=10)
    leapfrog_steps.doc = "the number of leapfrog substeps per trajectory"

    step_size = altar.properties.float(default=0.01)
    step_size.doc = "the leapfrog step size epsilon; adapted after each trajectory by {stepsizer}"

    # step size regulator
    stepsizer = altar.bayesian.stepsizer()
    stepsizer.doc = "the step size regulator that adjusts {step_size} based on acceptance statistics"


    # protocol obligations
    @altar.export
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

        # initialize the step size regulator and record the initial step size
        self.step_size = self.stepsizer.initialize(value=self.step_size)

        # the distribution used to draw momentum, one N(0,1) per (sample, parameter)
        self.uninormal = altar.pdf.ugaussian(rng=rng)
        # the distribution for the Metropolis-Hastings acceptance draws; strictly positive so
        # {log(dice)} never sees zero
        self.uniform = altar.pdf.uniform_pos(rng=rng)
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
        self.walk_chains(annealer=annealer, step=step)
        # notify we are done sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_finish, controller=annealer)
        # all done
        return self.statistics


    @altar.export
    def update(self, annealer, statistics):
        """
        Notification that a β step is complete; a no-op for {Hmc} since {step_size} is
        already adjusted after every trajectory in {walk_chains} -- CATMIP's β schedule can
        reach β=1 in just a handful of outer iterations (see {COV}), which is nowhere near
        enough adaptation opportunities for a fixed-in-advance leapfrog step, so {Hmc} adapts
        on the much shorter per-trajectory timescale instead, unlike {Metropolis}
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
        for _ in range(self.steps):
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
            # draw fresh momentum for this trajectory
            candidate.momentum.random(pdf=self.uninormal)
            # the kinetic energy of the freshly drawn momentum, one scalar per chain
            ke_old = self._kinetic(candidate.momentum)

            # the gradient of the log posterior at the trajectory's starting point
            self._evaluate(annealer=annealer, model=model, candidate=candidate, samples=samples)

            # leapfrog integration: a half momentum step, {leapfrog_steps} full position
            # steps (each followed by a fresh gradient evaluation and, except on the last
            # substep, a full momentum step), and a final half momentum step
            ε = self.step_size
            half = 0.5 * ε
            self._axpy(half, candidate.grad_posterior, candidate.momentum)
            for k in range(self.leapfrog_steps):
                self._axpy(ε, candidate.momentum, candidate.theta)
                self._evaluate(annealer=annealer, model=model, candidate=candidate, samples=samples)
                if k < self.leapfrog_steps - 1:
                    self._axpy(ε, candidate.grad_posterior, candidate.momentum)
            self._axpy(half, candidate.grad_posterior, candidate.momentum)

            # the kinetic energy at the end of the trajectory
            ke_new = self._kinetic(candidate.momentum)

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
            # docstring for why this can't wait for the once-per-β-step external call
            self.step_size = self.stepsizer.adjust(attempts=samples, accepted=trajectory_accepted)

            # notify we are done advancing the chains
            dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)

        # store statistics on self for access by update() and the controller
        self.statistics = Statistics(accepted, 0, rejected)
        # all done
        return


    # implementation details
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


    def _kinetic(self, momentum):
        """
        The kinetic energy 0.5 * sum(momentum**2) of each chain, as a plain numpy array; this
        is the one spot where reading through {momentum.ndarray()} (a zero-copy view over the
        same gsl-owned storage) is more direct than a gsl-level reduction
        """
        import numpy
        return 0.5 * numpy.sum(momentum.ndarray() ** 2, axis=1)


    # public data
    @property
    def scaling(self):
        """
        {Annealer} logs/archives a generic "scaling" value after every β step, reaching
        directly into {Metropolis}'s proposal scale; alias it to {step_size} so the same
        archiving code works unmodified for {Hmc} too
        """
        return self.step_size


    # private data
    steps = 1            # the number of trajectories per call to {sample_posterior}
    info = None           # the application info channel
    uninormal = None      # the distribution used to draw momentum
    uniform = None        # the distribution of the Metropolis-Hastings acceptance draws
    statistics = None     # (accepted, invalid, rejected) from the last walk_chains call


# end of file
