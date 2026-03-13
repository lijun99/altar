# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

"""
CPU Adaptive Metropolis sampler.

Instead of a fixed number of steps, each β step runs until the Pearson
correlation between the starting and current sample positions drops below
target_correlation (i.e., the chains are effectively de-correlated), or
max_mc_steps is reached.

The scaling is managed by the stepsizer component.  For the adaptive behaviour
that mirrors the CUDA version, configure the stepsizer as TargetedRate:

    sampler.stepsizer = altar.bayesian.stepsizers.targetedrate
    sampler.stepsizer.target = 0.234   # or whatever acceptance target
    sampler.stepsizer.gain   = <gain_function(target)>
"""

# externals
import math
import numpy
# the package
import altar

# base class
from .Metropolis import Metropolis, Statistics


# declaration
class AdaptiveMetropolis(Metropolis,
                         family="altar.samplers.adaptivemetropolis"):
    """
    Adaptive Metropolis sampler: each β step terminates when the sample
    correlation drops below target_correlation or max_mc_steps is reached.
    Inherits the proposal mechanism and stepsizer from Metropolis.
    """

    # adaptive termination parameters
    max_mc_steps = altar.properties.int(default=10000)
    max_mc_steps.doc = 'maximum Monte-Carlo steps per β step'

    min_mc_steps = altar.properties.int(default=1000)
    min_mc_steps.doc = 'minimum steps before the first correlation check'

    corr_check_steps = altar.properties.int(default=1000)
    corr_check_steps.doc = 'MC steps between successive correlation checks'

    target_correlation = altar.properties.float(default=0.6)
    target_correlation.doc = 'correlation threshold below which the chain is considered de-correlated'

    max_mc_steps_stage2 = altar.properties.int(default=None)
    max_mc_steps_stage2.doc = 'max steps when β > beta_stage2 (defaults to max_mc_steps)'

    beta_stage2 = altar.properties.float(default=1.0)
    beta_stage2.doc = 'β threshold above which to use max_mc_steps_stage2'


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # chain up to set up proposal, stepsizer, uniform distribution
        super().initialize(application=application)

        # resolve the stage-2 step limit
        if self.max_mc_steps_stage2 is None:
            self.max_mc_steps_stage2 = self.max_mc_steps

        # grab a journal channel for progress reporting
        self.info = application.info

        # all done
        return self


    def walk_chains(self, annealer, step):
        """
        Run the Adaptive Metropolis algorithm: iterate in blocks of
        corr_check_steps until the sample correlation is below
        target_correlation or max_mc_steps is reached.
        """
        # get the model and dispatcher
        model = annealer.model
        dispatcher = annealer.dispatcher

        # unpack what we need from the cooling step
        β = step.beta
        θ = step.theta
        prior = step.prior
        data = step.data
        posterior = step.posterior
        # sample geometry
        samples = step.samples
        parameters = step.parameters
        # math helpers
        log = math.log

        # reset accept/reject counters
        accepted = rejected = unlikely = 0

        # allocate workspace (reused across MC steps)
        cprior = altar.vector(shape=samples)
        cdata  = altar.vector(shape=samples)
        cpost  = altar.vector(shape=samples)
        csigma = altar.matrix(shape=(parameters, parameters))
        rejects = altar.vector(shape=samples)
        dice    = altar.vector(shape=samples)

        # snapshot of starting positions for correlation check
        θstart = θ.clone()

        # determine max steps for this β
        max_steps = (self.max_mc_steps_stage2 if β > self.beta_stage2
                     else self.max_mc_steps)

        # adaptive outer loop
        correlation = 1.0
        mcsteps = 0

        while correlation > self.target_correlation and mcsteps < max_steps:

            # inner block of corr_check_steps M-H steps
            for _ in range(self.corr_check_steps):
                # notify we are advancing the chains
                dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)

                # propose a displacement
                cθ = self.proposal.propose(sampler=self, step=step, annealer=annealer)
                # initialize the likelihoods and covariance scratch space
                likelihoods = cprior.zero(), cdata.zero(), cpost.zero()
                csigma.zero()
                # build a candidate state
                candidate = self.CoolingStep(beta=β, theta=cθ,
                                             likelihoods=likelihoods, sigma=csigma)

                # verify candidates against model constraints
                dispatcher.notify(event=dispatcher.verify_start, controller=annealer)
                model.verify(step=candidate, mask=rejects.zero())
                # replace invalid candidates with copies of the current samples
                for index, flag in enumerate(rejects):
                    if flag:
                        cθ.setRow(index, θ.getRow(index))
                dispatcher.notify(event=dispatcher.verify_finish, controller=annealer)

                # compute likelihoods for candidates
                model.likelihoods(annealer=annealer, step=candidate)

                # Metropolis acceptance criterion
                diff = cpost.clone()
                diff -= posterior
                dice.random(self.uniform)

                dispatcher.notify(event=dispatcher.accept_start, controller=annealer)

                for sample in range(samples):
                    if rejects[sample]:
                        rejected += 1
                        continue
                    if log(dice[sample]) > diff[sample]:
                        unlikely += 1
                        continue
                    # accept
                    accepted += 1
                    θ.setRow(sample, cθ.getRow(sample))
                    prior[sample]     = cprior[sample]
                    data[sample]      = cdata[sample]
                    posterior[sample] = cpost[sample]

                dispatcher.notify(event=dispatcher.accept_finish, controller=annealer)
                dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)

            mcsteps += self.corr_check_steps

            # check correlation once minimum steps are reached
            if mcsteps >= self.min_mc_steps:
                correlation = self._maxCorrelation(θstart, θ, parameters)
                self.info.log(
                    f"Adaptive Metropolis: correlation {correlation:.4f} at step {mcsteps}")

        # store statistics for update() and the controller
        self.statistics = Statistics(accepted, rejected, unlikely)
        # all done
        return


    # implementation details
    def _maxCorrelation(self, theta_start, theta, parameters):
        """
        Return the maximum absolute Pearson correlation between corresponding
        parameter columns of theta_start and theta (both samples × parameters).
        """
        n = theta_start.rows
        # extract both matrices into numpy arrays column by column
        ts = numpy.array([[theta_start[i, j] for j in range(parameters)]
                          for i in range(n)])
        tc = numpy.array([[theta[i, j] for j in range(parameters)]
                          for i in range(n)])
        # centre
        ts -= ts.mean(axis=0)
        tc -= tc.mean(axis=0)
        # Pearson r per parameter
        numer = (ts * tc).sum(axis=0)
        denom = numpy.sqrt((ts**2).sum(axis=0) * (tc**2).sum(axis=0))
        # guard against zero variance (constant column)
        safe_denom = numpy.where(denom > 0, denom, 1.0)
        corr = numpy.abs(numer / safe_denom)
        return float(corr.max())


    # private data
    info = None  # journal channel, set in initialize()


# end of file
