# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# externals
import math
from collections import namedtuple
# the package
import altar
# my protocol
from .Sampler import Sampler as sampler

# acceptance statistics container
Statistics = namedtuple('Statistics', ['accepted', 'rejected', 'unlikely'])


# declaration
class Metropolis(altar.component, family="altar.samplers.metropolis", implements=sampler):
    """
    The Metropolis algorithm as a sampler of the posterior distribution
    """

    # proposal mechanism (optional depending on sampler)
    proposal = altar.bayesian.proposal()
    proposal.doc = "the proposal mechanism used by this sampler"

    # types
    from ..states.CoolingStep import CoolingStep

    # step size regulator
    stepsizer = altar.bayesian.stepsizer()
    stepsizer.doc = "the step size regulator that adjusts the proposal scaling based on acceptance statistics"


    # protocol obligations
    @altar.export
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # pull the chain length from the job specification
        self.steps = application.job.steps
        # get the capsule of the random number generator
        rng = application.rng.rng

        # initialize the step size regulator and record the initial scaling
        self.scaling = self.stepsizer.initialize()

        # initialize the proposal mechanism
        self.proposal.initialize(application=application)

        # set up the distribution for building the sample multiplicities; use a strictly
        # positive distribution to avoid generating candidates with zero displacement
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


    @altar.provides
    def update(self, annealer, statistics):
        """
        Update my parameters based on the results of walking my Markov chains
        """
        # unpack the statistics
        accepted, rejected, unlikely = statistics
        # delegate step size adjustment to the stepsizer
        self.scaling = self.stepsizer.adjust(
            attempts=accepted + rejected + unlikely,
            accepted=accepted)
        # all done
        return


    def walk_chains(self, annealer, step):
        """
        Run the Metropolis algorithm on the Markov chains
        """
        # get the model
        model = annealer.model
        # and the event dispatcher
        dispatcher = annealer.dispatcher

        # unpack what i need from the cooling step
        β = step.beta
        θ = step.theta
        prior = step.prior
        data = step.data
        posterior = step.posterior
        # the sample geometry
        samples = step.samples
        parameters = step.parameters
        # a couple of functions from the math module
        exp = math.exp
        log = math.log

        # reset the accept/reject counters
        accepted = rejected = unlikely = 0

        # allocate some vectors that we use throughout the following
        # candidate likelihoods
        cprior = altar.vector(shape=samples)
        cdata = altar.vector(shape=samples)
        cpost = altar.vector(shape=samples)
        # a fake covariance matrix for the candidate steps, just so we don't have to rebuild it
        # every time
        csigma = altar.matrix(shape=(parameters,parameters))
        # the mask of samples rejected due to model constraint violations
        rejects = altar.vector(shape=samples)
        # and a vector with random numbers for the Metropolis acceptance
        dice = altar.vector(shape=samples)

        # step all chains together
        for _ in range(self.steps):
            # notify we are advancing the chains
            dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)

            # initialize the candidate sample by randomly displacing the current one
            cθ = self.proposal.propose(sampler=self, step=step, annealer=annealer)
            # initialize the likelihoods
            likelihoods = cprior.zero(), cdata.zero(), cpost.zero()
            # and the covariance matrix
            csigma.zero()
            # build a candidate state
            candidate = self.CoolingStep(beta=β, theta=cθ,
                                         likelihoods=likelihoods, sigma=csigma)

            # the random displacement may have generated candidates that are outside the
            # support of the model, so we must give it an opportunity to reject them;
            # notify we are starting the verification process
            dispatcher.notify(event=dispatcher.verify_start, controller=annealer)
            # reset the mask and ask the model to verify the sample validity
            model.verify(step=candidate, mask=rejects.zero())
            # make the candidate a consistent set by replacing the rejected samples with copies
            # of the originals from {θ}
            for index, flag in enumerate(rejects):
                # if this sample was rejected
                if flag:
                    # copy the corresponding row from {θ} into {candidate}
                    cθ.setRow(index, θ.getRow(index))
            # notify that the verification process is finished
            dispatcher.notify(event=dispatcher.verify_finish, controller=annealer)

            # compute the likelihoods
            model.likelihoods(annealer=annealer, step=candidate)

            # build a vector to hold the difference of the two posterior likelihoods
            diff = cpost.clone()
            # subtract the previous posterior
            diff -= posterior
            # randomize the Metropolis acceptance vector
            dice.random(self.uniform)

            # notify we are starting accepting samples
            dispatcher.notify(event=dispatcher.accept_start, controller=annealer)

            # accept/reject: go through all the samples
            for sample in range(samples):
                # a candidate is rejected if the model considered it invalid
                if rejects[sample]:
                    # nothing to do: θ, priorL, dataL, and postL contain the right statistics
                    # for this sample; just update the rejection count
                    rejected += 1
                    # and move on
                    continue
                # a candidate is also rejected if the model considered it less likely than the
                # original and it wasn't saved by the {dice}
                if log(dice[sample]) > diff[sample]:
                    # nothing to do: θ, priorL, dataL, and postL contain the right statistics
                    # for this sample; just update the unlikely count
                    unlikely += 1
                    # and move on
                    continue

                # otherwise, update the acceptance count
                accepted += 1
                # copy the candidate sample
                θ.setRow(sample, cθ.getRow(sample))
                # and its likelihoods
                prior[sample] = cprior[sample]
                data[sample] = cdata[sample]
                posterior[sample] = cpost[sample]

            # notify we are done accepting samples
            dispatcher.notify(event=dispatcher.accept_finish, controller=annealer)

            # notify we are done advancing the chains
            dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)


        # store statistics on self for access by update() and the controller
        self.statistics = Statistics(accepted, rejected, unlikely)
        # all done
        return


    # private data
    steps = 1          # the length of each Markov chain
    scaling = 0.1      # current proposal scaling; updated by stepsizer after each update
    statistics = None  # (accepted, rejected, unlikely) from the last walk_chains call

    uniform = None     # the distribution of the sample multiplicities
    dispatcher = None  # a reference to the event dispatcher


# end of file
