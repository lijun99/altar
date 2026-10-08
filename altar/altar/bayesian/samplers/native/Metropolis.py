# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy
from collections import namedtuple
# the package
import altar
from altar.bayesian.states.CoolingStep import CoolingStep

# acceptance statistics container
Statistics = namedtuple('Statistics', ['accepted', 'invalid', 'rejected'])


# declaration
class Metropolis:
    """
    The cpu implementation of the Metropolis algorithm as a sampler of the posterior
    distribution. See {altar.bayesian.samplers.Metropolis}, the pyre component a {.pfg}
    actually selects, which picks me (or my cuda counterpart) once, at {initialize} time.
    """

    # types
    CoolingStep = CoolingStep


    # protocol-shaped obligations (called by the shim, not pyre-dispatched directly)
    def initialize(self, application):
        """
        Initialize me and my parts given an {application} context
        """
        # grab the info channel
        self.info = application.info
        # get the capsule of the random number generator
        rng = application.rng.rng

        # random-walk Metropolis's theoretically-optimal acceptance rate in high dimensions
        # (Roberts-Gelman-Gilks), unless the user picked a target explicitly
        if getattr(self.stepsizer, "target", None) is None:
            self.stepsizer.target = 0.234
        # initialize the step size regulator and record the initial scaling
        self.scaling = self.stepsizer.initialize(value=self.scaling)

        # initialize the step count regulator (e.g. {FixedSteps} fills in application.job.steps)
        self.stepcounter.initialize(application=application)

        # initialize the proposal mechanism
        self.proposal.initialize(application=application)

        # set up the distribution for building the sample multiplicities; use a strictly
        # positive distribution to avoid generating candidates with zero displacement
        self.uniform = altar.pdf.uniform_pos(rng=rng)
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
        # let the proposal adapt to the samples this walk starts from
        self.proposal.new_walk()
        # walk the chains; statistics stored on self
        self.walk_chains(annealer=annealer, step=step)
        # notify we are done sampling the posterior
        dispatcher.notify(event=dispatcher.sample_posterior_finish, controller=annealer)
        # all done
        return self.statistics


    def update(self, annealer, statistics):
        """
        Update my parameters based on the results of walking my Markov chains
        """
        # unpack the statistics
        accepted, invalid, rejected = statistics
        # delegate step size adjustment to the stepsizer
        self.scaling = self.stepsizer.adjust(
            attempts=accepted + invalid + rejected,
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
        # a reparameterized model walks in sampling space, where every proposal is in the
        # support; the target there is the posterior plus the log-jacobian of the map
        reparameterized = step.has_reparametrization
        if reparameterized:
            θs = step.theta_sampling
            jacobian = step.jacobian
            jacobian.zero()
            model.eval_prior_with_physical(step=step, likelihood=jacobian)
            cjacobian = altar.vector(shape=samples)
            # the proposal moves the sampling-space chains
            walker = self.CoolingStep(beta=β, theta=θs, likelihoods=(prior, data, posterior))
            walker.weights = getattr(step, "weights", None)
            walker.weighted_theta = getattr(step, "weighted_theta", None)
        else:
            walker = step
        # reset the accept/reject counters
        accepted = invalid = rejected = 0

        # allocate some vectors that we use throughout the following
        # candidate likelihoods
        cprior = altar.vector(shape=samples)
        cdata = altar.vector(shape=samples)
        cpost = altar.vector(shape=samples)
        # a fake covariance matrix for the candidate steps, just so we don't have to rebuild it
        # every time
        # the mask of samples rejected due to model constraint violations
        rejects = altar.vector(shape=samples)
        # and a vector with random numbers for the Metropolis acceptance
        dice = altar.vector(shape=samples)

        # the step count regulator decides how many MC steps to run, in blocks, before
        # checking whether this β step is done ({FixedSteps}: one block, the fixed count;
        # {DecorrelatingSteps}: repeated blocks until decorrelated)
        self.stepcounter.start(theta=θ, beta=β)
        mcsteps = 0

        while not self.stepcounter.done(mcsteps=mcsteps, theta=θ, annealer=annealer):
            block = self.stepcounter.block_size()
            for _ in range(block):
                # notify we are advancing the chains
                dispatcher.notify(event=dispatcher.chain_advance_start, controller=annealer)

                # initialize the candidate sample by randomly displacing the current one
                cθ = self.proposal.propose(sampler=self, step=walker, annealer=annealer)
                # in sampling space, keep the candidate there and map a copy to physical
                if reparameterized:
                    cθs = cθ
                    cθ = cθs.clone()
                    model.to_physical(theta=cθ)
                # initialize the likelihoods
                likelihoods = cprior.zero(), cdata.zero(), cpost.zero()
                # build a candidate state
                candidate = self.CoolingStep(beta=β, theta=cθ, likelihoods=likelihoods)

                # the random displacement may have generated candidates that are outside the
                # support of the model, so we must give it an opportunity to reject them;
                # notify we are starting the verification process
                dispatcher.notify(event=dispatcher.verify_start, controller=annealer)
                # reset the mask and ask the model to verify the sample validity
                model.verify(step=candidate, mask=rejects.zero())
                # make the candidate a consistent set by replacing the rejected samples with
                # copies of the originals from {θ}
                invalids = numpy.asarray(rejects) != 0
                numpy.asarray(cθ)[invalids] = numpy.asarray(θ)[invalids]
                if reparameterized:
                    numpy.asarray(cθs)[invalids] = numpy.asarray(θs)[invalids]
                # notify that the verification process is finished
                dispatcher.notify(event=dispatcher.verify_finish, controller=annealer)

                # compute the likelihoods
                model.likelihoods(annealer=annealer, step=candidate)

                # build a vector to hold the difference of the two posterior likelihoods
                diff = cpost.clone()
                # subtract the previous posterior
                diff -= posterior
                # and, in sampling space, add the difference of the log-jacobians
                if reparameterized:
                    model.eval_prior_with_physical(step=candidate, likelihood=cjacobian.zero())
                    diff += cjacobian
                    diff -= jacobian
                # randomize the Metropolis acceptance vector
                dice.random(self.uniform)

                # notify we are starting accepting samples
                dispatcher.notify(event=dispatcher.accept_start, controller=annealer)

                # accept/reject: a candidate is invalid if the model considered it outside its
                # support, rejected if it was less likely than the original and it wasn't saved
                # by the {dice}, and accepted otherwise
                unlucky = numpy.log(numpy.asarray(dice)) > numpy.asarray(diff)
                rejections = ~invalids & unlucky
                accepts = ~invalids & ~unlucky
                # update the counts
                invalid += int(invalids.sum())
                rejected += int(rejections.sum())
                accepted += int(accepts.sum())
                # copy the accepted candidates, and their likelihoods
                numpy.asarray(θ)[accepts] = numpy.asarray(cθ)[accepts]
                if reparameterized:
                    numpy.asarray(θs)[accepts] = numpy.asarray(cθs)[accepts]
                    numpy.asarray(jacobian)[accepts] = numpy.asarray(cjacobian)[accepts]
                for current, proposed in ((prior, cprior), (data, cdata), (posterior, cpost)):
                    numpy.asarray(current)[accepts] = numpy.asarray(proposed)[accepts]

                # notify we are done accepting samples
                dispatcher.notify(event=dispatcher.accept_finish, controller=annealer)

                # notify we are done advancing the chains
                dispatcher.notify(event=dispatcher.chain_advance_finish, controller=annealer)

            mcsteps += block

        # store statistics on self for access by update() and the controller
        self.statistics = Statistics(accepted, invalid, rejected)
        # all done
        return


    # private data; the component-typed attributes are set by the shim's initialize()
    # before it calls mine (see {altar.bayesian.samplers.Metropolis._makeImpl})
    proposal = None    # the proposal mechanism used by this sampler
    stepsizer = None   # the step size regulator
    stepcounter = None # the step count regulator

    scaling = 0.1      # current proposal scaling; updated by stepsizer after each update
    statistics = None  # (accepted, invalid, rejected) from the last walk_chains call

    info = None        # the application info channel
    uniform = None     # the distribution of the sample multiplicities
    dispatcher = None  # a reference to the event dispatcher


# end of file
