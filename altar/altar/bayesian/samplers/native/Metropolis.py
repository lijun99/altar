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
from __future__ import annotations
import typing
import numpy
# the package
from altar.bayesian.states.CoolingStep import CoolingStep

if typing.TYPE_CHECKING:
    import journal
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.proposals.Proposal import Proposal
    from altar.bayesian.stepcounters.StepCounter import StepCounter
    from altar.bayesian.stepsizers.StepSizer import StepSizer
    from altar.shells.Application import Application


# acceptance statistics container
class Statistics(typing.NamedTuple):
    accepted: int
    invalid: int
    rejected: int


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
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # grab the info channel
        self.info = application.info
        # and the random number generator
        self.rng = application.rng.rng

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

        # all done
        return self


    def sample_posterior(self, annealer: Annealer, step: CoolingStep) -> Statistics:
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


    def update(self, annealer: Annealer, statistics: Statistics) -> None:
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


    def walk_chains(self, annealer: Annealer, step: CoolingStep) -> None:
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
        # a reparameterized model walks in sampling space, where every proposal is in the
        # support; the target there is the posterior plus the log-jacobian of the map
        reparameterized = step.has_reparametrization
        if reparameterized:
            θs = step.theta_sampling
            jacobian = step.jacobian
            jacobian[...] = 0
            model.eval_prior_with_physical(step=step, likelihood=jacobian)
            cjacobian = numpy.zeros(samples)
            # the proposal moves the sampling-space chains
            walker = self.CoolingStep(beta=β, theta=θs, likelihoods=(prior, data, posterior))
            walker.weights = getattr(step, "weights", None)
            walker.weighted_theta = getattr(step, "weighted_theta", None)
        else:
            walker = step
        # reset the accept/reject counters
        accepted = invalid = rejected = 0

        # the candidate likelihoods
        cprior = numpy.zeros(samples)
        cdata = numpy.zeros(samples)
        cpost = numpy.zeros(samples)
        # the mask of samples rejected due to model constraint violations
        rejects = numpy.zeros(samples)

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
                    cθ = cθs.copy()
                    model.to_physical(theta=cθ)
                # initialize the likelihoods
                for likelihood in (cprior, cdata, cpost):
                    likelihood[...] = 0
                # build a candidate state
                candidate = self.CoolingStep(beta=β, theta=cθ, likelihoods=(cprior, cdata, cpost))

                # the random displacement may have generated candidates that are outside the
                # support of the model, so we must give it an opportunity to reject them;
                # notify we are starting the verification process
                dispatcher.notify(event=dispatcher.verify_start, controller=annealer)
                # reset the mask and ask the model to verify the sample validity
                rejects[...] = 0
                model.verify(step=candidate, mask=rejects)
                # make the candidate a consistent set by replacing the rejected samples with
                # copies of the originals from {θ}
                invalids = rejects != 0
                cθ[invalids] = θ[invalids]
                if reparameterized:
                    cθs[invalids] = θs[invalids]
                # notify that the verification process is finished
                dispatcher.notify(event=dispatcher.verify_finish, controller=annealer)

                # compute the likelihoods
                model.likelihoods(annealer=annealer, step=candidate)

                # the difference of the two posterior likelihoods
                diff = cpost - posterior
                # and, in sampling space, the difference of the log-jacobians
                if reparameterized:
                    cjacobian[...] = 0
                    model.eval_prior_with_physical(step=candidate, likelihood=cjacobian)
                    diff += cjacobian - jacobian
                # roll the Metropolis dice, in (0, 1]
                dice = 1.0 - self.rng.random(size=samples)

                # notify we are starting accepting samples
                dispatcher.notify(event=dispatcher.accept_start, controller=annealer)

                # accept/reject: a candidate is invalid if the model considered it outside its
                # support, rejected if it was less likely than the original and it wasn't saved
                # by the {dice}, and accepted otherwise
                unlucky = numpy.log(dice) > diff
                rejections = ~invalids & unlucky
                accepts = ~invalids & ~unlucky
                # update the counts
                invalid += int(invalids.sum())
                rejected += int(rejections.sum())
                accepted += int(accepts.sum())
                # copy the accepted candidates, and their likelihoods
                θ[accepts] = cθ[accepts]
                if reparameterized:
                    θs[accepts] = cθs[accepts]
                    jacobian[accepts] = cjacobian[accepts]
                for current, proposed in ((prior, cprior), (data, cdata), (posterior, cpost)):
                    current[accepts] = proposed[accepts]

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
    proposal: Proposal       # the proposal mechanism used by this sampler
    stepsizer: StepSizer     # the step size regulator
    stepcounter: StepCounter # the step count regulator

    scaling: float = 0.1     # current proposal scaling; updated by stepsizer after each update
    statistics: Statistics | None = None  # from the last walk_chains call

    info: journal.info | None = None  # the application info channel
    rng: numpy.random.Generator       # the generator of the dice
    dispatcher: typing.Any = None     # a reference to the event dispatcher


# end of file
