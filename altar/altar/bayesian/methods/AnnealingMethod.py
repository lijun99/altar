# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

from __future__ import annotations
import typing
from datetime import datetime

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.methods.Diagnostics import Diagnostics
    from altar.shells.Application import Application

# declaration
class AnnealingMethod:
    """
    Base class for the various annealing implementation strategies
    """


    # types
    from ..states.CoolingStep import CoolingStep


    # public data
    step: CoolingStep | None = None # the current state of the solver
    iteration: int = 0 # my iteration counter

    wid: int = 0 # my worker id
    workers: int | None = None # the total number of chain processors

    @property
    def beta(self) -> float:
        """
        Return the temperature of my current step
        """
        # easy enough
        return self.step.beta


    # interface
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # borrow the canonical journal channels from the application
        self.info = application.info
        self.warning = application.warning
        self.error = application.error
        self.debug = application.debug
        self.firewall = application.firewall

        # all done
        return self


    def start(self, annealer: Annealer) -> typing.Self:
        """
        Start the annealing process from scratch
        """
        # reset my iteration count
        self.iteration = 0
        # a pool of the states the chains keep needs a method that keeps them
        if getattr(annealer, "pool", 1) > 1 and not self.pools:
            raise NotImplementedError(
                f"controller.pool = {annealer.pool} needs the gpu (job.gpus = 1)")
        # all done
        return self


    def restart(self, annealer: Annealer, checkpoint: typing.Any,
                share: tuple[int, int] | None = None) -> typing.Self:
        """
        Start the annealing process from a {checkpoint}, a step an earlier run archived; my
        rows of its population are {share}, (the population size, my first row), all of it
        by default
        """
        # the same checks as a fresh start
        AnnealingMethod.start(self, annealer=annealer)
        # and carry on counting from the step
        self.iteration = checkpoint.iteration
        # all done
        return self


    def produce(self, annealer: Annealer) -> typing.Self:
        """
        Prepare the production walk, at the current β, without resampling: its pool keeps states
        evenly spread over its {annealer.production} steps, and the sampler adapts to the current
        population, not to the importance weights of the last step
        """
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.spread = False
            pool.interval = max(1, annealer.production // pool.size)
        if self.wid == 0:
            self.iteration += 1
            self.step.weights = self.step.weighted_theta = self.step.ancestors = None
            # the population the chains walk from
            diagnostics = self.diagnose(annealer=annealer)
            if diagnostics is not None:
                diagnostics.begin(θ=getattr(self.step, "theta_sampling", self.step.theta))
        return self


    def top(self, annealer: Annealer) -> typing.Self:
        """
        Notification that we are at the beginning of an update
        """
        # notify the model
        annealer.model.top(annealer=annealer)
        # notify the archiver
        annealer.archiver.top(step=self.step, iteration=self.iteration, psets=getattr(annealer.model, "psets", None))

        # all done
        return self


    def cool(self, annealer: Annealer) -> typing.Self:
        """
        Push my state forward along the cooling schedule
        """
        # get the scheduler
        scheduler = annealer.scheduler
        # ask it to update my step
        scheduler.update(step=self.step)
        # update the iteration counter
        self.iteration += 1
        # the population the chains walk from, and the rows it was resampled from
        diagnostics = self.diagnose(annealer=annealer)
        if diagnostics is not None:
            if self.step.ancestors is not None:
                diagnostics.resampled(rows=self.step.ancestors)
            diagnostics.begin(θ=getattr(self.step, "theta_sampling", self.step.theta))
        # all done
        return self


    def walk(self, annealer: Annealer) -> typing.Any:
        """
        Explore configuration space by walking the Markov chains
        """
        # let the model update anything that depends on the samples, e.g. its C_p
        if annealer.model.update_model(annealer=annealer, step=self.step):
            self.densities(annealer=annealer)
        # get the sampler
        sampler = annealer.sampler
        # ask it to sample the posterior pdf
        stats = sampler.sample_posterior(annealer=annealer, step=self.step)
        # return the acceptance statistics
        return stats


    def densities(self, annealer: Annealer) -> typing.Self:
        """
        Recompute the prior, data and posterior densities of my step, after the model changed
        """
        step = self.step
        step.prior[...] = 0
        step.data[...] = 0
        step.posterior[...] = 0
        annealer.model.likelihoods(annealer=annealer, step=step)
        return self


    def resample(self, annealer: Annealer, statistics: typing.Any) -> typing.Any:
        """
        Analyze the acceptance statistics and take the problem state to the end of the
        annealing step
        """
        # get the sampler
        sampler = annealer.sampler
        # ask it to adjust the sample statistics
        sampler.update(annealer=annealer, statistics=statistics)
        # all done
        return statistics

    def archive(self, annealer: Annealer, scaling: float, stats: typing.Any) -> typing.Self:
        """
        Notify archiver to record
        """
        info={'iteration': self.iteration,
                    'beta' : self.beta,
                    'scaling' : scaling,
                    'stats' : stats}
        channel = annealer.info;
        channel.log(f"time: {datetime.now().isoformat()}")
        channel.log(f"iteration: {info['iteration']}, beta: {info['beta']}, scaling: {info['scaling']}")
        channel.log(f"stats(accepted/invalid/rejected): {info['stats']}")
        # how well the walk mixed
        diagnostics = self.diagnose(annealer=annealer)
        if diagnostics is not None and diagnostics.start is not None:
            pool = getattr(self, "pool", None)
            walk = diagnostics.end(θ=getattr(self.step, "theta_sampling", self.step.theta),
                final=0 if pool is None else pool.final, iteration=self.iteration, beta=self.beta)
            channel.log(f"diagnostics: parents {walk['parents']}, lineages {walk['lineages']}, "
                        f"correlation with the start median {walk['correlation']:.3f}, "
                        f"max {walk['correlation_max']:.3f}"
                        + (f", effective states per chain {walk['states']:.2f}"
                           if diagnostics.pool > 1 else ""))
        self.step.log_evidence = getattr(annealer.scheduler, "log_evidence", None)
        annealer.archiver.recordstep(step=self.step, stats=info, psets=getattr(annealer.model, "psets", None))
        # all done
        return self


    def diagnose(self, annealer: Annealer) -> Diagnostics | None:
        """
        My diagnostics of the walks, made on first use, on the manager; none if my population is
        not the chains of every task, {pool} states each
        """
        if self.diagnostics is None:
            tasks = getattr(self, "tasks", 1)
            chains = annealer.model.job.chains
            pool, remainder = divmod(self.step.samples, tasks * chains)
            if remainder or pool != getattr(annealer, "pool", 1):
                return None
            from .Diagnostics import Diagnostics
            self.diagnostics = Diagnostics(tasks=tasks, chains=chains, pool=pool)
            annealer.archiver.register(self.diagnostics)
        return self.diagnostics


    def bottom(self, annealer: Annealer) -> typing.Self:
        """
        Notification that we are at the bottom of an update
        """
        # notify the model
        annealer.model.bottom(annealer=annealer)

        if self.wid == 0: # only master
            # get the state of the solution
            step = self.step
            # calculate the statistics of samples
            step.statistics()
            # print a summary of current state
            step.print(channel=annealer.info)

        # notify the archiver
        annealer.archiver.bottom(step=self.step, iteration=self.iteration, psets=getattr(annealer.model, "psets", None))

        # all done
        return self


    def finish(self, annealer: Annealer) -> typing.Self:
        """
        Notification that the simulation is over
        """
        # get the state of the solution
        step = self.step
        # ask it to render itself to the screen
        step.print(channel=annealer.info)
        # the evidence, from a scheduler that estimates it
        step.log_evidence = getattr(annealer.scheduler, "log_evidence", None)
        if step.log_evidence is not None:
            annealer.info.log(f"log evidence: {step.log_evidence}")
        # ask the recorder to record it
        annealer.archiver.final(step=step, iteration=None, psets=getattr(annealer.model, "psets", None))
        # all done
        return self


    # whether i keep a pool of the states of the chains
    pools: bool = False
    # how well the walks mix, kept by the manager
    diagnostics: Diagnostics | None = None


    # meta-methods
    def __init__(self, annealer: Annealer, **kwds) -> None:
        # chain up; absorb the {annealer}
        super().__init__(**kwds)
        # all done
        return


# end of file
