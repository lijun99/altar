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


    def restart(self, annealer: Annealer) -> typing.Self:
        """
        Start the annealing process from a checkpoint
        """
        # NYI
        raise NotImplementedError()


    def top(self, annealer: Annealer) -> typing.Self:
        """
        Notification that we are at the beginning of an update
        """
        # notify the model
        annealer.model.top(annealer=annealer)
        # notify the archiver
        annealer.archiver.top(step=self.step, iteration=self.iteration, psets=annealer.model.psets)

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
        annealer.archiver.recordstep(step=self.step, stats=info, psets=annealer.model.psets)
        # all done
        return self


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
        annealer.archiver.bottom(step=self.step, iteration=self.iteration, psets=annealer.model.psets)

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
        # ask the recorder to record it
        annealer.archiver.final(step=step, iteration=None, psets=annealer.model.psets)
        # all done
        return self


    # whether i keep a pool of the states of the chains
    pools: bool = False


    # meta-methods
    def __init__(self, annealer: Annealer, **kwds) -> None:
        # chain up; absorb the {annealer}
        super().__init__(**kwds)
        # all done
        return


# end of file
