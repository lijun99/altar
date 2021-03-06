# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#


# declaration
class AnnealingMethod:
    """
    Base class for the various annealing implementation strategies
    """


    # types
    from .CoolingStep import CoolingStep


    # public data
    step = None # the current state of the solver
    iteration = 0 # my iteration counter

    wid = 0 # my worker id
    workers = None # the total number of chain processors

    @property
    def beta(self):
        """
        Return the temperature of my current step
        """
        # easy enough
        return self.step.beta


    # interface
    def initialize(self, application):
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


    def start(self, annealer):
        """
        Start the annealing process from scratch
        """
        # reset my iteration count
        self.iteration = 0
        # all done
        return self


    def restart(self, annealer):
        """
        Start the annealing process from a checkpoint
        """
        # NYI
        raise NotImplementedError()


    def top(self, annealer):
        """
        Notification that we are at the beginning of an update
        """
        # notify the model
        annealer.model.top(annealer=annealer)

        # all done
        return self


    def cool(self, annealer):
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


    def walk(self, annealer):
        """
        Explore configuration space by walking the Markov chains
        """
        # get the sampler
        sampler = annealer.sampler
        # ask it to sample the posterior pdf
        stats = sampler.samplePosterior(annealer=annealer, step=self.step)
        # return the acceptance statistics
        return stats


    def resample(self, annealer, statistics):
        """
        Analyze the acceptance statistics and take the problem state to the end of the
        annealing step
        """
        # get the sampler
        sampler = annealer.sampler
        # ask it to adjust the sample statistics
        sampler.resample(annealer=annealer, statistics=statistics)
        # all done
        return self

    def archive(self, annealer, scaling, stats):
        """
        Notify archiver to record
        """
        info={'iteration': self.iteration,
                    'beta' : self.beta,
                    'scaling' : scaling,
                    'stats' : stats}
        channel = annealer.info;
        channel.log(f"iteration: {info['iteration']}, beta: {info['beta']}, scaling: {info['scaling']}")
        channel.log(f"stats(accepted/invalid/rejected): {info['stats']}")
        annealer.archiver.recordstep(step=self.step, stats=info, psets=annealer.model.psets)
        # all done
        return self


    def bottom(self, annealer):
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

        # all done
        return self


    def finish(self, annealer):
        """
        Notification that the simulation is over
        """
        # get the state of the solution
        step = self.step
        # ask it to render itself to the screen
        step.print(channel=annealer.info)
        # ask the recorder to record it
        annealer.archiver.record(step=step, iteration=self.iteration, psets=annealer.model.psets)
        # all done
        return self


    # meta-methods
    def __init__(self, annealer, **kwds):
        # chain up; absorb the {annealer}
        super().__init__(**kwds)
        # all done
        return


# end of file
