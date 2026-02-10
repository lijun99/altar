# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

from datetime import datetime

# declaration
class LangevinMethod:
    """
    Base class for the various Langevin implementation strategies (workers)
    """


    # types
    from ..states.LangevinStep import LangevinStep


    # public data
    step = None # the current state of the solver
    iteration = 0 # my iteration counter

    wid = 0 # my worker id
    workers = None # the total number of chain processors


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


    def start(self, controller):
        """
        Start the annealing process from scratch
        """
        # reset my iteration count
        self.iteration = 0
        # all done
        return self


    def restart(self, controller):
        """
        Start the annealing process from a checkpoint
        """
        # NYI
        raise NotImplementedError()


    def top(self, controller):
        """
        Notification that we are at the beginning of an update
        """
        # notify the model
        controller.model.top(controller=controller)

        # all done
        return self


    def cool(self, controller):
        """
        Push my state forward along the cooling schedule
        """
        # get the scheduler
        scheduler = controller.scheduler
        # ask it to update my step
        scheduler.update(step=self.step)
        # update the iteration counter
        self.iteration += 1
        # all done
        return self


    def walk(self, controller):
        """
        Explore configuration space by walking the Markov chains
        """
        # get the sampler
        sampler = controller.sampler
        # ask it to sample the posterior pdf
        stats = sampler.samplePosterior(controller=controller, step=self.step)
        # return the acceptance statistics
        return stats


    def resample(self, controller, statistics):
        """
        Analyze the acceptance statistics and take the problem state to the end of the
        annealing step
        """
        # get the sampler
        sampler = controller.sampler
        # ask it to adjust the sample statistics
        sampler.resample(controller=controller, statistics=statistics)
        # all done
        return statistics

    def archive(self, controller, scaling, stats):
        """
        Notify archiver to record
        """
        info={'iteration': self.iteration,
                    'beta' : self.beta,
                    'scaling' : scaling,
                    'stats' : stats}
        channel = controller.info;
        channel.log(f"time: {datetime.now().isoformat()}")
        channel.log(f"iteration: {info['iteration']}, beta: {info['beta']}, scaling: {info['scaling']}")
        channel.log(f"stats(accepted/invalid/rejected): {info['stats']}")
        controller.archiver.recordstep(step=self.step, stats=info, psets=controller.model.psets)
        # all done
        return self


    def bottom(self, controller):
        """
        Notification that we are at the bottom of an update
        """
        # notify the model
        # controller.model.bottom(controller=controller)

        if self.wid == 0: # only master
            # get the state of the solution
            step = self.step
            # calculate the statistics of samples
            step.statistics()
            # print a summary of current state
            step.print(channel=controller.info)

        # all done
        return self


    def finish(self, controller):
        """
        Notification that the simulation is over
        """
        # get the state of the solution
        step = self.step
        # ask it to render itself to the screen
        step.print(channel=controller.info)
        # ask the recorder to record it
        controller.archiver.record(step=step, iteration=self.iteration, psets=controller.model.psets)
        # all done
        return self


    # meta-methods
    def __init__(self, controller, **kwds):
        # chain up; absorb the {controller}
        super().__init__(**kwds)
        # all done
        return


# end of file
