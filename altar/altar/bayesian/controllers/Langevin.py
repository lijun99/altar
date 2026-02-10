# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# the package
import altar
# my protocol
from .Controller import Controller as controller
# my event dispatcher
from ..monitoring.Notifier import Notifier
# my archiver
from ..archivers.Recorder import Recorder


# my declaration
class Langevin(altar.component, family="altar.controllers.langevin", implements=controller):
    """
    A Bayesian controller that uses Langevin Dynamics Based algorithms
    """

    dispatcher = altar.simulations.dispatcher(default=Notifier)
    dispatcher.doc = "the event dispatcher that activates the registered handlers"

    archiver = altar.simulations.archiver(default=Recorder)
    archiver.doc = "the archiver of simulation state"

    scheduler = altar.bayesian.langevinscheduler()
    scheduler.doc = "the scheduler for epsilon_t"

    tsteps = altar.properties.int(default=100)
    tsteps.doc = "time steps"

    tsteps_report = altar.properties.int(default=None)
    tsteps_report.doc = "number of tsteps to create a report"

    sweeps = altar.properties.int(default=1)
    sweeps.doc = "number of sweeps at a fixed t"

    # public data
    epsilon_t = None

    # protocol obligations
    @altar.export
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

        # initialize the dispatcher
        self.dispatcher.initialize(application=application)

        # initialize scheduler
        self.scheduler.initialize(application=application)

        # deduce my annealing method
        self.worker = self.deduceAnnealingMethod(job=application.job)
        # and initialize it
        self.worker.initialize(application=application)

        self.tsteps_report = self.tsteps_report or self.tsteps//10

        # initialize my other parts
        self.archiver.initialize(application=application)

        # go through the registered monitors
        for monitor in application.monitors.values():
            # initialize them
            monitor.initialize(application=application)
            # and register them with the {dispatcher}
            self.dispatcher.register(monitor=monitor)

        # all done
        return self


    @altar.export
    def posterior(self, model):
        """
        Sample the posterior distribution
        """
        # record the model so that everybody has easy access to it
        self.model = model
        # unpack what we need
        tolerance = model.job.tolerance
        # get my worker
        worker = self.worker
        # and my dispatcher
        dispatcher = self.dispatcher
        # and my scheduler
        scheduler = self.scheduler

        # notify all interested parties that the simulation is about to start
        dispatcher.notify(event=dispatcher.start, controller=self)
        # start the process
        # initialize samples
        worker.start(controller=self)
        # collect and record samples
        # worker.archive(controller=self, scaling=self.sampler.scaling, stats=(0,0,0))
        # bottom process: compute mean,sd and print a summary
        worker.bottom(controller=self)

        # to estimate initial sampling rate, if requested
        scheduler.start(controller=self)

        # iterate t to tsteps
        for t in range(self.tsteps):
            # step size
            self.epsilon_t = scheduler.epsilon_t(t)
            # walk the chains
            worker.walk(controller=self)
            # e.g., print out the statistics, calculate the mean model in Cp
            worker.bottom(controller=self)

        # and finish up
        worker.finish(controller=self)

        # forget the model
        self.model = None

        # all done; indicate success
        return 0


    # implementation details
    def deduceAnnealingMethod(self, job):
        """
        Instantiate an annealing method compatible the user choices
        """
        # the machine layout part of the {job} parameters has already been vetted; if we get
        # this far, we have what the user asked for; unpack the parameters we use
        mode = job.mode
        hosts = job.hosts
        tasks = job.tasks
        gpus = job.gpus

        # first let's figure out the base worker factory: if the user asked for gpus and we
        # have them, go CUDA, else use plain vanilla sequential

        # currently, only simple gpu is supported
        worker = self.cuda
        # ask the factory for a worker instance
        worker = worker()

        # all done
        return worker


    def cuda(self):
        """
        Instantiate a CUDA aware annealing method
        """
        # import the CUDA annealing method
        from ..methods.CUDASGLD import CUDASGLD
        # instantiate it and return it
        return CUDASGLD(controller=self)


    # private data
    model = None  # the model i'm sampling
    worker = None # the annealing method
    # journal channels shared with the application
    info = None
    warning = None
    error = None
    debug = None
    firewall = None


# end of file
