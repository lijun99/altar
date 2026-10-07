# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
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
class Annealer(altar.component, family="altar.controllers.annealer", implements=controller):
    """
    A Bayesian controller that uses an annealing schedule and MCMC to approximate the posterior
    distribution of a model
    """


    # user configurable state
    sampler = altar.bayesian.sampler()
    sampler.doc = "the sampler of the posterior distribution"

    scheduler = altar.bayesian.scheduler()
    scheduler.doc = "the generator of the annealing schedule"

    dispatcher = altar.simulations.dispatcher(default=Notifier)
    dispatcher.doc = "the event dispatcher that activates the registered handlers"

    archiver = altar.simulations.archiver(default=Recorder)
    archiver.doc = "the archiver of simulation state"

    pool = altar.properties.int(default=1)
    pool.validators = altar.constraints.isGreaterEqual(value=1)
    pool.doc = "the states each chain keeps from a β step, its last ones, {pool_interval} " \
               "steps apart: the population the scheduler, the resampling and the proposal " \
               "work on, and that the archiver records (gpu only); 1, the default, keeps the " \
               "final states only, which turns pooling off"

    pool_interval = altar.properties.int(default=1)
    pool_interval.validators = altar.constraints.isGreaterEqual(value=1)
    pool_interval.doc = "the MC steps, or HMC trajectories, between the states a chain keeps"


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

        # deduce my annealing method
        self.worker = self.deduce_annealing_method(job=application.job)
        # and initialize it
        self.worker.initialize(application=application)

        # initialize the archiver first: components initialized below (e.g. the sampler's
        # proposal) may self-register with it, and registration must land on the archiver's
        # own post-initialize() state, not be wiped out by it running afterwards
        self.archiver.initialize(application=application)
        # initialize my other parts
        self.sampler.initialize(application=application)
        self.scheduler.initialize(application=application)

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

        # notify all interested parties that the simulation is about to start
        dispatcher.notify(event=dispatcher.start, controller=self)
        # start the process
        # initialize samples
        worker.start(annealer=self)
        # collect and record samples
        worker.archive(annealer=self, scaling=self.sampler.scaling, stats=(0,0,0))
        # bottom process: compute mean,sd and print a summary
        worker.bottom(annealer=self)

        # iterate until done, by default until β is sufficiently close to one; the count of
        # iterations is kept here, since under mpi only the manager's worker counts them
        iteration = 0
        while self.continuing(worker=worker, iteration=iteration, tolerance=tolerance):
            # count this one
            iteration += 1
            # notify that we are at the top of the current step
            dispatcher.notify(event=dispatcher.beta_start, controller=self)

            # compute a new temperature
            # resampling and distribute the samples
            # compute new covariance matrix for proposal
            worker.cool(annealer=self)

            # worker procedures before sampling starts
            worker.top(annealer=self)

            # notify we are about to walk the chains
            dispatcher.notify(event=dispatcher.walk_chains_start, controller=self)
            # walk the chains
            statistics = worker.walk(annealer=self)
            # notify we are done walking the chains
            dispatcher.notify(event=dispatcher.walk_chains_finish, controller=self)

            # notify we are about to resample
            dispatcher.notify(event=dispatcher.resample_start, controller=self)
            # resample: this only adjusts the scaling factor of proposal matrix
            # better use another name
            statistics = worker.resample(annealer=self, statistics=statistics)
            # notify we are done resampling
            dispatcher.notify(event=dispatcher.resample_finish, controller=self)

            # ask archiver to record statistics information
            worker.archive(annealer=self, scaling=self.sampler.scaling, stats=statistics)

            # notify the worker we are at the bottom of the current step
            # worker procedures after the sampling ends
            # e.g., print out the statistics, calculate the mean model in Cp
            worker.bottom(annealer=self)
            # and dispatch the matching event
            dispatcher.notify(event=dispatcher.beta_finish, controller=self)

        # and finish up
        worker.finish(annealer=self)
        # notify all interested parties that the simulation has finished
        dispatcher.notify(event=dispatcher.finish, controller=self)

        # forget the model
        self.model = None

        # all done; indicate success
        return 0


    def continuing(self, worker, iteration, tolerance):
        """
        Whether to take another step, after {iteration} of them: until β is within {tolerance}
        of one
        """
        return worker.beta + tolerance < 1


    # implementation details
    def deduce_annealing_method(self, job):
        """
        Instantiate an annealing method compatible the user choices
        """
        # the machine layout part of the {job} parameters has already been vetted: one task
        # per host without mpi, and at most one gpu per task; unpack the parameters we use
        mode = job.mode
        gpus = job.gpus

        # first let's figure out the base worker factory: if the user asked for gpus and we
        # have them, go CUDA, else use plain vanilla sequential
        worker = self.cuda if gpus > 0 else self.sequential
        # ask the factory for a worker instance
        worker = worker()

        # if we are running under mpi, wrap it in the mpi aware annealing method
        if mode == "mpi":
            worker = self.mpi(worker=worker)

        # all done
        return worker


    def sequential(self):
        """
        Instantiate the plain sequential annealing method
        """
        # import the sequential annealer
        from ..methods.SequentialAnnealing import SequentialAnnealing
        # instantiate it and return it
        return SequentialAnnealing(annealer=self)


    def cuda(self):
        """
        Instantiate a CUDA aware annealing method
        """
        # import the CUDA annealing method
        from ..methods.CUDAAnnealing import CUDAAnnealing
        # instantiate it and return it
        return CUDAAnnealing(annealer=self)


    def mpi(self, worker):
        """
        Instantiate the MPI aware annealing method
        """
        from ..methods.MPIAnnealing import MPIAnnealing
        # instantiate it and return it
        return MPIAnnealing(annealer=self, worker=worker)


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
