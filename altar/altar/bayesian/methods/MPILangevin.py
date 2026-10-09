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
import mpi
import journal
import numpy
# the framework
import altar
# superclass
from .LangevinMethod import LangevinMethod
# moving the chains among the processes
from .exchange import collect, excerpt

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Langevin import Langevin
    from altar.bayesian.states.LangevinStep import LangevinStep
    from altar.shells.Application import Application


# declaration
class MPILangevin(LangevinMethod):
    """
    A distributed implementation of the Langevin method that uses MPI
    """


    # interface
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # chain up
        super().initialize(application=application)

        # give my rank a random stream of its own
        application.rng.reseed(rank=self.rank)

        # show me
        application.info.log(f"mpi annealing: worker {self.wid} out of total {self.workers}, {self.worker}")

        # initialize worker
        self.worker.wid = self.rank
        self.worker.initialize(application=application)

        # turn off info channel for non-managers
        if self.rank != self.manager:
            application.info.active = False

        # all done
        return self


    def start(self, controller: Langevin) -> typing.Any:
        """
        Start the annealing process
        """
        # chain up
        super().start(controller=controller)
        # everybody has to get ready
        self.worker.start(controller=controller)
        # collect the global state: at the master task, I get the entire state of the problem;
        # at the other tasks, I just get a reference to the local state so I have uniform
        # access to the annealing temperature
        self.step = self.collect()
        # all done
        return self


    def top(self, controller: Langevin) -> typing.Any:
        """
        Notification that we are at the beginning of a β update
        """
        # if i am the manager
        if self.rank == self.manager:
            # chain up
            return super().top(controller=controller)
        # otherwise, do nothing
        return self


    def cool(self, controller: Langevin) -> typing.Any:
        """
        Push my state forward along the cooling schedule
        """
        # if I am the manager
        if self.rank == self.manager:
            # i have the global state; cool it
            super().cool(controller=controller)
        # all done
        return self


    def walk(self, controller: Langevin) -> typing.Any:
        """
        Explore configuration space by walking the Markov chains
        """
        # partition and synchronize my state
        self.partition()
        # all workers walk their chains
        stats = self.worker.walk(controller=controller)
        # collect my state
        self.step = self.collect()
        # return the statistics
        return stats


    def rate_statistics(self, controller: Langevin) -> typing.Any:
        """
        The statistics {estimate_rate} needs, pooled over the chains of every task, so that
        all tasks agree on the sampling rate
        """
        comm = self.communicator
        # my worker's own
        n, sum_x, sum_x2, max_gradient = self.worker.rate_statistics(controller=controller)
        # add them up across the tasks
        n = int(comm.sum(int(n)))
        sum_x = numpy.array([comm.sum(float(v)) for v in sum_x])
        sum_x2 = numpy.array([comm.sum(float(v)) for v in sum_x2])
        max_gradient = comm.max(float(max_gradient))
        # all done
        return n, sum_x, sum_x2, max_gradient


    def resample(self, controller: Langevin, statistics: typing.Any) -> typing.Any:
        """
        Analyze the acceptance statistics and take the problem state to the end of the
        annealing step
        """
        # who is the boss?
        manager = self.manager
        # unpack the acceptance/rejection statistics
        accepted, invalid, rejected = statistics

        # add up the acceptance/rejection statistics from all the nodes
        accepted = int(self.communicator.sum(accepted))
        invalid = int(self.communicator.sum(invalid))
        rejected = int(self.communicator.sum(rejected))

        # chain up
        statistics = super().resample(controller=controller, statistics=(accepted,invalid,rejected))

        # all done
        return statistics

    def archive(self, controller: Langevin, scaling: float, stats: typing.Any) -> typing.Self:
        """
        Notify archiver to record controller information
        """
        # if i am the manager
        if self.rank == self.manager:
            super().archive(controller=controller, scaling=scaling, stats=stats)
        # otherwise, do nothing
        return self

    def bottom(self, controller: Langevin) -> typing.Any:
        """
        Notification that we are at the end of a β update
        """
        # if i am the manager
        if self.rank == self.manager:
            # chain up
            super().bottom(controller=controller)
        # otherwise, do nothing
        return self


    def finish(self, controller: Langevin) -> typing.Any:
        """
        Shut down the annealing process
        """
        # if i am the manager
        if self.rank == self.manager:
            # chain up
            return super().finish(controller=controller)
        # otherwise, do nothing
        return self


    # for cuda worker
    @property
    def device(self) -> typing.Any:
        return self.worker.device

    @property
    def gstep(self) -> typing.Any:
        return self.worker.gstep


    # meta-methods
    def __init__(self, controller: Langevin, worker: LangevinMethod,
                 communicator: typing.Any = None, **kwds) -> None:
        # chain up
        super().__init__(controller=controller, **kwds)

        # make sure i have a valid communicator
        comm = communicator or mpi.world
        # attach it
        self.communicator = comm
        # store the number of tasks
        self.tasks = comm.size
        # and my rank
        self.rank = comm.rank

        # save the annealing method for each of my tasks
        self.worker = worker
        # assign them a worker id
        self.worker.wid = self.rank
        self.wid = self.rank

        # compute the total number workers
        workers = comm.sum(item=worker.workers)
        # the result is meaningful only on the manager task
        self.workers = int(workers)

        # all done
        return


    # implementation details
    def collect(self) -> LangevinStep:
        """
        Assemble my global state
        """
        # get the communicator
        communicator = self.communicator
        # who's the boss?
        manager = self.manager
        # ask my worker for its local state
        step = self.worker.step
        # get the temperature
        β = step.beta
        # assemble the sample set
        θ = collect(array=step.theta, communicator=communicator, destination=manager)
        # the likelihoods
        prior = collect(array=step.prior, communicator=communicator, destination=manager)
        data = collect(array=step.data, communicator=communicator, destination=manager)
        posterior = collect(array=step.posterior, communicator=communicator, destination=manager)
        # the gradients, so the archived state reflects the last sweep, not zeros; a cuda
        # worker's host copy of its state has none
        grad_prior = grad_data = None
        if hasattr(step, "grad_prior"):
            grad_prior = collect(array=step.grad_prior, communicator=communicator, destination=manager)
            grad_data = collect(array=step.grad_data, communicator=communicator, destination=manager)

        # if I am not the manager task
        if self.rank != self.manager:
            # just return the local state
            return step

        # the manager packs the state of the problem, with the current sampling rate, and returns it
        collected = self.LangevinStep(
            beta=β, theta=θ, likelihoods=(prior,data,posterior),
            gradients=None if grad_prior is None else (grad_prior,grad_data))
        collected.epsilon_t = step.epsilon_t
        return collected


    def partition(self) -> LangevinStep:
        """
        Distribute my global state
        """
        # who is the boss
        manager = self.manager
        # am i the boss?
        if self.rank == manager:
            # grab my global state
            step = self.step
            # unpack it
            β = step.beta
            θ = step.theta
            prior = step.prior
            data = step.data
            posterior = step.posterior
        # the others
        else:
            # know nothing
            β = θ = prior = data = posterior = None

        # cache my communicator
        comm = self.communicator
        # the partitioning modifies my local state, which kept on my behalf by the manager of
        # my local workers
        step = self.worker.step

        # everybody gets the temperature
        step.beta = comm.bcast(item=β, source=manager)

        # it is important not to disturb the memory held by the manager: threaded managers have
        # their workers set up views on the local state and we don't want to mess that up

        # grab my portion of the sample set
        excerpt(target=step.theta, array=θ, source=manager, communicator=comm)
        # my portion of the likelihoods
        excerpt(target=step.prior, array=prior, source=manager, communicator=comm)
        excerpt(target=step.data, array=data, source=manager, communicator=comm)
        excerpt(target=step.posterior, array=posterior, source=manager, communicator=comm)

        # all done
        return step


    # private data
    manager: int = 0 # the rank responsible for distributing and collecting the workload
    worker: LangevinMethod | None = None # the annealing method implementation; deduced at start up time


# end of file
