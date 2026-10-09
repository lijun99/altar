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
# the framework
import altar
# superclass
from .AnnealingMethod import AnnealingMethod
# moving the chains among the processes
from .exchange import collect, excerpt

if typing.TYPE_CHECKING:
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.bayesian.states.CoolingStep import CoolingStep
    from altar.shells.Application import Application


# declaration
class MPIAnnealing(AnnealingMethod):
    """
    A distributed implementation of the annealing method that uses MPI
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


    def start(self, annealer: Annealer) -> typing.Any:
        """
        Start the annealing process
        """
        # chain up
        super().start(annealer=annealer)
        # everybody has to get ready
        self.worker.start(annealer=annealer)
        # collect the global state: at the master task, I get the entire state of the problem;
        # at the other tasks, I just get a reference to the local state so I have uniform
        # access to the annealing temperature
        self.step = self.collect()
        # notify the archiver on the manager
        if self.rank == self.manager:
            annealer.archiver.start(step=self.step, iteration=self.iteration, psets=annealer.model.psets)
        # all done
        return self


    def top(self, annealer: Annealer) -> typing.Any:
        """
        Notification that we are at the beginning of a β update
        """
        # if i am the manager
        if self.rank == self.manager:
            # chain up
            return super().top(annealer=annealer)
        # otherwise, do nothing
        return self


    def cool(self, annealer: Annealer) -> typing.Any:
        """
        Push my state forward along the cooling schedule
        """
        # if I am the manager
        if self.rank == self.manager:
            # i have the global state; cool it
            super().cool(annealer=annealer)
        # all done
        return self


    def walk(self, annealer: Annealer) -> typing.Any:
        """
        Explore configuration space by walking the Markov chains
        """
        # partition and synchronize my state
        self.partition()
        # all workers walk their chains
        stats = self.worker.walk(annealer=annealer)
        # collect my state
        self.step = self.collect()
        # return the statistics
        return stats


    def resample(self, annealer: Annealer, statistics: tuple[int, int, int]) -> typing.Any:
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
        statistics = super().resample(annealer=annealer, statistics=(accepted,invalid,rejected))

        # all done
        return statistics

    def archive(self, annealer: Annealer, scaling: float, stats: typing.Any) -> typing.Self:
        """
        Notify archiver to record annealer information
        """
        # if i am the manager
        if self.rank == self.manager:
            super().archive(annealer=annealer, scaling=scaling, stats=stats)
        # otherwise, do nothing
        return self

    def bottom(self, annealer: Annealer) -> typing.Any:
        """
        Notification that we are at the end of a β update
        """
        # if i am the manager
        if self.rank == self.manager:
            # chain up
            super().bottom(annealer=annealer)
        # otherwise, do nothing
        return self


    def finish(self, annealer: Annealer) -> typing.Any:
        """
        Shut down the annealing process
        """
        # if i am the manager
        if self.rank == self.manager:
            # chain up
            return super().finish(annealer=annealer)
        # otherwise, do nothing
        return self


    # for cuda worker
    @property
    def device(self) -> typing.Any:
        return self.worker.device

    @property
    def gstep(self) -> typing.Any:
        return self.worker.gstep

    # my worker's pool, which the samplers reach through me; each rank pools its own chains
    @property
    def pool(self) -> typing.Any:
        return getattr(self.worker, "pool", None)

    @property
    def pools(self) -> bool:
        return getattr(self.worker, "pools", False)


    # meta-methods
    def __init__(self, annealer: Annealer, worker: AnnealingMethod,
                 communicator: typing.Any = None, **kwds) -> None:
        # chain up
        super().__init__(annealer=annealer, **kwds)

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
    def collect(self) -> CoolingStep:
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

        # and, when reparameterized, the sampling space and the jacobian
        reparameterized = getattr(step, "has_reparametrization", False)
        θ_sampling = jacobian = None
        if reparameterized:
            θ_sampling = collect(
                array=step.theta_sampling, communicator=communicator, destination=manager)
            jacobian = collect(array=step.jacobian, communicator=communicator, destination=manager)

        # if I am not the manager task
        if self.rank != self.manager:
            # just return the local state
            return step

        # the manager packs the state of the problem and returns it
        return self.CoolingStep(
            beta=β, theta=θ, theta_sampling=θ_sampling, jacobian=jacobian,
            likelihoods=(prior,data,posterior), has_reparametrization=reparameterized)


    def partition(self) -> CoolingStep:
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
            θ_sampling, jacobian = step.theta_sampling, step.jacobian
        # the others
        else:
            # know nothing
            β = θ = prior = data = posterior = θ_sampling = jacobian = None

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
        # and the sampling space and the jacobian, when reparameterized
        if getattr(step, "has_reparametrization", False):
            excerpt(target=step.theta_sampling, array=θ_sampling, source=manager, communicator=comm)
            excerpt(target=step.jacobian, array=jacobian, source=manager, communicator=comm)

        # NOTE: the proposal covariance Σ used to be broadcast here, back when it lived on
        # {step.sigma}; it now lives in GaussianProposal, computed per-rank from each rank's
        # local {step.theta}/{step.weights} after this partition. That means MPI ranks running
        # GaussianProposal will independently derive different Σ from their own local sample
        # shard instead of agreeing on one global covariance -- a known distributed-correctness
        # gap, left for the worker/parallelism generalization milestone to resolve.

        # all done
        return step


    # private data
    manager = 0 # the rank responsible for distributing and collecting the workload
    worker = None # the annealing method implementation; deduced at start up time


# end of file
