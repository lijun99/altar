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
# and the protocols
from .controllers.Controller import Controller as controller
from .samplers.Sampler import Sampler as sampler
from .schedulers.Scheduler import Scheduler as scheduler
from .proposals.Proposal import Proposal as proposal
from altar.simulations.Monitor import Monitor as monitor
from altar.simulations.Archiver import Archiver as archiver
from .solvers.Solver import Solver as solver
from .langevin_schedulers.LangevinScheduler import LangevinScheduler as langevinscheduler
from .stepsizers.StepSizer import StepSizer as stepsizer

# implementations
@altar.foundry(
    implements=controller,
    tip="a Bayesian controller that implements simulated annealing")
def annealer():
    # grab the factory
    from .controllers.Annealer import Annealer
    # attach its docstring
    __doc__ = Annealer.__doc__
    # and return it
    return Annealer

@altar.foundry(
    implements=controller,
    tip="a Bayesian controller that implements stochastic Stochastic gradient Langevin dynamics")
def langevin():
    # grab the factory
    from .controllers.Langevin import Langevin
    # attach its docstring
    __doc__ = Langevin.__doc__
    # and return it
    return Langevin

@altar.foundry(
    implements=langevinscheduler,
    tip="a Langevin sceduler")
def powerdecay():
    # grab the factory
    from .langevin_schedulers.PowerDecay import PowerDecay
    # attach its docstring
    __doc__ = PowerDecay.__doc__
    # and return it
    return PowerDecay

@altar.foundry(
    implements=langevinscheduler,
    tip="a Langevin sceduler")
def expdecay():
    # grab the factory
    from .langevin_schedulers.ExpDecay import ExpDecay
    # attach its docstring
    __doc__ = ExpDecay.__doc__
    # and return it
    return ExpDecay

@altar.foundry(
    implements=scheduler,
    tip="a Bayesian scheduler based on the COV algorithm")
def cov():
    # grab the factory
    from .schedulers.COV import COV
    # attach its docstring
    __doc__ = COV.__doc__
    # and return it
    return COV


@altar.foundry(
    implements=solver,
    tip="a solver for δβ based on a Brent minimizer from gsl")
def brent():
    # grab the factory
    from .solvers.Brent import Brent
    # attach its docstring
    __doc__ = Brent.__doc__
    # and return it
    return Brent


@altar.foundry(
    implements=solver,
    tip="a solver for δβ based on a naive grid search")
def grid():
    # grab the factory
    from .solvers.Grid import Grid
    # attach its docstring
    __doc__ = Grid.__doc__
    # and return it
    return Grid


@altar.foundry(
    implements=sampler,
    tip="a Bayesian sampler based on the Metropolis algorithm")
def metropolis():
    # grab the factory
    if altar.backends.active() == "cuda":
        try:
            from altar.cuda.bayesian.cudaMetropolis import cudaMetropolis as Metropolis
        except ImportError:
            from .samplers.Metropolis import Metropolis
    else:
        from .samplers.Metropolis import Metropolis
    # attach its docstring
    __doc__ = Metropolis.__doc__
    # and return it
    return Metropolis


@altar.foundry(
    implements=proposal,
    tip="a Gaussian proposal mechanism for sampling updates")
def gaussianproposal():
    # grab the factory
    from .proposals.GaussianProposal import GaussianProposal
    # attach its docstring
    __doc__ = GaussianProposal.__doc__
    # and return it
    return GaussianProposal


@altar.foundry(
    implements=stepsizer,
    tip="step size regulator with a fixed value")
def fixedstep():
    from .stepsizers.StepSizer import FixedStepSize
    __doc__ = FixedStepSize.__doc__
    return FixedStepSize


@altar.foundry(
    implements=stepsizer,
    tip="step size regulator with linear acceptance feedback a + b * r")
def linearrate():
    from .stepsizers.StepSizer import LinearRate
    __doc__ = LinearRate.__doc__
    return LinearRate


@altar.foundry(
    implements=stepsizer,
    tip="step size regulator targeting an acceptance rate")
def targetedrate():
    from .stepsizers.StepSizer import TargetedRate
    __doc__ = TargetedRate.__doc__
    return TargetedRate


@altar.foundry(
    implements=stepsizer,
    tip="dual-averaging step size regulator")
def dual():
    from .stepsizers.StepSizer import DualAveragingStepSize
    __doc__ = DualAveragingStepSize.__doc__
    return DualAveragingStepSize


@altar.foundry(
    implements=monitor,
    tip="a monitor that times the various simulation phases")
def profiler():
    # grab the factory
    from .monitoring.Profiler import Profiler
    # attach its docstring
    __doc__ = Profiler.__doc__
    # and return it
    return Profiler

@altar.foundry(
    implements=archiver,
    tip="an archiver to record the results and progress")
def recorder():
    # grab the factory
    from .archivers.Recorder import Recorder
    # attach its docstring
    __doc__ = Recorder.__doc__
    # and return it
    return Recorder

# end of file
