# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Protocol and implementations for regulating how many MC steps a Metropolis-family sampler
runs per β step -- paired with {altar.bayesian.stepsizers.StepSizer}: {stepsizer} regulates
the size of each step/jump, {stepcounter} regulates how many of them to take.

Backend-agnostic: {theta} may be a cpu numpy array or a cuda {altar.cuda.array.Array}, which
numpy can view, so {DecorrelatingSteps}'s correlation check works unchanged on either.
"""

from __future__ import annotations
import typing
import numpy
import altar

if typing.TYPE_CHECKING:
    import journal
    from altar.arrays import Array
    from altar.bayesian.controllers.Annealer import Annealer
    from altar.shells.Application import Application


class StepCounter(altar.protocol, family="altar.bayesian.stepcounters"):
    """
    Protocol for components that decide how many MC steps a sampler runs per β step.
    """

    @altar.provides
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me given an {application} context
        """

    @altar.provides
    def start(self, theta: Array, beta: float | None = None) -> typing.Self:
        """
        Called once per β step, before the first block, given the starting sample matrix
        """

    @altar.provides
    def block_size(self) -> int:
        """
        The number of MC steps to run before the next {done} check
        """

    @altar.provides
    def done(self, mcsteps: int, theta: Array, annealer: Annealer | None = None) -> bool:
        """
        Return whether the current β step's chain walk should stop, given the number of MC
        steps run so far ({mcsteps}) and the current sample matrix ({theta})
        """

    @classmethod
    def pyre_default(cls, **kwds) -> type:
        """
        Supply a default implementation
        """
        # by default, run a fixed number of steps
        return FixedSteps


class FixedSteps(altar.component, family="altar.bayesian.stepcounters.fixed", implements=StepCounter):
    """
    Run a fixed number of MC steps per β step, in a single block
    """

    steps = altar.properties.int(default=None)
    steps.doc = "the number of MC steps per β step; None lets the sampler fill in " \
                "application.job.steps"

    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        if self.steps is None:
            self.steps = application.job.steps
        return self

    @altar.export
    def start(self, theta: Array, beta: float | None = None) -> typing.Self:
        return self

    @altar.export
    def block_size(self) -> int:
        return self.steps

    @altar.export
    def done(self, mcsteps: int, theta: Array, annealer: Annealer | None = None) -> bool:
        return mcsteps >= self.steps


class DecorrelatingSteps(altar.component, family="altar.bayesian.stepcounters.decorrelating",
                         implements=StepCounter):
    """
    Run in blocks of {corr_check_steps} MC steps until the Pearson correlation between the
    starting and current sample positions drops below {target_correlation} (the chains are
    considered effectively de-correlated), or {max_mc_steps} is reached.
    """

    max_mc_steps = altar.properties.int(default=10000)
    max_mc_steps.doc = 'maximum MC steps per β step'

    min_mc_steps = altar.properties.int(default=1000)
    min_mc_steps.doc = 'minimum MC steps before the first correlation check'

    corr_check_steps = altar.properties.int(default=1000)
    corr_check_steps.doc = 'MC steps between successive correlation checks'

    target_correlation = altar.properties.float(default=0.6)
    target_correlation.doc = 'correlation threshold below which the chain is considered de-correlated'

    max_mc_steps_stage2 = altar.properties.int(default=None)
    max_mc_steps_stage2.doc = 'max steps when β > beta_stage2 (defaults to max_mc_steps)'

    beta_stage2 = altar.properties.float(default=1.0)
    beta_stage2.doc = 'β threshold above which to use max_mc_steps_stage2'

    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        if self.max_mc_steps_stage2 is None:
            self.max_mc_steps_stage2 = self.max_mc_steps
        self.info = application.info
        return self

    @altar.export
    def start(self, theta: Array, beta: float | None = None) -> typing.Self:
        # snapshot of starting positions for the correlation check
        self._theta_start = numpy.array(theta, dtype=float)
        # the max-step budget for this β (two-stage: a tighter budget once annealing is done)
        self._max_steps = (self.max_mc_steps_stage2 if (beta is not None and beta > self.beta_stage2)
                           else self.max_mc_steps)
        return self

    @altar.export
    def block_size(self) -> int:
        return self.corr_check_steps

    @altar.export
    def done(self, mcsteps: int, theta: Array, annealer: Annealer | None = None) -> bool:
        if mcsteps < self.min_mc_steps:
            return False
        if mcsteps >= self._max_steps:
            return True

        ts = self._theta_start
        tc = numpy.asarray(theta)
        ts = ts - ts.mean(axis=0)
        tc = tc - tc.mean(axis=0)
        numer = (ts * tc).sum(axis=0)
        denom = numpy.sqrt((ts ** 2).sum(axis=0) * (tc ** 2).sum(axis=0))
        # guard against zero variance (constant column)
        safe_denom = numpy.where(denom > 0, denom, 1.0)
        correlation = float(numpy.abs(numer / safe_denom).max())

        # under MPI, every worker must agree on when to stop
        workers = getattr(getattr(annealer, "worker", None), "workers", 1)
        if annealer is not None and workers > 1:
            import mpi
            correlation = mpi.world.max(item=correlation)

        self.info.log(f"{type(self).__name__}: correlation {correlation:.4f} at {mcsteps} steps")
        return correlation <= self.target_correlation

    # private data
    info: journal.info | None = None
    _theta_start: numpy.ndarray | None = None
    _max_steps: int | None = None

# end of file
