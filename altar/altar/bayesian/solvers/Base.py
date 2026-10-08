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
import math
import typing
import numpy
# the package
import altar
# my protocol
from .Solver import Solver as solver

if typing.TYPE_CHECKING:
    from altar.bayesian.schedulers.COV import COV
    from altar.shells.Application import Application


# declaration
class Base(altar.component, implements=solver):
    """
    The shared part of the δβ solvers: the COV of the weights w = exp(δβ (llk - max llk)),
    normalized, for a candidate δβ, and the temperature reached so far; a solver looks for the
    δβ whose COV is my scheduler's target, within my {tolerance}
    """


    # user configurable state
    tolerance = altar.properties.float(default=.01)
    tolerance.doc = 'the fractional tolerance for achieving convergence'

    maxiter = altar.properties.int(default=10**3)
    maxiter.doc = 'the maximum number of iterations while looking for a δβ'


    # protocol obligations
    @altar.export
    def initialize(self, application: Application, scheduler: COV) -> typing.Self:
        """
        Initialize me and my parts given an {application} context and a {scheduler}
        """
        self.target = scheduler.target
        self.beta = 0.0
        self.cov = 0.0
        return self


    @altar.export
    def solve(self, llk: numpy.ndarray, weight: numpy.ndarray) -> tuple[float, float]:
        """
        The next temperature, and the COV of the normalized weights it gives the data log
        likelihoods {llk}, which are left in {weight}
        """
        llk = numpy.asarray(llk, dtype=float)
        self._llk, self._llkmax = llk, llk.max()
        # can we go all the way to β = 1
        high = 1.0 - self.beta
        cov = self.cov_at(dbeta=high)
        if cov < self.target or abs(cov - self.target) < self.tolerance:
            dbeta = high
        else:
            dbeta = self.dbeta(high=high)
        # the weights and the COV of my answer
        self.cov = self.cov_at(dbeta=dbeta, weight=weight)
        self.beta = 1.0 if dbeta == high else self.beta + dbeta
        return self.beta, self.cov


    # implementation details
    def dbeta(self, high: float) -> float:
        """
        The δβ in [0, {high}] whose COV is my target; {high} itself overshoots it
        """
        raise NotImplementedError(f"class '{type(self).__name__}' must implement 'dbeta'")


    def cov_at(self, dbeta: float, weight: numpy.ndarray | None = None) -> float:
        """
        The COV of the normalized weights at {dbeta}, which are left in {weight} if given; the
        weights are offset by the largest log likelihood, which leaves them unchanged once
        normalized but keeps exp from overflowing
        """
        llk = self._llk
        # at δβ = 0 every sample weighs the same, even with a -inf log likelihood
        if dbeta == 0:
            w = numpy.full(llk.size, 1.0 / llk.size)
        else:
            w = numpy.exp(dbeta * (llk - self._llkmax))
            w /= w.sum()
        if weight is not None:
            weight[...] = w
        cov = w.std(ddof=1) / w.mean()
        # a COV that is not well defined is too large
        return cov if math.isfinite(cov) else 1e100


    # private data
    target: float = 1.0     # the COV i aim for, from my scheduler
    beta: float = 0.0       # the temperature reached so far
    cov: float = 0.0        # the COV reached by the last update
    _llk: numpy.ndarray     # the data log likelihoods of the current solve
    _llkmax: float


# end of file
