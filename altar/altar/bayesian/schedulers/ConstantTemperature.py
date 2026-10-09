# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#
# A scheduler that keeps beta fixed at a constant value (default 1)

# externals
from __future__ import annotations
import typing
import numpy
# the package
import altar
# my protocol
from .Scheduler import Scheduler as scheduler

if typing.TYPE_CHECKING:
    import journal
    from altar.bayesian.states.CoolingStep import CoolingStep
    from altar.shells.Application import Application


class ConstantTemperature(altar.component, family="altar.schedulers.constant", implements=scheduler):
    """
    A scheduler that keeps the temperature fixed at a constant beta value.

    During a burn-in of {burnin} walks, it replaces the chains stuck far from the others, whose
    log posterior falls more than {outlier_iqr} interquartile ranges below the first quartile,
    by copies of randomly chosen other chains, after ter Braak (2006); it leaves the chains
    alone afterwards, so that the walks that follow sample the posterior.
    """

    beta_start = altar.properties.float(default=1.0)
    beta_start.doc = "the fixed beta value used for the entire run"

    burnin = altar.properties.int(default=0)
    burnin.validators = altar.constraints.isGreaterEqual(value=0)
    burnin.doc = "the walks after which the outlier chains are replaced by copies of others; " \
                 "none by default"

    outlier_iqr = altar.properties.float(default=2.0)
    outlier_iqr.doc = "how far below the first quartile of the log posteriors, in interquartile " \
                      "ranges, a chain is an outlier"


    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me given an {application} context
        """
        # the random number generator, for picking the replacements
        self.rng = application.rng.rng
        # grab the info channel
        self.info = application.info
        # no walks yet
        self.walks = 0
        return self


    @altar.export
    def update(self, step: CoolingStep) -> CoolingStep:
        """
        Set the temperature of {step}, after replacing its outlier chains during the burn-in
        """
        # the first update comes before any walk
        if 0 < self.walks <= self.burnin:
            self.replace_outliers(step=step)
        self.walks += 1
        self.update_temperature(step=step)
        step.compute_posterior()
        return step


    @altar.export
    def update_temperature(self, step: CoolingStep) -> CoolingStep:
        step.beta = self.beta_start
        return step


    @altar.export
    def compute_covariance(self, step: CoolingStep) -> CoolingStep:
        return step


    @altar.export
    def rank(self, step: CoolingStep) -> CoolingStep:
        return step


    def replace_outliers(self, step: CoolingStep) -> CoolingStep:
        """
        Replace the chains of {step} whose log posterior is below the first quartile by more than
        {outlier_iqr} interquartile ranges by copies of randomly chosen chains that are not
        """
        posterior = numpy.asarray(step.posterior, dtype=numpy.float64)
        q1, q3 = numpy.percentile(posterior, [25, 75])
        floor = q1 - self.outlier_iqr * (q3 - q1)
        # NaN is an outlier too
        outliers = ~(posterior >= floor)
        count = int(outliers.sum())
        if count == 0:
            return step
        # each outlier becomes a copy of one of the others
        rows = numpy.arange(posterior.size)
        rows[outliers] = self.rng.choice(numpy.flatnonzero(~outliers), size=count)
        step.reorder(rows=rows)
        self.info.log(f"burn-in: replaced {count} outlier chains, with log posterior below "
                      f"{floor:.6g}, after walk {self.walks} of {self.burnin}")
        return step


    # private data
    rng: numpy.random.Generator
    info: journal.info
    walks: int = 0 # the walks so far


# end of file
