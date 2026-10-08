#!/usr/bin/env python3
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
import numpy
# the package
import altar
from ..statistics import multiplicities

if typing.TYPE_CHECKING:
    import journal
    from altar.bayesian.states.BayesianState import BayesianState
    from altar.shells.Application import Application


# declaration
class ImportanceResampler(altar.component, family="altar.bayesian.importanceresampler"):
    """
    Implementation of importance resampling strategy for Bayesian inference.
    """

    # user configurable state
    use_low_variance_resampler = altar.properties.bool(default=False)
    use_low_variance_resampler.doc = "whether to use equal spaced random numbers for resampling"

    beta_resampling_start = altar.properties.float(default=0)
    beta_resampling_start.doc = 'the beta threshold to start the resampling procedure'

    # protocol obligations
    @altar.export
    def initialize(self, application: Application) -> typing.Self:
        """
        Initialize me and my parts given an {application} context
        """
        # the random number generator, for resampling
        self.rng = application.rng.rng
        # grab the info channel
        self.info = application.info
        # all done
        return self

    @altar.export
    def resample(self, w: numpy.ndarray, step: BayesianState, β: float) -> bool:
        """
        Rebuild the sample and its statistics based on importance weights if β threshold is met
        """
        # check if resampling should be performed
        if β <= self.beta_resampling_start:
            return False

        counts = self.compute_sample_multiplicities(w=w, step=step)
        # the index of the old sample behind each new one, duplicated by its count, shuffled
        rows = numpy.repeat(numpy.arange(counts.size), counts)
        self.rng.shuffle(rows)

        self.info.log(f"resampling: unique samples {numpy.count_nonzero(counts)} out of {counts.size}")

        # update the step with resampled data
        step.prior[...] = step.prior[rows]
        step.data[...] = step.data[rows]
        step.theta[...] = step.theta[rows]

        # indicate resampling was performed
        return True

    def compute_sample_multiplicities(self, w: numpy.ndarray, step: BayesianState) -> numpy.ndarray:
        """
        How many copies of each sample to keep, given the importance weights {w}
        """
        return multiplicities(
            w=numpy.asarray(w, dtype=float), rng=self.rng,
            low_variance=self.use_low_variance_resampler)

    # private data
    rng: numpy.random.Generator
    info: journal.info

# end of file
