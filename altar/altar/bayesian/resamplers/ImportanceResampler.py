# -*- python -*-
# -*- coding: utf-8 -*-
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
# my protocol
from .Resampler import Resampler
from ..statistics import multiplicities, RESAMPLING

if typing.TYPE_CHECKING:
    import journal
    from altar.shells.Application import Application


# declaration
class ImportanceResampler(altar.component, family="altar.bayesian.resamplers.importance",
                          implements=Resampler):
    """
    Importance resampling: draw a new sample from the old one, each sample with the probability
    of its weight, so that it is copied as many times as its weight deserves on average; the
    {method} decides how far the counts stray from that average
    """

    # user configurable state
    method = altar.properties.str(default="systematic")
    method.validators = altar.constraints.isMember(*RESAMPLING)
    method.doc = "how to draw the copies: multinomial, independent draws; stratified, one draw " \
                 "in each of n equal slices; systematic, the default, one draw and its n equally " \
                 "spaced shifts, so that a sample of weight w gets the floor or the ceiling of n w " \
                 "copies; or residual, the floor of n w copies and the rest drawn"


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
    def multiplicities(self, w: numpy.ndarray) -> numpy.ndarray:
        """
        How many copies of each sample to keep, given their normalized weights {w}
        """
        return multiplicities(w=numpy.asarray(w, dtype=float), rng=self.rng, method=self.method)


    @altar.export
    def rows(self, w: numpy.ndarray) -> numpy.ndarray:
        """
        The sample behind each one of the new sample, in random order, given the weights {w}
        """
        counts = self.multiplicities(w=w)
        # the index of the old sample behind each new one, duplicated by its count, shuffled
        rows = numpy.repeat(numpy.arange(counts.size), counts)
        self.rng.shuffle(rows)
        self.info.log(f"resampling ({self.method}): unique samples {numpy.count_nonzero(counts)} "
                      f"out of {counts.size}")
        return rows


    # private data
    rng: numpy.random.Generator
    info: journal.info


# end of file
