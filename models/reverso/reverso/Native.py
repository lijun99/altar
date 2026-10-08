# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis (michael.aivazis@para-sim.com)
# grace bato           (mary.grace.p.bato@jpl.nasa.gov)
# eric m. gurrola      (eric.m.gurrola@jpl.nasa.gov)
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved


# externals
from __future__ import annotations
import typing
import numpy
# the pure python implementation
from .libreverso import REVERSO

if typing.TYPE_CHECKING:
    from .Reverso import Reverso


# declaration
class Native:
    """
    The pure python strategy: one sample at a time, through {libreverso}; a reference
    """


    def initialize(self, model: Reverso) -> typing.Self:
        """
        Unpack the observation geometry
        """
        self.model = model
        # all done
        return self


    def forward_model_batched(self, theta: numpy.ndarray, prediction: numpy.ndarray,
                              batch: int) -> typing.Self:
        """
        Fill the first {batch} rows of {prediction} with the displacements of {theta}
        """
        model = self.model
        t, x, y = model.stations.T
        for sample in range(batch):
            u = REVERSO(t, x, y, **model.source(theta[sample]), **model.medium())
            prediction[sample] = numpy.column_stack(u).ravel()
        # all done
        return self


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray, batch: int) -> typing.Self:
        """
        Flag in {mask} the first {batch} samples of {theta} whose deep chamber isn't below the
        shallow one
        """
        model = self.model
        for sample in range(batch):
            source = model.source(theta[sample])
            if source["H_d"] <= source["H_s"]:
                mask[sample] = 1
        # all done
        return self


    # private data
    model: Reverso


# end of file
