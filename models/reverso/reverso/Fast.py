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

if typing.TYPE_CHECKING:
    from .Reverso import Reverso


# declaration
class Fast:
    """
    The cpu strategy: the forward model of all samples at once, in c++
    """


    def initialize(self, model: Reverso) -> typing.Self:
        """
        Upload the observation geometry
        """
        # get the extension; {altar.models.reverso.ext} swallows a failed import
        from .ext import libreverso
        self.libreverso = libreverso
        self.model = model
        self.stations = numpy.ascontiguousarray(model.stations, dtype=numpy.float64)
        # all done
        return self


    def forward_model_batched(self, theta: numpy.ndarray, prediction: numpy.ndarray,
                              batch: int) -> typing.Self:
        """
        Fill the first {batch} rows of {prediction} with the displacements of {theta}
        """
        model = self.model
        self.libreverso.displacements(
            theta, self.stations, model.layout, model.G, model.v, model.mu, model.drho, model.g,
            model.shallow == "sill", model.deep == "sill", batch, prediction)
        # all done
        return self


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray, batch: int) -> typing.Self:
        """
        Flag in {mask} the first {batch} samples of {theta} whose deep chamber isn't below the
        shallow one
        """
        self.libreverso.verify(theta, self.model.layout, batch, mask)
        # all done
        return self


    # private data
    libreverso: typing.Any = None
    model: Reverso
    stations: numpy.ndarray # the observation geometry


# end of file
