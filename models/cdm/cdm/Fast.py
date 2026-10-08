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

if typing.TYPE_CHECKING:
    from .CDM import CDM as Model


# declaration
class Fast:
    """
    The cpu strategy: the forward model of all samples at once, in c++
    """


    def initialize(self, model: Model) -> typing.Self:
        """
        Upload the observation geometry
        """
        # get the extension; {altar.models.cdm.ext} swallows a failed import
        from .ext import libcdm
        self.libcdm = libcdm
        self.model = model
        self.stations = numpy.ascontiguousarray(model.stations, dtype=numpy.float64)
        # all done
        return self


    def forward_model_batched(self, theta: numpy.ndarray, prediction: numpy.ndarray,
                              batch: int) -> typing.Self:
        """
        Fill the first {batch} rows of {prediction} with the LOS displacements of {theta}
        """
        model = self.model
        self.libcdm.displacements(theta, self.stations, model.layout, model.nu, batch, prediction)
        # all done
        return self


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray, batch: int) -> typing.Self:
        """
        Flag in {mask} the first {batch} samples of {theta} whose source reaches above the free
        surface
        """
        self.libcdm.verify(theta, self.model.layout, batch, mask)
        # all done
        return self


    # private data
    libcdm: typing.Any = None
    model: Model
    stations: numpy.ndarray # the observation geometry


# end of file
