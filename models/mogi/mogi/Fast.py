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
    from .Mogi import Mogi


# declaration
class Fast:
    """
    The cpu strategy: the forward model of all samples at once, in c++
    """


    def initialize(self, model: Mogi) -> typing.Self:
        """
        Upload the observation geometry and the parameter layout
        """
        # get the extension; {altar.models.mogi.ext} swallows a failed import
        from .ext import libmogi
        self.libmogi = libmogi
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
        self.libmogi.displacements(
            theta, self.stations, model.xIdx, model.yIdx, model.dIdx, model.sIdx,
            model.log10_dV, model.nu, batch, prediction)
        # all done
        return self


    # private data
    libmogi: typing.Any = None
    model: Mogi
    stations: numpy.ndarray # the observation geometry


# end of file
