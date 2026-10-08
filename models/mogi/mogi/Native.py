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
# the pure python implementation of the Mogi source
from .Source import Source as source

if typing.TYPE_CHECKING:
    from .Mogi import Mogi


# declaration
class Native:
    """
    The pure python strategy: one sample at a time, through {Source}; slow, but a reference
    """


    def initialize(self, model: Mogi) -> typing.Self:
        """
        Unpack the observation geometry
        """
        self.model = model
        stations = model.stations
        self.locations = [tuple(row) for row in stations[:, :2]]
        self.los = stations[:, 2:5].copy()
        self.offsets = [int(column) for column in stations[:, 5]]
        # all done
        return self


    def forward_model_batched(self, theta: numpy.ndarray, prediction: numpy.ndarray,
                              batch: int) -> typing.Self:
        """
        Fill the first {batch} rows of {prediction} with the LOS displacements of {theta}
        """
        model = self.model
        for sample in range(batch):
            parameters = theta[sample]
            s = parameters[model.sIdx]
            mogi = source(x=parameters[model.xIdx], y=parameters[model.yIdx],
                          d=parameters[model.dIdx], dV=10**s if model.log10_dV else s,
                          nu=model.nu)
            u = mogi.displacements(locations=self.locations, los=self.los)
            # less the dataset offsets
            for obs, column in enumerate(self.offsets):
                if column >= 0:
                    u[obs] -= parameters[column]
            prediction[sample] = u
        # all done
        return self


    # private data
    model: Mogi
    locations: list[tuple[float, float]]
    los: numpy.ndarray  # the LOS unit vector of each observation
    offsets: list[int]  # the column of the dataset offset of each observation, or -1


# end of file
