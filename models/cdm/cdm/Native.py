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
import numpy
# the package
import altar
# the pure python implementation of the CDM source
from .libcdm import CDM


# the names of the source parameters, in the order of {CDM.layout}
NAMES = ("X0", "Y0", "depth", "opening", "ax", "ay", "az", "omegaX", "omegaY", "omegaZ")


# declaration
class Native:
    """
    The pure python strategy: one sample at a time, through {libcdm}; slow, but a reference
    """


    def initialize(self, model):
        """
        Unpack the observation geometry
        """
        self.model = model
        self.stations = model.stations
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the LOS displacements of {theta}
        """
        stations = self.stations
        for sample in range(batch):
            parameters = numpy.asarray(theta.getRow(sample).ndarray())
            ue, un, uv = CDM(X=stations[:, 0], Y=stations[:, 1], nu=self.model.nu,
                             **self.source(parameters))
            u = ue*stations[:, 2] + un*stations[:, 3] + uv*stations[:, 4]
            # less the dataset offsets
            shifted = stations[:, 5] >= 0
            u[shifted] -= parameters[stations[shifted, 5].astype(int)]
            prediction.setRow(sample, self.model.io.toGsl(u))
        # all done
        return self


    def verify(self, theta, mask, batch):
        """
        Flag in {mask} the first {batch} samples of {theta} whose source reaches above the free
        surface
        """
        for sample in range(batch):
            parameters = numpy.asarray(theta.getRow(sample).ndarray())
            try:
                CDM(X=numpy.zeros(1), Y=numpy.zeros(1), nu=self.model.nu,
                    **self.source(parameters))
            except ValueError:
                mask[sample] = 1
        # all done
        return self


    # implementation details
    def source(self, parameters):
        """
        The source parameters of a sample, by name
        """
        return dict(zip(NAMES, (parameters[column] for column in self.model.layout)))


    # private data
    model = None
    stations = None


# end of file
