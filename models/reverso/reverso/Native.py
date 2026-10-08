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
import numpy
# the package
import altar
# the pure python implementation
from .libreverso import REVERSO


# declaration
class Native:
    """
    The pure python strategy: one sample at a time, through {libreverso}; a reference
    """


    def initialize(self, model):
        """
        Unpack the observation geometry
        """
        self.model = model
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the displacements of {theta}
        """
        model = self.model
        t, x, y = model.stations.T
        for sample in range(batch):
            parameters = numpy.asarray(theta.getRow(sample).ndarray())
            u = REVERSO(t, x, y, **model.source(parameters), **model.medium())
            prediction.setRow(sample, model.io.toGsl(numpy.column_stack(u).ravel()))
        # all done
        return self


    def verify(self, theta, mask, batch):
        """
        Flag in {mask} the first {batch} samples of {theta} whose deep chamber isn't below the
        shallow one
        """
        model = self.model
        for sample in range(batch):
            source = model.source(numpy.asarray(theta.getRow(sample).ndarray()))
            if source["H_d"] <= source["H_s"]:
                mask[sample] = 1
        # all done
        return self


    # private data
    model = None


# end of file
