# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis (michael.aivazis@para-sim.com)
# grace bato           (mary.grace.p.bato@jpl.nasa.gov)
# eric m. gurrola      (eric.m.gurrola@jpl.nasa.gov)
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved


# the package
import altar


# declaration
class Fast:
    """
    The cpu strategy: the forward model of all samples at once, in c++
    """


    def initialize(self, model):
        """
        Upload the observation geometry
        """
        # get the extension; {altar.models.reverso.ext} swallows a failed import
        from .ext import libreverso
        self.libreverso = libreverso
        self.model = model
        self.stations = model.io.toGsl(model.stations)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the displacements of {theta}
        """
        model = self.model
        self.libreverso.displacements(
            theta, self.stations, model.layout, model.G, model.v, model.mu, model.drho, model.g,
            model.shallow == "sill", model.deep == "sill", batch, prediction)
        # all done
        return self


    def verify(self, theta, mask, batch):
        """
        Flag in {mask} the first {batch} samples of {theta} whose deep chamber isn't below the
        shallow one
        """
        self.libreverso.verify(theta, self.model.layout, batch, mask)
        # all done
        return self


    # private data
    libreverso = None
    model = None
    stations = None # the observation geometry, as a gsl matrix


# end of file
