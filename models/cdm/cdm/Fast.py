# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


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
        # get the extension; {altar.models.cdm.ext} swallows a failed import
        from .ext import libcdm
        self.libcdm = libcdm
        self.model = model
        self.stations = model.io.toGsl(model.stations)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the LOS displacements of {theta}
        """
        model = self.model
        self.libcdm.displacements(theta, self.stations, model.layout, model.nu, batch, prediction)
        # all done
        return self


    def verify(self, theta, mask, batch):
        """
        Flag in {mask} the first {batch} samples of {theta} whose source reaches above the free
        surface
        """
        self.libcdm.verify(theta, self.model.layout, batch, mask)
        # all done
        return self


    # private data
    libcdm = None
    model = None
    stations = None # the observation geometry, as a gsl matrix


# end of file
