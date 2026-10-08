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
        Upload the observation geometry and the parameter layout
        """
        # get the extension; {altar.models.mogi.ext} swallows a failed import
        from .ext import libmogi
        self.libmogi = libmogi
        self.model = model
        self.stations = model.io.toGsl(model.stations)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
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
    libmogi = None
    model = None
    stations = None # the observation geometry, as a gsl matrix


# end of file
