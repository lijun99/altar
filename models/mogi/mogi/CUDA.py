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
import altar.cuda


# declaration
class CUDA:
    """
    The cuda strategy: the forward model of all samples at once, one thread per observation
    """


    def initialize(self, model):
        """
        Upload the observation geometry and the parameter layout
        """
        # get the extension; {altar.models.mogi.ext} swallows a failed import
        from .ext import libcudamogi
        self.libcudamogi = libcudamogi
        self.model = model
        self.stations = altar.cuda.matrix(source=model.stations, dtype=model.precision)
        # all done
        return self


    def forward_model_batched(self, theta, prediction, batch):
        """
        Fill the first {batch} rows of {prediction} with the LOS displacements of {theta}
        """
        model = self.model
        self.libcudamogi.displacements(
            theta.grid, self.stations.grid, model.xIdx, model.yIdx, model.dIdx, model.sIdx,
            model.log10_dV, model.nu, batch, prediction.grid)
        # all done
        return self


    # private data
    libcudamogi = None
    model = None
    stations = None # the observation geometry, on the device


# end of file
