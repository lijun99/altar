# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
# Author(s): Lijun Zhu
#

# get the package
import altar
# get my model package
import altar.models.seismic


# declaration
class Forward(altar.panel(), family='altar.actions.forward'):
    """
    Sample the posterior distribution of a model
    """


    # commands
    @altar.export(tip="perform the forward modeling with a given parameter set")
    def default(self, plexus, **kwds):
        """
        Sample the model posterior distribution
        """
        # get the model
        model = plexus.model
        # set the model forwardonly flag
        model.forwardonly = True
        # call the forwardProblem method
        return model.forwardProblem(application=plexus)


# end of file
