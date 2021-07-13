# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# get the package
import altar

# administrivia
@altar.foundry(implements=altar.action, tip="display information about this application")
def about():
    # get the command panel
    from .About import About
    # attach the docstring
    __doc__ = About.__doc__
    # and  return the panel
    return About


# sample the posterior distribution of a model
@altar.foundry(implements=altar.action, tip="sample the posterior distribution of a model")
def sample():
    # get the command panel
    from .Sample import Sample
    # attach the docstring
    __doc__ = Sample.__doc__
    # and  return the panel
    return Sample

# sample the posterior distribution of a model
@altar.foundry(implements=altar.action, tip="perform the forward modeling with a given parameter set")
def forward():
    # get the command panel
    from .Forward import Forward
    # attach the docstring
    __doc__ = Forward.__doc__
    # and  return the panel
    return Forward

# end of file
