# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
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

# perform the forward modeling with a given parameter set
@altar.foundry(implements=altar.action, tip="perform the forward modeling with a given parameter set")
def forward():
    # get the command panel
    from .Forward import Forward
    # attach the docstring
    __doc__ = Forward.__doc__
    # and  return the panel
    return Forward

# measure how well the data resolve the parameters
@altar.foundry(implements=altar.action, tip="measure how well the data resolve the parameters")
def resolution():
    # get the command panel
    from .Resolution import Resolution
    # attach the docstring
    __doc__ = Resolution.__doc__
    # and  return the panel
    return Resolution

# make synthetic data from a true model
@altar.foundry(implements=altar.action, tip="make synthetic data from a true model")
def synthetic():
    # get the command panel
    from .Synthetic import Synthetic
    # attach the docstring
    __doc__ = Synthetic.__doc__
    # and  return the panel
    return Synthetic

# compare a posterior with a true model
@altar.foundry(implements=altar.action, tip="compare a posterior with a true model")
def recover():
    # get the command panel
    from .Recover import Recover
    # attach the docstring
    __doc__ = Recover.__doc__
    # and  return the panel
    return Recover

# check the annealing schedule of a run, and its posterior against a reference
@altar.foundry(implements=altar.action, tip="check the annealing schedule of a run")
def diagnose():
    # get the command panel
    from .Diagnose import Diagnose
    # attach the docstring
    __doc__ = Diagnose.__doc__
    # and  return the panel
    return Diagnose

# end of file
