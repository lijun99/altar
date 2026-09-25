# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar
import altar.cuda

# the base
from altar.models.Model import Model as model
from altar.models.ParameterSet import ParameterSet as parameters

# implementations

@altar.foundry(implements=model, tip="a cuda AlTar model")
def bayesian():
    # grab the factory
    from .cudaBayesian import cudaBayesian as bayesian
    # attach its docstring
    __doc__ = bayesian.__doc__
    # and publish it
    return bayesian

@altar.foundry(implements=model, tip="a collection of cuda AlTar model")
def bayesianensemble():
    # grab the factory
    from .cudaBayesianEnsemble import cudaBayesianEnsemble as bayesianensemble
    # attach its docstring
    __doc__ = bayesianensemble.__doc__
    # and publish it
    return bayesianensemble


# {Contiguous}/{ParameterEnsemble} are unified components now: each picks its cpu/cuda
# implementation internally, at {initialize} time, so these two foundries exist only for
# {.pfg} files that still spell out the explicit "altar.cuda.models.parameterset"/
# "...parameterensemble" path; they resolve to the very same classes {altar.models.contiguous}
# and {altar.models.parameterensemble} do
@altar.foundry(implements=parameters, tip="a contiguous parameter set")
def parameterset():
    # grab the factory
    from altar.models.Contiguous import Contiguous as parameterset
    # attach its docstring
    __doc__ = parameterset.__doc__
    # and publish it
    return parameterset


@altar.foundry(implements=parameters, tip="an ensemble of parameter sets")
def parameterensemble():
    # grab the factory
    from altar.models.ParameterEnsemble import ParameterEnsemble as parameterensemble
    # attach its docstring
    __doc__ = parameterensemble.__doc__
    # and publish it
    return parameterensemble


# end of file
