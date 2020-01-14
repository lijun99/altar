# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# Author(s): Lijun Zhu

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


@altar.foundry(implements=parameters, tip="a cuda parameter set")
def parameterset():
    # grab the factory
    from .cudaParameterSet import cudaParameterSet as parameterset
    # attach its docstring
    __doc__ = parameterset.__doc__
    # and publish it
    return parameterset


# end of file
