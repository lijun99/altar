# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#


# the package
import altar


# the protocols
from .Model import Model as model
from .ParameterSet import ParameterSet as parameters


# the model base class
from .Bayesian import Bayesian as bayesian


# implementations
@altar.foundry(implements=model, tip="a trivial AlTar model")
def null():
    # grab the factory
    from .Null import Null as null
    # attach its docstring
    __doc__ = null.__doc__
    # and publish it
    return null


@altar.foundry(implements=model, tip="a collection of models that comprise an AlTar model")
def ensemble():
    # grab the factory
    if altar.backends.active() == "cuda":
        try:
            from altar.cuda.models.cudaBayesianEnsemble import cudaBayesianEnsemble as ensemble
        except ImportError:
            from .Ensemble import Ensemble as ensemble
    else:
        from .Ensemble import Ensemble as ensemble
    # attach its docstring
    __doc__ = ensemble.__doc__
    # and publish it
    return ensemble

@altar.foundry(implements=model, tip="a models that implements psets and dataobs with l2 norm")
def bayesianl2():
    # grab the factory
    from .BayesianL2 import BayesianL2 as bayesianl2
    # attach its docstring
    __doc__ = bayesianl2.__doc__
    # and publish it
    return bayesianl2


@altar.foundry(implements=parameters, tip="a contiguous parameter set")
def contiguous():
    # grab the factory
    if altar.backends.active() == "cuda":
        try:
            from altar.cuda.models.cudaParameterSet import cudaParameterSet as contiguous
        except ImportError:
            from .Contiguous import Contiguous as contiguous
    else:
        from .Contiguous import Contiguous as contiguous
    # attach its docstring
    __doc__ = contiguous.__doc__
    # and publish it
    return contiguous


@altar.foundry(implements=parameters, tip="an ensemble of parameter sets")
def parameterensemble():
    # grab the factory
    from .ParameterEnsemble import ParameterEnsemble as parameterensemble
    # attach its docstring
    __doc__ = parameterensemble.__doc__
    # and publish it
    return parameterensemble


# end of file
