# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2018 parasim inc
# (c) 2010-2018 california institute of technology
# all rights reserved
#

# the package
import altar
import altar.cuda

# publish the protocol for probability distributions
from altar.cuda.distributions import cudaDistribution as distribution
from altar.cuda.models.cudaBayesian import cudaBayesian as model

# implementations
@altar.foundry(implements=distribution, tip="the Moment Magnitude distribution")
def moment():
    # grab the factory
    from .cudaMoment import cudaMoment as moment
    # attach its docstring
    __doc__ = moment.__doc__
    # and return it
    return moment

# implementations
@altar.foundry(implements=model, tip="static inversion model")
def static():
    # grab the factory
    from .cudaStatic import cudaStatic as static
    # attach its docstring
    __doc__ = static.__doc__
    # and return it
    return static

# implementations
@altar.foundry(implements=model, tip="static inversion model with Cp")
def staticcp():
    # grab the factory
    from .cudaStaticCp import cudaStaticCp as staticcp
    # attach its docstring
    __doc__ = staticcp.__doc__
    # and return it
    return staticcp

# implementations
@altar.foundry(implements=model, tip="kinematic inversion model")
def kinematicg():
    # grab the factory
    from .cudaKinematicG import cudaKinematicG as kinematicg
    # attach its docstring
    __doc__ = kinematicg.__doc__
    # and return it
    return kinematicg

# implementations
@altar.foundry(implements=model, tip="cascaded kinematic inversion model")
def cascaded():
    # grab the factory
    from .cudaCascaded import cudaCascaded as cascaded
    # attach its docstring
    __doc__ = cascaded.__doc__
    # and return it
    return cascaded

# end of file
