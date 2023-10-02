# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2023 parasim inc
# (c) 2010-2023 california institute of technology
# all rights reserved
#

# the package
import altar
import altar.cuda

# publish the protocol for probability distributions
from altar.cuda.distributions import cudaDistribution as distribution  # noqa: F401
from altar.cuda.models.cudaBayesian import cudaBayesian as model


# implementations
@altar.foundry(implements=model, tip="static inversion model")
def linearviscous():
    # grab the factory
    from .cudaLinearViscous import cudaLinearViscous as linearviscous
    # attach its docstring
    __doc__ = linearviscous.__doc__  # noqa: F841
    # and return it
    return linearviscous


# implementations
@altar.foundry(implements=model, tip="Rate-dependent 3D SEAS Simulation")
def ratedependent():
    # grab the factory
    from .cudaRateDependent import cudaRateDependent as ratedependent
    # attach its docstring
    __doc__ = ratedependent.__doc__  # noqa: F841
    # and return it
    return ratedependent

# end of file
