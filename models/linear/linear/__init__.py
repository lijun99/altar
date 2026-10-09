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


# implementations
@altar.foundry(implements=altar.models.model, tip="a linear model")
def linear():
    # grab the factory
    from .Linear import Linear as linear
    # attach its docstring
    __doc__ = linear.__doc__
    # and return it
    return linear


@altar.foundry(implements=altar.models.model, tip="the linear model, with its forward model and gradient in jax")
def jax():
    # grab the factory
    from .LinearJax import LinearJax as jax
    # attach its docstring
    __doc__ = jax.__doc__
    # and return it
    return jax


# end of file
