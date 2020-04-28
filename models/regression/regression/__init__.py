# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2020 parasim inc
# (c) 2010-2020 california institute of technology
# all rights reserved
#
# author(s): Lijun Zhu

# the package
import altar

# the protocol
from altar.models.Model import Model as model


# implementations
@altar.foundry(implements=model, tip="a linear regression model")
def linear():
    # grab the factory
    from .Linear import Linear as linear
    # attach its docstring
    __doc__ = linear.__doc__
    # and return it
    return linear


# end of file
