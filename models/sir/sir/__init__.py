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


# implementations
@altar.foundry(implements=altar.models.model, tip="a SIR model in epidemiology")
def sir():
    # grab the factory
    from .SIR import SIR as sir
    # attach its docstring
    __doc__ = sir.__doc__
    # and return it
    return sir


# end of file
