# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2022 parasim inc
# (c) 2010-2022 california institute of technology
# all rights reserved
#
# author: Tobias Köhne

import altar


# the models; all of them run on the gpu, see {altar.models.seas.cuda}
@altar.foundry(implements=altar.models.model,
               tip="Sequences of Earthquakes and Aseismic Slip, on a 3d fault")
def seas3d():
    # grab the factory
    from .cuda.SEAS3D import SEAS3D as seas3d
    # attach its docstring
    __doc__ = seas3d.__doc__  # noqa: F841
    # and return it
    return seas3d


@altar.foundry(implements=altar.models.model,
               tip="a creeping fault with linear viscous rheology")
def linearviscous():
    # grab the factory
    from .cuda.LinearViscous import LinearViscous as linearviscous
    # attach its docstring
    __doc__ = linearviscous.__doc__  # noqa: F841
    # and return it
    return linearviscous


# end of file
