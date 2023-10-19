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


@altar.foundry(implements=altar.models.model,
               tip="Sequences of Earthquakes and Aseismic Slip")
def seas():
    # grab the factory
    from .SEAS import SEAS as seas
    # attach its docstring
    __doc__ = seas.__doc__  # noqa: F841
    # and return it
    return seas



