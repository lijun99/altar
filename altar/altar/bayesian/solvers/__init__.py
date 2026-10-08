# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from .Solver import Solver as solver

@altar.foundry(
    implements=solver,
    tip="a solver for δβ based on Brent's root finding method")
def brent():
    from .Brent import Brent
    __doc__ = Brent.__doc__
    return Brent

@altar.foundry(
    implements=solver,
    tip="a solver for δβ based on a naive grid search")
def grid():
    from .Grid import Grid
    __doc__ = Grid.__doc__
    return Grid

# end of file
