# -*- python -*-
# -*- coding: utf-8 -*-

import altar
from .Controller import Controller as controller

@altar.foundry(
    implements=controller,
    tip="a Bayesian controller that implements simulated annealing")
def annealer():
    from .Annealer import Annealer
    __doc__ = Annealer.__doc__
    return Annealer

@altar.foundry(
    implements=controller,
    tip="a Bayesian controller that implements stochastic gradient Langevin dynamics")
def langevin():
    from .Langevin import Langevin
    __doc__ = Langevin.__doc__
    return Langevin

# end of file
