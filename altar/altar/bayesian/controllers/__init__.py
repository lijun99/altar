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
    tip="the CATMIP controller: simulated annealing driven by the COV schedule")
def catmip():
    from .Catmip import Catmip
    __doc__ = Catmip.__doc__
    return Catmip

@altar.foundry(
    implements=controller,
    tip="a Bayesian controller that implements stochastic gradient Langevin dynamics")
def langevin():
    from .Langevin import Langevin
    __doc__ = Langevin.__doc__
    return Langevin

@altar.foundry(
    implements=controller,
    tip="Hamiltonian Monte Carlo with no annealing ladder (beta fixed at 1)")
def plainhmc():
    from .Hmc import Hmc
    __doc__ = Hmc.__doc__
    return Hmc

@altar.foundry(
    implements=controller,
    tip="CATMIP annealing (the COV schedule) with the sampler pinned to Hamiltonian Monte Carlo")
def catmiphmc():
    from .CatmipHmc import CatmipHmc
    __doc__ = CatmipHmc.__doc__
    return CatmipHmc

# end of file
