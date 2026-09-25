# -*- python -*-
# -*- coding: utf-8 -*-

import altar
from .Sampler import Sampler as sampler

@altar.foundry(
    implements=sampler,
    tip="a Bayesian sampler based on the Metropolis algorithm")
def metropolis():
    if altar.backends.active() == "cuda":
        try:
            from .cuda.Metropolis import Metropolis
        except ImportError:
            from .Metropolis import Metropolis
    else:
        from .Metropolis import Metropolis
    __doc__ = Metropolis.__doc__
    return Metropolis

@altar.foundry(
    implements=sampler,
    tip="a Bayesian sampler based on Hamiltonian Monte Carlo")
def hmc():
    if altar.backends.active() == "cuda":
        try:
            from .cuda.HMC import HMC
        except ImportError:
            from .HMC import HMC
    else:
        from .HMC import HMC
    __doc__ = HMC.__doc__
    return HMC

# end of file
