# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import altar
from .Sampler import Sampler as sampler

@altar.foundry(
    implements=sampler,
    tip="a Bayesian sampler based on the Metropolis algorithm")
def metropolis():
    from .Metropolis import Metropolis
    __doc__ = Metropolis.__doc__
    return Metropolis

@altar.foundry(
    implements=sampler,
    tip="a Bayesian sampler based on Hamiltonian Monte Carlo")
def hmc():
    from .HMC import HMC
    __doc__ = HMC.__doc__
    return HMC

# end of file
