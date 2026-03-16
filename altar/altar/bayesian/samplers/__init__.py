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
            from altar.cuda.bayesian.cudaMetropolis import cudaMetropolis as Metropolis
        except ImportError:
            from .Metropolis import Metropolis
    else:
        from .Metropolis import Metropolis
    __doc__ = Metropolis.__doc__
    return Metropolis

# end of file
