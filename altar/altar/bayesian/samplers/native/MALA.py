# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# my base
from .HMC import HMC


class MALA(HMC):
    """
    The cpu implementation of the Metropolis-adjusted Langevin algorithm: {HMC} with one
    leapfrog step per proposal, see {altar.bayesian.samplers.MALA}
    """

    # the optimal acceptance rate of MALA (Roberts and Rosenthal, 1998)
    target_acceptance = 0.574


# end of file
