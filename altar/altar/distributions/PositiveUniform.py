# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# get the package
import altar

# get the protocol
from . import distribution
# and my base class
from .Base import Base as base


# the declaration
class PositiveUniform(base, family="altar.distributions.positiveuniform"):
    """
    The uniform probability distribution over the open interval (0, 1), guaranteed to never
    land on 0 exactly, so a caller may safely take its log -- gsl's own {gsl_rng_uniform_pos}
    exists for exactly this reason, unlike the plain uniform generator, which can return 0.

    My actual numerics live in {altar.distributions.native.PositiveUniform.PositiveUniform};
    there is no cuda counterpart today.
    """


    # a finite interval is a strict subset of the reals
    bounded = True


# end of file
