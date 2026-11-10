# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# get the package
import numpy
import altar

# get the protocol
from . import distribution
# and my base class
from .Base import Base as base


# the declaration
class UnitGaussian(base, family="altar.distributions.ugaussian"):
    """
    Special case of the Gaussian probability distribution with σ = 1

    My actual numerics live in {altar.distributions.native.UnitGaussian.UnitGaussian}; there
    is no cuda counterpart today, matching the status quo before this refactor.
    """


    # interface
    def gaussian_form(self):
        """
        A zero mean and the identity, unbounded
        """
        n = self.parameters
        return numpy.zeros(n), numpy.identity(n), numpy.full(n, -numpy.inf), numpy.full(n, numpy.inf)


# end of file
