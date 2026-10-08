# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import math
import numpy
# get the package
import altar

# and my base class
from .Base import Base as base


# the declaration
class UnitGaussian(base):
    """
    The cpu implementation of the unit Gaussian probability distribution (σ = 1)
    """


    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator
        """
        # set up my pdf
        self.pdf = altar.pdf.ugaussian(rng=rng.rng)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # all samples are valid, so there is nothing to do
        return mask


    def log_density(self, x):
        """
        The log density of each entry of {x}
        """
        return -0.5 * x * x - 0.5 * math.log(2 * math.pi)


# end of file
