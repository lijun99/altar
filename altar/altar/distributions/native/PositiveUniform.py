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
import altar

# and my base class
from .Base import Base as base


# the declaration
class PositiveUniform(base):
    """
    The cpu implementation of the uniform probability distribution over (0, 1), excluding 0
    """


    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator
        """
        # set up my pdf
        self.pdf = altar.pdf.uniform_pos(rng=rng.rng)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # every sample the generator itself produces is already in (0, 1), so there is
        # nothing to check
        return mask


# end of file
