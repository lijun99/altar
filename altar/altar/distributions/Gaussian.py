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
class Gaussian(base, family="altar.distributions.gaussian"):
    """
    The Gaussian probability distribution

    My actual numerics live in {altar.distributions.native.Gaussian.Gaussian} (cpu) or
    {altar.distributions.cuda.Gaussian.Gaussian}; see {Base} for how one gets picked.
    """


    # user configurable state
    mean = altar.properties.float(default=0)
    mean.doc = "the mean value of the distribution"

    sigma = altar.properties.float(default=1)
    sigma.doc = "the standard deviation of the distribution"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.mean = self.mean
        impl.sigma = self.sigma
        return impl


# end of file
