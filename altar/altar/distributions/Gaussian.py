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


    # interface
    def gaussian_form(self):
        """
        My mean and precision, unbounded
        """
        n = self.parameters
        return (numpy.full(n, float(self.mean)), numpy.identity(n) / self.sigma ** 2,
                numpy.full(n, -numpy.inf), numpy.full(n, numpy.inf))


# end of file
