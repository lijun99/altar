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

# get the protocol
from . import distribution
# and my base class
from .Base import Base as base


# the declaration
class Uniform(base, family="altar.distributions.uniform"):
    """
    The uniform probability distribution

    My actual numerics live in {altar.distributions.native.Uniform.Uniform} (cpu) or
    {altar.distributions.cuda.Uniform.Uniform}; see {Base} for how one gets picked.
    """


    # user configurable state
    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the prior distribution"

    # a finite interval is a strict subset of the reals
    bounded = True


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.support = self.support
        return impl


# end of file
