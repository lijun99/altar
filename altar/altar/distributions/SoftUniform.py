# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the package
import altar
# my base class
from .Base import Base as base


# the declaration
class SoftUniform(base, family="altar.distributions.softuniform"):
    """
    A uniform distribution with logistic edges (Minson, 2024, eq. 16): the normalized difference
    of two logistic functions of sharpness k, at the ends of {support}. It approaches the uniform
    distribution over {support} as k grows, but is smooth and positive everywhere, so that raising
    it to a power reshapes it, as cross-fade sampling does to the prior

    My actual numerics live in {altar.distributions.native.SoftUniform.SoftUniform} (cpu) or
    {altar.distributions.cuda.SoftUniform.SoftUniform}; see {Base} for how one gets picked.
    """

    # user configurable state
    support = altar.properties.array(default=(0,1))
    support.doc = "the interval the distribution is nearly uniform over"

    sharpness = altar.properties.float(default=None)
    sharpness.doc = "the steepness k of the logistic edges, in inverse units of the parameter; " \
                    "by default 100 over the width of {support}"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.support = self.support
        low, high = self.support
        impl.sharpness = self.sharpness if self.sharpness is not None else 100 / (high - low)
        return impl


# end of file
