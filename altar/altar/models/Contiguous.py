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


# component
class Contiguous(base, family="altar.models.parameters.contiguous"):
    """
    A contiguous parameter set

    My actual numerics live in {altar.models.native.Contiguous.Contiguous} (cpu) or
    {altar.models.cuda.Contiguous.Contiguous}; see {Base} for how one gets picked.
    """


    # user configurable state
    count = altar.properties.int(default=1)
    count.doc = "the number of parameters in this set"

    prior = altar.distributions.distribution()
    prior.doc = "the prior distribution"

    prep = altar.distributions.distribution(default=None)
    prep.doc = "the distribution to use to initialize this parameter set; falls back to " \
               "{prior} when not given"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.prior = self.prior
        impl.prep = self.prep
        return impl


# end of file
