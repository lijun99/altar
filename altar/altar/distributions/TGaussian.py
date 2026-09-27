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
class TGaussian(base, family="altar.distributions.tgaussian"):
    """
    The Gaussian probability distribution, truncated to a finite support

    My actual numerics live in {altar.distributions.native.TGaussian.TGaussian} (cpu) or
    {altar.distributions.cuda.TGaussian.TGaussian}; see {Base} for how one gets picked.
    """


    # user configurable state
    mean = altar.properties.float(default=0)
    mean.doc = "the mean value of the underlying (untruncated) distribution"

    sigma = altar.properties.float(default=1)
    sigma.doc = "the standard deviation of the underlying (untruncated) distribution"

    support = altar.properties.array(default=(0,1))
    support.doc = "the support interval of the truncated distribution"

    # a finite interval is a strict subset of the reals
    bounded = True

    # reparameterization, for gradient-based samplers (HMC, SGLD)
    reparameterize = altar.properties.bool(default=False)
    reparameterize.doc = \
        "whether to reparameterize to an unconstrained sampling space via {transform}, " \
        "for use by gradient-based samplers"

    transform = altar.distributions.transform()
    transform.doc = "the transform used to map between physical and sampling space, when " \
                     "{reparameterize} is set"


    # implementation details
    def _makeImpl(self):
        """
        Hand my implementation my own extra state, beyond what {Base} already copies over
        """
        impl = super()._makeImpl()
        impl.mean = self.mean
        impl.sigma = self.sigma
        impl.support = self.support
        impl.reparameterize = self.reparameterize
        impl.transform = self.transform
        return impl


# end of file
