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
class Uniform(base):
    """
    The cpu implementation of the uniform probability distribution
    """


    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator
        """
        # set up my pdf
        self.pdf = altar.pdf.uniform(rng=rng.rng, support=self.support)
        # set up my transform, if reparameterizing
        self._initialize_transform(application=application)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones;
        {theta} is physical, reparameterized or not
        """
        # mark the samples with a parameter outside my support
        return self.outside(theta=theta, mask=mask, support=self.support)


    def log_density(self, x):
        """
        The log density of each entry of {x}: -log(high - low) on [low, high), as gsl's
        """
        low, high = self.support
        return numpy.where((x >= low) & (x < high), -math.log(high - low), -numpy.inf)


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta; when reparameterized,
        this is exactly the transform's jacobian-gradient, since a uniform prior's
        physical-space gradient is always zero
        """
        g = self.restrict(theta=gradient)
        if self.reparameterize:
            θ = self.restrict(theta=theta)
            self.transform.jacobian_gradient(theta=θ, gradient=g, batch=batch)
        else:
            g.zero()
        return self


    # private data, set by the shim before {initialize} runs
    support = None


# end of file
