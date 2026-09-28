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
import numpy
# get the package
import altar
# and my base class
from .Base import Base as base


# the declaration
class SoftUniform(base):
    """
    The cpu implementation of the uniform distribution with logistic edges
    """

    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator; the initial samples are uniform
        over my support
        """
        self.pdf = altar.pdf.uniform(rng=rng.rng, support=self.support)
        return self


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Add the log prior probabilities of my portion of the samples in {theta} to {likelihood}:
        -log(b - a) + log(1 - exp(-k (b - a))) - softplus(-k (x - a)) - softplus(k (x - b))
        """
        low, high = self.support
        k = self.sharpness
        x = numpy.asarray(self.restrict(theta=theta))
        constant = -numpy.log(high - low) + numpy.log1p(-numpy.exp(-k * (high - low)))
        logpdf = constant - numpy.logaddexp(0, -k * (x - low)) - numpy.logaddexp(0, k * (x - high))
        numpy.asarray(likelihood)[:] += logpdf.sum(axis=1)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta =
        k (sigmoid(-k (x - a)) - sigmoid(k (x - b)))
        """
        low, high = self.support
        k = self.sharpness
        x = numpy.asarray(self.restrict(theta=theta))
        g = numpy.asarray(self.restrict(theta=gradient))
        g[:] = k * (0.5 * (1 + numpy.tanh(-0.5 * k * (x - low))) - 0.5 * (1 + numpy.tanh(0.5 * k * (x - high))))
        return self


    def verify(self, theta, mask, batch=None):
        """
        Nothing to check: I am positive everywhere
        """
        return mask


    # private data, set by the shim before {initialize} runs
    support = None
    sharpness = None


# end of file
