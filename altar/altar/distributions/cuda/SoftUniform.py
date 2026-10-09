# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# and my base class
from .Base import Base as base


# the declaration
class SoftUniform(base):
    """
    The cuda implementation of the uniform distribution with logistic edges
    """

    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values, uniform over my support
        """
        low, high = self.support
        self.libcudaaltar.cudaUniform_sample(self._grid(theta), self.idx_begin, self.idx_end, low, high)
        return self


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Add the log prior probabilities of my portion of the samples in {theta} to {likelihood}
        """
        low, high = self.support
        self.libcudaaltar.cudaUniform_softlogpdf(
            self._grid(theta), self._grid(likelihood), self.idx_begin, self.idx_end,
            low, high, self.sharpness)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta
        """
        low, high = self.support
        self.libcudaaltar.cudaUniform_softgradient(
            self._grid(theta), self._grid(gradient), self.idx_begin, self.idx_end,
            low, high, self.sharpness)
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
