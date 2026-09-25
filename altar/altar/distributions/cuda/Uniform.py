# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# and my base class
from .Base import Base as base


# the declaration
class Uniform(base):
    """
    The cuda implementation of the uniform probability distribution

    {initialize} is inherited unchanged: i have no cuda-side state of my own beyond the
    generic setup {Base} already does.
    """


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values from my distribution
        """
        low, high = self.support
        self.libcudaaltar.cudaUniform_sample(self._grid(theta), self.idx_begin, self.idx_end, low, high)
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones;
        {mask} must be an int32 grid
        """
        low, high = self.support
        self.libcudaaltar.cudaRanged_verify(
            self._grid(theta), self._grid(mask), self.idx_begin, self.idx_end, low, high)
        return mask


    def constrain(self, theta, batch=None):
        """
        Force my portion of the samples in {theta} back within my support, in place
        """
        low, high = self.support
        self.libcudaaltar.cudaRanged_constrain(self._grid(theta), self.idx_begin, self.idx_end, low, high)
        return self


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the prior log-probabilities of the samples in
        {theta}

        {logpdf} *accumulates* into {likelihood}: the caller must have zeroed it first if
        this is meant to be the only contribution
        """
        low, high = self.support
        self.libcudaaltar.cudaUniform_logpdf(
            self._grid(theta), self._grid(likelihood), self.idx_begin, self.idx_end, low, high)
        return self


    # private data, set by the shim before {initialize} runs
    support = None


# end of file
