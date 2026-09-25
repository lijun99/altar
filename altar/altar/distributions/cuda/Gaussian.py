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
class Gaussian(base):
    """
    The cuda implementation of the Gaussian probability distribution

    {initialize} is inherited unchanged: i have no cuda-side state of my own beyond the
    generic setup {Base} already does.
    """


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values from my distribution

        {batch} is accepted for interface conformance with the other distributions'
        callers, but not forwarded: the extension always processes every row of {theta}, not
        a partial batch
        """
        self.libcudaaltar.cudaGaussian_sample(
            self._grid(theta), self.idx_begin, self.idx_end, self.mean, self.sigma)
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # all samples are valid, so there is nothing to do
        return mask


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the prior log-probabilities of the samples in
        {theta}

        {logpdf} *accumulates* into {likelihood}: the caller must have zeroed it first if
        this is meant to be the only contribution
        """
        self.libcudaaltar.cudaGaussian_logpdf(
            self._grid(theta), self._grid(likelihood), self.idx_begin, self.idx_end, self.mean, self.sigma)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta

        Unlike {eval_prior}, this is a plain assignment over columns [idx_begin, idx_end),
        not an accumulation
        """
        self.libcudaaltar.cudaGaussian_logpdfgradient(
            self._grid(theta), self._grid(gradient), self.idx_begin, self.idx_end, self.mean, self.sigma)
        return self


    # private data, set by the shim before {initialize} runs
    mean = None
    sigma = None


# end of file
