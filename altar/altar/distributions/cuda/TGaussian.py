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
from math import erf, sqrt

# and my base class
from .Base import Base as base


# the declaration
class TGaussian(base):
    """
    The cuda implementation of the Gaussian probability distribution, truncated to a finite
    support
    """


    def initialize(self, rng, application=None):
        """
        The generic cuda-side setup, plus my own: the support normalized to the underlying
        (untruncated) distribution's cdf, which is the form the sampling kernel needs
        """
        # the generic setup
        super().initialize(rng=rng, application=application)
        # Phi(x) = 1/2 (1 + erf((x - mean) / (sqrt(2) sigma)))
        sqrt2sigma = sqrt(2.0) * self.sigma
        Phi = lambda x: 0.5 + 0.5 * erf((x - self.mean) / sqrt2sigma)
        low, high = self.support
        self.support_normalized = (Phi(low), Phi(high))
        # all done
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values from my distribution
        """
        low, high = self.support_normalized
        self.libcudaaltar.cudaTGaussian_sample(
            self._grid(theta), self.idx_begin, self.idx_end, self.mean, self.sigma, low, high)
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones;
        {mask} must be an int32 grid. Verification happens in physical space, against the raw
        (not normalized) {support}, unlike {initialize_sample}/{eval_prior}
        """
        low, high = self.support
        self.libcudaaltar.cudaRanged_verify(
            self._grid(theta), self._grid(mask), self.idx_begin, self.idx_end, low, high)
        return mask


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the prior log-probabilities of the samples in
        {theta}

        {logpdf} *accumulates* into {likelihood}: the caller must have zeroed it first if
        this is meant to be the only contribution
        """
        low, high = self.support_normalized
        self.libcudaaltar.cudaTGaussian_logpdf(
            self._grid(theta), self._grid(likelihood), self.idx_begin, self.idx_end, self.mean, self.sigma, low, high)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta. Truncation only rescales
        the normalization constant, so within the support this is the same kernel the plain
        (untruncated) Gaussian uses. A plain assignment over columns [idx_begin, idx_end),
        not an accumulation.
        """
        self.libcudaaltar.cudaGaussian_logpdfgradient(
            self._grid(theta), self._grid(gradient), self.idx_begin, self.idx_end, self.mean, self.sigma)
        return self


    # private data, set by the shim before {initialize} runs
    mean = None
    sigma = None
    support = None
    # set by {initialize}
    support_normalized = None


# end of file
