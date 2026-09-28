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
class Uniform(base):
    """
    The cuda implementation of the uniform probability distribution

    {initialize} is inherited unchanged: i have no cuda-side state of my own beyond the
    generic setup {Base} already does.
    """


    def initialize(self, rng, application=None):
        """
        The generic cuda-side setup, plus, if reparameterizing, handing my transform its
        bounds and letting it initialize
        """
        super().initialize(rng=rng, application=application)
        self._initialize_transform(application=application)
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values from my distribution, in
        physical space -- the model bridges sampling/physical space for reparameterized
        priors once at startup (see {altar.models.BayesianL2.initialize_sample})
        """
        low, high = self.support
        self.libcudaaltar.cudaUniform_sample(self._grid(theta), self.idx_begin, self.idx_end, low, high)
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones;
        {mask} must be an int32 grid; {theta} is physical, reparameterized or not
        """
        low, high = self.support
        self.libcudaaltar.cudaRanged_verify(
            self._grid(theta), self._grid(mask), self.idx_begin, self.idx_end, low, high)
        return mask


    def constrain(self, theta, batch=None):
        """
        Force my portion of the samples in {theta} back within my support, in place; a no-op
        when reparameterized
        """
        if self.reparameterize:
            return self
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


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta; when reparameterized,
        this is exactly the transform's jacobian-gradient, since a uniform prior's
        physical-space gradient is always zero
        """
        if self.reparameterize:
            self.transform.jacobian_gradient(theta=theta, gradient=gradient, batch=batch)
        return self


    # private data, set by the shim before {initialize} runs
    support = None


# end of file
