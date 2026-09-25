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
class Contiguous(base):
    """
    The cpu implementation of a contiguous parameter set
    """


    def initialize(self, model, offset, application=None):
        """
        Initialize my state given the {model} that owns me
        """
        # set my offset
        self.offset = offset

        # get my count
        count = self.count
        # adjust the number of parameters of my distributions
        self.prior.parameters = count
        # get the random number generator
        rng = model.rng
        # initialize my prior
        self.prior.initialize(rng=rng)

        # a parameter set with no {prep} of its own initializes samples from its prior instead
        if self.prep is not None:
            self.prep.parameters = count
            self.prep.initialize(rng=rng)
        else:
            self.prep = self.prior

        # return my parameter count so the next set can be initialized properly
        return count


    def initialize_sample(self, theta, batch=None):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # grab the portion of the sample that belongs to me
        θ = self.restrict(theta=theta)
        # fill it with random numbers from my {prep} distribution
        self.prep.initialize_sample(theta=θ)
        # all done
        return self


    def eval_prior(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # delegate
        self.prior.eval_prior(theta=θ, likelihood=prior)
        # all done
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill {gradient} with d\log P(\theta)/d\theta for my portion of the samples in
        {theta}, for use by gradient-based samplers (e.g. SGLD)
        """
        # grab the portion of the sample and gradient that are mine
        θ = self.restrict(theta=theta)
        g = self.restrict(theta=gradient)
        # delegate
        self.prior.prior_gradient(theta=θ, gradient=g)
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether the samples in {theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # ask my prior to verify my samples
        self.prior.verify(theta=θ, mask=mask)
        # all done; return the rejection map
        return mask


    # private data, set by the shim before {initialize} runs
    prior = None
    prep = None


# end of file
