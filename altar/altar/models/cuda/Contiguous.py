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
class Contiguous(base):
    """
    The cuda implementation of a contiguous parameter set
    """


    def initialize(self, model, offset, application=None):
        """
        Initialize my state given the current {application}; {model} is unused on cuda
        """
        # set my offset
        self.offset = offset

        # get my count
        count = self.count
        # adjust the number of parameters of my distributions
        self.prior.parameters = count
        self.prior.offset = offset
        # initialize my prior; {rng} is unused on cuda, so pass nothing meaningful
        self.prior.initialize(rng=None, application=application)

        # a parameter set with no {prep} of its own initializes samples from its prior instead
        if self.prep is not None:
            self.prep.parameters = count
            self.prep.offset = offset
            self.prep.initialize(rng=None, application=application)
        else:
            self.prep = self.prior

        # return my parameter count so the next set can be initialized properly
        return count


    def initialize_sample(self, theta, batch=None):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        self.prep.initialize_sample(theta=theta, batch=batch)
        return self


    def eval_prior(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        self.prior.eval_prior(theta=theta, likelihood=prior, batch=batch)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill {gradient} with d\log P(\theta)/d\theta for my portion of the samples in {theta}
        """
        self.prior.prior_gradient(theta=theta, gradient=gradient, batch=batch)
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether the samples in {theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        self.prior.verify(theta=theta, mask=mask, batch=batch)
        return mask


    def constrain(self, theta, batch=None):
        """
        Force the samples in {theta} back within my constraints, in place
        """
        self.prior.constrain(theta=theta, batch=batch)
        return self


    def eval_prior_with_physical(self, theta, prior, batch=None):
        """
        Add any prior contributions that depend on physical parameters, beyond what
        {eval_prior} already contributed
        """
        self.prior.eval_prior_with_physical(theta=theta, likelihood=prior, batch=batch)
        return self


    def eval_prior_physical(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta}, given in physical space
        """
        self.prior.eval_prior_physical(theta=theta, likelihood=prior, batch=batch)
        return self


    def to_physical(self, theta, batch=None):
        """
        Transform {theta} from sampling space to physical space, in place; only distributions
        with reparameterization actually do anything here
        """
        if self.prior.has_reparametrization:
            self.prior.to_physical(theta=theta, batch=batch)
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform {theta} from physical space to sampling space, in place; the inverse of
        {to_physical}
        """
        if self.prior.has_reparametrization:
            self.prior.to_sampling(theta=theta, batch=batch)
        return self


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill {jacobian} with d(physical)/d(sampling) for my portion of {theta}, or leave my
        portion at its default of 1 when i'm not reparameterized
        """
        if self.prior.has_reparametrization:
            self.prior.jacobian(theta=theta, jacobian=jacobian, batch=batch)
        return self


    # private data, set by the shim before {initialize} runs
    prior = None
    prep = None


# end of file
