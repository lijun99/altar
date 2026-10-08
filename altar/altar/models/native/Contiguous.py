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
from __future__ import annotations
import typing
import numpy

if typing.TYPE_CHECKING:
    from altar.distributions.Distribution import Distribution
    from altar.models.BayesianL2 import BayesianL2
    from altar.shells.Application import Application

# and my base class
from .Base import Base as base


# the declaration
class Contiguous(base):
    """
    The cpu implementation of a contiguous parameter set
    """


    def initialize(self, model: BayesianL2, offset: int,
                   application: Application | None = None) -> int:
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
        self.prior.initialize(rng=rng, application=application)

        # a parameter set with no {prep} of its own initializes samples from its prior instead
        if self.prep is not None:
            self.prep.parameters = count
            self.prep.initialize(rng=rng, application=application)
        else:
            self.prep = self.prior

        # return my parameter count so the next set can be initialized properly
        return count


    def initialize_sample(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        # grab the portion of the sample that belongs to me
        θ = self.restrict(theta=theta)
        # fill it with random numbers from my {prep} distribution
        self.prep.initialize_sample(theta=θ)
        # all done
        return self


    def eval_prior(self, theta: numpy.ndarray, prior: numpy.ndarray,
                   batch: int | None = None) -> typing.Self:
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # delegate
        self.prior.eval_prior(theta=θ, likelihood=prior)
        # all done
        return self


    def prior_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
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


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray,
               batch: int | None = None) -> numpy.ndarray:
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


    def eval_prior_with_physical(self, theta: numpy.ndarray, prior: numpy.ndarray,
                                 batch: int | None = None) -> typing.Self:
        """
        Add any prior contributions that depend on physical parameters, e.g. the log-jacobian
        of a reparameterized prior
        """
        self.prior.eval_prior_with_physical(theta=self.restrict(theta=theta), likelihood=prior, batch=batch)
        return self


    def jacobian(self, theta: numpy.ndarray, jacobian: numpy.ndarray,
                 batch: int | None = None) -> typing.Self:
        """
        Fill my portion of {jacobian} with d(physical)/d(sampling), when reparameterized
        """
        self.prior.jacobian(
            theta=self.restrict(theta=theta), jacobian=self.restrict(theta=jacobian), batch=batch)
        return self


    def to_physical(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Transform my portion of {theta} from sampling space to physical space, in place
        """
        self.prior.to_physical(theta=self.restrict(theta=theta), batch=batch)
        return self


    def to_sampling(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Transform my portion of {theta} from physical space to sampling space, in place
        """
        self.prior.to_sampling(theta=self.restrict(theta=theta), batch=batch)
        return self


    # private data, set by the shim before {initialize} runs
    prior: Distribution | None = None
    prep: Distribution | None = None


# end of file
