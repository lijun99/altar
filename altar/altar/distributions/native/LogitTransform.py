# -*- python -*-
# -*- coding: utf-8 -*-
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
    from altar.shells.Application import Application


# the declaration
class LogitTransform:
    """
    The cpu implementation of {altar.distributions.transforms.Transform.LogitTransform}:
    physical = a + (b-a)*sigmoid(sampling); sampling = logit((physical-a)/(b-a)). The samples
    {theta} are the (samples x parameters) columns of the distribution that owns me, and
    {likelihood} a (samples,) array
    """

    def initialize(self, application: Application | None = None) -> typing.Self:
        """
        Nothing to set up beyond {support}, already handed to me by my owning distribution
        """
        return self


    def to_physical(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        theta <- a + (b-a)*sigmoid(theta), in place
        """
        a, b = self.support
        theta[...] = a + (b - a) / (1.0 + numpy.exp(-theta))
        return self


    def to_sampling(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        theta <- logit((theta-a)/(b-a)), in place; the inverse of {to_physical}
        """
        a, b = self.support
        theta[...] = numpy.log((theta - a) / (b - theta))
        return self


    def log_jacobian(self, theta: numpy.ndarray, likelihood: numpy.ndarray,
                     batch: int | None = None) -> typing.Self:
        """
        Add the standard-logistic log-pdf into {likelihood}, summed over the parameters this
        transform owns. {theta} is PHYSICAL space; see the shim's docstring for why.
        """
        a, b = self.support
        sig = (theta - a) / (b - a)
        # log(sig) + log(1-sig); dropping the constant log(b-a) term this omits changes
        # nothing downstream -- it cancels exactly in any delta-H/acceptance decision
        contribution = numpy.log(sig) + numpy.log(1.0 - sig)
        likelihood[:theta.shape[0]] += contribution.sum(axis=1)
        return self


    def jacobian(self, theta: numpy.ndarray, jacobian: numpy.ndarray,
                 batch: int | None = None) -> typing.Self:
        """
        Fill {jacobian} with d(physical)/d(sampling) = (b-a)*sig*(1-sig)
        """
        a, b = self.support
        sig = (theta - a) / (b - a)
        jacobian[...] = (b - a) * sig * (1.0 - sig)
        return self


    def jacobian_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                          batch: int | None = None) -> typing.Self:
        """
        Fill {gradient} with d/d(sampling)[log(sig) + log(1-sig)] = 1 - 2*sig
        """
        a, b = self.support
        sig = (theta - a) / (b - a)
        gradient[...] = 1.0 - 2.0 * sig
        return self


    def chain_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
        """
        gradient <- gradient*(b-a)*sig*(1-sig) + (1 - 2*sig), in place
        """
        a, b = self.support
        sig = (theta - a) / (b - a)
        gradient[...] = gradient * (b - a) * sig * (1.0 - sig) + 1.0 - 2.0 * sig
        return self


    # private data, set by the shim before {initialize} runs
    support: tuple[float, float]
    idx_begin: int | None = None  # unused on cpu; native's caller already hands me a pre-sliced view
    idx_end: int | None = None


# end of file
