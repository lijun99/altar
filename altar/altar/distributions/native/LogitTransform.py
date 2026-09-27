# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

import numpy


# the declaration
class LogitTransform:
    """
    The cpu implementation of {altar.distributions.transforms.Transform.LogitTransform}:
    physical = a + (b-a)*sigmoid(sampling); sampling = logit((physical-a)/(b-a)). Plain numpy
    math on a gsl matrix already *is* native cpu computation, so this is unchanged from (and
    numerically identical to) the shim's own body before it grew a native/cuda split.
    """

    def initialize(self, application=None):
        """
        Nothing to set up beyond {support}, already handed to me by my owning distribution
        """
        return self


    def to_physical(self, theta, batch=None):
        """
        theta <- a + (b-a)*sigmoid(theta), in place
        """
        a, b = self.support
        arr = numpy.asarray(theta)
        arr[:] = a + (b - a) / (1.0 + numpy.exp(-arr))
        return self


    def to_sampling(self, theta, batch=None):
        """
        theta <- logit((theta-a)/(b-a)), in place; the inverse of {to_physical}
        """
        a, b = self.support
        arr = numpy.asarray(theta)
        arr[:] = numpy.log((arr - a) / (b - arr))
        return self


    def log_jacobian(self, theta, likelihood, batch=None):
        """
        Add the standard-logistic log-pdf into {likelihood}, summed over the parameters this
        transform owns. {theta} is PHYSICAL space; see the shim's docstring for why.
        """
        a, b = self.support
        x = numpy.asarray(theta)
        sig = (x - a) / (b - a)
        # log(sig) + log(1-sig); dropping the constant log(b-a) term this omits changes
        # nothing downstream -- it cancels exactly in any delta-H/acceptance decision
        contribution = numpy.log(sig) + numpy.log(1.0 - sig)
        numpy.asarray(likelihood)[:] += contribution.sum(axis=1)
        return self


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill {jacobian} with d(physical)/d(sampling) = (b-a)*sig*(1-sig)
        """
        a, b = self.support
        x = numpy.asarray(theta)
        sig = (x - a) / (b - a)
        numpy.asarray(jacobian)[:] = (b - a) * sig * (1.0 - sig)
        return self


    def jacobian_gradient(self, theta, gradient, batch=None):
        """
        Fill {gradient} with d/d(sampling)[log(sig) + log(1-sig)] = 1 - 2*sig
        """
        a, b = self.support
        x = numpy.asarray(theta)
        sig = (x - a) / (b - a)
        numpy.asarray(gradient)[:] = 1.0 - 2.0 * sig
        return self


    def chain_gradient(self, theta, gradient, batch=None):
        """
        gradient <- gradient*(b-a)*sig*(1-sig) + (1 - 2*sig), in place
        """
        a, b = self.support
        sig = (numpy.asarray(theta) - a) / (b - a)
        g = numpy.asarray(gradient)
        g[:] = g * (b - a) * sig * (1.0 - sig) + 1.0 - 2.0 * sig
        return self


    # private data, set by the shim before {initialize} runs
    support = None
    idx_begin = None  # unused on cpu; native's caller already hands me a pre-sliced view
    idx_end = None


# end of file
