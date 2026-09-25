# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-2021 parasim inc
# (c) 2010-2021 california institute of technology
# all rights reserved
#

# externals
import math


# the declaration
class Base:
    """
    Shared support for the cpu implementation of a distribution: a plain class, not a pyre
    component -- the registered component is the shim in {altar.distributions} that owns
    {parameters}/{offset} as configurable traits and copies their values down to me, once, at
    {initialize} time.

    A concrete distribution overrides {initialize} and {verify}; the rest are sensible
    defaults (a flat prior, forwarding to {self.pdf}) that most distributions never need to
    touch.
    """


    def initialize(self, rng, application=None):
        # being abstract, i don't know what to do here
        raise NotImplementedError(
            f"class '{type(self).__name__}' must implement 'initialize'")


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # fill it with random numbers from my initializer
        self.pdf.matrix(matrix=θ)
        # and return
        return self


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the log prior probabilities of the samples in
        {theta}
        """
        # get my pdf implementation
        pdf = self.pdf
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # find out how may samples there are
        samples = θ.rows

        # for each one
        for sample in range(samples):
            # fill the vector with the log likelihoods; a sample far enough into the tail can
            # underflow {density} to exactly zero (e.g. a gradient-based sampler's trajectory
            # straying through an extreme, strongly-rejected region), and {math.log} raises
            # rather than returning {-inf} for that, so guard it explicitly
            likelihood[sample] += sum(
                math.log(density) if (density := pdf.density(parameter)) > 0 else float('-inf')
                for parameter in θ.getRow(sample))

        # all done
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta, elementwise, for the
        samples in {theta}. {gradient} has the same shape as {theta}.
        """
        # default: assume a flat (improper) prior, so the gradient is 0
        self.restrict(theta=gradient).zero()
        # all done
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # being abstract, i don't know what to do here
        raise NotImplementedError(
            f"class '{type(self).__name__}' must implement 'verify'")


    def constrain(self, theta, batch=None):
        """
        Force my portion of the samples in {theta} back within my constraints, in place. A
        cpu sampler rejects through {verify} instead, so there is nothing to do here.
        """
        return self


    def eval_prior_with_physical(self, theta, likelihood, batch=None):
        """
        Add any prior contributions to {likelihood} that depend on my physical parameters,
        beyond what {eval_prior} already contributed. Default: nothing further to add.
        """
        return self


    def eval_prior_physical(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the log prior probabilities of the samples in
        {theta}, given in physical space. Without reparameterization, physical space is
        sampling space, so the default is just {eval_prior}.
        """
        return self.eval_prior(theta=theta, likelihood=likelihood, batch=batch)


    def to_physical(self, theta, batch=None):
        """
        Transform my portion of {theta} from sampling space to physical space, in place.
        Without reparameterization, the two coincide, so the default is a no-op.
        """
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform my portion of {theta} from physical space to sampling space, in place. The
        inverse of {to_physical}; the default is likewise a no-op.
        """
        return self


    # the forwarding interface
    def sample(self):
        """
        Sample the distribution using a random number generator
        """
        # ask my pdf
        return self.pdf.sample()


    def density(self, x):
        """
        Compute the probability density of the distribution at {x}
        """
        # ask my pdf
        return self.pdf.density(x)


    def vector(self, vector):
        """
        Fill {vector} with random values
        """
        # ask my pdf
        return self.pdf.vector(vector)


    def matrix(self, matrix):
        """
        Fill {matrix} with random values
        """
        # ask my pdf
        return self.pdf.matrix(matrix)


    # implementation details
    def restrict(self, theta):
        """
        Return my portion of the {theta}
        """
        # find out how many samples in the set
        samples = theta.rows
        # find where my samples live within the overall sample matrix, and how wide my slice is
        start = 0, self.offset
        shape = samples, self.parameters
        # return the portion of the sample that's mine
        return theta.view(start=start, shape=shape)


    # private data, set by the shim before any other method runs
    parameters = None
    offset = None
    # set by {initialize}
    pdf = None


# end of file
