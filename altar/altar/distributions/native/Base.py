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
import numpy


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
        # grab the portion of the sample that's mine
        θ = numpy.asarray(self.restrict(theta=theta))
        # sum the log densities of each sample's parameters
        numpy.asarray(likelihood)[:θ.shape[0]] += self.log_density(θ).sum(axis=1)
        # all done
        return self


    def log_density(self, x):
        """
        The log density of each entry of the numpy array {x}; the default asks {self.pdf}, one
        entry at a time, and a distribution with a closed form overrides it
        """
        density = numpy.vectorize(self.pdf.density, otypes=[float])(x)
        # a density that underflows to zero, far into a tail, is a log density of -inf
        with numpy.errstate(divide="ignore"):
            return numpy.log(density)


    def outside(self, theta, mask, support):
        """
        Mark in {mask} the samples in {theta} with a parameter outside {support}; a NaN is
        outside too
        """
        θ = numpy.asarray(self.restrict(theta=theta))
        low, high = support
        inside = ((θ >= low) & (θ <= high)).all(axis=1)
        numpy.asarray(mask)[:θ.shape[0]] += ~inside
        return mask


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


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill my portion of {jacobian} with d(physical)/d(sampling) when reparameterized;
        otherwise {jacobian} already holds 1, the right value
        """
        if self.reparameterize:
            self.transform.jacobian(
                theta=self.restrict(theta=theta), jacobian=self.restrict(theta=jacobian), batch=batch)
        return self


    def eval_prior_with_physical(self, theta, likelihood, batch=None):
        """
        Add my log|J| into {likelihood} when reparameterized; otherwise nothing to add
        """
        if self.reparameterize:
            self.transform.log_jacobian(
                theta=self.restrict(theta=theta), likelihood=likelihood, batch=batch)
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
        Transform my portion of {theta} from sampling space to physical space, in place; a
        no-op unless reparameterized
        """
        if self.reparameterize:
            self.transform.to_physical(theta=self.restrict(theta=theta), batch=batch)
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform my portion of {theta} from physical space to sampling space, in place; a
        no-op unless reparameterized
        """
        if self.reparameterize:
            self.transform.to_sampling(theta=self.restrict(theta=theta), batch=batch)
        return self


    def _initialize_transform(self, application=None):
        """
        For a bounded distribution with a {reparameterize} trait: hand my transform my
        {support} and let it initialize
        """
        if self.reparameterize:
            self.has_reparametrization = True
            self.transform.support = self.support
            self.transform.initialize(application=application)
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
    # mirrored back onto the shim after {initialize}; a concrete distribution sets this to
    # True in its own {initialize} when reparameterizing (see {Uniform})
    has_reparametrization = False
    # set by the shim of a distribution that supports reparameterization (e.g. {Uniform})
    reparameterize = False
    transform = None
    # set by {initialize}
    pdf = None


# end of file
