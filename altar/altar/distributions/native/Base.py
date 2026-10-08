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
    from altar.shells.Application import Application
    from altar.simulations.NumpyRNG import NumpyRNG
    from ..transforms.Transform import Transform


# the declaration
class Base:
    """
    Shared support for the cpu implementation of a distribution: a plain class, not a pyre
    component -- the registered component is the shim in {altar.distributions} that owns
    {parameters}/{offset} as configurable traits and copies their values down to me, once, at
    {initialize} time.

    A concrete distribution provides {draw}, {log_density} and {verify}; the rest are sensible
    defaults that most distributions never need to touch. The samples {theta} are
    (samples x parameters) arrays, of which I own the columns {offset} to {offset+parameters};
    {likelihood} and {mask} are (samples,) arrays.
    """


    def initialize(self, rng: NumpyRNG, application: Application | None = None) -> typing.Self:
        """
        Hold on to the numpy generator of the {rng} component
        """
        self.rng = rng.rng
        return self


    def initialize_sample(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # fill it with random numbers from my initializer
        θ[...] = self.draw(shape=θ.shape)
        # and return
        return self


    def draw(self, shape: tuple[int, ...]) -> numpy.ndarray:
        """
        An array of {shape} with random values from my distribution
        """
        # being abstract, i don't know what to do here
        raise NotImplementedError(
            f"class '{type(self).__name__}' must implement 'draw'")


    def eval_prior(self, theta: numpy.ndarray, likelihood: numpy.ndarray,
                   batch: int | None = None) -> typing.Self:
        """
        Add to {likelihood} the log prior probabilities of my portion of the samples in {theta}
        """
        # grab the portion of the sample that's mine
        θ = self.restrict(theta=theta)
        # sum the log densities of each sample's parameters
        likelihood[:θ.shape[0]] += self.log_density(θ).sum(axis=1)
        # all done
        return self


    def log_density(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        The log density of each entry of {x}
        """
        # being abstract, i don't know what to do here
        raise NotImplementedError(
            f"class '{type(self).__name__}' must implement 'log_density'")


    def outside(self, theta: numpy.ndarray, mask: numpy.ndarray,
                support: tuple[float, float]) -> numpy.ndarray:
        """
        Mark in {mask} the samples in {theta} with a parameter outside {support}; a NaN is
        outside too
        """
        θ = self.restrict(theta=theta)
        low, high = support
        inside = ((θ >= low) & (θ <= high)).all(axis=1)
        mask[:θ.shape[0]] += ~inside
        return mask


    def prior_gradient(self, theta: numpy.ndarray, gradient: numpy.ndarray,
                       batch: int | None = None) -> typing.Self:
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta, elementwise, for the
        samples in {theta}. {gradient} has the same shape as {theta}.
        """
        # default: assume a flat (improper) prior, so the gradient is 0
        self.restrict(theta=gradient)[...] = 0
        # all done
        return self


    def verify(self, theta: numpy.ndarray, mask: numpy.ndarray,
               batch: int | None = None) -> numpy.ndarray:
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        # being abstract, i don't know what to do here
        raise NotImplementedError(
            f"class '{type(self).__name__}' must implement 'verify'")


    def constrain(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Force my portion of the samples in {theta} back within my constraints, in place. A
        cpu sampler rejects through {verify} instead, so there is nothing to do here.
        """
        return self


    def jacobian(self, theta: numpy.ndarray, jacobian: numpy.ndarray,
                 batch: int | None = None) -> typing.Self:
        """
        Fill my portion of {jacobian} with d(physical)/d(sampling) when reparameterized;
        otherwise {jacobian} already holds 1, the right value
        """
        if self.reparameterize:
            self.transform.jacobian(
                theta=self.restrict(theta=theta), jacobian=self.restrict(theta=jacobian), batch=batch)
        return self


    def eval_prior_with_physical(self, theta: numpy.ndarray, likelihood: numpy.ndarray,
                                 batch: int | None = None) -> typing.Self:
        """
        Add my log|J| into {likelihood} when reparameterized; otherwise nothing to add
        """
        if self.reparameterize:
            self.transform.log_jacobian(
                theta=self.restrict(theta=theta), likelihood=likelihood, batch=batch)
        return self


    def eval_prior_physical(self, theta: numpy.ndarray, likelihood: numpy.ndarray,
                            batch: int | None = None) -> typing.Self:
        """
        Add to {likelihood} the log prior probabilities of my portion of the samples in
        {theta}, given in physical space. Without reparameterization, physical space is
        sampling space, so the default is just {eval_prior}.
        """
        return self.eval_prior(theta=theta, likelihood=likelihood, batch=batch)


    def to_physical(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Transform my portion of {theta} from sampling space to physical space, in place; a
        no-op unless reparameterized
        """
        if self.reparameterize:
            self.transform.to_physical(theta=self.restrict(theta=theta), batch=batch)
        return self


    def to_sampling(self, theta: numpy.ndarray, batch: int | None = None) -> typing.Self:
        """
        Transform my portion of {theta} from physical space to sampling space, in place; a
        no-op unless reparameterized
        """
        if self.reparameterize:
            self.transform.to_sampling(theta=self.restrict(theta=theta), batch=batch)
        return self


    def _initialize_transform(self, application: Application | None = None) -> typing.Self:
        """
        For a bounded distribution with a {reparameterize} trait: hand my transform my
        {support} and let it initialize
        """
        if self.reparameterize:
            self.has_reparametrization = True
            self.transform.support = self.support
            self.transform.initialize(application=application)
        return self


    # the forwarding interface of the protocol
    def sample(self) -> float:
        """
        A single value drawn from me
        """
        return float(self.draw(shape=()))


    def density(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        My probability density at {x}
        """
        return numpy.exp(self.log_density(numpy.asarray(x, dtype=float)))


    def vector(self, vector: numpy.ndarray) -> numpy.ndarray:
        """
        Fill {vector} with values drawn from me
        """
        vector[...] = self.draw(shape=vector.shape)
        return vector


    def matrix(self, matrix: numpy.ndarray) -> numpy.ndarray:
        """
        Fill {matrix} with values drawn from me
        """
        matrix[...] = self.draw(shape=matrix.shape)
        return matrix


    # implementation details
    def restrict(self, theta: numpy.ndarray) -> numpy.ndarray:
        """
        Return my portion of the {theta}, a view of my columns
        """
        return theta[:, self.offset:self.offset + self.parameters]


    # private data, set by the shim before any other method runs
    parameters: int
    offset: int
    # mirrored back onto the shim after {initialize}; a concrete distribution sets this to
    # True in its own {initialize} when reparameterizing (see {Uniform})
    has_reparametrization: bool = False
    # set by the shim of a distribution that supports reparameterization (e.g. {Uniform})
    reparameterize: bool = False
    transform: Transform
    # the support of a bounded distribution, set by its shim
    support: tuple[float, float]
    # set by {initialize}
    rng: numpy.random.Generator


# end of file
