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
from importlib import import_module
# get the package
import altar

# get the protocol
from . import distribution


# the declaration
class Base(altar.component, implements=distribution):
    """
    The base class for probability distributions

    I am the one piece pyre ever registers and a {.pfg} ever names. My own job is small: pick
    my backend implementation once, in {initialize}, and forward every protocol method to it
    from then on -- no backend check anywhere outside this one place.

    My implementation lives in a same-named class in {altar.distributions.native} (the cpu
    default) or {altar.distributions.cuda}, e.g. {Uniform} here means
    {altar.distributions.native.Uniform.Uniform} or its cuda counterpart. Neither is a pyre
    component; each is a plain class holding the actual numerics, so the two backends never
    share a file, and adding a backend never means writing a second component.

    A concrete distribution (e.g. {Uniform}) only ever needs to declare its own configurable
    traits and override {_makeImpl} to copy any of its own traits down to the implementation
    it builds (see {Uniform.py}); everything else here is generic.
    """


    # protocol obligations
    # user configurable state
    parameters = altar.properties.int()
    parameters.doc = "the number of model parameters that belong to me"

    offset = altar.properties.int(default=0)
    offset.doc = "the starting point of my parameters in the overall model state"

    # whether my support is a strict subset of the reals; see the protocol docstring in
    # Distribution.py. False by default; a bounded distribution (e.g. Uniform) overrides it
    bounded = False

    # whether i reparameterize my samples to an unconstrained sampling space (e.g. a logit
    # transform of a bounded support), for gradient-based samplers. False by default; see
    # {to_physical}/{to_sampling}/{eval_prior_physical} below
    has_reparametrization = False


    # configuration
    @altar.export
    def initialize(self, rng, application=None):
        """
        Initialize with the given random number generator. On cuda, {application} is also
        required, so my implementation's generic cuda-side setup can run; see
        {cuda.Base.activate}.
        """
        # pick my backend implementation, once
        self._impl = self._makeImpl()
        # and let it initialize itself
        self._impl.initialize(rng=rng, application=application)
        # all done
        return self

    def _makeImpl(self):
        """
        Build my backend implementation: a same-named class in {native} (the cpu default) or
        {cuda}, picked once, here, based on {altar.backends.active()}
        """
        # my own class name is also my implementation's
        name = type(self).__name__
        # the package it lives in
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        # reach it
        module = import_module(f"altar.distributions.{backend}.{name}")
        factory = getattr(module, name)
        # build it
        impl = factory()
        # hand it the state every distribution shares
        impl.parameters = self.parameters
        impl.offset = self.offset
        # all done
        return impl


    @altar.export
    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values from my distribution.
        """
        return self._impl.initialize_sample(theta=theta, batch=batch)


    @altar.export
    def eval_prior(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the log prior probabilities of the samples in
        {theta}
        """
        return self._impl.eval_prior(theta=theta, likelihood=likelihood, batch=batch)


    @altar.export
    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta, elementwise, for the
        samples in {theta}. {gradient} has the same shape as {theta}.
        """
        return self._impl.prior_gradient(theta=theta, gradient=gradient, batch=batch)


    @altar.export
    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        return self._impl.verify(theta=theta, mask=mask, batch=batch)


    @altar.export
    def constrain(self, theta, batch=None):
        """
        Force my portion of the samples in {theta} back within my constraints, in place. Only
        meaningful on cuda; the cpu implementation's default is a no-op, since cpu samplers
        reject through {verify} instead.
        """
        return self._impl.constrain(theta=theta, batch=batch)


    @altar.export
    def eval_prior_with_physical(self, theta, likelihood, batch=None):
        """
        Add any prior contributions to {likelihood} that depend on my physical parameters,
        beyond what {eval_prior} already contributed. Only distributions with
        {has_reparametrization} ever need this; the default is nothing further to add.
        """
        return self._impl.eval_prior_with_physical(theta=theta, likelihood=likelihood, batch=batch)


    @altar.export
    def eval_prior_physical(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the log prior probabilities of the samples in
        {theta}, given in physical space. Without reparameterization, physical space is
        sampling space, so the default is just {eval_prior}.
        """
        return self._impl.eval_prior_physical(theta=theta, likelihood=likelihood, batch=batch)


    @altar.export
    def to_physical(self, theta, batch=None):
        """
        Transform my portion of {theta} from sampling space to physical space, in place.
        Without reparameterization, the two coincide, so the default is a no-op.
        """
        return self._impl.to_physical(theta=theta, batch=batch)


    @altar.export
    def to_sampling(self, theta, batch=None):
        """
        Transform my portion of {theta} from physical space to sampling space, in place. The
        inverse of {to_physical}; the default is likewise a no-op.
        """
        return self._impl.to_sampling(theta=theta, batch=batch)


    # the forwarding interface (cpu only: on cuda, samples are drawn straight into a grid by
    # the kernels behind {initialize_sample}, never one distribution at a time)
    @altar.export
    def sample(self):
        """
        Sample the distribution using a random number generator
        """
        return self._impl.sample()


    @altar.export
    def density(self, x):
        """
        Compute the probability density of the distribution at {x}
        """
        return self._impl.density(x)


    @altar.export
    def vector(self, vector):
        """
        Fill {vector} with random values
        """
        return self._impl.vector(vector)


    @altar.export
    def matrix(self, matrix):
        """
        Fill {matrix} with random values
        """
        return self._impl.matrix(matrix)


    # private data
    _impl = None # my backend implementation, chosen once, in {initialize}


# end of file
