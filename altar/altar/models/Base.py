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
from importlib import import_module
# get the package
import altar

# get the protocol
from .ParameterSet import ParameterSet as parameters


# the declaration
class Base(altar.component, implements=parameters):
    """
    The base class for parameter sets

    Same shape as {altar.distributions.Base}: I am the one piece pyre ever registers, and my
    only job is to pick my backend implementation once, in {initialize}, and forward every
    protocol method to it from then on. My implementation lives in a same-named class in
    {altar.models.native} (the cpu default) or {altar.models.cuda}; neither is a pyre
    component, just the numerics.
    """


    # protocol obligations
    # user configurable state; the protocol declares these directly, so every conforming
    # component carries them, even a leaf parameter set that shares one prior/prep pair
    # (see {Contiguous}) or an ensemble that doesn't use them at all (see {ParameterEnsemble})
    count = altar.properties.int(default=1)
    count.doc = "the number of parameters in this set"

    prior = altar.distributions.distribution()
    prior.doc = "the prior distribution"

    prep = altar.distributions.distribution(default=None)
    prep.doc = "the distribution to use to initialize this parameter set"


    # configuration
    @altar.export
    def initialize(self, model, offset, application=None):
        """
        Initialize my state given the {model} that owns me. On cuda, {application} is also
        required, so my implementation's generic cuda-side setup can run.
        """
        # pick my backend implementation, once
        self._impl = self._makeImpl()
        # and let it initialize itself
        count = self._impl.initialize(model=model, offset=offset, application=application)
        # keep my own {offset} in sync, the way the old implementations did
        self.offset = offset
        # all done
        return count

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
        module = import_module(f"altar.models.{backend}.{name}")
        factory = getattr(module, name)
        # build it and hand it my count, which every parameter set shares
        impl = factory()
        impl.count = self.count
        # all done
        return impl


    @altar.export
    def initialize_sample(self, theta, batch=None):
        """
        Fill {theta} with an initial random sample from my prior distribution.
        """
        return self._impl.initialize_sample(theta=theta, batch=batch)


    @altar.export
    def eval_prior(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta} in my prior distribution
        """
        return self._impl.eval_prior(theta=theta, prior=prior, batch=batch)


    @altar.export
    def verify(self, theta, mask, batch=None):
        """
        Check whether the samples in {theta} are consistent with the model requirements and
        update the {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        return self._impl.verify(theta=theta, mask=mask, batch=batch)


    @altar.export
    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill {gradient} with d\log P(\theta)/d\theta for my portion of the samples in {theta},
        for use by gradient-based samplers (e.g. SGLD)
        """
        return self._impl.prior_gradient(theta=theta, gradient=gradient, batch=batch)


    @altar.export
    def constrain(self, theta, batch=None):
        """
        Force the samples in {theta} back within my constraints, in place. Only meaningful on
        cuda; see {altar.distributions.Base.constrain}.
        """
        return self._impl.constrain(theta=theta, batch=batch)


    @altar.export
    def eval_prior_with_physical(self, theta, prior, batch=None):
        """
        Add any prior contributions that depend on physical parameters, beyond what
        {eval_prior} already contributed
        """
        return self._impl.eval_prior_with_physical(theta=theta, prior=prior, batch=batch)


    @altar.export
    def eval_prior_physical(self, theta, prior, batch=None):
        """
        Fill {prior} with the log likelihoods of the samples in {theta}, given in physical space
        """
        return self._impl.eval_prior_physical(theta=theta, prior=prior, batch=batch)


    @altar.export
    def to_physical(self, theta, batch=None):
        """
        Transform {theta} from sampling space to physical space, in place
        """
        return self._impl.to_physical(theta=theta, batch=batch)


    @altar.export
    def to_sampling(self, theta, batch=None):
        """
        Transform {theta} from physical space to sampling space, in place
        """
        return self._impl.to_sampling(theta=theta, batch=batch)


    # private data
    offset = 0 # kept in sync with my implementation's, for callers that read it directly
    _impl = None # my backend implementation, chosen once, in {initialize}


# end of file
