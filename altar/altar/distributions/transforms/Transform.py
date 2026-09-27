# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Protocol and implementations for reparameterizing a bounded distribution to an unconstrained
sampling space, for use by gradient-based samplers (HMC, SGLD) that can't handle a bounded
prior directly.

{LogitTransform} is the one thing pyre ever registers and a {.pfg} ever names; like
{altar.distributions.Base}, its own job is small: pick a backend implementation once, in
{initialize}, and forward every protocol method to it from then on. Its implementation lives
in a same-named class in {altar.distributions.native}/{altar.distributions.cuda} -- the same
packages {Uniform}/{Gaussian}/etc. already live in -- so cuda runs real CUDA kernels
(`altar/lib/libcudaaltar/distributions/cudaLogitTransform.cu`) instead of the
`numpy.asarray(theta)` approach this used to take unconditionally: that worked (managed
memory is host-visible), but ran the actual arithmetic on the cpu and forced a host/device
migration on every touch, in what's a per-leapfrog-substep hot loop.
"""

from importlib import import_module
import altar


class Transform(altar.protocol, family="altar.distributions.transforms"):
    """
    Protocol for components that map a distribution's samples between physical space (the
    distribution's own, possibly bounded, coordinates) and sampling space (unconstrained,
    what a gradient-based sampler actually walks).
    """

    @altar.provides
    def initialize(self, application=None):
        """
        Initialize me given an {application} context
        """

    @altar.provides
    def to_physical(self, theta, batch=None):
        """
        Transform {theta} from sampling space to physical space, in place
        """

    @altar.provides
    def to_sampling(self, theta, batch=None):
        """
        Transform {theta} from physical space to sampling space, in place; the inverse of
        {to_physical}
        """

    @altar.provides
    def log_jacobian(self, theta, likelihood, batch=None):
        """
        {theta} is in sampling space; add log|d(physical)/d(sampling)| into {likelihood}
        (accumulates, matching {altar.distributions.Base.eval_prior}'s convention)
        """

    @altar.provides
    def jacobian(self, theta, jacobian, batch=None):
        """
        {theta} is in sampling space; fill {jacobian} with the raw elementwise
        d(physical)/d(sampling) value (overwrites, matching {prior_gradient}'s convention)
        """

    @altar.provides
    def jacobian_gradient(self, theta, gradient, batch=None):
        """
        {theta} is in sampling space; fill {gradient} with d/d(sampling)[log|jacobian|]
        (overwrites, matching {prior_gradient}'s convention)
        """

    @altar.provides
    def chain_gradient(self, theta, gradient, batch=None):
        """
        {theta} is physical space; turn {gradient}, a physical-space prior gradient, into the
        sampling-space one in place: gradient <- gradient*J + d/d(sampling)[log|J|]
        """

    @classmethod
    def pyre_default(cls, **kwds):
        """
        Supply a default implementation
        """
        # by default, a logit/expit transform to/from a bounded interval
        return LogitTransform


class LogitTransform(altar.component, family="altar.distributions.transforms.logit",
                     implements=Transform):
    """
    physical = a + (b-a)*sigmoid(sampling); sampling = logit((physical-a)/(b-a)).

    The log-jacobian and its gradient reduce to exactly the standard Logistic(0,1) log-pdf
    and score: log|d(physical)/d(sampling)| = log(b-a) + log(sigmoid(x)) + log(1-sigmoid(x)),
    and the (b-a) term is a per-sample additive constant that plays no role in any gradient or
    acceptance decision downstream, so it is dropped here -- a Uniform(a,b) pushed through this
    map is standard-logistic-distributed in sampling space regardless of a/b.

    My actual numerics live in {altar.distributions.native.LogitTransform.LogitTransform}
    (cpu) or {altar.distributions.cuda.LogitTransform.LogitTransform}; see this module's own
    docstring for how one gets picked. {idx_begin}/{idx_end} are handed to me by my owning
    distribution (e.g. {Uniform}), alongside {support} -- only the cuda implementation uses
    them (its kernels need a column-range bound, like every other cuda distribution).
    """

    support = altar.properties.array(default=(0, 1))
    support.doc = "the physical-space support interval [a, b]"


    @altar.export
    def initialize(self, application=None):
        """
        Pick my backend implementation and let it initialize itself
        """
        self._impl = self._makeImpl()
        self._impl.support = self.support
        self._impl.idx_begin = self.idx_begin
        self._impl.idx_end = self.idx_end
        self._impl.initialize(application=application)
        return self

    def _makeImpl(self):
        """
        Build my backend implementation: a same-named class in {native} (the cpu default) or
        {cuda}, picked once, here, based on {altar.backends.active()}
        """
        name = type(self).__name__
        backend = "cuda" if altar.backends.active() == "cuda" else "native"
        module = import_module(f"altar.distributions.{backend}.{name}")
        return getattr(module, name)()


    @altar.export
    def to_physical(self, theta, batch=None):
        """
        theta <- a + (b-a)*sigmoid(theta), in place
        """
        return self._impl.to_physical(theta=theta, batch=batch)


    @altar.export
    def to_sampling(self, theta, batch=None):
        """
        theta <- logit((theta-a)/(b-a)), in place; the inverse of {to_physical}
        """
        return self._impl.to_sampling(theta=theta, batch=batch)


    @altar.export
    def log_jacobian(self, theta, likelihood, batch=None):
        """
        Add the standard-logistic log-pdf into {likelihood}, summed over the parameters this
        transform owns. {theta} here is PHYSICAL space (unlike {to_physical}/{to_sampling}) --
        every real caller (e.g. {altar.models.BayesianL2.eval_prior_with_physical}, via
        {Uniform.eval_prior_with_physical}) already has physical-space theta on hand (the
        model/gradient layer works in physical space throughout), and
        sig := sigmoid(sampling) = (physical-a)/(b-a) by construction, so computing sig
        directly from physical space is exactly equivalent to (and cheaper than) converting
        to sampling space and back
        """
        return self._impl.log_jacobian(theta=theta, likelihood=likelihood, batch=batch)


    @altar.export
    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill {jacobian} with d(physical)/d(sampling) = (b-a)*sig*(1-sig). {theta} is PHYSICAL
        space; see {log_jacobian}'s docstring for why.
        """
        return self._impl.jacobian(theta=theta, jacobian=jacobian, batch=batch)


    @altar.export
    def jacobian_gradient(self, theta, gradient, batch=None):
        """
        Fill {gradient} with d/d(sampling)[log(sig) + log(1-sig)] = 1 - 2*sig. {theta} is
        PHYSICAL space; see {log_jacobian}'s docstring for why.
        """
        return self._impl.jacobian_gradient(theta=theta, gradient=gradient, batch=batch)


    @altar.export
    def chain_gradient(self, theta, gradient, batch=None):
        """
        gradient <- gradient*(b-a)*sig*(1-sig) + (1 - 2*sig), in place; {theta} is PHYSICAL
        space, {gradient} a physical-space prior gradient on the way in
        """
        return self._impl.chain_gradient(theta=theta, gradient=gradient, batch=batch)


    # private data
    idx_begin = None   # set by my owning distribution, alongside support; cuda-only
    idx_end = None
    _impl = None       # my backend implementation, chosen once, in {initialize}


# end of file
