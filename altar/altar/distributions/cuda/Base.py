# -*- python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the declaration
class Base:
    """
    Shared support for the cuda implementation of a distribution: a plain class, not a pyre
    component -- the registered component is the shim in {altar.distributions} that owns
    {parameters}/{offset} as configurable traits and copies their values down to me, once, at
    {initialize} time.

    A concrete distribution overrides {initialize_sample} and {verify}; the rest are sensible
    defaults (a flat prior) that most distributions never need to touch. {initialize} itself
    is generic -- most distributions need nothing beyond the setup it already does -- so a
    concrete distribution overrides it only if it has cuda-side state of its own to set up.
    """


    def initialize(self, rng, application=None):
        """
        The generic cuda-side setup every distribution needs
        """
        # my slice of the overall parameter vector, as separate (idx_begin, idx_end) bounds --
        # every cudaX_* binding below takes them as two positional arguments, not a tuple
        self.idx_begin, self.idx_end = self.offset, self.offset + self.parameters
        # the compiled extension's {distributions} submodule, imported lazily so a cpu-only
        # build never has to load it; {altar.cuda.libcudaaltar} is the same extension module
        # {altar.norms.cuda.L2} and friends already import their own bindings from
        import altar.cuda
        # which device i run on; this runs during model initialization, before
        # {application.controller} exists, so it can't be read off the worker the way
        # {altar.bayesian.methods.CUDAAnnealing}/{CUDASGLD} do -- altar assumes one gpu per
        # process anyway (see {altar.cuda.get_current_device}), so there is exactly one
        # sensible answer regardless of which worker ends up owning this distribution
        self.device = altar.cuda.get_current_device()
        # my precision
        self.precision = application.job.gpuprecision
        self.libcudaaltar = altar.cuda.libcudaaltar.distributions
        # all done
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with initial random values from my distribution
        """
        raise NotImplementedError(
            f"class '{type(self).__name__}' must implement 'initialize_sample'")


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Fill my portion of {likelihood} with the log prior probabilities of the samples in
        {theta}
        """
        # default: flat prior, nothing to add
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P(\theta)/d\theta. The caller is trusted to
        have zeroed {gradient} already, exactly as cuda's elementwise kernels expect
        """
        # default: gradient = 0
        return self


    def verify(self, theta, mask, batch=None):
        """
        Check whether my portion of the samples in {theta} are consistent with my constraints, and
        update {mask}, a vector with zeroes for valid samples and non-zero for invalid ones
        """
        raise NotImplementedError(
            f"class '{type(self).__name__}' must implement 'verify'")


    def constrain(self, theta, batch=None):
        """
        Force my portion of the samples in {theta} back within my constraints, in place
        """
        # default: nothing to do
        return self


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill my portion of {jacobian} with d(physical)/d(sampling) when reparameterized;
        otherwise {jacobian} already holds 1, the right value
        """
        if self.reparameterize:
            self.transform.jacobian(theta=theta, jacobian=jacobian, batch=batch)
        return self


    def eval_prior_with_physical(self, theta, likelihood, batch=None):
        """
        Add my log|J| into {likelihood} when reparameterized; otherwise nothing to add
        """
        if self.reparameterize:
            self.transform.log_jacobian(theta=theta, likelihood=likelihood, batch=batch)
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
            self.transform.to_physical(theta=theta, batch=batch)
        return self


    def to_sampling(self, theta, batch=None):
        """
        Transform my portion of {theta} from physical space to sampling space, in place; a
        no-op unless reparameterized
        """
        if self.reparameterize:
            self.transform.to_sampling(theta=theta, batch=batch)
        return self


    def _initialize_transform(self, application=None):
        """
        For a bounded distribution with a {reparameterize} trait: hand my transform my
        {support} and column range, and let it initialize; my transform works on the full
        buffer over [idx_begin, idx_end), like every other cuda kernel here
        """
        if self.reparameterize:
            self.has_reparametrization = True
            self.transform.support = self.support
            self.transform.idx_begin = self.idx_begin
            self.transform.idx_end = self.idx_end
            self.transform.initialize(application=application)
        return self


    @staticmethod
    def _grid(buffer):
        """
        The bare {pyre.grid} grid underneath {buffer}, for handing to a cuda extension
        binding: {buffer} may already be one, or it may be an {altar.cuda.array.Array} (the
        state layer's {altar.cuda.matrix}/{altar.cuda.vector} instances)
        """
        return buffer.grid if hasattr(buffer, "grid") else buffer


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
    device = None
    idx_begin = None
    idx_end = None
    precision = None
    libcudaaltar = None


# end of file
