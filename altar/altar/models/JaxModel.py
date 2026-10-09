# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy
# the package
import altar


# declaration
class JaxModel:
    """
    A mixin for {BayesianL2} models whose forward model is written in jax: {jax_forward} maps
    one sample to its predicted data, and i provide the batched forward model and the gradient
    of the data log likelihood, by {jax.grad}, on the gpu or the cpu

    On the gpu, jax reads the managed memory of the chains in place, through its cuda array
    interface, and the results come back to the managed buffers by device to device copies.
    My predictions are raw, so {dataobs} computes the likelihood itself, as for any model.
    """


    # interface
    def jax_forward(self, theta):
        """
        The predicted data, (observations,), of one sample, (parameters,); in jax.numpy
        """
        raise NotImplementedError(f"model '{type(self).__name__}' must implement 'jax_forward'")


    # the {BayesianL2} hooks
    def forward_model_batched(self, theta, prediction, batch=None):
        """
        Fill {prediction}, (samples x observations), with the raw predicted data of every row of
        {theta}; all the rows, so that jax compiles once, whatever the batch
        """
        self._jax()
        predicted = self._forward(self._to_jax(theta))
        self._store(predicted, prediction)
        return self


    def gradient(self, controller, step, batch=None):
        """
        Fill the prior and data gradients of {step} with respect to its {theta}; the data part
        is {jax.grad} of the same log likelihood {dataobs} computes
        """
        # gradient-based samplers need unbounded priors, or reparameterized bounded ones
        if not self.embedded and not self.checked_unbounded_priors:
            self.verify_unbounded_priors()
        # the cuda states call them {prior_gradient}/{data_gradient}, the cpu ones {grad_prior}/{grad_data}
        cuda = altar.backends.active() == "cuda"
        θ = self.restrict(theta=step.theta)
        grad_prior = self.restrict(theta=step.prior_gradient if cuda else step.grad_prior)
        grad_data = self.restrict(theta=step.data_gradient if cuda else step.grad_data)
        for name in ([] if self.embedded else self.psets_list):
            if cuda:
                self.psets[name].prior_gradient(theta=θ, gradient=grad_prior, batch=batch)
            else:
                self.psets[name].prior_gradient(theta=θ, gradient=grad_prior)
        # the data part
        self._jax()
        data, kind, terms = self._noise()
        self._store(self._gradients[kind](self._to_jax(θ), data, *terms), grad_data)
        return self


    # implementation details
    def _jax(self):
        """
        Import jax and build my compiled functions, once
        """
        if self._compiled:
            import jax
            return jax
        import jax
        import jax.numpy as jnp
        # float64 chains need jax's 64-bit mode, before any array is made
        if self.precision == "float64" or altar.backends.active() != "cuda":
            jax.config.update("jax_enable_x64", True)
        forward = self.jax_forward

        # the data log likelihood of one sample, for each way {dataobs} weighs the residual
        def diagonal(theta, data, c):
            # a constant covariance: c holds 1/sigma^2, times the mask
            r = forward(theta) - data
            return -0.5 * jnp.sum(c * r * r)

        def full(theta, data, factor, before, after):
            # r @ factor whitens the residual; the mask applies before (cpu) or after (cuda)
            r = (forward(theta) - data) * before
            w = r @ factor.astype(r.dtype)
            return -0.5 * jnp.sum(after * w * w)

        self._forward = jax.jit(jax.vmap(forward))
        self._gradients = {
            "diagonal": jax.jit(jax.vmap(jax.grad(diagonal), in_axes=(0, None, None))),
            "full": jax.jit(jax.vmap(jax.grad(full), in_axes=(0, None, None, None, None))),
        }
        self._compiled = True
        return jax


    def _noise(self):
        """
        The observed data and the terms of the data log likelihood, as {dataobs} has them now;
        read on every call, since a model covariance update changes them
        """
        import jax.numpy as jnp
        impl = self.dataobs._impl
        n = self.observations
        ones = numpy.ones(n)
        if altar.backends.active() == "cuda":
            data = jnp.asarray(impl._dataobs_raw)
            cd = impl.cd_inv
            if isinstance(cd, float):
                return data, "diagonal", (jnp.asarray(impl._weight_scaled),)
            # Cd_inv = U^T U, U in the upper triangle of {cd_inv}: the rows of r @ U^T are U r
            mask = impl._weight
            after = jnp.asarray(mask) if mask is not None else jnp.asarray(ones, dtype=data.dtype)
            factor = jnp.triu(jnp.asarray(cd)).T
            return data, "full", (factor, jnp.asarray(ones, dtype=data.dtype), after)
        # the cpu: raw data, and Cd_inv = L L^T with L in the lower triangle of {cd_inv}
        data = jnp.asarray(numpy.asarray(impl._observed, dtype=float))
        mask = impl.mask
        weight = ones if mask is None else numpy.asarray(mask, dtype=float)
        cd = impl.cd_inv
        if isinstance(cd, float):
            return data, "diagonal", (jnp.asarray(weight * cd * cd),)
        factor = jnp.tril(jnp.asarray(numpy.asarray(cd.ndarray(), dtype=float)))
        return data, "full", (factor, jnp.asarray(numpy.sqrt(weight)), jnp.asarray(ones))


    def _to_jax(self, array):
        """
        A jax view of {array}: in place for managed gpu memory, a copy of the host values otherwise
        """
        import jax.numpy as jnp
        if altar.backends.active() == "cuda":
            return jnp.asarray(array)
        return jnp.asarray(numpy.asarray(array.ndarray() if hasattr(array, "ndarray") else array))


    def _store(self, value, target):
        """
        Copy the jax array {value} into the altar buffer {target}, of the same shape
        """
        value.block_until_ready()
        if altar.backends.active() == "cuda":
            from cuda.bindings import runtime as cudart
            nbytes = value.size * value.dtype.itemsize
            status, = cudart.cudaMemcpy(target.address, value.unsafe_buffer_pointer(), nbytes,
                                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
            if status != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"jax model: copying the results back failed: {status}")
            return target
        out = target.ndarray() if hasattr(target, "ndarray") else numpy.asarray(target)
        out[...] = numpy.asarray(value)
        return target


    # private data
    _compiled = False # whether my jax functions are built
    _forward = None # the batched, compiled forward model
    _gradients = None # the batched, compiled data gradients, by covariance kind


# end of file
