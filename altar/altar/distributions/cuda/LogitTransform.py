# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# the declaration
class LogitTransform:
    """
    The cuda implementation of {altar.distributions.transforms.Transform.LogitTransform}:
    physical = a + (b-a)*sigmoid(sampling); sampling = logit((physical-a)/(b-a)). Every
    method here calls a real CUDA kernel
    (`altar/lib/libcudaaltar/distributions/cudaLogitTransform.cu`) over my
    [idx_begin, idx_end) column range of the full buffer, the same "full grid + idx_begin/
    idx_end" convention every other cuda distribution uses (see
    {altar.distributions.cuda.Uniform}) -- unlike the numpy-on-managed-memory approach this
    used to take, which ran on the cpu and forced a host/device migration on every touch.
    """

    def initialize(self, application=None):
        """
        The only cuda-side setup i need: the extension module my kernels live in
        """
        import altar.cuda
        self.libcudaaltar = altar.cuda.libcudaaltar.distributions
        return self


    def to_physical(self, theta, batch=None):
        """
        theta[:, idx_begin:idx_end] <- a + (b-a)*sigmoid(theta), in place
        """
        a, b = self.support
        self.libcudaaltar.cudaLogitTransform_tophysical(
            self._grid(theta), self.idx_begin, self.idx_end, a, b)
        return self


    def to_sampling(self, theta, batch=None):
        """
        theta[:, idx_begin:idx_end] <- logit((theta-a)/(b-a)), in place; the inverse of
        {to_physical}
        """
        a, b = self.support
        self.libcudaaltar.cudaLogitTransform_tosampling(
            self._grid(theta), self.idx_begin, self.idx_end, a, b)
        return self


    def log_jacobian(self, theta, likelihood, batch=None):
        """
        Add the standard-logistic log-pdf into {likelihood}, summed over
        [idx_begin, idx_end). {theta} is PHYSICAL space; see the shim's docstring for why.
        """
        a, b = self.support
        self.libcudaaltar.cudaLogitTransform_logjacobian(
            self._grid(theta), self._grid(likelihood), self.idx_begin, self.idx_end, a, b)
        return self


    def jacobian(self, theta, jacobian, batch=None):
        """
        Fill {jacobian[:, idx_begin:idx_end]} with d(physical)/d(sampling)
        """
        a, b = self.support
        self.libcudaaltar.cudaLogitTransform_jacobian(
            self._grid(theta), self._grid(jacobian), self.idx_begin, self.idx_end, a, b)
        return self


    def jacobian_gradient(self, theta, gradient, batch=None):
        """
        Fill {gradient[:, idx_begin:idx_end]} with d/d(sampling)[log(sig) + log(1-sig)]
        """
        a, b = self.support
        self.libcudaaltar.cudaLogitTransform_jacobiangradient(
            self._grid(theta), self._grid(gradient), self.idx_begin, self.idx_end, a, b)
        return self


    # implementation details
    @staticmethod
    def _grid(buffer):
        """
        The bare {pyre.grid} grid underneath {buffer}, for handing to a cuda extension
        binding: {buffer} may already be one, or it may be an {altar.cuda.array.Array} (the
        state layer's {altar.cuda.matrix}/{altar.cuda.vector} instances)
        """
        return buffer.grid if hasattr(buffer, "grid") else buffer


    # private data, set by the shim before {initialize} runs
    support = None
    idx_begin = None
    idx_end = None
    libcudaaltar = None


# end of file
