# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
import numpy
import altar

from .Base import Base as base
from ..native.Correlated import Correlated as host


class Correlated(base):
    """
    The cuda implementation of a gaussian distribution correlated in space: the covariance and
    the initial samples on the host, by the cpu implementation; the density and its gradient
    on the device, by gemms with matrices as wide as {theta}, zero outside my columns, since
    the samples of all the parameter sets share one buffer
    """


    def initialize(self, rng, application=None):
        """
        The generic cuda-side setup, then my covariance, built on the host
        """
        super().initialize(rng=rng, application=application)
        # the numerics on the host, without the transform, which runs here on the device
        self.host = host()
        for name in ["parameters", "offset", "mean", "sigma", "kernel", "length", "coordinates",
                     "dataset", "support", "jitter", "sweeps", "normalization_draws"]:
            setattr(self.host, name, getattr(self, name))
        # {rng} is unused on cuda; the initial samples come from the application's generator
        generator = rng if rng is not None else getattr(application, "rng", None)
        self.host.rng = generator.rng if generator is not None else numpy.random.default_rng()
        self.host.build(application=application)
        self.bounded = self.host.bounded
        self._initialize_transform(application=application)
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Fill my portion of {theta} with samples drawn on the host; a one-off host write into
        managed memory
        """
        θ = numpy.asarray(self._grid(theta))
        θ[:, self.idx_begin:self.idx_end] = self.host.draw(shape=(θ.shape[0], self.parameters))
        return self


    def verify(self, theta, mask, batch=None):
        """
        Mark in {mask} the samples with a parameter outside my support; {mask} must be an int32
        grid; {theta} is physical, reparameterized or not
        """
        if self.bounded:
            low, high = self.support
            self.libcudaaltar.cudaRanged_verify(
                self._grid(theta), self._grid(mask), self.idx_begin, self.idx_end, low, high)
        return mask


    def constrain(self, theta, batch=None):
        """
        Force my portion of the samples in {theta} back within my support, in place; a no-op
        when reparameterized or unbounded
        """
        if self.bounded and not self.reparameterize:
            low, high = self.support
            self.libcudaaltar.cudaRanged_constrain(
                self._grid(theta), self.idx_begin, self.idx_end, low, high)
        return self


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Add to {likelihood} the log prior probabilities of my portion of the samples in
        {theta}: z = L^-1 (θ - mean), then log N = constant - z.z / 2
        """
        grid = self._grid(theta)
        rows, width = grid.shape
        batch = rows if batch is None else batch
        self._prepare(rows=rows, width=width)
        n = self.parameters
        work = self._work
        work.copy(self._shift)
        # the rows of θ are the column-major columns of θ^T: work^T += W θ^T, with W = L^-1 on
        # my columns and zero elsewhere
        self._gemm(altar.cuda.cublas_handle(), self._N, self._N, n, batch, width, 1.0,
                   self._whitener.grid, n, grid, width, 1.0, work.grid, n)
        altar.cuda.libcudaaltar.norms.cudaL2_normllk(work.grid, self._density.grid, batch, self.host.constant)
        altar.cuda.cublas.axpy(1.0, self._density, likelihood, batch=batch)
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        r"""
        Fill my portion of {gradient} with d\log P/d\theta = -C^-1 (\theta - mean); when
        reparameterized, its chain rule to the sampling space
        """
        grid = self._grid(theta)
        rows, width = grid.shape
        batch = rows if batch is None else batch
        self._prepare(rows=rows, width=width)
        handle = altar.cuda.cublas_handle()
        # s^T = A θ^T + c 1^T, with A = -C^-1 and c = C^-1 mean on my columns, zero elsewhere
        s = self._gradient.zero()
        self._gemm(handle, self._N, self._N, width, batch, width, 1.0,
                   self._precision.grid, width, grid, width, 0.0, s.grid, width)
        self._gemm(handle, self._N, self._N, width, batch, 1, 1.0,
                   self._offset.grid, width, self._ones.grid, 1, 1.0, s.grid, width)
        # my columns of {gradient} replaced by those of s, the others kept
        g = self._grid(gradient)
        g *= self._keep.grid
        g += s.grid
        if self.reparameterize:
            self.transform.chain_gradient(theta=theta, gradient=gradient, batch=batch)
        return self


    # implementation details
    def _prepare(self, rows, width):
        """
        The device matrices for samples {width} parameters wide, and the scratch for {rows} of
        them, made on first use and again if either changes
        """
        dtype = self.precision
        n, b, e = self.parameters, self.idx_begin, self.idx_end
        if self._width != width:
            # stored row-major as the transposes of what the column-major gemms read
            whitener = numpy.zeros((width, n))
            whitener[b:e, :] = self.host.whitener.T
            self._whitener = altar.cuda.matrix(source=whitener, dtype=dtype)
            precision = numpy.zeros((width, width))
            precision[b:e, b:e] = -self.host.precision_matrix
            self._precision = altar.cuda.matrix(source=precision, dtype=dtype)
            offset = numpy.zeros((width, 1))
            offset[b:e, 0] = self.host.precision_matrix @ self.host.mean_vector
            self._offset = altar.cuda.matrix(source=offset, dtype=dtype)
            self._width = width
            self._rows = None
        if self._rows != rows:
            shift = -(self.host.whitener @ self.host.mean_vector)
            self._shift = altar.cuda.matrix(source=numpy.tile(shift, (rows, 1)), dtype=dtype)
            self._work = altar.cuda.matrix(shape=(rows, n), dtype=dtype)
            self._density = altar.cuda.vector(shape=rows, dtype=dtype)
            self._ones = altar.cuda.matrix(source=numpy.ones((rows, 1)), dtype=dtype)
            self._gradient = altar.cuda.matrix(shape=(rows, width), dtype=dtype)
            keep = numpy.ones((rows, width))
            keep[:, b:e] = 0
            self._keep = altar.cuda.matrix(source=keep, dtype=dtype)
            self._rows = rows
        cublas = altar.cuda.cublas
        self._gemm = cublas.dgemm if dtype == "float64" else cublas.sgemm
        self._N = cublas.Operation.N
        return self


    # private data, set by the shim before {initialize} runs
    mean = None
    sigma = None
    kernel = None
    length = None
    coordinates = None
    dataset = None
    support = None
    jitter = None
    sweeps = None
    normalization_draws = 10000
    bounded = False
    # set by {initialize} and {_prepare}
    host = None
    _width = None
    _rows = None
    _whitener = None
    _precision = None
    _offset = None
    _shift = None
    _work = None
    _density = None
    _ones = None
    _gradient = None
    _keep = None
    _gemm = None
    _N = None


# end of file
