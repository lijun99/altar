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
import altar.cuda
from altar.cuda import nvrtc
# my base class
from altar.distributions.cuda.Base import Base as base
# the geometry, and the host numerics for the initial samples
from .. import pooling
from ..native.Pooled import resolve


SOURCE = r"""
// the pooled energies of one sample (blockIdx.y), one thread per (station s, pool q, column j),
// j fastest: e = sum over the neighbors k of s of W_sk sum_{c in q} theta[k, c, j]^2; stores
// f = phi'(r) / r and adds -phi(r) into logp[sample], r = sqrt(e + epsilon^2)
template <typename real>
__global__ void energies(const real * theta, int width, int offset, int columns,
                         int stations, int components, int pools,
                         const int * block, const int * poolstart, const int * poolcomps,
                         const int * indptr, const int * indices, const real * weights,
                         double epsilon, double scale, double strength, int shape,
                         real * f, double * logp)
{
    int n = blockIdx.y;
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    long count = (long) stations * pools * columns;
    if (i >= count) return;
    int j = i % columns;
    int q = (i / columns) % pools;
    int s = i / ((long) columns * pools);
    const real * sample = theta + (long) n * width + offset;
    double e = 0;
    for (int a = indptr[s]; a < indptr[s + 1]; ++a) {
        int k = indices[a];
        double sum = 0;
        for (int b = poolstart[q]; b < poolstart[q + 1]; ++b) {
            int blk = block[k * components + poolcomps[b]];
            if (blk >= 0) {
                double t = sample[(long) blk * columns + j];
                sum += t * t;
            }
        }
        e += weights[a] * sum;
    }
    double r = sqrt(e + epsilon * epsilon);
    double phi, factor;
    if (shape == 0) {
        phi = r / scale;
        factor = 1.0 / (scale * r);
    } else {
        phi = strength * log1p(r / scale);
        factor = strength / ((scale + r) * r);
    }
    f[(long) n * count + i] = (real) factor;
    atomicAdd(logp + n, -phi);
}

// the gradient of one sample (blockIdx.y), one thread per (station k, component c, column j):
// d log P / d theta[k, c, j] = -theta[k, c, j] sum over s of W_sk f[s, pool(c), j], with the
// transposed neighbor lists
template <typename real>
__global__ void gradients(const real * theta, real * gradient, int width, int offset,
                          int columns, int stations, int components, int pools,
                          const int * block, const int * poolof,
                          const int * indptr, const int * indices, const real * weights,
                          const real * f)
{
    int n = blockIdx.y;
    long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long) stations * components * columns) return;
    int j = i % columns;
    int c = (i / columns) % components;
    int k = i / ((long) columns * components);
    int blk = block[k * components + c];
    if (blk < 0) return;
    int q = poolof[c];
    long count = (long) stations * pools * columns;
    const real * fn = f + (long) n * count;
    double acc = 0;
    for (int a = indptr[k]; a < indptr[k + 1]; ++a) {
        int s = indices[a];
        acc += weights[a] * fn[((long) s * pools + q) * columns + j];
    }
    long at = (long) n * width + offset + (long) blk * columns + j;
    gradient[at] = (real) (-theta[at] * acc);
}

// likelihood[n] += logp[n]
template <typename real>
__global__ void accumulate(real * likelihood, const double * logp, int samples)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < samples) likelihood[n] += (real) logp[n];
}
"""

THREADS = 256


class Pooled(base):
    """
    The cuda implementation of the spatially pooled sparsity prior: the geometry built on the
    host, the energies and gradients by NVRTC kernels over compressed neighbor lists
    """


    def initialize(self, rng, application=None):
        """
        Build the geometry on the host, upload it, compile my kernels
        """
        super().initialize(rng=rng, application=application)
        path = resolve(self.design_file, application)
        g = pooling.load(path, self.parameters, self.length, self.radius, self.pooling)
        self.geometry = g
        dtype = self.precision
        ints = lambda a: altar.cuda.vector(source=numpy.asarray(a, dtype=numpy.int32),
                                           dtype="int32")
        self._block = ints(g.block.ravel())
        starts = numpy.cumsum([0] + [len(p) for p in g.pools])
        self._poolstart = ints(starts)
        self._poolcomps = ints(numpy.concatenate(g.pools))
        self._poolof = ints(g.pool_of)
        self._rows = (ints(g.rows[0]), ints(g.rows[1]),
                      altar.cuda.vector(source=g.rows[2], dtype=dtype))
        self._cols = (ints(g.cols[0]), ints(g.cols[1]),
                      altar.cuda.vector(source=g.cols[2], dtype=dtype))
        real = nvrtc.real[dtype]
        self.names = (f"energies<{real}>", f"gradients<{real}>", f"accumulate<{real}>")
        self.kernels = nvrtc.kernels(SOURCE, self.names)
        generator = rng if rng is not None else getattr(application, "rng", None)
        self.host_rng = generator.rng if generator is not None else numpy.random.default_rng()
        info = getattr(application, "info", None)
        if info is not None:
            info.log(f"pooled prior, {self.shape}, scale {self.scale:g}: {g.describe()}")
        return self


    def initialize_sample(self, theta, batch=None):
        """
        Independent Laplace draws of my scale, written on the host into managed memory
        """
        θ = numpy.asarray(self._grid(theta))
        θ[:, self.idx_begin:self.idx_end] = self.host_rng.laplace(
            0.0, self.scale, size=(θ.shape[0], self.parameters))
        return self


    def verify(self, theta, mask, batch=None):
        """
        Unbounded: every sample is valid
        """
        return mask


    def _energies(self, theta, batch):
        """
        The pooled energies of the first {batch} samples: f in my scratch, -sum phi in logp
        """
        g = self.geometry
        grid = self._grid(theta)
        rows, width = grid.shape
        batch = rows if batch is None else batch
        count = len(g.stations) * len(g.pools) * g.width
        if self._f is None or self._f.shape != (rows, count):
            self._f = altar.cuda.matrix(shape=(rows, count), dtype=self.precision)
            self._logp = altar.cuda.vector(shape=rows, dtype="float64")
        self._logp.zero()
        indptr, indices, weights = self._rows
        nvrtc.launch(self.kernels[self.names[0]], (nvrtc.blocks(count, THREADS), batch), THREADS,
                     grid, numpy.int32(width), numpy.int32(self.idx_begin),
                     numpy.int32(g.width), numpy.int32(len(g.stations)),
                     numpy.int32(len(g.components)), numpy.int32(len(g.pools)),
                     self._block, self._poolstart, self._poolcomps, indptr, indices, weights,
                     numpy.float64(self.epsilon), numpy.float64(self.scale),
                     numpy.float64(self.strength), numpy.int32(0 if self.shape == "laplace" else 1),
                     self._f, self._logp)
        return grid, width, batch


    def eval_prior(self, theta, likelihood, batch=None):
        """
        Add the log prior of each sample to {likelihood}
        """
        _, _, batch = self._energies(theta, batch)
        nvrtc.launch(self.kernels[self.names[2]], nvrtc.blocks(batch, THREADS), THREADS,
                     self._grid(likelihood), self._logp, numpy.int32(batch))
        return self


    def prior_gradient(self, theta, gradient, batch=None):
        """
        Fill my columns of {gradient} with d log P / d theta
        """
        g = self.geometry
        grid, width, batch = self._energies(theta, batch)
        count = len(g.stations) * len(g.components) * g.width
        indptr, indices, weights = self._cols
        nvrtc.launch(self.kernels[self.names[1]], (nvrtc.blocks(count, THREADS), batch), THREADS,
                     grid, self._grid(gradient), numpy.int32(width), numpy.int32(self.idx_begin),
                     numpy.int32(g.width), numpy.int32(len(g.stations)),
                     numpy.int32(len(g.components)), numpy.int32(len(g.pools)),
                     self._block, self._poolof, indptr, indices, weights, self._f)
        return self


    # private data, set by the shim before {initialize} runs
    design_file = None
    scale = None
    shape = None
    strength = None
    epsilon = None
    length = None
    radius = None
    pooling = None
    # set by {initialize}
    geometry = None
    host_rng = None
    names = None
    kernels = None
    _block = None
    _poolstart = None
    _poolcomps = None
    _poolof = None
    _rows = None
    _cols = None
    _f = None # phi'(r) / r of the pooled energies, (samples x stations pools columns)
    _logp = None # the log prior of each sample, in double precision


# end of file
