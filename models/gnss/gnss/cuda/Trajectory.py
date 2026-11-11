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
import altar.cuda
from . import cublasx


# declaration
class Trajectory:
    """
    The cuda numerics of the gnss trajectory model: per group, one strided batched gemm over
    the blocks, the samples as the columns, reading the coefficients in place in theta

    Column major, as cublas sees the row-major managed arrays: theta is (P x S), ld P; the
    predictions (B n x S), ld B n; a shared group's G, stored (n x p) row major, is (p x n),
    ld p, the same for every block (stride 0); a per-block group's, stored (B n x k), is block
    b's (k x n) at b n k, ld k. Forward, for each block b and group g at offset o:
        pred[b n:(b+1) n, :] (+)= G_b^T theta[o + b p:o + (b+1) p, :]
    then the rows are scaled by the weights w = 1/sigma (dgmm) and the whitened data, already
    in {dataobs_batch}, subtracted. The data gradient, -J^T r with J = diag(w) G:
        grad[o + b p:o + (b+1) p, :] = -G_b (w * r)[b n:(b+1) n, :]
    """


    def initialize(self, model, application):
        """
        Upload the design matrices and the weights
        """
        d = model.design
        self.precision = model.precision
        self.blocks, self.days = d.blocks, d.days
        self.parameters = model.parameters
        self.offsets = model.offsets
        self._dataobs = model.dataobs
        self.groups = []
        for g in d.groups:
            G = g.G if g.shared else g.G.reshape(d.blocks * d.days, g.width)
            self.groups.append((g, altar.cuda.matrix(source=numpy.ascontiguousarray(G),
                                                     dtype=self.precision)))
        self.weight = altar.cuda.vector(source=d.weight.ravel(), dtype=self.precision)
        cublasx.library()
        return self


    def _predict(self, theta, out, samples):
        """
        {out} = w * (G theta) for the first {samples} columns, whitened predictions
        """
        handle = altar.cuda.cublas_handle()
        prec, n, B, P = self.precision, self.days, self.blocks, self.parameters
        addr = lambda a, o=0: cublasx.address(a, o, prec)
        for i, (g, G) in enumerate(self.groups):
            o, p = self.offsets[g.name], g.width
            cublasx.gemm_strided_batched(
                handle, prec, cublasx.OP_T, cublasx.OP_N, n, samples, p, 1.0,
                addr(G), p, 0 if g.shared else n * p,
                addr(theta, o), P, p,
                0.0 if i == 0 else 1.0,
                addr(out), B * n, n, B)
        cublasx.dgmm(handle, prec, cublasx.SIDE_LEFT, B * n, samples, addr(out), B * n,
                     addr(self.weight), 1, addr(out), B * n)
        return out


    def forward_model_batched(self, model, theta, prediction, batch=None):
        """
        The whitened residuals of the first {batch} rows of {theta}
        """
        samples = theta.shape[0] if batch is None else batch
        self._predict(theta, prediction, samples)
        prediction -= self._dataobs.dataobs_batch
        return self


    def gradient(self, model, controller, step, batch=None):
        """
        The prior gradient from my parameter sets, and the data gradient -J^T r, in the cuda
        state's {prior_gradient} and {data_gradient}
        """
        θ = step.theta
        samples = θ.shape[0] if batch is None else batch
        for name in model.psets_list:
            model.psets[name].prior_gradient(theta=θ, gradient=step.prior_gradient, batch=batch)
        shape = (θ.shape[0], self.blocks * self.days)
        if self._residual is None or self._residual.shape != shape:
            self._residual = altar.cuda.matrix(shape=shape, dtype=self.precision)
        r = self._residual
        self.forward_model_batched(model=model, theta=θ, prediction=r, batch=batch)
        handle = altar.cuda.cublas_handle()
        prec, n, B, P = self.precision, self.days, self.blocks, self.parameters
        addr = lambda a, o=0: cublasx.address(a, o, prec)
        # w * r, in place
        cublasx.dgmm(handle, prec, cublasx.SIDE_LEFT, B * n, samples, addr(r), B * n,
                     addr(self.weight), 1, addr(r), B * n)
        for g, G in self.groups:
            o, p = self.offsets[g.name], g.width
            cublasx.gemm_strided_batched(
                handle, prec, cublasx.OP_N, cublasx.OP_N, p, samples, n, -1.0,
                addr(G), p, 0 if g.shared else n * p,
                addr(r), B * n, n,
                0.0,
                addr(step.data_gradient, o), P, p, B)
        return self


    # private data
    groups = None # (group, its design matrix on the device)
    weight = None # 1/sigma of each observation, 0 where missing, on the device
    offsets = None
    _residual = None # the residuals of {gradient}, reused
    _dataobs = None


# end of file
