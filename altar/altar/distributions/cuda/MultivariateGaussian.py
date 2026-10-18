# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import typing
import numpy
# the package
import altar
# the factorization and the sampling, on the host
from ..native.MultivariateGaussian import MultivariateGaussian as base

if typing.TYPE_CHECKING:
    from altar.cuda.array import Array


# the declaration
class MultivariateGaussian(base):
    """
    The cuda implementation of the multivariate normal distribution: the log density of the
    samples on the device; the factorization and the sampling stay on the host
    """


    def log_density(self, theta: Array, out: Array, batch: int | None = None) -> Array:
        """
        Fill the first {batch} entries of {out} with log N(θ_s; mean, covariance), for each
        row θ_s of {theta}: z = L^-1 (θ - mean) by one gemm, then the l2 log likelihood of its rows
        """
        rows, parameters = theta.shape
        batch = rows if batch is None else batch
        # the scratch rows, filled with -L^-1 mean, the offset of the whitened samples
        if self._shift is None or self._shift.shape[0] != rows:
            offset = -(self.whitener @ self.mean)
            self._shift = altar.cuda.matrix(
                source=numpy.tile(offset, (rows, 1)), dtype=self.precision)
            self._work = altar.cuda.matrix(shape=(rows, parameters), dtype=self.precision)
        work = self._work
        work.copy(self._shift)
        # the row-major rows of θ are the column-major columns of θ^T: work^T += L^-1 θ^T
        cublas = altar.cuda.cublas
        gemm = cublas.dgemm if self.precision == "float64" else cublas.sgemm
        gemm(altar.cuda.cublas_handle(), cublas.Operation.N, cublas.Operation.N,
             parameters, batch, parameters, 1.0,
             self._whitener.grid, parameters, theta.grid, parameters,
             1.0, work.grid, parameters)
        altar.cuda.libcudaaltar.norms.cudaL2_normllk(work.grid, out.grid, batch, self.constant)
        return out


    # meta-methods
    def __init__(self, mean: numpy.ndarray, covariance: numpy.ndarray,
                 precision: str = "float64", **kwds) -> None:
        super().__init__(mean=mean, covariance=covariance, precision=precision, **kwds)
        # stored row-major as (L^-1)^T, which a column-major gemm reads as L^-1
        self._whitener = altar.cuda.matrix(source=self.whitener.T.copy(), dtype=precision)
        return


    # private data
    _whitener: Array
    _shift: Array | None = None
    _work: Array | None = None


# end of file
