# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Support for cross-fade sampling (Minson, 2024, GJI 239, 1629): a model with a conjugate prior
p_conj(θ) = N(m, C_m), whose posterior p_conj(θ|d) = N(m*, C*) is known in closed form, samples
f_β(θ) ∝ p_conj(θ|d) [p(θ) / p_conj(θ)]^β from β = 0 to 1, i.e., the annealer's prior slot holds
log p_conj(θ|d) and its data slot log p(θ) - log p_conj(θ); the data likelihood is never
evaluated, and the evidence is p(d) = p_conj(d) Π_m <w_m>
"""

# externals
from __future__ import annotations
import math
import typing
import numpy
# the package
import altar

if typing.TYPE_CHECKING:
    from altar.arrays import Array
    from altar.models.BayesianL2 import BayesianL2


class Gaussian:
    """
    A multivariate normal distribution over the parameters of a model, whose log density is
    evaluated for a batch of samples at once, on the cpu or the gpu
    """

    def log_density(self, theta: Array, out: Array, batch: int | None = None) -> Array:
        """
        Fill the first {batch} entries of {out} with log N(θ_s; mean, covariance), for each
        row θ_s of {theta}
        """
        batch = theta.shape[0] if batch is None else batch
        if self.cuda:
            return self._log_density_cuda(theta=theta, out=out, batch=batch)
        x = numpy.asarray(theta[:batch], dtype=numpy.float64)
        z = (x - self.mean) @ self.whitener.T
        out[:batch] = self.constant - 0.5 * (z * z).sum(axis=1)
        return out


    def sample(self, rows: int, rng: numpy.random.Generator) -> numpy.ndarray:
        """
        A (rows x parameters) array of random samples, drawn with {rng}
        """
        z = rng.standard_normal(size=(rows, self.mean.size))
        return self.mean + z @ self.factor.T


    # implementation details
    def _log_density_cuda(self, theta: Array, out: Array, batch: int) -> Array:
        """
        z = L^-1 (θ - mean) by one gemm, then the l2 log likelihood of its rows
        """
        rows, parameters = theta.shape
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
        super().__init__(**kwds)
        self.mean = numpy.asarray(mean, dtype=float)
        covariance = numpy.asarray(covariance, dtype=float)
        # covariance = L L^T, and its inverse factor L^-1
        self.factor = numpy.linalg.cholesky(covariance)
        self.whitener = numpy.linalg.inv(self.factor)
        # the normalization, -P/2 log 2π - log |L|
        self.constant = (-0.5 * self.mean.size * math.log(2 * math.pi)
                         - numpy.log(numpy.diag(self.factor)).sum())
        self.precision = precision
        self.cuda = altar.backends.active() == "cuda"
        if self.cuda:
            # stored row-major as (L^-1)^T, which a column-major gemm reads as L^-1
            self._whitener = altar.cuda.matrix(source=self.whitener.T.copy(), dtype=precision)
        return


    # private data
    _shift: Array | None = None
    _work: Array | None = None
    _whitener: Array | None = None


class CrossFade:
    """
    The state of a model in cross-fade sampling: its conjugate prior and posterior, and the
    evidence of the conjugate model, log p_conj(d)
    """

    def log_densities(self, model: BayesianL2, theta: Array, prior: Array, data: Array,
                      batch: int | None = None) -> typing.Self:
        """
        Fill {prior} with log p_conj(θ|d), and {data} with log p(θ) - log p_conj(θ), where
        log p(θ) is the prior of the parameter sets of {model}
        """
        rows = theta.shape[0]
        batch = rows if batch is None else batch
        self.posterior.log_density(theta=theta, out=prior, batch=batch)
        self._zero(data)
        for name in model.psets_list:
            model.psets[name].eval_prior(theta=theta, prior=data, batch=batch)
        scratch = self._scratch(rows=rows, dtype=data.dtype)
        self.prior.log_density(theta=theta, out=scratch, batch=batch)
        if self.cuda:
            altar.cuda.cublas.axpy(alpha=-1.0, x=scratch, y=data, batch=batch)
        else:
            data[:batch] -= scratch[:batch]
        # outside the support of a bounded prior, the target vanishes: no weight, but finite, so
        # that β = 0 still gives the conjugate posterior
        mask = self._zero(self._mask(rows=rows))
        model.verify_theta(theta=theta, mask=mask, batch=batch)
        outside = numpy.asarray(mask)[:batch] != 0
        ratio = numpy.asarray(data)[:batch]
        ratio[outside] = self.floor
        numpy.maximum(ratio, self.floor, out=ratio)
        return self


    def draw(self, model: BayesianL2, rows: int, limit: int = 10**6) -> float:
        """
        Draw {rows} initial samples from the conjugate posterior within the support of the
        prior of {model}, by rejection, the limit of the target as β -> 0; return the fraction
        of the conjugate posterior within the support
        """
        parameters = self.posterior.mean.size
        if self.cuda:
            theta = altar.cuda.matrix(shape=(rows, parameters), dtype=self.precision)
        else:
            theta = numpy.zeros((rows, parameters), dtype=self.precision)
        mask = self._mask(rows=rows)
        kept, count, drawn = [], 0, 0
        while count < rows and drawn < limit:
            candidates = self.posterior.sample(rows=rows, rng=self.rng)
            numpy.asarray(theta)[...] = candidates
            model.verify_theta(theta=theta, mask=self._zero(mask), batch=rows)
            inside = numpy.asarray(mask)[:rows] == 0
            kept.append(candidates[inside])
            count += int(inside.sum())
            drawn += rows
        if count < rows:
            raise ValueError(
                f"only {count} of {drawn} samples of the conjugate posterior lie within the support "
                f"of the prior; the posterior is too far from the conjugate posterior, try "
                f"a softuniform prior or catmip")
        self.pool = numpy.concatenate(kept)[:rows]
        return count / drawn


    # the log density ratio of the samples outside the support of the prior
    floor = -1e30


    # implementation details
    def _zero(self, array: Array) -> Array:
        """
        Fill {array} with zeroes, on the device or the host
        """
        if self.cuda:
            return array.zero()
        array[...] = 0
        return array


    def _mask(self, rows: int) -> Array:
        """
        A (rows,) mask for the support check of the prior
        """
        if self._flags is None or self._flags.shape[0] != rows:
            if self.cuda:
                self._flags = altar.cuda.vector(shape=rows, dtype="int32").zero()
            else:
                self._flags = numpy.zeros(rows)
        return self._flags


    def _scratch(self, rows: int, dtype: typing.Any) -> Array:
        """
        A (rows,) vector for the conjugate prior
        """
        if self._vector is None or self._vector.shape[0] != rows:
            if self.cuda:
                self._vector = altar.cuda.vector(shape=rows, dtype=dtype).zero()
            else:
                self._vector = numpy.zeros(rows, dtype=dtype)
        return self._vector


    # meta-methods
    def __init__(self, prior: Gaussian, posterior: Gaussian, log_evidence: float,
                 rng: numpy.random.Generator, precision: str = "float64", **kwds) -> None:
        super().__init__(**kwds)
        # the conjugate prior N(m, C_m) and posterior N(m*, C*), as {Gaussian} instances
        self.prior = prior
        self.posterior = posterior
        # log p_conj(d)
        self.log_evidence = log_evidence
        # the numpy generator for the initial samples
        self.rng = rng
        # the precision of the samples
        self.precision = precision
        self.cuda = altar.backends.active() == "cuda"
        return


    # private data
    pool: numpy.ndarray | None = None # the initial samples
    _vector: Array | None = None
    _flags: Array | None = None


# end of file
