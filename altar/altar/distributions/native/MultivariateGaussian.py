# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# externals
from __future__ import annotations
import math
import numpy


# the declaration
class MultivariateGaussian:
    """
    The cpu implementation of the multivariate normal distribution N(mean, covariance) over the
    columns of a (samples x parameters) array, evaluated for a batch of samples at once

    Not a pyre component: its users build it directly, with a mean and a covariance they
    computed, e.g. the conjugate prior and posterior of cross-fade sampling
    """


    def log_density(self, theta: numpy.ndarray, out: numpy.ndarray,
                    batch: int | None = None) -> numpy.ndarray:
        """
        Fill the first {batch} entries of {out} with log N(θ_s; mean, covariance), for each
        row θ_s of {theta}
        """
        batch = theta.shape[0] if batch is None else batch
        # whiten the samples, z = L^-1 (θ - mean), in double precision
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
        # the precision of the samples
        self.precision = precision
        return


    # public data
    mean: numpy.ndarray
    factor: numpy.ndarray # L, with covariance = L L^T
    whitener: numpy.ndarray # L^-1
    constant: float
    precision: str


# end of file
