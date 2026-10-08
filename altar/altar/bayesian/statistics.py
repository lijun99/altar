# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
from __future__ import annotations
import numpy


# the weighted moments of a population, on the host
def weighted_variance(θ: numpy.ndarray, w: numpy.ndarray) -> numpy.ndarray:
    """
    The variance of each column of {θ} over its rows, with weights {w}
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    δ = θ - w @ θ
    return w @ (δ * δ)


def weighted_covariance(θ: numpy.ndarray, w: numpy.ndarray) -> numpy.ndarray:
    """
    The covariance Σ_i w_i (θ_i - θ̄)(θ_i - θ̄)^T of the rows of {θ}, with weights {w}
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    δ = θ - w @ θ
    return (δ * w[:, None]).T @ δ


def effective_size(w: numpy.ndarray) -> float:
    """
    The effective number of samples 1/Σ_i w_i² of the normalized weights {w}
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    return 1 / (w @ w)


def shrunk_correlation(θ: numpy.ndarray, w: numpy.ndarray,
                       λ: float | None = None) -> tuple[numpy.ndarray, float]:
    """
    The correlation of the columns of {θ} with weights {w}, shrunk toward the identity by {λ},
    or by the intensity of Schäfer & Strimmer (2005) if none, and that λ
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    δ = θ - w @ θ
    z = δ / numpy.sqrt(w @ (δ * δ))
    r = (z * w[:, None]).T @ z
    if λ is None:
        # the variance of each correlation over the samples, from the weighted moments of z_i z_j
        z2 = z * z
        var = ((z2 * w[:, None]).T @ z2 - r * r) / effective_size(w)
        off = ~numpy.eye(r.shape[0], dtype=bool)
        λ = float(numpy.clip(var[off].sum() / (r[off] ** 2).sum(), 0, 1))
    shrunk = (1 - λ) * r
    numpy.fill_diagonal(shrunk, 1)
    return shrunk, λ


def condition_covariance(Σ: numpy.ndarray, ratio: float) -> numpy.ndarray:
    """
    The symmetric positive definite version of {Σ}: its eigenvalues below {ratio} times the
    one of largest magnitude raised to that floor
    """
    λ, V = numpy.linalg.eigh(Σ)
    floor = ratio * λ[numpy.argmax(numpy.abs(λ))]
    conditioned = (V * numpy.maximum(λ, floor)) @ V.T
    return 0.5 * (conditioned + conditioned.T)


def multiplicities(w: numpy.ndarray, rng: numpy.random.Generator,
                   low_variance: bool = False) -> numpy.ndarray:
    """
    How many copies of each sample to keep, resampling by the normalized weights {w}: uniform
    draws in [0, 1), or the equally spaced (u + i)/n of the low variance resampler, counted in
    the intervals of the cumulative weights
    """
    n = w.size
    r = (rng.random() + numpy.arange(n)) / n if low_variance else rng.random(size=n)
    edges = numpy.concatenate(([0.0], numpy.cumsum(w)))
    # a draw past a cumulative sum that rounding left short of one belongs to the last sample
    bins = numpy.clip(numpy.searchsorted(edges, r, side="right") - 1, 0, n - 1)
    return numpy.bincount(bins, minlength=n)


# end of file
