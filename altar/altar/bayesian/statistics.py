# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


# externals
import numpy


# the weighted moments of a population, on the host
def weighted_variance(θ, w):
    """
    The variance of each column of {θ} over its rows, with weights {w}
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    δ = θ - w @ θ
    return w @ (δ * δ)


def weighted_covariance(θ, w):
    """
    The covariance Σ_i w_i (θ_i - θ̄)(θ_i - θ̄)^T of the rows of {θ}, with weights {w}
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    δ = θ - w @ θ
    return (δ * w[:, None]).T @ δ


def effective_size(w):
    """
    The effective number of samples 1/Σ_i w_i² of the normalized weights {w}
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    return 1 / (w @ w)


def shrunk_correlation(θ, w):
    """
    The correlation of the columns of {θ} with weights {w}, shrunk toward the identity with the
    intensity λ of Schäfer & Strimmer (2005), and that λ
    """
    w = numpy.asarray(w, dtype=float)
    w = w / w.sum()
    δ = θ - w @ θ
    z = δ / numpy.sqrt(w @ (δ * δ))
    r = (z * w[:, None]).T @ z
    # the variance of each correlation over the samples, from the weighted moments of z_i z_j
    z2 = z * z
    var = ((z2 * w[:, None]).T @ z2 - r * r) / effective_size(w)
    off = ~numpy.eye(r.shape[0], dtype=bool)
    λ = float(numpy.clip(var[off].sum() / (r[off] ** 2).sum(), 0, 1))
    shrunk = (1 - λ) * r
    numpy.fill_diagonal(shrunk, 1)
    return shrunk, λ


# end of file
