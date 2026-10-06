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


# end of file
