#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Check the weighted moments of {altar.bayesian.statistics} against numpy
"""


def test():
    import numpy
    from altar.bayesian.statistics import (
        effective_size, shrunk_correlation, weighted_covariance, weighted_variance)

    rng = numpy.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    θ = rng.normal(size=(2000, 6)) @ A + 3
    # unnormalized weights, which the moments normalize
    w = 5 * rng.random(2000)

    Σ = numpy.cov(θ.T, aweights=w, bias=True)
    assert numpy.allclose(weighted_covariance(θ, w), Σ)
    assert numpy.allclose(weighted_variance(θ, w), numpy.diag(Σ))
    # uniform weights count every sample
    assert numpy.isclose(effective_size(numpy.ones(100)), 100)
    assert effective_size(w) < 2000

    # no shrinkage is the correlation, full shrinkage the identity
    r = Σ / numpy.sqrt(numpy.outer(numpy.diag(Σ), numpy.diag(Σ)))
    shrunk, λ = shrunk_correlation(θ, w, 0.0)
    assert λ == 0 and numpy.allclose(shrunk, r)
    shrunk, λ = shrunk_correlation(θ, w, 1.0)
    assert numpy.allclose(shrunk, numpy.eye(6))
    # the automatic intensity shrinks little with many samples, more with few
    _, many = shrunk_correlation(θ, numpy.ones(2000))
    _, few = shrunk_correlation(θ[:20], numpy.ones(20))
    assert 0 <= many < few <= 1

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
