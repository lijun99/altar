#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.cuda.libcudaaltar.distributions's cudaGaussian bindings, against a real
GPU.

Covers the full set: {sample} (drawn values land in the requested [idx_begin, idx_end) range
and match N(mean, sigma^2) statistically; untouched columns stay exactly at their prior
value), {logpdf} (an accumulation into a (samples,) vector, checked against the closed-form
Gaussian log pdf), {logpdfgradient_i} (single-parameter gradient, accumulated, and a no-op
when {index} falls outside [idx_begin, idx_end)), and {logpdfgradient} (the full
(samples x parameters) gradient matrix, a plain assignment restricted to the requested
column range -- columns outside it are left untouched, unlike {logpdfgradient_i}'s
accumulation into a single (samples,) vector).
"""


def test():
    import numpy
    import pyre.grid
    import altar
    import altar.cuda

    distributions = altar.cuda.libcudaaltar.distributions

    rng = numpy.random.default_rng(7)
    samples, parameters = 20000, 4
    idx_begin, idx_end = 1, 3
    mean, sigma = 2.0, 1.5

    theta = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0

    distributions.cudaGaussian_sample(theta, idx_begin, idx_end, mean, sigma)

    arr = numpy.asarray(theta)
    # columns outside [idx_begin, idx_end) are untouched
    assert numpy.all(arr[:, 0] == 0.0)
    assert numpy.all(arr[:, 3] == 0.0)
    sampled = arr[:, idx_begin:idx_end]
    assert abs(sampled.mean() - mean) < 0.05
    assert abs(sampled.std() - sigma) < 0.05

    # logpdf: an accumulation into a (samples,) vector
    probability = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(probability)[:] = 0.0
    distributions.cudaGaussian_logpdf(theta, probability, idx_begin, idx_end, mean, sigma)
    got = numpy.asarray(probability)
    expected = numpy.zeros(samples)
    for i in range(idx_begin, idx_end):
        x = arr[:, i]
        expected += -0.5 * numpy.log(2 * numpy.pi * sigma**2) - 0.5 * ((x - mean) / sigma) ** 2
    assert numpy.allclose(got, expected, atol=1e-9)

    # logpdfgradient_i: single-parameter gradient, accumulated
    probability = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(probability)[:] = 0.0
    distributions.cudaGaussian_logpdfgradient_i(theta, probability, idx_begin, idx_end, 1, mean, sigma)
    got = numpy.asarray(probability)
    expected = (mean - arr[:, 1]) / sigma**2
    assert numpy.allclose(got, expected, atol=1e-9)

    # a no-op when {index} falls outside [idx_begin, idx_end)
    probability = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(probability)[:] = 0.0
    distributions.cudaGaussian_logpdfgradient_i(theta, probability, idx_begin, idx_end, 3, mean, sigma)
    assert numpy.all(numpy.asarray(probability) == 0.0)

    # logpdfgradient: the full gradient matrix, a plain assignment restricted to
    # [idx_begin, idx_end); columns outside it are left untouched
    gradient = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(gradient)[:, :] = -999.0
    distributions.cudaGaussian_logpdfgradient(theta, gradient, idx_begin, idx_end, mean, sigma)
    gp = numpy.asarray(gradient)
    for i in range(idx_begin, idx_end):
        expected = (mean - arr[:, i]) / sigma**2
        assert numpy.allclose(gp[:, i], expected, atol=1e-9)
    assert numpy.all(gp[:, 0] == -999.0)
    assert numpy.all(gp[:, 3] == -999.0)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
