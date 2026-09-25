#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.cuda.libcudaaltar.distributions's cudaTGaussian bindings, against a real
GPU.

Covers {sample} (drawn values' normal CDF -- computed independently here via math.erf, not
through the extension -- lands inside the requested normalized [low, high) support) and
{logpdf} (an accumulation, checked against the closed-form truncated-Gaussian log pdf).
"""


def test():
    import math
    import numpy
    import pyre.grid
    import altar
    import altar.cuda

    erf = numpy.vectorize(math.erf)
    distributions = altar.cuda.libcudaaltar.distributions

    samples, parameters = 20000, 3
    idx_begin, idx_end = 0, 2
    mean, sigma = 1.0, 2.0
    low, high = 0.1, 0.9  # normalized (Phi-space) support

    theta = pyre.grid.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0
    distributions.cudaTGaussian_sample(theta, idx_begin, idx_end, mean, sigma, low, high)
    arr = numpy.asarray(theta)
    assert numpy.all(arr[:, 2] == 0.0)

    sub = arr[:, idx_begin:idx_end]
    phi = 0.5 * (1 + erf((sub - mean) / (sigma * numpy.sqrt(2))))
    assert phi.min() >= low - 1e-6 and phi.max() < high + 1e-6

    probability = pyre.grid.managed(shape=(samples,), cell="float64")
    numpy.asarray(probability)[:] = 0.0
    distributions.cudaTGaussian_logpdf(theta, probability, idx_begin, idx_end, mean, sigma, low, high)
    got = numpy.asarray(probability)
    expected = numpy.zeros(samples)
    c1 = -numpy.log(sigma * numpy.sqrt(2 * numpy.pi) * (high - low))
    for i in range(idx_begin, idx_end):
        x = arr[:, i]
        expected += c1 - 0.5 * ((x - mean) / sigma) ** 2
    assert numpy.allclose(got, expected, atol=1e-8)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
