#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.cuda.libcudaaltar.distributions's cudaUniform bindings, against a real
GPU.

Covers {sample} (drawn values land in [low, high) and only in the requested column range),
{logpdf} (constant per sample, -log(high - low) * (idx_end - idx_begin), accumulated), and
their per-parameter-bounds counterparts {sample_unique}/{logpdf_unique}.
"""


def test():
    import numpy
    import altar.cuda
    import altar

    distributions = altar.cuda.libcudaaltar.distributions

    samples, parameters = 20000, 4
    idx_begin, idx_end = 1, 3
    low, high = -2.0, 5.0

    theta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0
    distributions.cudaUniform_sample(theta, idx_begin, idx_end, low, high)
    arr = numpy.asarray(theta)
    assert numpy.all(arr[:, 0] == 0.0) and numpy.all(arr[:, 3] == 0.0)
    sub = arr[:, idx_begin:idx_end]
    assert sub.min() >= low and sub.max() < high
    assert abs(sub.mean() - (low + high) / 2) < 0.1

    probability = altar.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(probability)[:] = 0.0
    distributions.cudaUniform_logpdf(theta, probability, idx_begin, idx_end, low, high)
    got = numpy.asarray(probability)
    expected = numpy.full(samples, -numpy.log(high - low) * (idx_end - idx_begin))
    assert numpy.allclose(got, expected)

    # per-parameter bounds
    n = idx_end - idx_begin
    lows = altar.cuda.managed(shape=(n,), cell="float64")
    highs = altar.cuda.managed(shape=(n,), cell="float64")
    numpy.asarray(lows)[:] = [-1.0, 0.0]
    numpy.asarray(highs)[:] = [3.0, 10.0]

    theta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = 0.0
    distributions.cudaUniform_sample_unique(theta, idx_begin, idx_end, lows, highs)
    arr = numpy.asarray(theta)
    assert numpy.all(arr[:, 0] == 0.0) and numpy.all(arr[:, 3] == 0.0)
    assert arr[:, 1].min() >= -1.0 and arr[:, 1].max() < 3.0
    assert arr[:, 2].min() >= 0.0 and arr[:, 2].max() < 10.0

    probability = altar.cuda.managed(shape=(samples,), cell="float64")
    numpy.asarray(probability)[:] = 0.0
    distributions.cudaUniform_logpdf_unique(theta, probability, idx_begin, idx_end, lows, highs)
    got = numpy.asarray(probability)
    expected = numpy.full(samples, -numpy.log(3.0 - (-1.0)) - numpy.log(10.0 - 0.0))
    assert numpy.allclose(got, expected)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
