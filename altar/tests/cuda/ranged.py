#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.cuda.libcudaaltar.distributions's cudaRanged bindings, against a real GPU.

Covers {verify} (flags samples outside [low, high], leaves an already-flagged sample alone
even if its parameters are now in range), {constrain} (clamps in place, only within the
requested column range), and the per-parameter-bounds counterpart {verify_unique}.
"""


def test():
    import numpy
    import altar.cuda
    import altar

    distributions = altar.cuda.libcudaaltar.distributions

    rng = numpy.random.default_rng(3)
    samples, parameters = 10000, 4
    idx_begin, idx_end = 1, 3
    low, high = -1.0, 1.0

    theta = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta)[:, :] = rng.uniform(-2, 2, size=(samples, parameters))
    arr = numpy.asarray(theta)

    invalid = altar.cuda.managed(shape=(samples,), cell="int32")
    numpy.asarray(invalid)[:] = 0
    distributions.cudaRanged_verify(theta, invalid, idx_begin, idx_end, low, high)
    inv = numpy.asarray(invalid)
    expected = numpy.any(
        (arr[:, idx_begin:idx_end] < low) | (arr[:, idx_begin:idx_end] > high), axis=1
    ).astype("int32")
    assert numpy.array_equal(inv, expected)

    # a pre-flagged sample stays flagged even when its parameters are in range
    invalid = altar.cuda.managed(shape=(samples,), cell="int32")
    numpy.asarray(invalid)[:] = 0
    numpy.asarray(invalid)[0] = 1
    theta_ok = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(theta_ok)[:, :] = 0.0
    distributions.cudaRanged_verify(theta_ok, invalid, idx_begin, idx_end, low, high)
    assert numpy.asarray(invalid)[0] == 1
    assert numpy.all(numpy.asarray(invalid)[1:] == 0)

    # constrain: clamps only the requested columns
    clamped = altar.cuda.managed(shape=(samples, parameters), cell="float64")
    numpy.asarray(clamped)[:, :] = arr
    distributions.cudaRanged_constrain(clamped, idx_begin, idx_end, low, high)
    c = numpy.asarray(clamped)
    expected = numpy.clip(arr[:, idx_begin:idx_end], low, high)
    assert numpy.allclose(c[:, idx_begin:idx_end], expected)
    assert numpy.array_equal(c[:, 0], arr[:, 0]) and numpy.array_equal(c[:, 3], arr[:, 3])

    # per-parameter bounds
    n = idx_end - idx_begin
    lows = altar.cuda.managed(shape=(n,), cell="float64")
    highs = altar.cuda.managed(shape=(n,), cell="float64")
    numpy.asarray(lows)[:] = [-0.5, -1.5]
    numpy.asarray(highs)[:] = [0.5, 1.5]
    invalid = altar.cuda.managed(shape=(samples,), cell="int32")
    numpy.asarray(invalid)[:] = 0
    distributions.cudaRanged_verify_unique(theta, invalid, idx_begin, idx_end, lows, highs)
    inv = numpy.asarray(invalid)
    lo_arr, hi_arr = numpy.asarray(lows), numpy.asarray(highs)
    expected = numpy.zeros(samples, dtype="int32")
    for j in range(n):
        col = arr[:, idx_begin + j]
        expected |= ((col < lo_arr[j]) | (col > hi_arr[j])).astype("int32")
    assert numpy.array_equal(inv, expected)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
