#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: the pybind11+AnyGrid cudaL2 norm bindings (altar.cuda.libcudaaltar.norms),
against a real GPU, at both float64 and float32.

This is the first piece of altar.cuda.ext.cudaaltar ported from the old raw-cpython,
capsule-based bindings to pybind11, operating on altar.cuda.managed() directly. The rest of
the old extension (distributions, metropolis, leapfrog, langevin) is not yet ported -- see
the note in altar/ext/cuda/cudaaltar.cc.
"""


def check(precision):
    import numpy
    import altar.cuda

    norms = altar.cuda.libcudaaltar.norms

    samples, observations = 5, 4
    rng = numpy.random.default_rng(1)
    data_np = rng.random((samples, observations))

    data = altar.cuda.managed(shape=(samples, observations), cell=precision)
    result = altar.cuda.managed(shape=(samples,), cell=precision)
    numpy.asarray(data)[:, :] = data_np

    tol = 1e-8 if precision == "float64" else 1e-5

    # ||data||, one row of data per sample
    norms.cudaL2_norm(data, result, samples)
    expected = numpy.linalg.norm(data_np, axis=1)
    assert numpy.allclose(numpy.asarray(result), expected, atol=tol)

    # constant - 0.5 ||data||^2
    result2 = altar.cuda.managed(shape=(samples,), cell=precision)
    constant = 3.0
    norms.cudaL2_normllk(data, result2, samples, constant)
    expected2 = constant - 0.5 * expected**2
    assert numpy.allclose(numpy.asarray(result2), expected2, atol=tol)

    # a partial batch: only the first {batch} rows get touched
    batch = 3
    result3 = altar.cuda.managed(shape=(samples,), cell=precision)
    numpy.asarray(result3)[:] = -1.0
    norms.cudaL2_norm(data, result3, batch)
    got3 = numpy.asarray(result3)
    assert numpy.allclose(got3[:batch], expected[:batch], atol=tol)
    assert numpy.all(got3[batch:] == -1.0)

    # all done
    return


def test():
    # check both precisions altar cares about
    check("float64")
    check("float32")
    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
