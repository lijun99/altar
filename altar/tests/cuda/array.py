#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved


"""
Sanity check: {altar.cuda.matrix}/{altar.cuda.vector} (the {pyre.grid}-backed replacement for
the old capsule-based buffers, in {altar/cuda/array.py}) and {altar.cuda.cublas.axpy}, against
a real GPU.

Covers every method the bayesian state/sampler layer (BayesianState, HMCState, ...) actually
calls: {zero}/{fill}/{clone}/{copy}/{copy_from_host}/{copy_to_host}/{mean_sd}, the {source=}
constructor, cpu <-> gpu interop (a cpu {altar.matrix} supports the buffer protocol directly,
so {copy_from_host}/{copy_to_host} work on it with no special-casing), attribute passthrough
to the underlying grid for anything not explicitly wrapped, {axpy} (used by
{compute_posterior}: posterior = prior + beta*data), and, added for the Metropolis sampler
family: {sum}, in-place {*=}/{+=}/{-=}, {cholesky} (against a numpy reference, careful to mask
the untouched triangle before reconstructing), and {altar.cuda.curand.uniform}/{gaussian}.
"""


def test():
    import numpy
    import altar
    import altar.cuda

    # shape/dtype accessors
    m = altar.cuda.matrix(shape=(5, 3), dtype="float64")
    assert m.shape == (5, 3) and m.rows == 5 and m.cols == 3 and m.dtype == "float64"

    # zero / fill
    m.zero()
    assert numpy.all(numpy.asarray(m) == 0.0)
    m.fill(2.5)
    assert numpy.all(numpy.asarray(m) == 2.5)

    v = altar.cuda.vector(shape=5, dtype="int32")
    assert v.shape == (5,)
    v.zero()
    assert numpy.all(numpy.asarray(v) == 0)

    # clone: independent memory
    clone = m.clone()
    assert numpy.array_equal(numpy.asarray(clone), numpy.asarray(m))
    clone.fill(9.0)
    assert not numpy.array_equal(numpy.asarray(clone), numpy.asarray(m))

    # copy: overwrite in place from another Array
    dest = altar.cuda.matrix(shape=(5, 3), dtype="float64").zero()
    dest.copy(m)
    assert numpy.array_equal(numpy.asarray(dest), numpy.asarray(m))

    # cpu <-> gpu interop: a cpu altar.matrix supports the buffer protocol directly
    cpu_source = altar.matrix(shape=(5, 3))
    numpy.asarray(cpu_source)[:, :] = 7.0
    gpu = altar.cuda.matrix(shape=(5, 3), dtype="float64").zero()
    gpu.copy_from_host(source=cpu_source)
    assert numpy.all(numpy.asarray(gpu) == 7.0)

    cpu_target = altar.matrix(shape=(5, 3))
    gpu.copy_to_host(target=cpu_target)
    assert numpy.all(numpy.asarray(cpu_target) == 7.0)

    host_copy = gpu.copy_to_host(type="numpy")
    assert isinstance(host_copy, numpy.ndarray) and numpy.all(host_copy == 7.0)

    # the source= constructor
    source = numpy.arange(12).reshape(3, 4).astype("float64")
    built = altar.cuda.matrix(source=source)
    assert numpy.array_equal(numpy.asarray(built), source)

    # mean_sd: per-column statistics
    rng = numpy.random.default_rng(1)
    stats = altar.cuda.matrix(shape=(2000, 3), dtype="float64")
    numpy.asarray(stats)[:, :] = rng.normal(loc=[1, 2, 3], scale=1, size=(2000, 3))
    mean, sd = stats.mean_sd()
    assert numpy.allclose(mean, [1, 2, 3], atol=0.1)
    assert numpy.allclose(sd, [1, 1, 1], atol=0.1)

    # attribute passthrough to the underlying grid, for anything not explicitly wrapped
    assert isinstance(m.address, int)

    # cublas.axpy: y = alpha*x + y
    x = altar.cuda.vector(shape=100, dtype="float64")
    numpy.asarray(x)[:] = 2.0
    y = altar.cuda.vector(shape=100, dtype="float64")
    numpy.asarray(y)[:] = 3.0
    altar.cuda.cublas.axpy(alpha=2.0, x=x, y=y, batch=100)
    assert numpy.allclose(numpy.asarray(y), 3.0 + 2.0 * 2.0)

    # sum
    flags = altar.cuda.vector(shape=10, dtype="int32")
    numpy.asarray(flags)[:] = [0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
    assert flags.sum() == 6

    # in-place arithmetic
    a = altar.cuda.matrix(shape=(3, 3), dtype="float64").fill(2.0)
    a *= 3.0
    assert numpy.all(numpy.asarray(a) == 6.0)
    b = altar.cuda.matrix(shape=(3, 3), dtype="float64").fill(1.0)
    a += b
    assert numpy.all(numpy.asarray(a) == 7.0)
    a -= b
    assert numpy.all(numpy.asarray(a) == 6.0)

    # cholesky: self = U^T U, U in the row-major upper triangle; the untouched lower
    # triangle must be masked before reconstructing, or the check would use stale data
    rng = numpy.random.default_rng(5)
    n = 4
    root = rng.random((n, n))
    spd = root @ root.T + n * numpy.eye(n)
    factored = altar.cuda.matrix(shape=(n, n), dtype="float64")
    numpy.asarray(factored)[:, :] = spd
    factored.cholesky()
    upper = numpy.triu(numpy.asarray(factored))
    assert numpy.allclose(upper.T @ upper, spd, atol=1e-8)

    # curand.uniform/gaussian
    uniform_draws = altar.cuda.vector(shape=20000, dtype="float64")
    altar.cuda.curand.uniform(out=uniform_draws)
    ud = numpy.asarray(uniform_draws)
    assert ud.min() > 0.0 and ud.max() <= 1.0
    assert abs(ud.mean() - 0.5) < 0.02

    gaussian_draws = altar.cuda.matrix(shape=(200, 100), dtype="float64")
    altar.cuda.curand.gaussian(out=gaussian_draws, mean=1.0, stddev=2.0)
    gd = numpy.asarray(gaussian_draws)
    assert abs(gd.mean() - 1.0) < 0.05
    assert abs(gd.std() - 2.0) < 0.05

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
