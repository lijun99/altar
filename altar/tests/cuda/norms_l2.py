#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-2025 parasim inc
# (c) 2010-2025 california institute of technology
# all rights reserved


"""
Sanity check: altar.norms.cuda.L2, against a real GPU.

Covers the plain norm/likelihood path (which just calls the ported cudaL2 kernels, see
tests/cuda/norm.py) and the covariance-weighted path, which goes through
pyre.cuda.cublas.dtrmm/strmm -- the row-major "swap trick" translation of the old
cuBlas.trmm-based code. The expected value here is derived from the swap trick itself (v @
L.T, where L.T is what a caller stores in sigma_inv's upper triangle), not from the
"textbook" whitened norm (v @ L) -- the two coincide only once DataL2's own covariance
computation is also ported, since that's what decides which convention actually ends up in
sigma_inv. This test locks in that the trmm translation itself is faithful to the pre-port
behavior, confirmed once by comparing against the real GPU (see the session notes).
"""


def test():
    import numpy
    import pyre.grid
    from altar.norms.cuda.L2 import L2

    rng = numpy.random.default_rng(3)
    samples, observations = 4, 3

    # a real SPD covariance and its cholesky-factored inverse
    A = rng.random((observations, observations))
    Cd = A @ A.T + observations * numpy.eye(observations)
    Cd_inv = numpy.linalg.inv(Cd)
    L = numpy.linalg.cholesky(Cd_inv)  # Cd_inv = L @ L.T, L lower triangular

    v_np = rng.random((samples, observations))

    v = pyre.grid.managed(shape=(samples, observations), cell="float64")
    numpy.asarray(v)[:, :] = v_np

    # store L.T in the row-major upper triangle (row <= col); the lower triangle is never
    # read, so it's left at whatever pyre.grid.managed() happened to give it
    sigma_inv = pyre.grid.managed(shape=(observations, observations), cell="float64")
    sig = numpy.asarray(sigma_inv)
    for a in range(observations):
        for b in range(a, observations):
            sig[a, b] = L.T[a, b]

    l2 = L2()

    # plain norm, no covariance
    out = l2.eval(v=v, sigma_inv=None)
    assert numpy.allclose(numpy.asarray(out), numpy.linalg.norm(v_np, axis=1))

    # covariance-weighted norm: v_new = v @ L.T, then the row-wise l2 norm
    v2 = pyre.grid.managed(shape=(samples, observations), cell="float64")
    numpy.asarray(v2)[:, :] = v_np
    out2 = l2.eval(v=v2, sigma_inv=sigma_inv)
    expected2 = numpy.linalg.norm(v_np @ L.T, axis=1)
    assert numpy.allclose(numpy.asarray(out2), expected2)

    # eval_likelihood: constant - 0.5 * norm^2
    v3 = pyre.grid.managed(shape=(samples, observations), cell="float64")
    numpy.asarray(v3)[:, :] = v_np
    constant = 2.0
    out3 = l2.eval_likelihood(v=v3, constant=constant, sigma_inv=sigma_inv)
    expected3 = constant - 0.5 * expected2**2
    assert numpy.allclose(numpy.asarray(out3), expected3)

    # all done
    return


# main
if __name__ == "__main__":
    test()


# end of file
