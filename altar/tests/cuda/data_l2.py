#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
Sanity check: altar.data.cuda.DataL2, against a real GPU.

Exercises {update_covariance} (the cusolver dpotrf/dpotri/dpotrf sequence that computes the
Cholesky-decomposed inverse covariance and the l2 normalization), {merge_cdto_data} (the
cublas dtrmv "opposite triangle" translation), {eval_likelihood}, and {release_cd} -- all
against numpy references, without going through the pyre component layer (blocked for
array-trait components by a pre-existing pyre bug; see the session notes), instantiating the
plain backend class directly instead, exactly as {norms_l2.py} does for {altar.norms.cuda.L2}.

{eval_likelihood}'s {residual=False} path assumes {prediction} is already in the same
covariance-whitened space as {dataobs_batch} (i.e. pre-multiplied by the same U the forward
model would apply to its own Green's function) -- it does not itself apply the covariance, by
design (see the class docstring). The whitened-space check here mirrors that convention.
"""


def check(precision):
    import numpy
    import altar.cuda
    from altar.norms.cuda.L2 import L2
    from altar.data.cuda.DataL2 import DataL2

    dtype = precision
    tol = 1e-9 if precision == "float64" else 2e-5

    n = 5
    samples = 6
    rng = numpy.random.default_rng(42)

    A = rng.random((n, n))
    Cchi = (A @ A.T + n * numpy.eye(n)).astype(dtype)
    dataobs = rng.random(n).astype(dtype)

    class _Channel:
        def log(self, msg):
            pass

    d = DataL2()
    d.observations = n
    d.cd_dtype = precision
    d.precision = precision
    d.cd = Cchi
    d.dataobs = dataobs
    d.samples = samples
    d.error = _Channel()
    d.info = _Channel()
    d.norm = L2()

    d._dataobs_batch = altar.cuda.managed(shape=(samples, n), cell=precision)
    d.update_covariance()

    # reference: Cd_inv = L L^T (numpy, lower L); the class stores U = L^T in cd_inv's
    # row-major upper triangle
    Cd_inv = numpy.linalg.inv(Cchi.astype("float64"))
    L = numpy.linalg.cholesky(Cd_inv)
    U = L.T.astype(dtype)

    got_U = numpy.asarray(d.cd_inv)
    iu = numpy.triu_indices(n)
    assert numpy.allclose(got_U[iu], U[iu], atol=tol)

    logdet_expected = -0.5 * numpy.log(2 * numpy.pi) * n + numpy.log(numpy.diag(L)).sum()
    assert abs(d.normalization - logdet_expected) < tol

    # merge_cdto_data: every row of dataobs_batch equals U @ dataobs, duplicated
    merged_expected = U @ dataobs
    batch = numpy.asarray(d._dataobs_batch)
    for s in range(samples):
        assert numpy.allclose(batch[s], merged_expected, atol=tol)

    # eval_likelihood, in the whitened space dataobs_batch itself lives in
    theta_raw = rng.random((samples, n)).astype(dtype)
    theta_whitened = (theta_raw @ U.T).astype(dtype)
    theta_whitened[0] = batch[0]  # sample 0: an exact match, residual 0

    prediction = altar.cuda.managed(shape=(samples, n), cell=precision)
    numpy.asarray(prediction)[:, :] = theta_whitened

    likelihood = altar.cuda.managed(shape=(samples,), cell=precision)
    d.eval_likelihood(prediction=prediction, likelihood=likelihood, residual=False, batch=samples)

    lik = numpy.asarray(likelihood)
    assert abs(lik[0] - d.normalization) < tol

    resid = theta_whitened.astype("float64") - batch.astype("float64")
    expected = d.normalization - 0.5 * numpy.sum(resid**2, axis=1)
    assert numpy.allclose(lik, expected, atol=max(tol, 1e-4))

    # release_cd
    d.release_cd()
    assert d.cd_inv is None

    # all done
    return


def test():
    check("float64")
    check("float32")
    return


# main
if __name__ == "__main__":
    test()


# end of file
