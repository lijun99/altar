#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#


"""
The correlated gaussian prior, {altar.distributions.correlated}: its covariance and density
against direct numpy, its gradient against finite differences, in physical and in logit space,
its exact draws against its covariance, and its draws within a support, by elliptical slice
sampling, against rejection sampling; with --gpu, the cuda density and gradient against the
cpu ones, for a block of columns in the middle of a wider {theta}.

Built without pyre's component machinery, as {altar/tests/cuda/framework_distributions.py}
does: the backend classes, configured by hand as the shim's {_makeImpl} would.
"""

import math
import os
import sys
import tempfile
import numpy


class _RNG:
    def __init__(self, seed):
        self.rng = numpy.random.default_rng(seed)


def make(backend, coordinates, offset=0, support=(-math.inf, math.inf), reparameterize=False,
         kernel="exponential", length=(2.0,), mean=1.0, sigma=0.5, seed=1, application=None):
    """
    A correlated prior of {backend}, over the locations in the file {coordinates}
    """
    module = __import__(f"altar.distributions.{backend}.Correlated", fromlist=["Correlated"])
    prior = module.Correlated()
    prior.offset = offset
    prior.parameters = numpy.loadtxt(coordinates, ndmin=2).shape[0]
    prior.mean, prior.sigma, prior.kernel, prior.length = mean, sigma, kernel, length
    prior.coordinates, prior.dataset = coordinates, None
    prior.support, prior.jitter, prior.sweeps = support, 1e-6, 100
    prior.reparameterize = reparameterize
    if reparameterize:
        transform = __import__(f"altar.distributions.{backend}.LogitTransform",
                               fromlist=["LogitTransform"]).LogitTransform()
        transform.support = support
        prior.transform = transform
    prior.initialize(rng=_RNG(seed) if backend == "native" else None, application=application)
    return prior


def check(label, error, tolerance):
    status = "ok" if error <= tolerance else "FAILED"
    print(f"  {label}: {error:.2e} (tolerance {tolerance:.0e}) {status}")
    return error <= tolerance


def cpu(coordinates, x):
    passed = []
    n = x.shape[0]
    # the covariance and the density, against their definitions
    prior = make("native", coordinates)
    r = numpy.abs(x[:, None, 0] - x[None, :, 0]) / 2.0
    C = 0.25 * (numpy.exp(-r) + 1e-6 * numpy.identity(n))
    passed.append(check("covariance", numpy.abs(prior.covariance - C).max(), 1e-14))
    rng = numpy.random.default_rng(2)
    θ = 1.0 + rng.standard_normal((5, n)) @ numpy.linalg.cholesky(C).T
    d = θ - 1.0
    expected = (-0.5 * numpy.einsum("si,ij,sj->s", d, numpy.linalg.inv(C), d)
                - 0.5 * numpy.linalg.slogdet(2 * math.pi * C)[1])
    likelihood = numpy.zeros(5)
    prior.eval_prior(theta=θ, likelihood=likelihood)
    passed.append(check("log density", numpy.abs(likelihood - expected).max(), 1e-9))
    # the gradient, against central differences of the density
    gradient = numpy.zeros_like(θ)
    prior.prior_gradient(theta=θ, gradient=gradient)
    h = 1e-6
    fd = numpy.zeros_like(θ)
    for j in range(n):
        step = numpy.zeros(n)
        step[j] = h
        fd[:, j] = (prior.log_density(θ + step) - prior.log_density(θ - step)) / (2 * h)
    passed.append(check("gradient", numpy.abs(gradient - fd).max() / numpy.abs(fd).max(), 1e-7))
    # reparameterized: the gradient in logit space, of log P(x(φ)) + log|J(φ)|
    support = (0.0, 3.0)
    bounded = make("native", coordinates, support=support, reparameterize=True)
    x0 = numpy.clip(θ, 0.1, 2.9)
    gradient = numpy.zeros_like(x0)
    bounded.prior_gradient(theta=x0, gradient=gradient)
    a, b = support
    phi = numpy.log((x0 - a) / (b - x0))
    physical = lambda p: a + (b - a) / (1 + numpy.exp(-p))
    total = lambda p: (bounded.log_density(physical(p))
                       + numpy.log((physical(p) - a) * (b - physical(p)) / (b - a)).sum(axis=1))
    fd = numpy.zeros_like(phi)
    for j in range(n):
        step = numpy.zeros(n)
        step[j] = h
        fd[:, j] = (total(phi + step) - total(phi - step)) / (2 * h)
    passed.append(check("gradient in logit space", numpy.abs(gradient - fd).max() / numpy.abs(fd).max(), 1e-6))
    # exact draws: the sample covariance against C
    draws = prior.draw(shape=(200000, n))
    passed.append(check("draws, covariance", numpy.abs(numpy.cov(draws.T) - C).max() / C.max(), 2e-2))
    # within a support: elliptical slice sampling against rejection sampling
    low, high = 0.8, 1.6
    truncated = make("native", coordinates, support=(low, high))
    ess = truncated.draw(shape=(20000, n))
    inside = ((ess >= low) & (ess <= high)).all()
    raw = prior.draw(shape=(2000000, n))
    kept = raw[((raw >= low) & (raw <= high)).all(axis=1)]
    print(f"  {kept.shape[0]} of {raw.shape[0]} untruncated draws inside the support")
    passed.append(check("truncated draws, inside", 0.0 if inside else 1.0, 0.0))
    passed.append(check("truncated draws, means", numpy.abs(ess.mean(0) - kept.mean(0)).max(), 1e-2))
    passed.append(check("truncated draws, sds", numpy.abs(ess.std(0) - kept.std(0)).max(), 1e-2))
    return all(passed)


class _Application:
    class job:
        gpuprecision = "float64"
    rng = _RNG(3)


def gpu(coordinates):
    import altar.cuda
    passed = []
    rows, offset = 64, 3
    for reparameterize in [False, True]:
        support = (0.0, 3.0) if reparameterize else (-math.inf, math.inf)
        host = make("native", coordinates, offset=offset, support=support, reparameterize=reparameterize)
        device = make("cuda", coordinates, offset=offset, support=support, reparameterize=reparameterize,
                      application=_Application())
        n = host.parameters
        # my columns in the middle of a wider {theta}
        width = offset + n + 2
        rng = numpy.random.default_rng(4)
        θ = rng.uniform(0.2, 2.8, size=(rows, width))
        label = "reparameterized" if reparameterize else "unbounded"
        # the density, added to what is there
        expected = numpy.full(rows, 0.5)
        host.eval_prior(theta=θ, likelihood=expected)
        theta = altar.cuda.matrix(source=θ, dtype="float64")
        likelihood = altar.cuda.vector(source=numpy.full(rows, 0.5), dtype="float64")
        device.eval_prior(theta=theta, likelihood=likelihood)
        passed.append(check(f"cuda log density, {label}", numpy.abs(numpy.array(likelihood) - expected).max(), 1e-9))
        # the gradient: my columns replaced, the others kept, twice in a row
        other = rng.standard_normal((rows, width))
        expected = other.copy()
        host.prior_gradient(theta=θ, gradient=expected)
        gradient = altar.cuda.matrix(source=other, dtype="float64")
        for _ in range(2):
            device.prior_gradient(theta=theta, gradient=gradient)
        passed.append(check(f"cuda gradient, {label}", numpy.abs(numpy.array(gradient) - expected).max(), 1e-9))
        # the initial samples, in my columns only
        theta = altar.cuda.matrix(source=numpy.zeros((rows, width)), dtype="float64")
        device.initialize_sample(theta=theta)
        values = numpy.array(theta)
        outside = numpy.abs(numpy.delete(values, range(offset, offset + n), axis=1)).max()
        passed.append(check(f"cuda initial samples, other columns, {label}", outside, 0.0))
    return all(passed)


def main():
    # twelve parameters along a line, a correlation length of 2
    x = numpy.linspace(0.0, 11.0, 12)
    with tempfile.TemporaryDirectory() as scratch:
        coordinates = os.path.join(scratch, "coordinates.txt")
        numpy.savetxt(coordinates, x[:, None])
        print("cpu:")
        ok = cpu(coordinates, x[:, None])
        if "--gpu" in sys.argv:
            print("gpu:")
            ok = gpu(coordinates) and ok
    print("correlated prior:", "passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())


# end of file
