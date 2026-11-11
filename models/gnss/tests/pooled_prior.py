#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# check the spatially pooled sparsity prior of the gnss trajectory model on a small case: an
# isolated coefficient costs what a plain Laplace prior charges; the log density against a
# brute-force evaluation of its formula; the gradient against central differences, for both
# shapes and every pooling; and, with a gpu, the cuda log density and gradient against the cpu
#     python pooled_prior.py [--gpu]
import pathlib
import sys
import tempfile
import types

import numpy

STATIONS = {"ALBH": (-123.49, 48.39), "SC02": (-123.02, 48.55), "P435": (-123.71, 48.05),
            "PGC5": (-123.45, 48.65), "SEAT": (-122.31, 47.65), "NEAH": (-124.62, 48.30)}
OFFSET = 5                   # my parameters start after 5 columns of another parameter set


def case(path):
    from altar.models.gnss import design
    names = list(STATIONS)
    station = numpy.repeat(names, 3)
    component = numpy.tile(["east", "north", "up"], len(names))
    lon = numpy.repeat([STATIONS[s][0] for s in names], 3)
    lat = numpy.repeat([STATIONS[s][1] for s in names], 3)
    days = numpy.arange(60.0)
    raw = numpy.zeros((station.size, days.size))
    groups = [("transient", "isplines", {"t_start": 0.0, "t_end": 59.0, "scales": [8.0],
                                         "spacing": 4.0, "degree": 2})]
    design.write(path, days, station, component, raw, numpy.ones_like(raw), groups, lon=lon,
                 lat=lat)
    return design.load(path)


def native(path, parameters, shape, pooling, epsilon=0.1, scale=1.5, strength=2.0):
    from altar.models.gnss.native.Pooled import Pooled
    p = Pooled()
    p.parameters, p.offset = parameters, OFFSET
    p.design_file, p.scale, p.shape, p.strength = str(path), scale, shape, strength
    p.epsilon, p.length, p.radius, p.pooling = epsilon, 30.0, 75.0, pooling
    p.initialize(rng=types.SimpleNamespace(rng=numpy.random.default_rng(0)))
    return p


def brute(p, x):
    """
    The log prior of one sample, its parameters only, straight from the formula
    """
    g = p.geometry
    total = 0.0
    for s in range(len(g.stations)):
        for pool in g.pools:
            for j in range(g.width):
                e = 0.0
                for k in range(len(g.stations)):
                    for c in pool:
                        b = g.block[k, c]
                        if b >= 0:
                            e += g.W[s, k] * x[b * g.width + j] ** 2
                r = numpy.sqrt(e + p.epsilon ** 2)
                total += r / p.scale if p.shape == "laplace" else \
                    p.strength * numpy.log1p(r / p.scale)
    return -total


def report(name, value, tolerance):
    good = value <= tolerance
    print(f"{name:58s} {'ok' if good else 'FAIL':4s} {value:.2e} (tolerance {tolerance:.0e})")
    return 0 if good else 1


def cpu_checks(path, P, rng):
    failures = 0
    # an isolated coefficient, with a vanishing smoothing
    p = native(path, P, "laplace", "components", epsilon=1e-9)
    x = numpy.zeros(P)
    x[7] = 2.5
    base = p.log_density(numpy.zeros(P))[0]
    failures += report("isolated coefficient costs |x| / scale",
                       abs(p.log_density(x)[0] - base + 2.5 / 1.5), 1e-6)
    q = native(path, P, "log", "components", epsilon=1e-9)
    base = q.log_density(numpy.zeros(P))[0]
    # for the log shape, the neighbors' terms don't add up to one: sum_s log(1 + sqrt(W_sk) x / b)
    g = q.geometry
    k = int(numpy.nonzero(g.block == 7 // g.width)[0][0])
    expected = 2.0 * numpy.log1p(numpy.sqrt(g.W[:, k]) * 2.5 / 1.5).sum()
    failures += report("isolated coefficient, log shape",
                       abs(q.log_density(x)[0] - base + expected), 1e-6)
    for shape in ("laplace", "log"):
        for pooling in ("components", "horizontal", "all"):
            p = native(path, P, shape, pooling)
            θ = numpy.zeros((3, OFFSET + P))
            θ[:, OFFSET:] = rng.laplace(0, 1.0, (3, P)) * (rng.random((3, P)) < 0.3)
            ll = numpy.zeros(3)
            p.eval_prior(θ, ll)
            e = abs(ll[0] - brute(p, θ[0, OFFSET:])) / abs(ll[0])
            failures += report(f"{shape:7s} {pooling:10s} log density vs formula", e, 1e-12)
            grad = numpy.zeros_like(θ)
            p.prior_gradient(θ, grad)
            h, picks = 1e-6, rng.choice(P, 25, replace=False)
            fd = []
            for j in picks:
                tp, tm = θ[:1].copy(), θ[:1].copy()
                tp[0, OFFSET + j] += h
                tm[0, OFFSET + j] -= h
                fd.append((p.log_density(tp[:, OFFSET:])[0] - p.log_density(tm[:, OFFSET:])[0])
                          / (2 * h))
            g = grad[0, OFFSET + picks]
            e = numpy.abs(numpy.array(fd) - g).max() / numpy.abs(g).max()
            failures += report(f"{shape:7s} {pooling:10s} gradient vs central differences",
                               e, 1e-6)
    return failures


def gpu_checks(path, P, rng, precision, tolerance):
    import altar
    import altar.cuda
    import cuda.core
    from altar.models.gnss.cuda.Pooled import Pooled
    failures = 0
    for shape in ("laplace", "log"):
        for pooling in ("components", "horizontal"):
            host = native(path, P, shape, pooling)
            dev = Pooled()
            dev.parameters, dev.offset = P, OFFSET
            for name in ("design_file", "scale", "shape", "strength", "epsilon", "length",
                         "radius", "pooling"):
                setattr(dev, name, getattr(host, name))
            app = types.SimpleNamespace(job=types.SimpleNamespace(gpuprecision=precision))
            dev.initialize(rng=types.SimpleNamespace(rng=numpy.random.default_rng(1)),
                           application=app)
            θ = numpy.zeros((4, OFFSET + P + 3))
            θ[:, OFFSET:OFFSET + P] = rng.laplace(0, 1.0, (4, P)) * (rng.random((4, P)) < 0.3)
            gθ = altar.cuda.matrix(source=θ, dtype=precision)
            ll = altar.cuda.vector(source=numpy.full(4, 3.0), dtype=precision)
            grad = altar.cuda.matrix(source=numpy.full(θ.shape, 7.0), dtype=precision)
            dev.eval_prior(gθ, ll)
            dev.prior_gradient(gθ, grad)
            cuda.core.Device().sync()
            want = numpy.full(4, 3.0)
            host.eval_prior(θ[:, :OFFSET + P], want)
            got = numpy.array(ll, dtype=float)
            failures += report(f"cuda {precision} {shape:7s} {pooling:10s} log density",
                               numpy.abs(got - want).max() / numpy.abs(want).max(), tolerance)
            wg = numpy.zeros((4, OFFSET + P))
            host.prior_gradient(θ[:, :OFFSET + P], wg)
            gg = numpy.array(grad, dtype=float)
            e = numpy.abs(gg[:, OFFSET:OFFSET + P] - wg[:, OFFSET:]).max() / numpy.abs(wg).max()
            # the other columns of {gradient} are not mine and must be left alone
            untouched = numpy.all(gg[:, :OFFSET] == 7.0) and numpy.all(gg[:, OFFSET + P:] == 7.0)
            failures += report(f"cuda {precision} {shape:7s} {pooling:10s} gradient",
                               e if untouched else numpy.inf, tolerance)
    return failures


def main():
    rng = numpy.random.default_rng(5)
    with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as scratch:
        path = pathlib.Path(scratch) / "trajectory.h5"
        d = case(path)
        P = d.parameters("transient")
        from altar.models.gnss import pooling
        print(pooling.load(path, P, 30.0, 75.0, "horizontal").describe())
        failures = cpu_checks(path, P, rng)
        if "--gpu" in sys.argv:
            sys.argv = sys.argv[:1]
            failures += gpu_checks(path, P, rng, "float64", 1e-12)
            failures += gpu_checks(path, P, rng, "float32", 1e-4)
    print("all ok" if not failures else f"{failures} failed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())


# end of file
