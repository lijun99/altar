#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# check the numerics of the gnss trajectory model on a small synthetic case with every kind of
# group: the cpu predictions against an explicit block-diagonal design matrix, the cpu data
# gradient against central differences of the log likelihood, and, with a gpu, the cuda
# residuals and data gradient against the cpu ones, in double and single precision
#     python trajectory_numerics.py [--gpu]
import pathlib
import sys
import tempfile
import types

import numpy


def case(path, rng):
    """
    A small case: 4 stations x 3 components, 120 days, a few missing, every kind of group
    """
    from altar.models.gnss import design
    days = numpy.arange(120.0)
    station = numpy.repeat(["AAAA", "BBBB", "CCCC", "DDDD"], 3)
    component = numpy.tile(["east", "north", "up"], 4)
    B = station.size
    raw = rng.standard_normal((B, days.size)).cumsum(axis=1)
    sigma = rng.uniform(0.5, 2.0, (B, days.size))
    raw[rng.random(raw.shape) < 0.05] = numpy.nan
    steps = numpy.full((B, 2), numpy.nan)
    steps[0:3, 0], steps[6:9, :] = 40.0, [[30.0, 80.0]] * 3
    groups = [("secular", "polynomial", {"order": 1, "t0": 60.0, "unit": 365.25}),
              ("seasonal", "sinusoid", {"periods": [365.25, 182.625], "t0": 0.0}),
              ("transient", "isplines", {"t_start": 0.0, "t_end": 119.0, "scales": [4.0, 16.0],
                                         "spacing": 2.0, "degree": 2}),
              ("steps", "steps", {"times": steps}),
              ("postseismic", "logarithmic", {"times": steps[:, :1], "tau": 10.0})]
    design.write(path, days, station, component, raw, sigma, groups, reference="2015-07-01")
    return design.load(path)


def dense(d):
    """
    The block-diagonal whitened design matrix, (observations x parameters), group by group
    """
    cols = []
    for g in d.groups:
        for b in range(d.blocks):
            Gb = g.G if g.shared else g.G[b]
            for j in range(g.width):
                c = numpy.zeros((d.blocks, d.days))
                c[b] = Gb[:, j] * d.weight[b]
                cols.append(c.ravel())
    return numpy.column_stack(cols)


def offsets(d):
    out, o = {}, 0
    for g in d.groups:
        out[g.name] = o
        o += d.blocks * g.width
    return out


def cpu_checks(d, rng):
    off = offsets(d)
    P = d.parameters()
    theta = rng.standard_normal((3, P))
    J = dense(d)
    failures = 0
    pred = d.predict(theta, off)
    e = numpy.abs(pred - theta @ J.T).max() / numpy.abs(pred).max()
    failures += report("cpu prediction vs block-diagonal matrix", e, 1e-12)
    # the data gradient of log L = -|J theta - d|^2 / 2
    r = pred - d.data
    grad = numpy.zeros_like(theta)
    d.adjoint(-r, off, grad)
    e = numpy.abs(grad + r @ J).max() / numpy.abs(grad).max()
    failures += report("cpu gradient vs -J^T r", e, 1e-12)
    loglik = lambda t: -0.5 * ((d.predict(t, off) - d.data) ** 2).sum()
    h, picks = 1e-5, rng.choice(P, 20, replace=False)
    fd = []
    for j in picks:
        tp, tm = theta[:1].copy(), theta[:1].copy()
        tp[0, j] += h
        tm[0, j] -= h
        fd.append((loglik(tp) - loglik(tm)) / (2 * h))
    e = numpy.abs(numpy.array(fd) - grad[0, picks]).max() / numpy.abs(grad[0, picks]).max()
    failures += report("cpu gradient vs central differences", e, 1e-6)
    return failures, theta, off


def gpu_checks(d, theta, off, precision, tolerance):
    import altar
    import altar.cuda
    from altar.models.gnss.cuda.Trajectory import Trajectory
    S, P = theta.shape
    data = numpy.tile(d.data, (S, 1))
    model = types.SimpleNamespace(
        design=d, precision=precision, parameters=P, offsets=off, psets_list=[],
        dataobs=types.SimpleNamespace(
            dataobs_batch=altar.cuda.matrix(source=data, dtype=precision)))
    impl = Trajectory().initialize(model=model, application=None)
    gθ = altar.cuda.matrix(source=theta, dtype=precision)
    pred = altar.cuda.matrix(shape=(S, d.observations), dtype=precision)
    impl.forward_model_batched(model=model, theta=gθ, prediction=pred)
    sync()
    want = d.predict(theta, off) - d.data
    got = numpy.array(pred, dtype=float)
    failures = report(f"cuda residuals vs cpu, {precision}",
                      numpy.abs(got - want).max() / numpy.abs(want).max(), tolerance)
    step = types.SimpleNamespace(theta=gθ,
                                 prior_gradient=altar.cuda.matrix(shape=(S, P), dtype=precision),
                                 data_gradient=altar.cuda.matrix(shape=(S, P), dtype=precision))
    impl.gradient(model=model, controller=None, step=step)
    grad = numpy.zeros_like(theta)
    d.adjoint(-want, off, grad)
    sync()
    got = numpy.array(step.data_gradient, dtype=float)
    failures += report(f"cuda gradient vs cpu, {precision}",
                       numpy.abs(got - grad).max() / numpy.abs(grad).max(), tolerance)
    return failures


def sync():
    """
    Wait for the device, before reading managed memory on the host
    """
    import cuda.core
    cuda.core.Device().sync()
    return


def report(name, value, tolerance):
    good = value <= tolerance
    print(f"{name:45s} {'ok' if good else 'FAIL':4s} {value:.2e} (tolerance {tolerance:.0e})")
    return 0 if good else 1


def main():
    rng = numpy.random.default_rng(3)
    with tempfile.TemporaryDirectory(dir=pathlib.Path.cwd()) as scratch:
        d = case(pathlib.Path(scratch) / "trajectory.h5", rng)
    for line in d.describe():
        print(line)
    failures, theta, off = cpu_checks(d, rng)
    if "--gpu" in sys.argv:
        sys.argv = sys.argv[:1]
        failures += gpu_checks(d, theta, off, "float64", 1e-12)
        failures += gpu_checks(d, theta, off, "float32", 1e-4)
    print("all ok" if not failures else f"{failures} failed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())


# end of file
