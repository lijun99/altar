#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check the numerics of the gnss slip model ({altar.models.gnss.tdslip}) on a small random case, small
enough to form its posterior precision A densely: A symmetric positive definite, the conjugate-
gradient solve with the preconditioner against the dense solve, the perturbations N(0, A), the log
density against A (its gradient is b - A theta), and the data constant of a block against the
marginal gaussian of its observations; without and with the noise correlated between stations

    python slip_numerics.py
"""

import numpy
from scipy.stats import multivariate_normal

from altar.models.gnss import design as designs
from altar.models.gnss import tdslip as slips
from altar.bayesian.samplers.native.Perturbation import pcg


def case(rng, stations=6, days=120):
    """
    A small case: offset, trend, an annual term, two I-spline scales; random Green's functions
    """
    station = numpy.repeat([f"S{i}" for i in range(stations)], 3)
    component = numpy.tile(list(slips.COMPONENTS), stations)
    B = station.size
    t = numpy.arange(float(days))
    groups = [("offset", "polynomial", {"order": 0, "t0": 0.0, "unit": 365.25}),
              ("trend", "polynomial", {"order": 1, "first": 1, "t0": days / 2, "unit": 365.25}),
              ("seasonal", "sinusoid", {"periods": [365.25], "t0": 0.0}),
              ("transient_s16", "isplines", {"t_start": 0.0, "t_end": days - 1.0, "scales": [16.0],
                                             "spacing": 16.0, "degree": 2}),
              ("transient_s32", "isplines", {"t_start": 0.0, "t_end": days - 1.0, "scales": [32.0],
                                             "spacing": 32.0, "degree": 2})]
    built = [designs.build_group(n, k, s, t, B) for n, k, s in groups]
    sigma = numpy.tile([1.0, 1.0, 3.0], stations)[:, None] * numpy.ones((B, days))
    raw = rng.normal(0, 5, (B, days))
    raw[rng.random(raw.shape) < 0.05] = numpy.nan
    mask = numpy.isfinite(raw)
    data = numpy.where(mask, raw / sigma, 0.0)
    with numpy.errstate(divide="ignore"):
        weight = numpy.where(mask, 1 / sigma, 0.0)
    lat = numpy.repeat(rng.uniform(40, 41, stations), 3)
    lon = numpy.repeat(rng.uniform(-124, -123, stations), 3)
    d = designs.Design(t, station, component, weight, built, data=data.ravel(), mask=mask.ravel(),
                       lon=lon, lat=lat)
    N = 4
    greens = rng.normal(0, 0.05, (B, 2 * N))
    centroids = rng.uniform(0, 50, (N, 3))
    rows = [{"w": 1.0, "amps_mm": [0.7]} for _ in range(B)]
    return d, greens, centroids, ([20.0], rows)


def dense(op):
    n = op.size
    A = op.apply(numpy.identity(n))
    return 0.5 * (A + A.T), numpy.abs(A - A.T).max() / numpy.abs(A).max()


def main():
    rng = numpy.random.default_rng(3)
    d, greens, cen, noise = case(rng)
    nuisance = {"offset": 10.0, "trend": 2.0, "seasonal": 3.0}
    groups = ["transient_s16", "transient_s32"]
    failures = 0
    for spatial in (None, {c: {"c0": 0.3, "ell_km": 200.0} for c in slips.COMPONENTS}):
        s = slips.Slip(d, groups, nuisance, greens, cen, noise=noise, sigma0=5.0, length=20.0,
                       rho=0.5, spatial=spatial)
        op = s.operator(1.0)
        A, asym = dense(op)
        eig = numpy.linalg.eigvalsh(A)
        b = op.rhs()
        x_dense = numpy.linalg.solve(A, b)
        x_cg, it, res = pcg(op.apply, op.precondition, b[None], 1e-12, 5000)
        e = op.perturb(20000, rng)
        cov = e.T @ e / e.shape[0]
        theta = x_dense + rng.normal(0, 1, s.size)
        v = rng.normal(0, 1, s.size)
        f = lambda th: s.log_prior(th[None])[0] + s.log_likelihood(th[None])[0]
        fd = (f(theta + 1e-4 * v) - f(theta - 1e-4 * v)) / 2e-4
        an = v @ (b - A @ theta)
        checks = {
            "symmetric": asym < 1e-12,
            "positive definite": eig.min() > 0,
            "cg against dense": (numpy.abs(x_cg[0] - x_dense).max()
                                 / numpy.abs(x_dense).max()) < 1e-8,
            "perturbations ~ N(0, A)": (numpy.linalg.norm(cov - A) / numpy.linalg.norm(A)) < 0.05,
            "gradient b - A theta": abs(fd - an) / max(abs(an), 1) < 1e-5,
        }
        if spatial is None:
            # the data constant of a block against the marginal gaussian of its observations
            bk = 0
            ok = d.weight[bk] > 0
            sig = 1 / d.weight[bk][ok]
            y = d.data.reshape(d.blocks, d.days)[bk][ok] * sig
            K = slips.block_covariance(sig, d.time[ok], noise[1][bk], noise[0])
            X = numpy.hstack([g.G for g in d.groups])[ok]
            Sn = numpy.diag(numpy.concatenate([numpy.full(g.width, (nuisance[g.name]) ** 2)
                                               for g in d.groups if g.name not in groups]))
            th = rng.normal(0, 1, s.J)
            direct = multivariate_normal(mean=X[:, s.tcols] @ th,
                                         cov=K + X[:, s.ncols] @ Sn @ X[:, s.ncols].T).logpdf(y)
            ours = -0.5 * th @ s.M[bk] @ th + s.h[bk] @ th + s.c[bk]
            checks["block marginal"] = abs(direct - ours) < 1e-8 * abs(direct)
        label = "with" if spatial else "without"
        for name, passed in checks.items():
            failures += not passed
            print(f"{label:7s} correlated noise between stations, {s.size:4d} parameters: "
                  f"{name:24s} {'ok' if passed else 'FAILED'}")
        print(f"        cg: {it} iterations")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

# end of file
