#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

"""
Check the device operator of the gnss slip model ({altar.models.gnss.cuda.Slip.Operator})
against the host's on the small random case of {slip_numerics}: the solves against the dense
solve, and the perturbations against N(0, A); without and with the noise correlated between
stations. Needs a gpu

    python slip_cuda.py
"""

import numpy

import altar.cuda  # noqa: F401
from altar.models.gnss import tdslip as slips
from altar.models.gnss.cuda.Slip import Operator
from altar.bayesian.samplers.native.Perturbation import pcg

from slip_numerics import case, dense


def main():
    rng = numpy.random.default_rng(5)
    d, greens, cen, noise = case(rng)
    nuisance = {"offset": 10.0, "trend": 2.0, "seasonal": 3.0}
    groups = ["transient_s16", "transient_s32"]
    failures = 0
    for spatial in (None, {c: {"c0": 0.3, "ell_km": 200.0} for c in slips.COMPONENTS}):
        s = slips.Slip(d, groups, nuisance, greens, cen, noise=noise, sigma0=5.0, length=20.0,
                       rho=0.5, spatial=spatial)
        host = s.operator(0.7)
        device = Operator(host)
        A, _ = dense(host)
        b = numpy.vstack([host.rhs(), rng.normal(0, 1, (4, s.size))])
        x_dense = numpy.linalg.solve(A, b.T).T
        x_host, it_host, _ = pcg(host.apply, host.precondition, b, 1e-12, 5000)
        x_dev, it_dev, res = device.solve(b, 1e-12, 5000)
        e = device.perturb(20000, rng)
        cov = e.T @ e / e.shape[0]
        error = numpy.abs(x_dev - x_dense).max() / numpy.abs(x_dense).max()
        checks = {
            "solve against dense": error < 1e-8,
            "iterations as the host": abs(it_dev - it_host) <= 2,
            "perturbations ~ N(0, A)": (numpy.linalg.norm(cov - A) / numpy.linalg.norm(A)) < 0.05,
        }
        label = "with" if spatial else "without"
        for name, passed in checks.items():
            failures += not passed
            print(f"{label:7s} correlated noise between stations, {s.size:4d} parameters: "
                  f"{name:24s} {'ok' if passed else 'FAILED'}")
        print(f"        cg: {it_dev} iterations on the device, {it_host} on the host; "
              f"error {error:.1e}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

# end of file
