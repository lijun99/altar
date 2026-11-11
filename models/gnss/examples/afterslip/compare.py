#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# the samples of afterslip.pfg (results/step_final.h5) against the truth of make_case.py: the
# afterslip by the last day of every patch (correlation, coverage of the 90 % intervals) and the
# moment of the afterslip at a few days (posterior mean and 5-95 % against the truth)
#     python compare.py
import json
import h5py
import numpy

from altar.models.gnss import design as designs

MU = 30e9


def main():
    truth = numpy.load("truth.npz")
    d = designs.load("trajectory.h5")
    basis = numpy.hstack([g.G for g in d.groups if g.kind == "isplines"])         # (days, J)
    with h5py.File("results/step_final.h5") as h:
        X = h["ParameterSets/slip_sampling"][:]
    N, J, k = truth["cen"].shape[0], basis.shape[1], X.shape[0]
    X = X.reshape(k, 2 * N, J)
    tris = truth["tris"]
    area = 0.5 * numpy.linalg.norm(numpy.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]),
                                   axis=1) * 1e6                                    # m2
    # slip at day t since 20 days before the earthquake (truth.npz "t_eq"): dip rows first,
    # then strike
    ref = int(truth["t_eq"]) - 20
    so_far = lambda t: X @ (basis[t] - basis[ref])                                  # (k, 2N)
    out = {"samples": k, "patches": N, "basis_functions": J}
    last = len(truth["days"]) - 1
    S = so_far(last)
    for name, rows, true in (("strike", slice(N, 2 * N), truth["final"][0]),
                             ("dip", slice(0, N), truth["final"][1])):
        s = S[:, rows]
        lo, hi = numpy.percentile(s, [5, 95], axis=0)
        out[f"final_{name}"] = {
            "true_max_mm": round(float(true.max()), 1),
            "mean_max_mm": round(float(s.mean(axis=0).max()), 1),
            "correlation": round(float(numpy.corrcoef(s.mean(axis=0), true)[0, 1]), 3),
            "rms_error_mm": round(float(numpy.sqrt(numpy.mean((s.mean(axis=0) - true) ** 2))), 2),
            "coverage_90": round(float(numpy.mean((true >= lo) & (true <= hi))), 3)}
    moments = []
    for t in (370, 380, 395, 455, 545, last):
        m = MU * (so_far(t)[:, N:] * 1e-3 * area).sum(axis=1)
        true = MU * (truth["slip"][0, :, t] * 1e-3 * area).sum()
        p5, p95 = numpy.percentile(m, [5, 95])
        moments.append({"day": t, "true": float(f"{true:.4g}"), "mean": float(f"{m.mean():.4g}"),
                        "p5": float(f"{p5:.4g}"), "p95": float(f"{p95:.4g}"),
                        "covered": bool(p5 <= true <= p95)})
    out["strike_moment"] = moments
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

# end of file
