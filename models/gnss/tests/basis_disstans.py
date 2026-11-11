#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# check the basis functions of the gnss trajectory model against DISSTANS' mappings, in an
# environment with disstans (altar isn't needed: basis.py is loaded from its file)
#     python basis_disstans.py
import importlib.util
import pathlib

import numpy
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("basis", HERE.parent / "gnss" / "basis.py")
basis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basis)


def report(name, value, tolerance=1e-12):
    good = value <= tolerance
    print(f"{name:40s} {'ok' if good else 'FAIL':4s} {value:.2e}")
    return 0 if good else 1


def main():
    from disstans import models
    ref = pd.Timestamp("2015-07-01")
    tvec = pd.date_range("2015-05-01", "2016-08-31", freq="D")
    t = ((tvec - ref) / pd.Timedelta(days=1)).values.astype(float)
    failures = 0

    m = models.Polynomial(order=2, t_reference=ref, time_unit="D")
    failures += report("polynomial", numpy.abs(
        basis.polynomial(t, 2, 0.0, 1.0) - m.get_mapping(tvec).toarray()).max())

    m = models.Sinusoid(period=365.25, t_reference=ref, time_unit="D")
    failures += report("sinusoid", numpy.abs(
        basis.sinusoid(t, 365.25, 0.0) - m.get_mapping(tvec).toarray()).max())

    steps = ["2015-08-10", "2016-02-01"]
    m = models.Step(steps)
    ts = [(pd.Timestamp(s) - ref).days for s in steps]
    failures += report("step", numpy.abs(
        basis.step(t, ts) - m.get_mapping(tvec).toarray()).max())

    m = models.Logarithmic(tau=10.0, t_reference="2015-08-10", time_unit="D")
    failures += report("logarithmic", numpy.abs(
        basis.logarithmic(t, [ts[0]], 10.0) - m.get_mapping(tvec).toarray()).max())

    for scale in (4.0, 16.0, 32.0):
        n = 12
        m = models.ISpline(degree=2, scale=scale, t_reference=ref, time_unit="D",
                           num_splines=n)
        # DISSTANS places the centers one scale apart from its t_reference
        centers = scale * numpy.arange(n)
        failures += report(f"ispline, scale {scale:g} d", numpy.abs(
            basis.ispline(t, centers, scale, 2) - m.get_mapping(tvec).toarray()).max())

    print("all ok" if not failures else f"{failures} failed")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())


# end of file
