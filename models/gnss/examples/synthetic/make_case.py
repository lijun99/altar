#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# a small synthetic case for the gnss trajectory model: 3 stations x (east, north, up), 200 days,
# a trend, an annual term, a slow slip ramp over ~3 weeks at two stations, an 8 mm step at one,
# white noise (1 mm horizontal, 3 mm vertical), 3% of the days missing; writes trajectory.h5,
# truth.npy (the true coefficients) and prints the parameter counts for the pfg files
#     python make_case.py
import numpy

from altar.models.gnss import design

GROUPS = [("secular", "polynomial", {"order": 1, "t0": 100.0, "unit": 365.25}),
          ("seasonal", "sinusoid", {"periods": [365.25], "t0": 0.0}),
          ("transient", "isplines", {"t_start": 0.0, "t_end": 199.0, "scales": [8.0, 16.0, 32.0],
                                     "spacing": 4.0, "degree": 2}),
          ("steps", "steps", {})]


def main():
    rng = numpy.random.default_rng(11)
    days = numpy.arange(200.0)
    station = numpy.repeat(["ALBH", "SC02", "P435"], 3)
    component = numpy.tile(["east", "north", "up"], 3)
    B = station.size
    steps = numpy.full((B, 1), numpy.nan)
    steps[3:6, 0] = 60.0                                     # SC02, all components
    groups = [(n, k, dict(s, times=steps) if k == "steps" else s) for n, k, s in GROUPS]
    # the truth, coefficient by coefficient, block by block
    G = {n: design.build_group(n, k, s, days, B, s.get("times")) for n, k, s in groups}
    truth = {n: numpy.zeros((B, g.width)) for n, g in G.items()}
    truth["secular"][:, 0] = rng.normal(0, 3, B)
    truth["secular"][:, 1] = numpy.tile([5.0, 3.0, 0.5], 3)  # mm/yr
    truth["seasonal"][:] = rng.normal(0, 1.5, (B, 2))
    # the slow slip: southwestward, ~4 mm, a 16-day I-spline centered on day 120
    cols = G["transient"].columns
    j = numpy.argmin(numpy.abs(cols["center"] - 120) + 1e3 * (cols["scale"] != 16))
    truth["transient"][0, j], truth["transient"][1, j] = -3.0, -2.5     # ALBH east, north
    truth["transient"][3, j], truth["transient"][4, j] = -2.0, -1.5     # SC02
    truth["steps"][3:6, 0] = [8.0, -4.0, 2.0]
    clean = sum((g.G[None] if g.shared else g.G) @ truth[n][..., None]
                for n, g in G.items())[..., 0]
    sigma = numpy.tile([1.0, 1.0, 3.0], 3)[:, None] * numpy.ones((B, days.size))
    raw = clean + sigma * rng.standard_normal(clean.shape)
    raw[rng.random(raw.shape) < 0.03] = numpy.nan
    lon = numpy.repeat([-123.49, -123.02, -123.71], 3)       # for the pooled prior
    lat = numpy.repeat([48.39, 48.55, 48.05], 3)
    design.write("trajectory.h5", days, station, component, raw, sigma, groups, lon=lon,
                 lat=lat, reference="2016-01-01")
    numpy.save("truth.npy", numpy.concatenate([truth[n].ravel() for n, _, _ in groups]))
    d = design.load("trajectory.h5")
    for line in d.describe():
        print(line)
    print(f"observations = {d.observations}")
    for g in d.groups:
        print(f"{g.name}.count = {d.blocks * g.width}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# end of file
