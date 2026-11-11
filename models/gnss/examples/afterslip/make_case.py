#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# a synthetic afterslip case for the gnss slip model, unlike the subduction slow slip the model was
# built for: a vertical strike-slip fault, 60 km long and 20 km deep, in 150 triangles; a 1 m
# coseismic rupture above 8 km on day 365, then a year of afterslip below it, mostly right-lateral,
# growing as log(1 + t / 10 d); 25 stations within 60 km, 730 days, offsets, trends, annual and
# semiannual terms, white noise (1 mm horizontal, 3 mm vertical) plus a process correlated over
# 30 days, 2 % of the days missing. The I-splines of the slip start at the earthquake, so the year of
# data before it pins the offsets and trends. Writes the model's inputs: trajectory.h5 (the
# coseismic step as a per-block steps group, nuisance), greens.h5 (cutde, triangular dislocations
# in a half-space) and noise.json; truth.npz; and prints the counts for afterslip.pfg
#     python make_case.py
import json
import numpy
import h5py
import cutde.halfspace as hs

from altar.models.gnss import design
from altar.models.gnss.tdslip import block_covariance

LAT0, LON0 = 35.0, -117.0
KM_PER_DEG = 111.195
DAYS = 730
T_EQ = 365.0
# I-spline scales (days) and the span after the earthquake each one covers
SCALES = ((4.0, 40.0), (8.0, 80.0), (16.0, 400.0), (32.0, 400.0), (64.0, 400.0), (128.0, 400.0))


def fault():
    """
    The triangles (N x 3 vertices x 3, km, z up) and their centroids (N x 3, x, y, depth)
    """
    ys, zs = numpy.linspace(-30, 30, 16), numpy.linspace(0, -20, 6)
    tris = []
    for i in range(15):
        for k in range(5):
            a, b = [0, ys[i], zs[k]], [0, ys[i + 1], zs[k]]
            c, d = [0, ys[i + 1], zs[k + 1]], [0, ys[i], zs[k + 1]]
            tris += [[a, b, c], [a, c, d]]
    tris = numpy.array(tris)
    cen = tris.mean(axis=1)
    return tris, numpy.column_stack([cen[:, 0], cen[:, 1], -cen[:, 2]])


def main():
    rng = numpy.random.default_rng(5)
    tris, cen = fault()
    N = len(tris)
    # the stations, away from the trace
    xy = []
    while len(xy) < 25:
        p = rng.uniform(-60, 60, 2)
        if abs(p[0]) > 3:
            xy.append(p)
    xy = numpy.array(xy)
    names = [f"S{i:02d}" for i in range(len(xy))]
    obs = numpy.column_stack([xy, numpy.zeros(len(xy))])
    # displacement per unit slip: (station, 3 east north up, 2 strike dip, patch)
    D = hs.disp_matrix(obs_pts=obs, tris=tris, nu=0.25)          # (S, 3, N, 3)
    gf = numpy.ascontiguousarray(D[:, :, :, :2].transpose(0, 1, 3, 2))
    # the truth: coseismic slip above 8 km, afterslip below, centered at y = 5 km, 12 km deep
    depth, y = cen[:, 2], cen[:, 1]
    coseismic = 1000.0 * numpy.exp(-(y / 15) ** 2) * (depth < 8)                 # mm, strike
    weight = numpy.exp(-((y - 5) / 12) ** 2 - ((depth - 12) / 5) ** 2)
    final = numpy.stack([120.0 * weight, 20.0 * weight])                        # mm, strike, dip
    days = numpy.arange(float(DAYS))
    tt = numpy.clip(days - T_EQ, 0, None)
    shape = numpy.log1p(tt / 10.0) / numpy.log1p((DAYS - 1 - T_EQ) / 10.0)
    slip_t = final[:, :, None] * shape[None, None, :]                            # (2, N, days)
    # the blocks: a station and a component each
    station = numpy.repeat(names, 3)
    component = numpy.tile(["east", "north", "up"], len(names))
    B = station.size
    sta = numpy.repeat(numpy.arange(len(names)), 3)
    comp = numpy.tile(numpy.arange(3), len(names))
    G = gf[sta, comp]                                                            # (B, 2, N)
    transient = numpy.einsum("bkn,knt->bt", G, slip_t)
    step = numpy.einsum("bn,n->b", G[:, 0], coseismic)
    # the trajectory groups, the coseismic step per block
    steps = numpy.full((B, 1), T_EQ)
    groups = [("offset", "polynomial", {"order": 0, "t0": 0.0, "unit": 365.25}),
              ("trend", "polynomial", {"order": 1, "first": 1, "t0": DAYS / 2, "unit": 365.25}),
              ("seasonal", "sinusoid", {"periods": [365.25, 182.625], "t0": 0.0}),
              ("steps", "steps", {"times": steps})]
    # the temporal basis of the slip, from the earthquake on: short I-splines only early, for the
    # fast start of the afterslip, longer ones over the whole year
    for s, end in SCALES:
        groups.append((f"transient_s{int(s)}", "isplines",
                       {"t_start": T_EQ, "t_end": min(T_EQ + end, DAYS - 1.0), "scales": [s],
                        "spacing": s / 4, "degree": 2}))
    built = {n: design.build_group(n, k, st, days, B, st.get("times")) for n, k, st in groups}
    seasonal = rng.normal(0, 1.5, (B, 4)) @ built["seasonal"].G.T
    nuisance = (rng.normal(0, 50, B)[:, None]
                + rng.normal(0, 3, B)[:, None] * (days - DAYS / 2)[None, :] / 365.25 + seasonal
                + step[:, None] * (days >= T_EQ)[None, :])
    # the noise: white, 1 mm horizontal and 3 mm vertical, plus 30-day correlated
    sigma = numpy.where(comp[:, None] == 2, 3.0, 1.0) * numpy.ones((B, DAYS))
    amps = numpy.where(comp == 2, 2.0, 0.7)
    rows = [{"station": str(station[b]), "component": str(component[b]), "w": 1.0,
             "amps_mm": [float(amps[b])]} for b in range(B)]
    noise = numpy.empty((B, DAYS))
    for b in range(B):
        K = block_covariance(sigma[b], days, rows[b], [30.0])
        noise[b] = numpy.linalg.cholesky(K) @ rng.standard_normal(DAYS)
    raw = transient + nuisance + noise
    raw[rng.random(raw.shape) < 0.02] = numpy.nan
    lat = LAT0 + xy[sta, 1] / KM_PER_DEG
    lon = LON0 + xy[sta, 0] / (KM_PER_DEG * numpy.cos(numpy.radians(LAT0)))
    design.write("trajectory.h5", days, station, component, raw, sigma, groups, lon=lon, lat=lat,
                 reference="2020-01-01")
    with h5py.File("greens.h5", "w") as h:
        h["stations"] = numpy.array(names, dtype="S")
        h["gf"] = gf
        h["centroids_km"] = cen
        h.attrs["layout"] = "station, component (east, north, up), rake (strike, dip), patch"
    with open("noise.json", "w") as f:
        json.dump({"taus_days": [30.0], "blocks": rows}, f, indent=1)
    numpy.savez("truth.npz", slip=slip_t, final=final, coseismic=coseismic, cen=cen, tris=tris,
                days=days, t_eq=T_EQ)
    J = sum(design.build_group(n, k, st, days, B).width for n, k, st in groups if k == "isplines")
    print(f"{B} blocks x {DAYS} days = {B * DAYS} observations; {N} patches, {J} basis functions:"
          f" slip count {2 * N * J}")


if __name__ == "__main__":
    main()

# end of file
