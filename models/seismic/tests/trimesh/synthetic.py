#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# a synthetic kinematic model on a triangulated fault: the mesh and the green's functions from csi,
# on a pyrocko store, the arrival times from this directory's prototype, and the data; checks the
# assembled G against a direct simulation of the rupture; needs csi (with TriangularPatchesKin),
# pyrocko and a store, e.g. chile_70km_crust:
#     python synthetic.py --store-dir=/path/to/stores --output=synthetic
import argparse
import os
import sys
import time
import numpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh import subdivide
from slip import SlipHistory

parser = argparse.ArgumentParser()
parser.add_argument("--store-dir", required=True)
parser.add_argument("--store-id", default="chile_70km_crust")
parser.add_argument("--output", default="synthetic")
parser.add_argument("--nstrike", type=int, default=8)
parser.add_argument("--ndip", type=int, default=6)
parser.add_argument("--refine", type=int, default=10, help="eikonal triangles per patch edge")
parser.add_argument("--points", type=int, default=3, help="source point divisions per patch edge")
parser.add_argument("--Nt", type=int, default=60)
parser.add_argument("--Dt", type=float, default=1.0)
parser.add_argument("--duration", type=float, default=160.0)
parser.add_argument("--noise", type=float, default=0.01, help="data noise, m")
parser.add_argument("--network", choices=["ring", "grid"], default="ring",
                    help="10 stations around the fault, or a grid over and around it")
parser.add_argument("--grid", type=int, default=8, help="stations along each side of the grid")
parser.add_argument("--margin", type=float, default=20.0, help="the grid past the fault, km")
parser.add_argument("--min-distance", type=float, default=12.0,
                    help="the least horizontal distance of a station from a source, km")
parser.add_argument("--points-gf", action="store_true",
                    help="green's functions from the source points of each patch, not its centroid")
parser.add_argument("--delta", type=float, default=None, help="the sampling interval of the data, s")
parser.add_argument("--truth", choices=["v1", "v2"], default="v1",
                    help="one asperity, or two, with rise times and rupture velocities that vary more")
options = parser.parse_args()

import csi

os.makedirs(options.output, exist_ok=True)
os.chdir(options.output)
rng = numpy.random.default_rng(0)

# the fault: 80 km along strike (north), 60 km down dip at 20 degrees, its top at 5 km
fault = csi.TriangularPatchesKin("synthetic", lon0=-71.5, lat0=-31.5, verbose=False)
fault.buildPlanarFault(lon=-71.8, lat=-31.9, depth=5.0, strike=0.0, dip=20.0, length=80.0,
                       width=60.0, nstrike=options.nstrike, ndip=options.ndip)
fault.setSourcePoints(options.points)
Np = len(fault.patch)
print(f"fault: {Np} patches, {fault.sources_uv.shape[0]} source points")

# the stations, at least {min_distance} from every source of the green's functions
centers = numpy.array([fault.getcenter(p)[:2] for p in fault.patch])
sources = fault.sources_xyz[:, :2] if options.points_gf else centers
x0, y0 = fault.Vertices[:, 0].min(), fault.Vertices[:, 1].min()
x1, y1 = fault.Vertices[:, 0].max(), fault.Vertices[:, 1].max()
if options.network == "ring":
    xs, ys = numpy.meshgrid(x0 + numpy.array([-25.0, 15.0, 40.0, 80.0]),
                            y0 + numpy.array([-25.0, 40.0, 105.0]), indexing="ij")
else:
    m = options.margin
    xs, ys = numpy.meshgrid(numpy.linspace(x0 - m, x1 + m, options.grid),
                            numpy.linspace(y0 - m, y1 + m, options.grid), indexing="ij")
xs, ys = xs.ravel(), ys.ravel()
far = numpy.array([numpy.hypot(sources[:, 0] - x, sources[:, 1] - y).min() for x, y in zip(xs, ys)])
keep = far >= options.min_distance
xs, ys, far = xs[keep], ys[keep], far[keep]
names = [f"S{k:02d}" for k in range(len(xs))]
data = csi.seismic("synthetic", lon0=-71.5, lat0=-31.5)
# csi's setStat converts XY with ll2xy, so give it lon, lat
data.setStat(names, *fault.xy2ll(xs, ys), loc_format="LL")
print(f"stations: {len(names)}, nearest {far.min():.1f} km from a source")

# the green's functions
engine = csi.pyrockoengine(options.store_id, options.store_dir, delta=options.delta)
npts = int(round(options.duration / engine.delta))
fault.setMu(engine.material)
start = time.perf_counter()
fault.buildKinGFs(data, engine, Dt=options.Dt, npts=npts, points=options.points_gf)
fault.buildBigGD(data, Nt=options.Nt, Dt=options.Dt)
print(f"green's functions: {time.perf_counter() - start:.1f} s; big G {fault.bigG.shape},"
      f" mu {fault.mu.min() / 1e9:.1f}-{fault.mu.max() / 1e9:.1f} GPa")

# the true model: a dip-slip asperity, a little strike slip, rise times, rupture velocities
uvc = numpy.array([fault.uv[f].mean(0) for f in fault.Faces])
if options.truth == "v1":
    asperity = numpy.exp(-((uvc - [45.0, 30.0]) ** 2).sum(1) / (2 * 15.0 ** 2))
    strikeslip = 0.3 * asperity * numpy.sin(uvc[:, 0] / 20.0)
    dipslip = 0.2 + 4.0 * asperity
    risetime = 6.0 + 4.0 * asperity
    velocity = 2.6 + 0.4 * numpy.cos(uvc[:, 1] / 25.0)
    hypocenter = numpy.array([25.0, 35.0])
else:
    # two asperities; the rise times follow the slip; the rupture slows along strike
    a = numpy.exp(-((uvc - [22.0, 22.0]) ** 2).sum(1) / (2 * 12.0 ** 2))
    b = numpy.exp(-((uvc - [58.0, 38.0]) ** 2).sum(1) / (2 * 12.0 ** 2))
    strikeslip = 0.5 * (a - b)
    dipslip = 0.3 + 3.0 * a + 4.0 * b
    risetime = 4.0 + 7.0 * (dipslip - dipslip.min()) / (dipslip.max() - dipslip.min())
    velocity = 3.2 - 1.2 * uvc[:, 0] / 80.0 + 0.2 * numpy.sin(uvc[:, 1] / 10.0)
    hypocenter = numpy.array([15.0, 30.0])
θ = numpy.concatenate([strikeslip, dipslip, risetime, velocity, hypocenter])
fault.setHypoOnFault(*hypocenter)

# the slip history, with the arrival times on the refined mesh
mesh = subdivide(fault.Vertices, fault.Faces, numpy.arange(Np), fault.uv, options.refine)
history = SlipHistory(mesh, fault.sources_uv, fault.sources_patch, Np, options.Nt, options.Dt,
                      numpy.zeros(Np), fault.sources_per_patch)
start = time.perf_counter()
Mb = history(θ)
T = history.eikonal.T
print(f"slip history: eikonal on {mesh.vertices.shape[0]} vertices in {time.perf_counter() - start:.2f} s,"
      f" rupture ends at {T.max():.1f} s; Mb sums to {Mb.reshape(options.Nt, 2, Np).sum(0).sum(1)} m")
assert T.max() + risetime.max() < options.Nt * options.Dt, "Nt * Dt doesn't cover the rupture"

# the data
clean = fault.bigG @ Mb
observed = clean + options.noise * rng.standard_normal(clean.size)
fault.bigD = observed
print(f"data: {clean.size} observations, rms {numpy.sqrt((clean ** 2).mean()):.3f} m,"
      f" max |d| {numpy.abs(clean).max():.3f} m, noise {options.noise} m")

# the check: a direct simulation of the rupture, every source point with its own arrival time and
# triangle, against G Mb; with the points at their patch centroids, it isolates the time stepping
TI0 = (history.weights * T[history.corners]).sum(1)
p = fault.sources_patch
rake = numpy.degrees(numpy.arctan2(dipslip, strikeslip))[p]
amount = numpy.hypot(strikeslip, dipslip)[p]
area = numpy.asarray(fault.area)[p] / fault.sources_per_patch
stations = {"name": names, "lon": data.lon, "lat": data.lat}
for label, xyz in [("source points at their patch centroids", centers[p]),
                   ("source points where they are", fault.sources_xyz)]:
    depth = (numpy.array([fault.getcenter(q)[2] for q in fault.patch])[p]
             if label.startswith("source points at") else fault.sources_xyz[:, 2])
    lon, lat = fault.xy2ll(xyz[:, 0], xyz[:, 1])
    sources = []
    for k in range(p.size):
        _, _, _, _, _, strike, dip = fault.getpatchgeometry(int(p[k]), center=True)
        sources.append({"lon": lon[k], "lat": lat[k], "depth": depth[k],
                        "strike": numpy.degrees(strike), "dip": numpy.degrees(dip), "rake": rake[k],
                        "moment": fault.mu[p[k]] * amount[k] * area[k] * 1.0e6,
                        "time": TI0[k], "duration": risetime[p[k]], "anchor": -1.0})
    start = time.perf_counter()
    # with the same lead as the green's functions, cropped to the data
    pre = int(round(40.0 / engine.delta))
    direct = engine.synthetics(sources, stations, npts + pre, tstart=-pre * engine.delta)
    direct = direct[..., pre:].sum(0).ravel()
    rel = numpy.linalg.norm(direct - clean) / numpy.linalg.norm(direct)
    print(f"direct simulation, {label}: {time.perf_counter() - start:.1f} s,"
          f" |direct - G Mb| / |direct| = {rel:.4f}; noise / signal {options.noise * numpy.sqrt(clean.size) / numpy.linalg.norm(direct):.4f}")

# save
fault.saveKinematic("mesh.h5")
numpy.savetxt("theta_true.txt", θ[None, :])
numpy.save("Mb_true.npy", Mb)
numpy.savetxt("stations.txt", numpy.column_stack([data.lon, data.lat]), header=" ".join(names))
print(f"saved to {os.getcwd()}: mesh.h5, kinematicG.gf.h5, kinematicG.data.h5, theta_true.txt")


# end of file
