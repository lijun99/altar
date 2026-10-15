#!/usr/bin/env python3
# -*- python -*-
# -*- coding: utf-8 -*-
#
# (c) 2013-present parasim inc
# (c) 2010-present california institute of technology
# all rights reserved
#

# a Pedernales-like synthetic (after Gombert et al. 2018): a slab whose dip steepens with depth,
# slip along and across the plate convergence, stations on land only, bandpassed waveforms, static
# gps and an InSAR-like line of sight; the mesh and green's functions from csi on a pyrocko store,
# the arrival times from this directory's prototype; needs csi (TriangularPatchesKin) and pyrocko:
#     python pedernales.py --store-dir=/path/to/stores --output=pedernales
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
parser.add_argument("--store-id", default="crust2_2hz_d0")
parser.add_argument("--output", default="pedernales")
parser.add_argument("--cells", type=int, default=8, help="cells along strike and down dip")
parser.add_argument("--refine", type=int, default=10)
parser.add_argument("--points", type=int, default=3)
parser.add_argument("--Nt", type=int, default=85)
parser.add_argument("--duration", type=float, default=160.0)
parser.add_argument("--bandpass", type=float, nargs=2, default=[0.015, 0.08])
parser.add_argument("--noise", type=float, default=0.005, help="waveform noise, m")
parser.add_argument("--gps-noise", type=float, default=0.005, help="static gps noise, m")
parser.add_argument("--insar-noise", type=float, default=0.01, help="line of sight noise, m")
parser.add_argument("--groups", type=int, default=4, help="velocity groups along strike and down dip")
options = parser.parse_args()

import csi

os.makedirs(options.output, exist_ok=True)
os.chdir(options.output)
rng = numpy.random.default_rng(2016)
lon0, lat0 = -80.0, 0.0

# the slab: strike N30E, its dip from 10 degrees above 10 km to 27 at 50 km, its top at 5 km
strike, length, width = 30.0, 160.0, 160.0
dip = lambda z: numpy.clip(10.0 + 17.0 * (z - 10.0) / 40.0, 10.0, 27.0)
fault = csi.TriangularPatchesKin("pedernales", lon0=lon0, lat0=lat0, verbose=False)
fault.buildCurvedFault(lon=-80.6, lat=-0.8, depth=5.0, strike=strike, dip=dip, length=length,
                       width=width, nstrike=options.cells, ndip=options.cells)
fault.setSourcePoints(options.points)
Np = len(fault.patch)
# slip along the convergence of the Nazca plate, N83E, and across it
parallel = fault.convergenceRakes(83.0)
rakes = (parallel + 90.0, parallel)
print(f"slab: {Np} patches, depth {fault.Vertices[:, 2].min():.0f}-{fault.Vertices[:, 2].max():.0f} km,"
      f" rake along the convergence {parallel.min():.0f}-{parallel.max():.0f} degrees")

# the frame: along strike, and east of the trench, across it
φ = numpy.radians(strike)
x0, y0 = fault.ll2xy(-80.6, -0.8)
along = lambda x, y: (x - x0) * numpy.sin(φ) + (y - y0) * numpy.cos(φ)
across = lambda x, y: (x - x0) * numpy.cos(φ) - (y - y0) * numpy.sin(φ)
point = lambda a, c: (x0 + a * numpy.sin(φ) + c * numpy.cos(φ), y0 + a * numpy.cos(φ) - c * numpy.sin(φ))
coast = 45.0
sources = fault.sources_xyz[:, :2]


def land(count, near, far, spread=20.0):
    """
    {count} places on land, {near} to {far} km east of the coast, at least 1 km from every source
    """
    places = []
    while len(places) < count:
        a, c = rng.uniform(-spread, length + spread), coast + rng.uniform(near, far)
        x, y = point(a, c)
        if numpy.hypot(sources[:, 0] - x, sources[:, 1] - y).min() >= 1.0:
            places.append((x, y))
    return numpy.array(places)


def stations(name, places, prefix):
    names = [f"{prefix}{k:03d}" for k in range(len(places))]
    data = csi.seismic(name, lon0=lon0, lat0=lat0)
    data.setStat(names, *fault.xy2ll(places[:, 0], places[:, 1]), loc_format="LL")
    return data


# the networks: high-rate gps and strong motion, static gps, and an InSAR-like grid
kinematic = stations("kinematic", land(18, 2.0, 120.0), "K")
gps = stations("gps", land(30, 2.0, 150.0), "G")
ga, gc = numpy.meshgrid(numpy.linspace(-10, length + 10, 12), coast + numpy.linspace(3, 100, 10), indexing="ij")
insar_xy = numpy.array([point(a, c) for a, c in zip(ga.ravel(), gc.ravel())])
insar_xy = insar_xy[[numpy.hypot(sources[:, 0] - x, sources[:, 1] - y).min() >= 1.0 for x, y in insar_xy]]
insar = stations("insar", insar_xy, "I")
los = numpy.array([0.62, -0.11, 0.78])  # east, north, up: a descending track
print(f"stations: {len(kinematic.sta_name)} kinematic, {len(gps.sta_name)} gps, {len(insar.sta_name)} line of sight")

# the green's functions
engine = csi.pyrockoengine(options.store_id, options.store_dir, delta=1.0)
fault.setMu(engine.material)
npts = int(round(options.duration / engine.delta))
start = time.perf_counter()
fault.buildKinGFs(kinematic, engine, Dt=1.0, npts=npts, rakes=rakes, points=True,
                  bandpass=tuple(options.bandpass))
fault.buildBigGD(kinematic, Nt=options.Nt, Dt=1.0)
Ggps = fault.buildStaticGFs(gps, engine, rakes=rakes, points=True)
Gins = fault.buildStaticGFs(insar, engine, rakes=rakes, points=True)
Glos = numpy.einsum("c,scp->sp", los, Gins.reshape(len(insar.sta_name), 3, -1))
print(f"green's functions: {time.perf_counter() - start:.0f} s; big G {fault.bigG.shape},"
      f" static {Ggps.shape[0]} gps + {Glos.shape[0]} los, mu {fault.mu.min() / 1e9:.1f}-{fault.mu.max() / 1e9:.1f} GPa")

# the true model: an asperity near the hypocenter in the north, a larger one to the south; the
# rise times follow the slip; a velocity for each of the groups x groups blocks of cells
uvc = numpy.array([fault.uv[f].mean(0) for f in fault.Faces])
a = numpy.exp(-((uvc - [118.0, 60.0]) ** 2).sum(1) / (2 * 15.0 ** 2))
b = numpy.exp(-((uvc - [55.0, 75.0]) ** 2).sum(1) / (2 * 18.0 ** 2))
slip_parallel = 0.2 + 3.0 * a + 6.0 * b
slip_across = 0.3 * (a - b)
risetime = 4.0 + 6.0 * (slip_parallel - slip_parallel.min()) / (slip_parallel.max() - slip_parallel.min())
n = options.groups
gi = numpy.clip((uvc[:, 0] / (length / n)).astype(int), 0, n - 1)
gj = numpy.clip((uvc[:, 1] / (width / n)).astype(int), 0, n - 1)
groups = gi * n + gj
centers = numpy.array([uvc[groups == g].mean(0) for g in range(n * n)])
per_group = 2.2 + 0.6 * centers[:, 0] / length + 0.3 * numpy.sin(centers[:, 1] / 30.0)
velocity = per_group[groups]
hypocenter = numpy.array([125.0, 55.0])
θ = numpy.concatenate([slip_across, slip_parallel, risetime, velocity, hypocenter])
fault.setHypoOnFault(*hypocenter)

# the slip history on the refined mesh, and the data
mesh = subdivide(fault.Vertices, fault.Faces, numpy.arange(Np), fault.uv, options.refine)
history = SlipHistory(mesh, fault.sources_uv, fault.sources_patch, Np, options.Nt, 1.0,
                      numpy.zeros(Np), fault.sources_per_patch)
start = time.perf_counter()
Mb = history(θ)
T = history.eikonal.T
print(f"slip history: eikonal on {mesh.vertices.shape[0]} vertices in {time.perf_counter() - start:.1f} s,"
      f" rupture ends at {T.max():.1f} s")
assert T.max() + risetime.max() < options.Nt, "Nt doesn't cover the rupture"
clean = fault.bigG @ Mb
waveforms = clean + options.noise * rng.standard_normal(clean.size)
slips = numpy.concatenate([slip_across, slip_parallel])
static_clean = numpy.concatenate([Ggps @ slips, Glos @ slips])
sigma = numpy.concatenate([numpy.full(Ggps.shape[0], options.gps_noise), numpy.full(Glos.shape[0], options.insar_noise)])
static = static_clean + sigma * rng.standard_normal(sigma.size)
Mw = 2 / 3 * (numpy.log10((fault.mu * numpy.asarray(fault.area) * 1e6 * numpy.hypot(slip_across, slip_parallel)).sum()) - 9.1)
print(f"Mw {Mw:.2f}; waveforms: {clean.size}, rms {numpy.sqrt((clean ** 2).mean()):.3f} m, noise {options.noise} m;"
      f" static: max |d| {numpy.abs(static_clean).max():.2f} m")

# save
import h5py
fault.saveKinematic("mesh.h5")
with h5py.File("mesh.h5", "a") as h5:
    h5["groups/velocity"] = groups
    h5["rakes"] = numpy.column_stack(rakes)
for name, values in [("kinematicG.data", waveforms), ("static.gf", numpy.vstack([Ggps, Glos])),
                     ("static.data", static)]:
    with h5py.File(f"{name}.h5", "w") as h5:
        h5[name] = values
numpy.savetxt("static.sigma.txt", sigma)
numpy.save("static.Cd.npy", numpy.diag(sigma ** 2))
with h5py.File("static.Cd.h5", "w") as h5:
    h5["static.Cd"] = numpy.diag(sigma ** 2)
numpy.savetxt("theta_true.txt", θ[None, :])
numpy.savetxt("theta_true_grouped.txt",
              numpy.concatenate([slip_across, slip_parallel, risetime, per_group, hypocenter])[None, :])
numpy.save("Mb_true.npy", Mb)
for name, data in [("kinematic", kinematic), ("gps", gps), ("insar", insar)]:
    numpy.savetxt(f"stations_{name}.txt", numpy.column_stack([data.lon, data.lat]), header=" ".join(data.sta_name))
print(f"saved to {os.getcwd()}")


# end of file
